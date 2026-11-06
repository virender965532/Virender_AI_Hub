"""VERIFY tools (run_tests, run_linter) and the DANGEROUS run_command (queued for approval).

Command safety:
- Only commands whose leading tokens match ``CA_COMMAND_ALLOWLIST`` run.
- ``subprocess`` with ``shell=False``, ``cwd`` inside the workspace, a timeout
  that kills the whole process tree, and output truncated to the last N lines.
- Path-like arguments must stay inside the working directory.
- The child environment is stripped of variables that look like secrets
  (so test code cannot read ``OPENAI_API_KEY``), and output is redacted.
- With ``CA_USE_DOCKER=true`` commands run in ``docker run --network none``.

Verification never runs in the real workspace: ``VerifyEnvironment`` makes a
private temporary copy (without secrets or ignored folders), overlays the
agent's proposed edits and runs the checks there.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from ..schemas import CheckResult, ProposedChange
from ..workspace.sandbox import IGNORED_DIRS, Sandbox, SandboxError, is_secret_path, is_within, redact
from .registry import RiskLevel, Tool, ToolContext, ToolError

if TYPE_CHECKING:
    from ..config import CodeAssistantSettings
    from .edit_tools import WorkingSet

_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION|PRIVATE)", re.IGNORECASE)
_SAFE_ENV = {"PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR", "HOME",
             "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)",
             "LANG", "LC_ALL", "PYTHONPATH", "VIRTUAL_ENV", "NODE_PATH", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
             "USERNAME", "USER", "LOGNAME", "SHELL", "TERM"}
_BATCH_UNSAFE = re.compile(r'[&|<>^%!"\r\n]')
_COPY_MAX_FILES = 20000
_COPY_MAX_BYTES = 300 * 1024 * 1024
_FAILURE_LINE = re.compile(r"^(FAILED|ERROR)\s+(\S+)")
_LINT_LINE = re.compile(r"^(?P<path>[^:\s][^:]*):\d+(?::\d+)?:\s*(?P<code>[A-Z]+\d+|error|warning)?\s*(?P<msg>.*)$")
_NUMBERS = re.compile(r"\d+")


class CommandNotAllowedError(ToolError):
    """The command is not in ``CA_COMMAND_ALLOWLIST`` or has unsafe arguments."""


@dataclass
class CommandResult:
    argv: list[str]
    exit_code: int | None
    output: str
    timed_out: bool
    duration_ms: int

    def render(self) -> str:
        status = "TIMED OUT" if self.timed_out else f"exit code {self.exit_code}"
        return f"$ {shlex.join(self.argv)}\n[{status}, {self.duration_ms} ms]\n{self.output}"


def tail(text: str, lines: int) -> str:
    rows = text.splitlines()
    if len(rows) <= lines:
        return text
    return f"[truncated: showing the last {lines} of {len(rows)} lines]\n" + "\n".join(rows[-lines:])


def failure_signatures(output: str) -> list[str]:
    """Normalised failure identifiers (pytest FAILED/ERROR ids, linter path+code+message without numbers)."""
    signatures: set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        match = _FAILURE_LINE.match(line)
        if match:
            signatures.add(f"{match.group(1)} {match.group(2)}")
            continue
        lint = _LINT_LINE.match(line)
        if lint and lint.group("code"):
            path = lint.group("path").replace("\\", "/")
            signatures.add(f"{path}: {lint.group('code')} {_NUMBERS.sub('#', lint.group('msg'))[:120]}")
    return sorted(signatures)


def child_env() -> dict[str, str]:
    """A minimal environment without secrets."""
    env = {k: v for k, v in os.environ.items() if k.upper() in _SAFE_ENV and not _SECRET_ENV.search(k)}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "CI": "1", "NO_COLOR": "1", "PYTHONIOENCODING": "utf-8"})
    return env


class CommandRunner:
    """Runs allowlisted commands with timeouts, optionally inside Docker."""

    def __init__(self, settings: CodeAssistantSettings) -> None:
        self.settings = settings

    # --- validation ---------------------------------------------------------------------------

    @staticmethod
    def split(command: str | list[str]) -> list[str]:
        if isinstance(command, list):
            if not all(isinstance(a, str) for a in command):
                raise CommandNotAllowedError("command arguments must be strings")
            return list(command)
        try:
            return shlex.split(command)
        except ValueError as exc:
            raise CommandNotAllowedError(f"could not parse command: {exc}") from exc

    def matching_entry(self, argv: list[str]) -> tuple[str, ...] | None:
        if not argv:
            return None
        head = [Path(argv[0]).stem.lower() if i == 0 else a for i, a in enumerate(argv)]
        for entry in self.settings.command_allowlist:
            if len(head) >= len(entry) and [Path(entry[0]).stem.lower(), *entry[1:]] == head[:len(entry)]:
                return entry
        return None

    def validate(self, argv: list[str], cwd: Path) -> None:
        if not argv:
            raise CommandNotAllowedError("empty command")
        if os.sep in argv[0] or "/" in argv[0]:
            raise CommandNotAllowedError("use a bare executable name such as 'pytest', not a path")
        if self.matching_entry(argv) is None:
            allowed = ", ".join(" ".join(e) for e in self.settings.command_allowlist)
            raise CommandNotAllowedError(f"command not allowed: {argv[0]!r}. Allowed commands: {allowed}")
        for arg in argv[1:]:
            if "\x00" in arg:
                raise CommandNotAllowedError("NUL byte in argument")
            value = arg.split("=", 1)[1] if arg.startswith("-") and "=" in arg else arg
            if value.startswith("-"):
                continue
            pathish = value.split("::", 1)[0]
            if "/" in pathish or "\\" in pathish or os.path.isabs(pathish) or pathish.startswith("~"):
                if ".." in pathish.replace("\\", "/").split("/"):
                    raise CommandNotAllowedError(f"'..' is not allowed in arguments: {arg}")
                target = Path(pathish) if os.path.isabs(pathish) else cwd / pathish
                if pathish.startswith("~") or not is_within(target, cwd):
                    raise CommandNotAllowedError(f"argument points outside the workspace: {arg}")
                if is_secret_path(pathish):
                    raise CommandNotAllowedError(f"argument points at a secret file: {arg}")

    # --- execution ----------------------------------------------------------------------------

    def _resolve_executable(self, name: str, workspace_root: Path) -> str:
        """Prefer the workspace's own virtualenv, then this interpreter for ``python``, then PATH."""
        for venv in (".venv", "venv", "env"):
            for sub in ("Scripts", "bin"):
                for candidate in (name, f"{name}.exe"):
                    path = workspace_root / venv / sub / candidate
                    if path.is_file():
                        return str(path)
        if name.lower() in ("python", "python3"):
            return sys.executable
        found = shutil.which(name)
        if not found:
            raise ToolError(f"executable not found on PATH: {name}")
        return found

    def run(self, argv: list[str], *, cwd: Path, workspace_root: Path | None = None, timeout: float | None = None) -> CommandResult:
        """Validate and run a command; never raises for non-zero exit codes or timeouts."""
        self.validate(argv, cwd)
        timeout = timeout or self.settings.command_timeout
        if self.settings.use_docker:
            name = f"code-assistant-{uuid.uuid4().hex[:10]}"
            full = ["docker", "run", "--rm", "--name", name, "--network", "none", "--memory", "2g", "--cpus", "2",
                    "-e", "CI=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
                    "-v", f"{cwd}:/workspace", "-w", "/workspace", self.settings.docker_image, *argv]
            cleanup = ["docker", "rm", "-f", name]
        else:
            exe = self._resolve_executable(argv[0], workspace_root or cwd)
            if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")) and any(_BATCH_UNSAFE.search(a) for a in argv[1:]):
                raise CommandNotAllowedError("unsafe characters in arguments for a batch-file command")
            full = [exe, *argv[1:]]
            cleanup = None
        started = time.monotonic()
        kwargs: dict = {"cwd": str(cwd), "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                        "shell": False, "env": child_env() if not self.settings.use_docker else None}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        else:
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(full, **kwargs)  # noqa: S603 - allowlisted argv, no shell
        except OSError as exc:
            raise ToolError(f"could not start {argv[0]}: {exc}") from exc
        timed_out = False
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_tree(proc)
            out, _ = proc.communicate()
            if cleanup:
                subprocess.run(cleanup, capture_output=True, timeout=30, check=False)  # noqa: S603
        text = redact((out or b"").decode("utf-8", errors="replace").replace("\r\n", "\n"))
        return CommandResult(
            argv=argv, exit_code=None if timed_out else proc.returncode,
            output=tail(text, self.settings.output_tail_lines), timed_out=timed_out,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=15, check=False)  # noqa: S603,S607
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            proc.kill()
        except OSError:
            pass


# --- the private verify copy -------------------------------------------------------------------

class VerifyEnvironment:
    """A temporary copy of the workspace with the working set overlaid. Created lazily, removed by ``cleanup``."""

    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox
        self.path: Path | None = None
        self._synced: set[str] = set()
        self._lock = threading.Lock()

    def _ignore(self, directory: str, names: list[str]) -> set[str]:
        rel_dir = self.sandbox.relative(Path(directory))
        skipped = set()
        for name in names:
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            full = os.path.join(directory, name)
            if os.path.isdir(full):
                if name in IGNORED_DIRS or os.path.islink(full):
                    skipped.add(name)
            elif is_secret_path(rel) or (os.path.islink(full) and not is_within(full, self.sandbox.root)):
                skipped.add(name)
        return skipped

    def _check_size(self) -> None:
        files = total = 0
        for dirpath, dirnames, filenames in os.walk(self.sandbox.root):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
            files += len(filenames)
            for name in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
            if files > _COPY_MAX_FILES or total > _COPY_MAX_BYTES:
                raise ToolError("workspace is too large to copy for verification; run the tests yourself after approving")

    def prepare(self) -> Path:
        with self._lock:
            if self.path is None:
                self._check_size()
                base = Path(tempfile.mkdtemp(prefix="code-assistant-verify-"))
                target = base / self.sandbox.root.name
                shutil.copytree(self.sandbox.root, target, ignore=self._ignore, symlinks=True)
                self.path = target
            return self.path

    def sync(self, working_set: WorkingSet) -> Path:
        """Make the copy reflect the original workspace plus the current proposals."""
        root = self.prepare()
        contents = working_set.contents()
        with self._lock:
            for rel in self._synced - set(contents):  # proposals that were dropped: restore the original
                original = self.sandbox.root / rel
                target = root / rel
                if original.is_file():
                    shutil.copy2(original, target)
                elif target.exists():
                    target.unlink()
            for rel, text in contents.items():
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(target, "w", encoding="utf-8", newline="") as handle:
                    handle.write(text)
            self._synced = set(contents)
        return root

    def cleanup(self) -> None:
        with self._lock:
            if self.path is not None:
                shutil.rmtree(self.path.parent, ignore_errors=True)
                self.path = None
                self._synced = set()


# --- choosing commands -------------------------------------------------------------------------

def _first_allowed(runner: CommandRunner, candidates: Iterable[list[str]]) -> list[str] | None:
    for argv in candidates:
        if runner.matching_entry(argv) is not None:
            return argv
    return None


def test_command(runner: CommandRunner, root: Path, target: str | None = None) -> list[str] | None:
    """Pick the project's test command from the allowlist (pytest for Python, npm test for Node)."""
    has_python = any(root.glob("*.py")) or (root / "pyproject.toml").exists() or (root / "pytest.ini").exists() \
        or (root / "setup.cfg").exists() or (root / "tests").is_dir()
    candidates: list[list[str]] = []
    if has_python:
        extra: list[str] = ["-q", "--maxfail=20", "--no-header", "-p", "no:cacheprovider"]
        if target:
            if (root / target.split("::", 1)[0]).exists():
                extra.append(target)
            else:
                extra += ["-k", target]
        candidates += [["python", "-m", "pytest", *extra], ["pytest", *extra]]
    if (root / "package.json").exists():
        candidates.append(["npm", "test", "--silent"])
    return _first_allowed(runner, candidates)


def lint_command(runner: CommandRunner, root: Path, paths: list[str]) -> list[str] | None:
    py_paths = [p for p in paths if p.endswith((".py", ".pyi"))] if paths else ["."]
    if not py_paths:
        return None
    candidates = [["ruff", "check", "--no-cache", "--output-format", "concise", *py_paths], ["flake8", *py_paths]]
    for argv in candidates:
        if runner.matching_entry(argv) is None:
            continue
        if runner.settings.use_docker or shutil.which(argv[0]) or any(
            (root / v / s / argv[0]).exists() or (root / v / s / f"{argv[0]}.exe").exists()
            for v in (".venv", "venv") for s in ("Scripts", "bin")
        ):
            return argv
    return None


def run_check(ctx: ToolContext, name: str, argv: list[str] | None, *, cwd: Path) -> CheckResult:
    if argv is None:
        return CheckResult(name=name, command=[], exit_code=None, passed=True, skipped=True,
                           output=f"No allowlisted {name} command applies to this project.")
    result = ctx.runner.run(argv, cwd=cwd, workspace_root=ctx.sandbox.root)
    no_tests = name == "tests" and result.exit_code == 5 and "pytest" in " ".join(argv)
    passed = result.exit_code == 0 or no_tests
    check = CheckResult(
        name=name, command=argv, exit_code=result.exit_code, passed=passed and not result.timed_out,
        skipped=no_tests, timed_out=result.timed_out, output=result.output,
        new_failures=[] if passed else failure_signatures(result.output),
    )
    ctx.trace.emit("test_result", {"check": name, "command": shlex.join(argv), "passed": check.passed,
                                   "exit_code": result.exit_code, "timed_out": result.timed_out,
                                   "duration_ms": result.duration_ms, "output": result.output[-3000:]})
    return check


# --- tool handlers -----------------------------------------------------------------------------

def run_tests(ctx: ToolContext, path_or_pattern: str | None = None) -> str:
    target = (path_or_pattern or "").split("::", 1)[0]
    if "/" in target or "\\" in target or target.endswith(".py"):
        ctx.sandbox.resolve(target)  # raises SandboxError if it escapes the workspace
    cwd = ctx.verify_env.sync(ctx.working_set)
    check = run_check(ctx, "tests", test_command(ctx.runner, cwd, path_or_pattern), cwd=cwd)
    verdict = "PASSED" if check.passed else "FAILED"
    return f"Tests {verdict} (run on a private copy that includes your proposed edits)\n$ {shlex.join(check.command)}\n{check.output}"


def run_linter(ctx: ToolContext, path: str | None = None) -> str:
    paths = [ctx.sandbox.relative(ctx.sandbox.resolve(path))] if path else (ctx.working_set.changed_paths() or [])
    cwd = ctx.verify_env.sync(ctx.working_set)
    check = run_check(ctx, "lint", lint_command(ctx.runner, ctx.sandbox.root, paths), cwd=cwd)
    verdict = "PASSED" if check.passed else "FAILED"
    return f"Lint {verdict}\n$ {shlex.join(check.command)}\n{check.output}"


def run_command(ctx: ToolContext, cmd: str, explanation: str = "") -> str:
    """Queue a command for user approval (validated now, executed only after approval)."""
    argv = ctx.runner.split(cmd)
    ctx.runner.validate(argv, ctx.sandbox.root)
    if ctx.session is None:
        raise ToolError("run_command needs a chat session")
    change = ProposedChange(
        change_id=uuid.uuid4().hex[:12], path=".", unified_diff="", explanation=explanation or f"Run {shlex.join(argv)}",
        risk="medium", kind="command", command=argv,
    )
    ctx.session.add_change(change)
    ctx.trace.emit("diff", {"path": ".", "command": argv, "change_id": change.change_id, "explanation": change.explanation, "proposed": True})
    return f"Command queued for user approval as change {change.change_id}: {shlex.join(argv)}. It has NOT run."


TOOLS = [
    Tool(
        "run_tests", "Run the project's tests on a private copy of the workspace that includes your proposed edits.",
        {"type": "object", "properties": {
            "path_or_pattern": {"type": "string", "description": "Optional test file/node id (tests/test_x.py::test_y) or -k pattern."},
        }, "required": []},
        RiskLevel.VERIFY, run_tests,
    ),
    Tool(
        "run_linter", "Run the linter (ruff/flake8) on a file, or on your changed files by default.",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": []},
        RiskLevel.VERIFY, run_linter,
    ),
    Tool(
        "run_command", "Ask the user to run an allowlisted command in the real workspace. It is queued for "
        "approval and does NOT run now; prefer run_tests / run_linter.",
        {"type": "object", "properties": {"cmd": {"type": "string"}, "explanation": {"type": "string"}}, "required": ["cmd"]},
        RiskLevel.DANGEROUS, run_command,
    ),
]

__all__ = [
    "CommandNotAllowedError", "CommandResult", "CommandRunner", "VerifyEnvironment", "child_env", "failure_signatures",
    "lint_command", "run_check", "tail", "test_command", "SandboxError", "TOOLS",
]
