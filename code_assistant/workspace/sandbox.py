"""Workspace sandbox: path validation, root enforcement, ignore rules and secret protection.

Security rules (enforced here, used by every tool):
- Every path is resolved with ``os.path.realpath`` and must stay inside the
  workspace root. ``..`` components are rejected outright; absolute paths
  outside the root and symlinks escaping it are rejected after resolution.
- A workspace itself must live inside one of ``CA_WORKSPACE_ROOTS``.
- ``.gitignore`` plus a default ignore list is applied to listing and search.
- Secret files (``.env``, keys, credentials) are never read, and strings that
  look like API keys or tokens are redacted before anything leaves the process.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator

from ..errors import SandboxError

try:
    import pathspec
except ImportError:  # pragma: no cover - pathspec is in requirements.txt
    pathspec = None  # type: ignore[assignment]


IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "venv", ".venv", "env", ".env", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", ".nox", ".idea", ".vscode", "dist", "build", ".next", ".nuxt",
    "target", "coverage", "htmlcov", ".gradle", ".terraform", "site-packages", ".eggs",
})

BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".psd", ".svgz",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac", ".ogg", ".webm",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".war", ".whl", ".egg",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".lib", ".obj", ".class", ".pyc", ".pyo", ".pyd",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".db", ".sqlite", ".sqlite3", ".pkl", ".pickle", ".npy", ".npz", ".h5", ".parquet", ".onnx", ".pt", ".pth",
    ".safetensors", ".ckpt", ".faiss", ".index",
})

LOCK_FILES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock", "Cargo.lock",
    "composer.lock", "Gemfile.lock", "uv.lock", "bun.lockb",
})
LOCK_FILE_MAX_BYTES = 1_000_000

SECRET_NAME_PATTERNS = (
    ".env", ".env.*", "*.env", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "*.ppk",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*", "credentials", "credentials.*", "*credentials*.json",
    ".netrc", ".npmrc", ".pypirc", ".htpasswd", "secrets.*", "*.secret", "*.secrets", "service-account*.json",
)
# Templates meant to be committed are not secrets.
SECRET_EXCEPTIONS = (".env.example", ".env.sample", ".env.template", ".env.dist")

_REDACTED = "[REDACTED]"
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), _REDACTED),
    (re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}"), _REDACTED),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"), _REDACTED),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"), _REDACTED),
    (re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"), _REDACTED),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), _REDACTED),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"), _REDACTED),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), _REDACTED),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"), _REDACTED),
    (re.compile(r"\bxkeysib-[A-Za-z0-9\-]{20,}\b"), _REDACTED),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"), _REDACTED),
    (re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9_\-\.=]{20,}"), r"\1 " + _REDACTED),
    (re.compile(r"(?i)(://[^:/\s@]+:)[^@\s/]{3,}(@)"), r"\1" + _REDACTED + r"\2"),
    (
        re.compile(
            r"""(?ix)
            \b([A-Z0-9_\-]*(?:api[_\-]?key|secret|token|passw(?:or)?d|pwd|private[_\-]?key|access[_\-]?key)[A-Z0-9_\-]*)
            (\s*[:=]\s*["']?)
            (?=[A-Za-z0-9_\-+/=.]*\d)          # token-like values contain a digit; identifiers rarely do
            ([A-Za-z0-9_\-+/=.]{12,})
            (?=["'\s,;]|$)                     # not followed by ( or [ -> not a function call / subscript
            """
        ),
        r"\1\2" + _REDACTED,
    ),
)


def redact(text: str) -> str:
    """Replace strings that look like credentials with ``[REDACTED]``."""
    if not text:
        return text
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def is_secret_path(rel_path: str) -> bool:
    """True for files that commonly hold credentials (checked on the file name and every path part)."""
    parts = PurePosixPath(rel_path.replace("\\", "/")).parts
    name = parts[-1] if parts else ""
    lowered = name.lower()
    if lowered in SECRET_EXCEPTIONS:
        return False
    return any(fnmatch.fnmatch(lowered, pattern) for pattern in SECRET_NAME_PATTERNS) or ".ssh" in parts


def is_within(path: str | os.PathLike[str], root: str | os.PathLike[str]) -> bool:
    """True if real ``path`` equals or is inside real ``root`` (case-insensitive on Windows)."""
    path_s = os.path.normcase(os.path.realpath(path))
    root_s = os.path.normcase(os.path.realpath(root))
    try:
        return os.path.commonpath([path_s, root_s]) == root_s
    except ValueError:  # different drives on Windows
        return False


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="surrogateescape")).hexdigest()


@dataclass(frozen=True)
class Workspace:
    """A registered project folder."""

    id: str
    root: Path
    name: str


def workspace_id_for(root: Path) -> str:
    return hashlib.sha1(os.path.normcase(str(root)).encode()).hexdigest()[:12]


class Sandbox:
    """All filesystem access for one workspace goes through this class."""

    def __init__(self, root: str | os.PathLike[str], *, max_file_bytes: int = 1_000_000) -> None:
        real = os.path.realpath(root)
        if not os.path.isdir(real):
            raise SandboxError(f"workspace folder does not exist: {root}")
        self.root = Path(real)
        self.max_file_bytes = max_file_bytes
        self._gitignore = self._load_gitignore()

    # --- paths --------------------------------------------------------------------------------

    def resolve(self, path: str, *, must_exist: bool = False) -> Path:
        """Resolve a workspace-relative (or absolute, inside-root) path; raise ``SandboxError`` if it escapes."""
        if not isinstance(path, str) or not path.strip():
            raise SandboxError("path must be a non-empty string")
        if "\x00" in path:
            raise SandboxError("path contains a NUL byte")
        cleaned = path.strip().replace("\\", "/")
        if any(part == ".." for part in cleaned.split("/")):
            raise SandboxError(f"'..' is not allowed in paths: {path}")
        candidate = Path(cleaned) if os.path.isabs(cleaned) else self.root / cleaned
        real = Path(os.path.realpath(candidate))
        if not is_within(real, self.root):
            raise SandboxError(f"path is outside the workspace: {path}")
        if must_exist and not real.exists():
            raise SandboxError(f"path does not exist: {self.relative(real)}")
        return real

    def relative(self, path: Path) -> str:
        """Workspace-relative POSIX path (``.`` for the root)."""
        rel = os.path.relpath(path, self.root).replace("\\", "/")
        return "." if rel in ("", ".") else rel

    # --- ignore rules -------------------------------------------------------------------------

    def _load_gitignore(self) -> object | None:
        gitignore = self.root / ".gitignore"
        if pathspec is None or not gitignore.is_file():
            return None
        try:
            lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
            if hasattr(pathspec, "GitIgnoreSpec"):
                return pathspec.GitIgnoreSpec.from_lines(lines)
            return pathspec.PathSpec.from_lines("gitwildmatch", lines)
        except Exception:  # noqa: BLE001 - a malformed .gitignore just means no extra rules
            return None

    def is_ignored(self, rel_path: str, *, is_dir: bool = False) -> bool:
        """Default ignore list + ``.gitignore`` (root file only)."""
        rel = rel_path.replace("\\", "/").strip("/")
        if not rel or rel == ".":
            return False
        parts = rel.split("/")
        if any(part in IGNORED_DIRS for part in (parts if is_dir else parts[:-1])):
            return True
        if not is_dir:
            name = parts[-1]
            if Path(name).suffix.lower() in BINARY_EXTENSIONS:
                return True
            if name in LOCK_FILES:
                try:
                    if (self.root / rel).stat().st_size > LOCK_FILE_MAX_BYTES:
                        return True
                except OSError:
                    return True
        if self._gitignore is not None:
            probe = rel + "/" if is_dir else rel
            if self._gitignore.match_file(probe):  # type: ignore[attr-defined]
                return True
        return False

    def is_readable_file(self, rel_path: str) -> bool:
        return not self.is_ignored(rel_path) and not is_secret_path(rel_path)

    def iter_files(self, start: str = ".", *, max_files: int = 20000) -> Iterator[str]:
        """Yield readable, non-ignored workspace-relative file paths (no symlink following)."""
        base = self.resolve(start, must_exist=True)
        count = 0
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            rel_dir = self.relative(Path(dirpath))
            dirnames[:] = sorted(
                d for d in dirnames
                if not self.is_ignored(d if rel_dir == "." else f"{rel_dir}/{d}", is_dir=True)
                and not os.path.islink(os.path.join(dirpath, d))
            )
            for name in sorted(filenames):
                rel = name if rel_dir == "." else f"{rel_dir}/{name}"
                full = os.path.join(dirpath, name)
                if os.path.islink(full) and not is_within(full, self.root):
                    continue
                if not self.is_readable_file(rel):
                    continue
                yield rel
                count += 1
                if count >= max_files:
                    return

    # --- reading ------------------------------------------------------------------------------

    def read_text(self, path: str) -> str:
        """Read a text file inside the workspace. Secrets, ignored binaries and huge files are refused."""
        real = self.resolve(path, must_exist=True)
        rel = self.relative(real)
        if is_secret_path(rel):
            raise SandboxError(f"refusing to read a secret file: {rel}")
        if not real.is_file():
            raise SandboxError(f"not a file: {rel}")
        if Path(rel).suffix.lower() in BINARY_EXTENSIONS:
            raise SandboxError(f"binary file: {rel}")
        size = real.stat().st_size
        if size > self.max_file_bytes:
            raise SandboxError(f"file too large ({size} bytes > {self.max_file_bytes}): {rel}")
        data = real.read_bytes()
        if b"\x00" in data[:8192]:
            raise SandboxError(f"binary file: {rel}")
        return data.decode("utf-8", errors="replace")

    def check_writable(self, path: str) -> Path:
        """Validate a write target (inside root, not secret, not in an ignored dir)."""
        real = self.resolve(path)
        rel = self.relative(real)
        if rel == ".":
            raise SandboxError("cannot write the workspace root")
        if is_secret_path(rel):
            raise SandboxError(f"refusing to write a secret file: {rel}")
        parts = rel.split("/")
        if any(part in IGNORED_DIRS for part in parts[:-1]):
            raise SandboxError(f"refusing to write inside an ignored folder: {rel}")
        return real


class WorkspaceRegistry:
    """Workspaces the user has registered; each must be inside ``CA_WORKSPACE_ROOTS``."""

    def __init__(self, allowed_roots: tuple[Path, ...], *, max_file_bytes: int = 1_000_000) -> None:
        self.allowed_roots = allowed_roots
        self.max_file_bytes = max_file_bytes
        self._workspaces: dict[str, Workspace] = {}
        self._lock = threading.Lock()

    def register(self, path: str) -> Workspace:
        if not isinstance(path, str) or not path.strip():
            raise SandboxError("workspace path must be a non-empty string")
        if not self.allowed_roots:
            raise SandboxError("no workspace roots are allowed; set CA_WORKSPACE_ROOTS")
        real = Path(os.path.realpath(os.path.expanduser(path.strip())))
        if not real.is_dir():
            raise SandboxError(f"workspace folder does not exist: {path}")
        if not any(is_within(real, root) for root in self.allowed_roots):
            raise SandboxError(f"folder is not inside CA_WORKSPACE_ROOTS: {path}")
        workspace = Workspace(id=workspace_id_for(real), root=real, name=real.name or str(real))
        with self._lock:
            self._workspaces[workspace.id] = workspace
        return workspace

    def get(self, workspace_id: str) -> Workspace:
        with self._lock:
            workspace = self._workspaces.get(workspace_id)
        if workspace is None:
            raise KeyError(f"unknown workspace id: {workspace_id}")
        return workspace

    def list(self) -> list[Workspace]:
        with self._lock:
            return list(self._workspaces.values())

    def sandbox(self, workspace_id: str) -> Sandbox:
        return Sandbox(self.get(workspace_id).root, max_file_bytes=self.max_file_bytes)
