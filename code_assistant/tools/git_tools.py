"""Git tools (git_status, git_diff) plus checkpoints, create_branch and rollback.

Checkpoints protect the user's files when an approved change is applied:
- the original content of every file about to be written is copied to
  ``CA_DATA_DIR/checkpoints/<id>/`` (works for non-git folders too);
- in a git repository, the current HEAD is recorded and a branch
  ``code-assistant/checkpoint-<id>`` is created at HEAD (``git branch``; the
  working tree and the user's checked-out branch are never touched, and
  ``git stash`` is not used because it would capture unrelated work);
- ``rollback`` restores the copies (and deletes files the change created),
  refusing if a file was modified again after the change was applied.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..errors import RollbackConflictError
from ..workspace.sandbox import Sandbox, is_secret_path, redact, sha256_text
from .registry import RiskLevel, Tool, ToolContext, ToolError

logger = logging.getLogger(__name__)

try:
    import git
except ImportError:  # pragma: no cover - gitpython is in requirements.txt
    git = None  # type: ignore[assignment]

_BRANCH_NAME = re.compile(r"^[A-Za-z0-9._/\-]{1,100}$")


def open_repo(root: Path):  # -> git.Repo | None
    """The repository whose working tree is ``root`` (or contains it), if any."""
    if git is None:
        return None
    try:
        return git.Repo(root, search_parent_directories=True)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError, git.GitCommandNotFound):
        return None


def _scoped(ctx: ToolContext, repo) -> str:
    """Pathspec limiting git output to the workspace folder (it may be a sub-folder of the repo)."""
    return os.path.relpath(ctx.sandbox.root, repo.working_tree_dir).replace("\\", "/")


def git_status(ctx: ToolContext) -> str:
    repo = open_repo(ctx.sandbox.root)
    if repo is None:
        return "Not a git repository."
    try:
        branch = repo.active_branch.name if not repo.head.is_detached else f"(detached at {repo.head.commit.hexsha[:8]})"
    except (TypeError, ValueError):
        branch = "(no commits yet)"
    status = repo.git.status("--porcelain=v1", "--", _scoped(ctx, repo))
    lines = [line for line in status.splitlines() if not is_secret_path(line[3:].strip('"'))]
    return f"Branch: {branch}\n" + ("\n".join(lines) if lines else "Working tree clean.")


def git_diff(ctx: ToolContext, path: str | None = None, staged: bool = False) -> str:
    repo = open_repo(ctx.sandbox.root)
    if repo is None:
        return "Not a git repository."
    target = _scoped(ctx, repo)
    if path:
        real = ctx.sandbox.resolve(path)
        rel = ctx.sandbox.relative(real)
        if is_secret_path(rel):
            raise ToolError(f"refusing to diff a secret file: {rel}")
        target = os.path.relpath(real, repo.working_tree_dir).replace("\\", "/")
    args = ["--cached"] if staged else []
    diff = repo.git.diff(*args, "--", target, ":(exclude)*.env", ":(exclude).env*", ":(exclude)*.pem", ":(exclude)*.key")
    return redact(diff) or "No changes."


def create_branch(sandbox: Sandbox, name: str) -> str:
    """Create a branch at HEAD without checking it out. Returns the branch name."""
    if not _BRANCH_NAME.match(name) or ".." in name or name.endswith((".lock", "/")):
        raise ToolError(f"invalid branch name: {name!r}")
    repo = open_repo(sandbox.root)
    if repo is None:
        raise ToolError("not a git repository")
    try:
        repo.git.branch(name)
    except git.GitCommandError as exc:  # type: ignore[union-attr]
        raise ToolError(f"could not create branch {name}: {exc.stderr.strip() if exc.stderr else exc}") from exc
    return name


# --- checkpoints -------------------------------------------------------------------------------

@dataclass
class Checkpoint:
    id: str
    workspace_root: str
    created_at: float
    files: dict[str, bool]  # path -> existed before the change
    applied_sha256: dict[str, str] = field(default_factory=dict)
    git_head: str | None = None
    git_branch: str | None = None
    rolled_back: bool = False


class CheckpointManager:
    """Creates and restores file-level checkpoints under ``directory``."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()

    def create(self, sandbox: Sandbox, paths: list[str], *, make_branch: bool = True) -> Checkpoint:
        checkpoint = Checkpoint(uuid.uuid4().hex[:12], str(sandbox.root), time.time(), {})
        folder = self.directory / checkpoint.id
        (folder / "files").mkdir(parents=True, exist_ok=True)
        for rel in paths:
            source = sandbox.check_writable(rel)
            exists = source.is_file()
            checkpoint.files[rel] = exists
            if exists:
                backup = folder / "files" / rel
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, backup)
        repo = open_repo(sandbox.root)
        if repo is not None:
            try:
                checkpoint.git_head = repo.head.commit.hexsha
                if make_branch:
                    checkpoint.git_branch = create_branch(sandbox, f"code-assistant/checkpoint-{checkpoint.id}")
            except (ValueError, ToolError) as exc:  # no commits yet / branch failure: file backup still protects
                logger.warning("Checkpoint %s: git branch not created (%s)", checkpoint.id, exc)
        self._save(checkpoint)
        return checkpoint

    def mark_applied(self, checkpoint: Checkpoint, sandbox: Sandbox) -> None:
        for rel in checkpoint.files:
            real = sandbox.resolve(rel)
            if real.is_file():
                checkpoint.applied_sha256[rel] = sha256_text(real.read_bytes().decode("utf-8", errors="replace"))
        self._save(checkpoint)

    def rollback(self, checkpoint: Checkpoint, sandbox: Sandbox, *, force: bool = False) -> list[str]:
        """Restore the files of ``checkpoint``. Returns the restored paths."""
        if checkpoint.rolled_back:
            return []
        folder = self.directory / checkpoint.id
        if not force:
            modified = []
            for rel, applied in checkpoint.applied_sha256.items():
                real = sandbox.resolve(rel)
                current = sha256_text(real.read_bytes().decode("utf-8", errors="replace")) if real.is_file() else None
                if current != applied:
                    modified.append(rel)
            if modified:
                raise RollbackConflictError(
                    "these files were modified after the change was applied: " + ", ".join(modified)
                    + ". Roll back with force=true to discard those modifications."
                )
        restored = []
        for rel, existed in checkpoint.files.items():
            target = sandbox.check_writable(rel)
            if existed:
                shutil.copy2(folder / "files" / rel, target)
            elif target.exists():
                target.unlink()
            restored.append(rel)
        checkpoint.rolled_back = True
        self._save(checkpoint)
        return restored

    def load(self, checkpoint_id: str) -> Checkpoint:
        if not re.fullmatch(r"[0-9a-f]{12}", checkpoint_id):
            raise KeyError(f"unknown checkpoint: {checkpoint_id}")
        data = json.loads((self.directory / checkpoint_id / "manifest.json").read_text(encoding="utf-8"))
        return Checkpoint(**data)

    def _save(self, checkpoint: Checkpoint) -> None:
        with self._lock:
            folder = self.directory / checkpoint.id
            folder.mkdir(parents=True, exist_ok=True)
            tmp = folder / "manifest.json.tmp"
            tmp.write_text(json.dumps(asdict(checkpoint), indent=2), encoding="utf-8")
            os.replace(tmp, folder / "manifest.json")


TOOLS = [
    Tool("git_status", "Show the current branch and changed files.", {"type": "object", "properties": {}, "required": []},
         RiskLevel.READ, git_status),
    Tool(
        "git_diff", "Show uncommitted changes in the workspace (optionally one file, or staged changes).",
        {"type": "object", "properties": {"path": {"type": "string"}, "staged": {"type": "boolean"}}, "required": []},
        RiskLevel.READ, git_diff,
    ),
]
