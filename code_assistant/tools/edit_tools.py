"""WRITE tools: propose_edit, create_file (proposals only) and apply_patch (approval required).

The agent never writes to the workspace. Its edits accumulate in a per-request
``WorkingSet`` (an in-memory overlay of the files it changed). The pipeline
turns the working set into ``ProposedChange`` records, one per file, each
holding a unified diff from the original to the final content. Only
``apply_change_to_disk`` (called after explicit user approval) writes files.

The diff applier is tolerant of the usual LLM diff mistakes: wrong line
numbers in ``@@`` headers, missing leading space on blank context lines,
trailing-whitespace differences and CRLF files. If a hunk's context cannot be
found, the agent gets a clear error and is told to re-read the file.
"""

from __future__ import annotations

import difflib
import os
import re
import threading
import uuid
from dataclasses import dataclass, field

from ..errors import ConflictError
from ..schemas import ProposedChange
from ..workspace.sandbox import Sandbox, SandboxError, sha256_text
from .registry import RiskLevel, Tool, ToolContext, ToolError

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_SENSITIVE_PATH = re.compile(
    r"(^|/)(setup\.py|setup\.cfg|pyproject\.toml|requirements[^/]*\.txt|package\.json|Dockerfile|docker-compose[^/]*"
    r"|\.github/|\.gitlab-ci|Makefile|manage\.py|settings\.py|config/|migrations/|auth|security|permission)",
    re.IGNORECASE,
)


class PatchError(ValueError):
    """A unified diff could not be applied to the current file content."""


# --- diff application --------------------------------------------------------------------------

@dataclass
class _Hunk:
    old_start: int | None
    old: list[str]
    new: list[str]


def _parse_hunks(diff: str) -> list[_Hunk]:
    hunks: list[_Hunk] = []
    current: _Hunk | None = None
    for raw in diff.replace("\r\n", "\n").split("\n"):
        if raw.startswith(("--- ", "+++ ", "diff --git", "index ")) and (current is None or not (current.old or current.new)):
            continue
        header = _HUNK_HEADER.match(raw)
        if header or raw.startswith("@@"):
            current = _Hunk(int(header.group(1)) if header else None, [], [])
            hunks.append(current)
            continue
        if current is None:
            if raw[:1] in (" ", "-", "+"):
                current = _Hunk(None, [], [])  # diff without @@ headers
                hunks.append(current)
            else:
                continue
        if raw.startswith("\\"):
            continue  # "\ No newline at end of file"
        tag, body = (raw[:1], raw[1:]) if raw[:1] in (" ", "-", "+") else (" ", raw)
        if tag in (" ", "-"):
            current.old.append(body)
        if tag in (" ", "+"):
            current.new.append(body)
    # A trailing empty "context" line is usually an artefact of the final newline.
    for hunk in hunks:
        while hunk.old and hunk.new and hunk.old[-1] == "" and hunk.new[-1] == "":
            hunk.old.pop()
            hunk.new.pop()
    return [h for h in hunks if h.old or h.new]


def _find(lines: list[str], block: list[str], start: int, hint: int | None) -> int | None:
    """Index where ``block`` occurs at/after ``start``; exact, then whitespace-tolerant. Closest to hint wins."""
    if not block:
        return None
    for normalise in (lambda s: s, lambda s: s.rstrip(), lambda s: s.strip()):
        target = [normalise(b) for b in block]
        matches = [
            i for i in range(start, len(lines) - len(block) + 1)
            if normalise(lines[i]) == target[0] and [normalise(x) for x in lines[i:i + len(block)]] == target
        ]
        if matches:
            if hint is None:
                return matches[0]
            return min(matches, key=lambda i: abs(i - hint))
    return None


def apply_unified_diff(original: str, diff: str) -> str:
    """Apply ``diff`` to ``original``; raise ``PatchError`` with a helpful message on failure."""
    hunks = _parse_hunks(diff)
    if not hunks:
        raise PatchError("no hunks found; send a unified diff with @@ headers and ' ', '-', '+' line prefixes")
    newline = "\r\n" if "\r\n" in original else "\n"
    had_final_newline = original.endswith(("\n", "\r\n")) or original == ""
    lines = original.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    cursor = 0
    for number, hunk in enumerate(hunks, start=1):
        hint = hunk.old_start - 1 if hunk.old_start else None
        if not hunk.old:  # pure insertion: "@@ -a,0 ..." inserts after line a; no header appends
            position = len(lines) if hunk.old_start is None else min(len(lines), hunk.old_start)
            lines[position:position] = hunk.new
            cursor = position + len(hunk.new)
            continue
        position = _find(lines, hunk.old, cursor, hint)
        if position is None:
            position = _find(lines, hunk.old, 0, hint)  # hunks given out of order
        if position is None:
            preview = "\n".join(hunk.old[:8])
            raise PatchError(
                f"hunk {number} does not match the current file: these lines were not found:\n{preview}\n"
                "Re-read the file with read_file and send a diff whose context and '-' lines match exactly."
            )
        lines[position:position + len(hunk.old)] = hunk.new
        cursor = position + len(hunk.new)
    text = newline.join(lines)
    if lines and had_final_newline:
        text += newline
    return text


def make_unified_diff(path: str, old: str | None, new: str) -> str:
    """Unified diff (3 lines of context) from ``old`` (None = new file) to ``new``."""
    old_lines = (old or "").replace("\r\n", "\n").splitlines(keepends=True)
    new_lines = new.replace("\r\n", "\n").splitlines(keepends=True)
    fromfile = "/dev/null" if old is None else f"a/{path}"
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=fromfile, tofile=f"b/{path}", n=3)
    return "".join(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in diff)


def assess_risk(path: str, diff: str, *, created: bool = False) -> str:
    changed = sum(1 for line in diff.splitlines() if line[:1] in "+-" and not line.startswith(("+++", "---")))
    if _SENSITIVE_PATH.search(path) or changed > 200:
        return "high"
    if changed > 40 or (created and changed > 150):
        return "medium"
    return "low"


# --- the working set ---------------------------------------------------------------------------

@dataclass
class WorkingSet:
    """Per-request overlay of proposed file contents (never written to the workspace)."""

    sandbox: Sandbox
    _original: dict[str, str | None] = field(default_factory=dict)  # None = file did not exist
    _current: dict[str, str] = field(default_factory=dict)
    _notes: dict[str, list[str]] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def get(self, rel: str) -> str | None:
        with self._lock:
            return self._current.get(rel)

    def changed_paths(self) -> list[str]:
        with self._lock:
            return [p for p, text in self._current.items() if text != self._original.get(p)]

    def created_paths(self) -> list[str]:
        with self._lock:
            return [p for p in self._current if self._original.get(p) is None]

    def contents(self) -> dict[str, str]:
        with self._lock:
            return {p: self._current[p] for p in self.changed_paths()}

    def is_empty(self) -> bool:
        return not self.changed_paths()

    def _base(self, rel: str) -> str:
        with self._lock:
            if rel in self._current:
                return self._current[rel]
        text = self.sandbox.read_text(rel)
        with self._lock:
            self._original.setdefault(rel, text)
        return text

    def propose_edit(self, path: str, unified_diff: str, explanation: str) -> tuple[str, str]:
        """Apply a diff to the overlay. Returns ``(relative_path, diff_of_this_step)``."""
        real = self.sandbox.check_writable(path)
        rel = self.sandbox.relative(real)
        with self._lock:
            known = rel in self._current
        if not known and not real.is_file():
            raise ToolError(f"{rel} does not exist; use create_file for new files")
        before = self._base(rel)
        after = apply_unified_diff(before, unified_diff)
        if after == before:
            raise ToolError("the diff produced no change")
        with self._lock:
            self._current[rel] = after
            self._notes.setdefault(rel, []).append(explanation.strip())
        return rel, make_unified_diff(rel, before, after)

    def create_file(self, path: str, content: str, explanation: str) -> str:
        real = self.sandbox.check_writable(path)
        rel = self.sandbox.relative(real)
        with self._lock:
            exists_in_overlay = rel in self._current
        if real.exists() or exists_in_overlay:
            raise ToolError(f"{rel} already exists; use propose_edit to change it")
        with self._lock:
            self._original[rel] = None
            self._current[rel] = content
            self._notes.setdefault(rel, []).append(explanation.strip())
        return rel

    def discard(self) -> None:
        with self._lock:
            self._original.clear()
            self._current.clear()
            self._notes.clear()

    def combined_diff(self) -> str:
        with self._lock:
            return "".join(make_unified_diff(p, self._original.get(p), self._current[p]) for p in self.changed_paths())

    def to_changes(self) -> list[ProposedChange]:
        """One ``ProposedChange`` per changed file (original -> final)."""
        changes: list[ProposedChange] = []
        with self._lock:
            for rel in self.changed_paths():
                original = self._original.get(rel)
                final = self._current[rel]
                diff = make_unified_diff(rel, original, final)
                changes.append(ProposedChange(
                    change_id=uuid.uuid4().hex[:12],
                    path=rel,
                    unified_diff=diff,
                    explanation=" ".join(n for n in self._notes.get(rel, []) if n) or "(no explanation given)",
                    risk=assess_risk(rel, diff, created=original is None),  # type: ignore[arg-type]
                    kind="create" if original is None else "edit",
                    original_content=original,
                    new_content=final,
                    original_sha256=sha256_text(original) if original is not None else None,
                ))
        return changes


# --- applying an approved change ---------------------------------------------------------------

def apply_change_to_disk(sandbox: Sandbox, change: ProposedChange) -> None:
    """Write an APPROVED edit/create change. Refuses if the file changed since the proposal."""
    if change.kind not in ("edit", "create") or change.new_content is None:
        raise ValueError(f"change {change.change_id} is not a file change")
    real = sandbox.check_writable(change.path)
    if change.kind == "create":
        if real.exists():
            raise ConflictError(f"{change.path} was created by someone else after the change was proposed")
    else:
        if not real.is_file():
            raise ConflictError(f"{change.path} no longer exists")
        current = real.read_bytes().decode("utf-8", errors="replace")
        if sha256_text(current) != change.original_sha256:
            raise ConflictError(f"{change.path} changed on disk after the change was proposed; ask again to regenerate it")
    real.parent.mkdir(parents=True, exist_ok=True)
    tmp = real.with_name(f".{real.name}.ca-tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(change.new_content)
    os.replace(tmp, real)


# --- tool handlers -----------------------------------------------------------------------------

def propose_edit(ctx: ToolContext, path: str, unified_diff: str, explanation: str) -> str:
    try:
        rel, step_diff = ctx.working_set.propose_edit(path, unified_diff, explanation)
    except PatchError as exc:
        raise ToolError(str(exc)) from exc
    ctx.trace.emit("diff", {"path": rel, "unified_diff": step_diff, "explanation": explanation, "proposed": True})
    return f"Proposed edit recorded for {rel} (not written to disk; the user will review it). Resulting change:\n{step_diff}"


def create_file(ctx: ToolContext, path: str, content: str, explanation: str) -> str:
    rel = ctx.working_set.create_file(path, content, explanation)
    ctx.trace.emit("diff", {"path": rel, "unified_diff": make_unified_diff(rel, None, content), "explanation": explanation, "proposed": True})
    return f"Proposed new file {rel} ({len(content.splitlines())} lines; not written to disk until the user approves)."


def apply_patch(ctx: ToolContext, change_id: str) -> str:
    raise ToolError(
        f"apply_patch({change_id}) requires explicit user approval; changes are applied only when the user "
        "approves them in the UI or via POST /api/changes/<change_id>/approve."
    )


TOOLS = [
    Tool(
        "propose_edit", "Propose an edit to an existing file as a unified diff against its CURRENT content "
        "(read it first). Not written to disk; the user approves every change.",
        {"type": "object", "properties": {
            "path": {"type": "string"},
            "unified_diff": {"type": "string", "description": "Unified diff with @@ hunk headers and 3 lines of context."},
            "explanation": {"type": "string", "description": "Why this change is needed."},
        }, "required": ["path", "unified_diff", "explanation"]},
        RiskLevel.WRITE, propose_edit,
    ),
    Tool(
        "create_file", "Propose a new file. Not written to disk until the user approves.",
        {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}, "explanation": {"type": "string"},
        }, "required": ["path", "content", "explanation"]},
        RiskLevel.WRITE, create_file,
    ),
    Tool(
        "apply_patch", "Apply an approved change to the workspace (only after explicit user approval).",
        {"type": "object", "properties": {"change_id": {"type": "string"}}, "required": ["change_id"]},
        RiskLevel.WRITE, apply_patch, agent_visible=False,
    ),
]

__all__ = [
    "ConflictError", "PatchError", "WorkingSet", "apply_change_to_disk", "apply_unified_diff",
    "assess_risk", "make_unified_diff", "SandboxError", "TOOLS",
]
