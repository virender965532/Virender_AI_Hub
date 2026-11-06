"""Per-user session memory: conversation, files read, pending changes, checkpoints, pending plan.

Sessions live in memory (bounded LRU with idle expiry). Old conversation turns
are summarized (not silently dropped) once the history grows past
``history_turns``; if the summarizer fails, a deterministic digest is kept.
Each session also has a "busy" lock so only one request runs per session.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from typing import Callable

from ..schemas import Plan, ProposedChange

Summarizer = Callable[[str], str]

_MAX_TURN_CHARS = 8000


class Session:
    """State for one conversation in one workspace."""

    def __init__(self, session_id: str, workspace_id: str) -> None:
        self.id = session_id
        self.workspace_id = workspace_id
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.turns: list[dict[str, str]] = []
        self.summary = ""
        self.files_read: OrderedDict[str, None] = OrderedDict()
        self.changes: OrderedDict[str, ProposedChange] = OrderedDict()
        self.change_checkpoints: dict[str, str] = {}
        self.checkpoints: list[str] = []
        self.pending_plan: Plan | None = None
        self.pending_task: str | None = None
        self.lock = threading.RLock()
        self.busy = threading.Lock()

    def touch(self) -> None:
        self.updated_at = time.time()

    # --- conversation -------------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        with self.lock:
            self.turns.append({"role": role, "content": content[:_MAX_TURN_CHARS]})
            self.touch()

    def compact(self, keep_turns: int, summarize: Summarizer | None) -> bool:
        """Fold turns older than the last ``keep_turns`` into ``summary``. Returns True if compacted."""
        with self.lock:
            if len(self.turns) <= keep_turns * 2:
                return False
            older, recent = self.turns[:-keep_turns], self.turns[-keep_turns:]
            transcript = "\n".join(f"{t['role']}: {t['content'][:1500]}" for t in older)
            source = (f"Earlier summary:\n{self.summary}\n\n" if self.summary else "") + transcript
        try:
            new_summary = summarize(source) if summarize else ""
        except Exception:  # noqa: BLE001 - fall back to a deterministic digest
            new_summary = ""
        if not new_summary.strip():
            digest = "\n".join(f"- {t['role']}: {t['content'][:200]}" for t in older)
            new_summary = ((self.summary + "\n") if self.summary else "") + digest
            new_summary = new_summary[-6000:]
        with self.lock:
            self.summary = new_summary.strip()
            self.turns = recent
        return True

    def history_text(self) -> str:
        """Summary + recent turns, for the agent's task context."""
        with self.lock:
            parts = []
            if self.summary:
                parts.append(f"Summary of earlier conversation:\n{self.summary}")
            if self.turns:
                parts.append("Recent conversation:\n" + "\n".join(f"{t['role']}: {t['content'][:2000]}" for t in self.turns))
            if self.files_read:
                parts.append("Files already read in this session: " + ", ".join(list(self.files_read)[-30:]))
            pending = [c for c in self.changes.values() if c.status == "pending"]
            if pending:
                parts.append("Changes still awaiting the user's approval: " + ", ".join(f"{c.path} ({c.change_id})" for c in pending))
            return "\n\n".join(parts)

    def note_file_read(self, path: str) -> None:
        with self.lock:
            self.files_read.pop(path, None)
            self.files_read[path] = None
            while len(self.files_read) > 200:
                self.files_read.popitem(last=False)

    # --- changes ------------------------------------------------------------------------------

    def add_change(self, change: ProposedChange) -> None:
        with self.lock:
            self.changes[change.change_id] = change
            self.touch()

    def get_change(self, change_id: str) -> ProposedChange | None:
        with self.lock:
            return self.changes.get(change_id)

    def list_changes(self, status: str | None = None) -> list[ProposedChange]:
        with self.lock:
            return [c for c in self.changes.values() if status is None or c.status == status]

    def supersede_pending(self, paths: set[str]) -> None:
        """New proposals for a file replace older pending ones for the same file."""
        with self.lock:
            for change in self.changes.values():
                if change.status == "pending" and change.path in paths and change.kind != "command":
                    change.status = "rejected"
                    change.review_concerns = (change.review_concerns + " Superseded by a newer proposal.").strip()


class SessionStore:
    """Thread-safe, bounded in-memory session store."""

    def __init__(self, *, max_sessions: int = 200, idle_ttl_seconds: float = 24 * 3600) -> None:
        self.max_sessions = max_sessions
        self.idle_ttl = idle_ttl_seconds
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def _evict(self) -> None:
        now = time.time()
        for sid in [sid for sid, s in self._sessions.items() if now - s.updated_at > self.idle_ttl and not s.busy.locked()]:
            del self._sessions[sid]
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)

    def get_or_create(self, session_id: str | None, workspace_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id) if session_id else None
            if session is not None and session.workspace_id != workspace_id:
                raise ValueError("session belongs to a different workspace; start a new session")
            if session is None:
                session = Session(session_id or uuid.uuid4().hex[:16], workspace_id)
                self._sessions[session.id] = session
            self._sessions.move_to_end(session.id)
            self._evict()
            return session

    def get(self, session_id: str) -> Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown session: {session_id}")
        return session

    def find_change(self, change_id: str) -> tuple[Session, ProposedChange]:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            change = session.get_change(change_id)
            if change is not None:
                return session, change
        raise KeyError(f"unknown change: {change_id}")
