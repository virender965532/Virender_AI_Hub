from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from enterpriseRAG.config.settings import MEMORY_DIR

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """In-memory conversation and retrieval context for current session."""

    def __init__(self) -> None:
        self.conversations: dict[str, list[dict[str, str]]] = {}
        self.retrieval_context: dict[str, list[dict[str, Any]]] = {}

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.conversations.setdefault(session_id, []).append({"role": role, "content": content})

    def get_messages(self, session_id: str, limit: int = 20) -> list[dict[str, str]]:
        return self.conversations.get(session_id, [])[-limit:]

    def set_retrieval_context(self, session_id: str, chunks: list[dict[str, Any]]) -> None:
        self.retrieval_context[session_id] = chunks

    def get_retrieval_context(self, session_id: str) -> list[dict[str, Any]]:
        return self.retrieval_context.get(session_id, [])


class EntityMemory:
    """Extract and persist entities (topics, technologies, companies, products)."""

    def __init__(self, persist_path: Path | None = None) -> None:
        self.persist_path = persist_path or MEMORY_DIR / "entities.json"
        self.entities: dict[str, list[str]] = {
            "topics": [],
            "technologies": [],
            "companies": [],
            "products": [],
        }
        self._load()

    def _load(self) -> None:
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text(encoding="utf-8"))
                self.entities.update(data)
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not load entity memory from %s", self.persist_path)

    def _save(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(
            json.dumps(self.entities, indent=2), encoding="utf-8"
        )

    def update(self, new_entities: dict[str, list[str]]) -> None:
        for category, values in new_entities.items():
            if category not in self.entities:
                self.entities[category] = []
            existing = set(self.entities[category])
            for val in values:
                if val and val not in existing:
                    self.entities[category].append(val)
                    existing.add(val)
        self._save()

    def get_all(self) -> dict[str, list[str]]:
        return dict(self.entities)


class SessionMemory:
    """User preferences and answer format per session."""

    def __init__(self, persist_path: Path | None = None) -> None:
        self.persist_path = persist_path or MEMORY_DIR / "sessions.json"
        self.sessions: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.persist_path.exists():
            try:
                self.sessions = json.loads(self.persist_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not load session memory from %s", self.persist_path)

    def _save(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(
            json.dumps(self.sessions, indent=2), encoding="utf-8"
        )

    def get_preferences(self, session_id: str) -> dict[str, Any]:
        return self.sessions.get(session_id, {})

    def set_preferences(self, session_id: str, preferences: dict[str, Any]) -> None:
        current = self.sessions.get(session_id, {})
        current.update(preferences)
        self.sessions[session_id] = current
        self._save()

    def get_role(self, session_id: str, default: str = "Enterprise Architect") -> str:
        prefs = self.get_preferences(session_id)
        return prefs.get("role", default)


class MemoryStore:
    """Unified memory facade."""

    def __init__(self) -> None:
        self.short_term = ShortTermMemory()
        self.entity = EntityMemory()
        self.session = SessionMemory()
