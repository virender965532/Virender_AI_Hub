"""Audit store for generated reports (``sf_reports`` collection).

Every report is saved with its timestamp, prompt version, model versions,
request metrics and a data snapshot (the exact numbers and article ids each
agent saw), so recommendations can be audited and back-tested later. Recent
reports are also kept in memory so ``GET /api/reports/<id>`` works even when
MongoDB is unreachable; a failed save never fails the request.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable

from pymongo import DESCENDING
from pymongo.collection import Collection

from .mongo import REPORTS_COLLECTION, get_collection, to_bson_safe
from .schemas import FinalReport

logger = logging.getLogger(__name__)

_MEMORY_LIMIT = 50


class ReportStore:
    def __init__(self, collection: Callable[[], Collection] | None = None) -> None:
        self._collection = collection or (lambda: get_collection(REPORTS_COLLECTION))
        self._recent: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def save(self, report: FinalReport, snapshot: dict[str, Any] | None = None) -> bool:
        """Persist ``report`` (+ data snapshot). Returns False (and logs) if MongoDB is unavailable."""
        doc = to_bson_safe({"report": report.model_dump(mode="json"), "data_snapshot": snapshot or {}})
        doc.update(_id=report.report_id, kind=report.kind, created_at=datetime.now(timezone.utc),
                   prompt_version=report.prompt_version, models=report.models, market=report.market)
        with self._lock:
            self._recent[report.report_id] = doc
            while len(self._recent) > _MEMORY_LIMIT:
                self._recent.popitem(last=False)
        try:
            self._collection().replace_one({"_id": report.report_id}, doc, upsert=True)
        except Exception as exc:  # noqa: BLE001 - auditing must never break the response
            logger.warning("Stock Finder report %s not saved to MongoDB: %s", report.report_id, type(exc).__name__)
            return False
        return True

    def get(self, report_id: str, *, include_snapshot: bool = False) -> dict[str, Any] | None:
        projection = None if include_snapshot else {"data_snapshot": 0}
        doc = None
        try:
            doc = self._collection().find_one({"_id": report_id}, projection)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stock Finder report lookup failed: %s", type(exc).__name__)
        if doc is None:
            with self._lock:
                doc = dict(self._recent.get(report_id) or {}) or None
            if doc is not None and not include_snapshot:
                doc.pop("data_snapshot", None)
        if doc is not None:
            doc["report_id"] = doc.pop("_id", report_id)
        return doc

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        try:
            cursor = self._collection().find({}, {"_id": 1, "kind": 1, "created_at": 1, "report.executive_summary": 1}) \
                .sort("created_at", DESCENDING).limit(limit)
            return [{"report_id": d["_id"], "kind": d.get("kind"), "created_at": d.get("created_at"),
                     "summary": (d.get("report") or {}).get("executive_summary", "")[:200]} for d in cursor]
        except Exception:  # noqa: BLE001
            with self._lock:
                docs = list(self._recent.values())[-limit:][::-1]
            return [{"report_id": d["_id"], "kind": d.get("kind"), "created_at": d.get("created_at"),
                     "summary": d["report"].get("executive_summary", "")[:200]} for d in docs]
