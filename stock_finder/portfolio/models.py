"""Holdings storage in the Stock Finder MongoDB (``sf_holdings`` collection).

One document per holding, ``_id`` = ``"<portfolio_id>:<symbol>"``. The hub has
no user accounts, so there is one ``"default"`` portfolio unless the caller
passes another id. Dates are stored as ISO strings. CSV import is supported
(columns: symbol, quantity, avg_cost, buy_date, thesis).
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import ValidationError
from pymongo.collection import Collection

from ..mongo import HOLDINGS_COLLECTION, get_collection, to_bson_safe
from ..schemas import Holding, HoldingIn

DEFAULT_PORTFOLIO = "default"
MAX_HOLDINGS = 200
CSV_COLUMNS = ("symbol", "quantity", "avg_cost", "buy_date", "thesis")


def _doc_id(portfolio_id: str, symbol: str) -> str:
    return f"{portfolio_id}:{symbol.upper()}"


class HoldingsRepository:
    """CRUD for holdings; symbols must already be normalized (e.g. ``TCS.NS``)."""

    def __init__(self, collection: Callable[[], Collection] | None = None) -> None:
        self._collection = collection or (lambda: get_collection(HOLDINGS_COLLECTION))

    def list(self, portfolio_id: str = DEFAULT_PORTFOLIO) -> list[Holding]:
        docs = self._collection().find({"portfolio_id": portfolio_id}, {"_id": 0, "portfolio_id": 0}).sort("symbol", 1)
        return [Holding.model_validate(doc) for doc in docs]

    def upsert(self, holding: HoldingIn, portfolio_id: str = DEFAULT_PORTFOLIO) -> Holding:
        """Insert or replace the holding for ``holding.symbol``."""
        now = datetime.now(timezone.utc)
        coll = self._collection()
        doc_id = _doc_id(portfolio_id, holding.symbol)
        existing = coll.find_one({"_id": doc_id}, {"created_at": 1})
        if existing is None and coll.count_documents({"portfolio_id": portfolio_id}) >= MAX_HOLDINGS:
            raise ValueError(f"a portfolio can hold at most {MAX_HOLDINGS} positions")
        doc = to_bson_safe(holding.model_dump(mode="json"))
        doc.update(portfolio_id=portfolio_id, updated_at=now,
                   created_at=(existing or {}).get("created_at") or now)
        coll.replace_one({"_id": doc_id}, doc, upsert=True)
        return Holding.model_validate({k: v for k, v in doc.items() if k != "portfolio_id"})

    def delete(self, symbol: str, portfolio_id: str = DEFAULT_PORTFOLIO) -> bool:
        return self._collection().delete_one({"_id": _doc_id(portfolio_id, symbol)}).deleted_count > 0

    def clear(self, portfolio_id: str = DEFAULT_PORTFOLIO) -> int:
        return self._collection().delete_many({"portfolio_id": portfolio_id}).deleted_count

    def replace_all(self, holdings: list[HoldingIn], portfolio_id: str = DEFAULT_PORTFOLIO) -> list[Holding]:
        """Replace the whole portfolio (CSV import / table save)."""
        if len(holdings) > MAX_HOLDINGS:
            raise ValueError(f"a portfolio can hold at most {MAX_HOLDINGS} positions")
        self.clear(portfolio_id)
        return [self.upsert(h, portfolio_id) for h in holdings]


def parse_holdings_csv(text: str) -> tuple[list[HoldingIn], list[str]]:
    """Parse a holdings CSV. Returns ``(valid_rows, errors)``; duplicate symbols keep the last row."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if not reader.fieldnames or not {"symbol", "quantity", "avg_cost"} <= {f.strip().lower() for f in reader.fieldnames}:
        return [], ["CSV must have the columns symbol, quantity, avg_cost (optional: buy_date, thesis)"]
    rows: dict[str, HoldingIn] = {}
    errors: list[str] = []
    for line_no, raw in enumerate(reader, start=2):
        row: dict[str, Any] = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
        if not any(row.values()):
            continue
        data = {k: row.get(k) or None for k in CSV_COLUMNS}
        try:
            holding = HoldingIn.model_validate(data)
        except ValidationError as exc:
            first = exc.errors()[0]
            errors.append(f"line {line_no}: {'.'.join(str(p) for p in first['loc'])}: {first['msg']}")
            continue
        rows[holding.symbol] = holding
    return list(rows.values()), errors
