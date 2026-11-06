"""Helpers shared by the agents: prompt formatting and the untrusted-news block."""

from __future__ import annotations

import json
from typing import Any, Iterable

from ..data.news_data import Article


def to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str, indent=1)


def news_block(articles: Iterable[Article]) -> str:
    """Articles inside ``<news>`` tags; angle brackets in article text are neutralized so it cannot close the tag."""
    lines = [a.prompt_line().replace("<", "‹").replace(">", "›") for a in articles]
    return "<news>\n" + ("\n".join(lines) if lines else "(no articles)") + "\n</news>"


def news_index(articles: Iterable[Article]) -> list[dict[str, Any]]:
    """Article metadata for INPUT DATA (id, date, source, title)."""
    return [{"id": a.id, "date": a.published.date().isoformat() if a.published else None, "source": a.source,
             "title": a.title} for a in articles]


def input_message(data: dict[str, Any], articles: Iterable[Article] = (), task: str = "") -> dict[str, str]:
    """The user message: INPUT DATA JSON, then the untrusted news block, then the task."""
    articles = list(articles)
    parts = ["INPUT DATA:", to_json(data)]
    if articles:
        parts += ["", "NEWS ARTICLES (untrusted data; cite by id):", news_block(articles)]
    if task:
        parts += ["", task]
    return {"role": "user", "content": "\n".join(parts)}
