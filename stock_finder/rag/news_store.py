"""In-memory vector store of recent news articles (FAISS inner-product over normalized embeddings).

Articles keep their id, source, title, URL and published date, so whatever an
agent retrieves can be cited exactly. The store is also the citation registry:
``citations(ids)`` resolves ids to ``Citation`` objects and reports unknown ids.

If embeddings are unavailable (no API key, OpenAI down) articles are still
stored and searched with a keyword-overlap fallback, so the pipeline degrades
instead of failing. The store is bounded; the oldest articles are evicted.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime
from typing import Any, Callable, Iterable

import numpy as np

from ..data.news_data import Article
from ..schemas import Citation

logger = logging.getLogger(__name__)

Embedder = Callable[[list[str]], list[list[float]]]
_WORD_RE = re.compile(r"[a-z0-9]{3,}")
_STOP = {"the", "and", "for", "with", "from", "that", "this", "are", "was", "has", "have", "its", "after", "over", "into"}


def _ts(article: Article) -> float:
    """Sort key: publish time as epoch seconds (unknown dates sort oldest)."""
    return article.published.timestamp() if article.published else 0.0


def _words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOP}


class NewsStore:
    """Thread-safe article registry + similarity search."""

    def __init__(self, embedder: Embedder | None = None, *, max_articles: int = 2000) -> None:
        self._embedder = embedder
        self.max_articles = max_articles
        self._articles: dict[str, Article] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._index: Any = None
        self._index_ids: list[str] = []
        self._dirty = True
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._articles)

    # --- writes -------------------------------------------------------------------------------

    def add(self, articles: Iterable[Article]) -> int:
        """Register articles (embedding the new ones). Returns how many were new."""
        with self._lock:
            new = [a for a in articles if a.id not in self._articles]
            for article in new:
                self._articles[article.id] = article
        if not new:
            return 0
        if self._embedder is not None:
            try:
                vectors = self._embedder([f"{a.title}. {a.summary}" for a in new])
            except Exception as exc:  # noqa: BLE001 - keyword search still works
                logger.warning("News embeddings unavailable (%s); using keyword search", type(exc).__name__)
            else:
                with self._lock:
                    for article, vector in zip(new, vectors):
                        arr = np.asarray(vector, dtype="float32")
                        norm = float(np.linalg.norm(arr))
                        if norm > 0:
                            self._vectors[article.id] = arr / norm
        with self._lock:
            self._dirty = True
            self._evict()
        return len(new)

    def _evict(self) -> None:
        overflow = len(self._articles) - self.max_articles
        if overflow <= 0:
            return
        oldest = sorted(self._articles.values(), key=_ts)[:overflow]
        for article in oldest:
            self._articles.pop(article.id, None)
            self._vectors.pop(article.id, None)

    # --- reads --------------------------------------------------------------------------------

    def get(self, article_id: str) -> Article | None:
        with self._lock:
            return self._articles.get(article_id)

    def citations(self, ids: Iterable[str]) -> tuple[list[Citation], list[str]]:
        """Resolve article ids to citations (deduplicated, in order). Returns ``(citations, unknown_ids)``."""
        found: list[Citation] = []
        unknown: list[str] = []
        seen: set[str] = set()
        for raw in ids:
            article_id = str(raw).strip().strip("[]")
            if article_id in seen:
                continue
            seen.add(article_id)
            article = self.get(article_id)
            if article is None:
                unknown.append(article_id)
            else:
                found.append(article.to_citation())
        return found, unknown

    def _build_index(self) -> None:
        import faiss

        ids = [i for i in self._articles if i in self._vectors]
        self._index_ids = ids
        if not ids:
            self._index = None
        else:
            matrix = np.stack([self._vectors[i] for i in ids]).astype("float32")
            self._index = faiss.IndexFlatIP(matrix.shape[1])
            self._index.add(matrix)
        self._dirty = False

    def _keyword_scores(self, query: str, candidates: Iterable[Article]) -> dict[str, float]:
        terms = _words(query)
        if not terms:
            return {}
        scores = {}
        for article in candidates:
            overlap = len(terms & _words(f"{article.title} {article.summary}"))
            if overlap:
                scores[article.id] = overlap / len(terms)
        return scores

    def search(self, query: str, k: int = 10, *, since: datetime | None = None) -> list[Article]:
        """Most relevant articles for ``query`` (vector search, keyword fallback), newest first on ties."""
        with self._lock:
            if not self._articles:
                return []
            if self._dirty and self._vectors:
                self._build_index()
            index, index_ids = self._index, list(self._index_ids)
            articles = dict(self._articles)
        scores: dict[str, float] = {}
        if index is not None and self._embedder is not None:
            try:
                qvec = np.asarray(self._embedder([query])[0], dtype="float32")
                qvec /= float(np.linalg.norm(qvec)) or 1.0
                sims, idx = index.search(qvec.reshape(1, -1), min(len(index_ids), max(k * 3, k)))
                scores.update({index_ids[i]: float(s) for s, i in zip(sims[0], idx[0]) if i >= 0})
            except Exception as exc:  # noqa: BLE001
                logger.warning("News vector search failed (%s); using keyword search", type(exc).__name__)
        # Articles without a vector (embedding failed) are ranked by keyword overlap, weighted
        # below vector hits when there are any.
        weight = 0.5 if scores else 1.0
        no_vector = [a for i, a in articles.items() if i not in self._vectors]
        for article_id, score in self._keyword_scores(query, no_vector).items():
            scores[article_id] = score * weight
        ranked = sorted(scores, key=lambda i: (scores[i], _ts(articles[i])), reverse=True)
        results = [articles[i] for i in ranked if since is None or articles[i].published is None or articles[i].published >= since]
        return results[:k]
