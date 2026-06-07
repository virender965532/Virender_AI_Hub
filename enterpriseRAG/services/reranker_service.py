from __future__ import annotations

import logging
import time
from typing import Any

from enterpriseRAG.config.settings import get_settings
from enterpriseRAG.services.observability import ObservabilityService

logger = logging.getLogger(__name__)

_cross_encoder = None
_bge_reranker = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        settings = get_settings()
        _cross_encoder = CrossEncoder(settings.cross_encoder_model)
        logger.info("Loaded CrossEncoder: %s", settings.cross_encoder_model)
    return _cross_encoder


def _get_bge_reranker():
    global _bge_reranker
    if _bge_reranker is None:
        from sentence_transformers import CrossEncoder

        settings = get_settings()
        _bge_reranker = CrossEncoder(settings.bge_reranker_model)
        logger.info("Loaded BGE Reranker: %s", settings.bge_reranker_model)
    return _bge_reranker


class RerankerService:
    """Retrieve Top-K → Rerank → Top-N pipeline."""

    def __init__(self, observability: ObservabilityService | None = None) -> None:
        self.settings = get_settings()
        self.obs = observability
        self._model_available: bool | None = None

    def _check_model(self) -> bool:
        if self._model_available is not None:
            return self._model_available
        try:
            import sentence_transformers  # noqa: F401

            self._model_available = True
        except ImportError:
            self._model_available = False
            logger.warning(
                "sentence-transformers not installed; using score-based reranking fallback."
            )
        return self._model_available

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_n: int | None = None,
        reranker_type: str | None = None,
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []

        top_n = top_n or self.settings.top_k_rerank
        reranker_type = reranker_type or self.settings.reranker_type
        t0 = time.perf_counter()

        if self._check_model():
            try:
                model = (
                    _get_bge_reranker()
                    if reranker_type == "bge"
                    else _get_cross_encoder()
                )
                pairs = [(query, c["content"]) for c in chunks]
                scores = model.predict(pairs)
                for chunk, score in zip(chunks, scores):
                    chunk["rerank_score"] = float(score)
                ranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
            except Exception:
                logger.exception("Neural reranker failed; falling back to retrieval scores.")
                ranked = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
        else:
            ranked = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)

        result = ranked[:top_n]
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        if self.obs:
            self.obs.record_tool(
                "reranker",
                elapsed_ms,
                reranker_type=reranker_type,
                input_count=len(chunks),
                output_count=len(result),
            )
        return result
