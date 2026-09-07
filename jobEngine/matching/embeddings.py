"""Semantic similarity stage (section 19/22): embeds the candidate profile
once per run and each pre-filtered job description, then ranks by cosine
similarity. This is one signal among several (combined with rule + LLM
scoring in ``scorer.py``) — never the sole basis for a decision.

Degrades gracefully: if ``OPENAI_API_KEY`` is unset or
``JOB_ENGINE_EMBEDDINGS_ENABLED=false``, ``embed_profile``/``embed_job``
return ``None`` and the caller (``graph/nodes/matching.py``) skips this
signal rather than failing the run.
"""

from __future__ import annotations

import logging
import math

from ..config.settings import EngineSettings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingMatcher:
    def __init__(self, settings: EngineSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None
        self._profile_vector_cache: dict[str, list[float]] = {}

    def available(self) -> bool:
        return self.settings.embeddings_enabled and bool(self.settings.openai_api_key)

    def _get_client(self):
        if self._client is None:
            from langchain_openai import OpenAIEmbeddings

            self._client = OpenAIEmbeddings(
                model=self.settings.embedding_model,
                api_key=self.settings.openai_api_key,
                timeout=self.settings.openai_timeout,
            )
        return self._client

    def embed_profile(self, profile_text: str) -> list[float] | None:
        if not self.available():
            return None
        cached = self._profile_vector_cache.get(profile_text)
        if cached is not None:
            return cached
        try:
            vector = self._get_client().embed_query(profile_text)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to embed candidate profile; degrading (no embedding signal)")
            return None
        self._profile_vector_cache[profile_text] = vector
        return vector

    def embed_job(self, job_text: str) -> list[float] | None:
        if not self.available():
            return None
        try:
            return self._get_client().embed_query(job_text)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to embed job text; degrading (no embedding signal)")
            return None

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    def similarity(self, profile_text: str, job_text: str) -> float | None:
        profile_vector = self.embed_profile(profile_text)
        if profile_vector is None:
            return None
        job_vector = self.embed_job(job_text)
        if job_vector is None:
            return None
        return self.cosine_similarity(profile_vector, job_vector)
