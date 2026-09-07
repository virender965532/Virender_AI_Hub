from __future__ import annotations

from jobEngine.config.settings import EngineSettings
from jobEngine.matching.embeddings import EmbeddingMatcher


def test_cosine_similarity_identical_vectors_is_one():
    assert EmbeddingMatcher.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert EmbeddingMatcher.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_handles_zero_vector():
    assert EmbeddingMatcher.cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_cosine_similarity_mismatched_lengths_returns_zero():
    assert EmbeddingMatcher.cosine_similarity([1.0], [1.0, 0.0]) == 0.0


def test_unavailable_without_api_key():
    settings = EngineSettings(openai_api_key="")
    matcher = EmbeddingMatcher(settings)
    assert matcher.available() is False
    assert matcher.embed_profile("some text") is None
    assert matcher.similarity("profile", "job") is None


def test_unavailable_when_disabled_even_with_key():
    settings = EngineSettings(openai_api_key="sk-fake", embeddings_enabled=False)
    matcher = EmbeddingMatcher(settings)
    assert matcher.available() is False
