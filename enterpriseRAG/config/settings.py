from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODULE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = MODULE_DIR / "uploads"
DATA_DIR = MODULE_DIR / "data"
MEMORY_DIR = MODULE_DIR / "data" / "memory"

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".docx", ".doc", ".xlsx", ".xls"})


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("ENTERPRISE_RAG_OPENAI_API_KEY")
        or os.getenv("OPENAI_API_KEY", "")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("ENTERPRISE_RAG_MODEL", "gpt-4o-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "ENTERPRISE_RAG_EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )
    chunk_size: int = field(
        default_factory=lambda: _env_int("ENTERPRISE_RAG_CHUNK_SIZE", 800)
    )
    chunk_overlap: int = field(
        default_factory=lambda: _env_int("ENTERPRISE_RAG_CHUNK_OVERLAP", 150)
    )
    parent_chunk_size: int = field(
        default_factory=lambda: _env_int("ENTERPRISE_RAG_PARENT_CHUNK_SIZE", 2000)
    )
    top_k_retrieve: int = field(
        default_factory=lambda: _env_int("ENTERPRISE_RAG_TOP_K_RETRIEVE", 30)
    )
    top_k_rerank: int = field(
        default_factory=lambda: _env_int("ENTERPRISE_RAG_TOP_K_RERANK", 5)
    )
    reranker_type: str = field(
        default_factory=lambda: os.getenv("ENTERPRISE_RAG_RERANKER_TYPE", "cross_encoder")
    )
    cross_encoder_model: str = field(
        default_factory=lambda: os.getenv(
            "ENTERPRISE_RAG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    bge_reranker_model: str = field(
        default_factory=lambda: os.getenv(
            "ENTERPRISE_RAG_BGE_RERANKER", "BAAI/bge-reranker-base"
        )
    )
    max_retrieval_loops: int = field(
        default_factory=lambda: _env_int("ENTERPRISE_RAG_MAX_RETRIEVAL_LOOPS", 2)
    )
    max_reflection_loops: int = field(
        default_factory=lambda: _env_int("ENTERPRISE_RAG_MAX_REFLECTION_LOOPS", 2)
    )
    confidence_threshold: float = field(
        default_factory=lambda: _env_float("ENTERPRISE_RAG_CONFIDENCE_THRESHOLD", 0.65)
    )
    hybrid_alpha: float = field(
        default_factory=lambda: _env_float("ENTERPRISE_RAG_HYBRID_ALPHA", 0.5)
    )
    default_role: str = field(
        default_factory=lambda: os.getenv("ENTERPRISE_RAG_DEFAULT_ROLE", "Enterprise Architect")
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
