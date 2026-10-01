"""Adapter registry: ``config/job_sources.yaml``'s ``adapter:`` field resolves
here. Adding a new source that needs custom logic means writing one adapter
class and registering it — the graph builder discovers it automatically.
"""

from __future__ import annotations

from .base import JobSourceAdapter

_REGISTRY: dict[str, type[JobSourceAdapter]] = {}


def register_adapter(name: str):
    def _decorator(cls: type[JobSourceAdapter]) -> type[JobSourceAdapter]:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"Adapter name {name!r} already registered to {_REGISTRY[name]!r}")
        _REGISTRY[name] = cls
        return cls

    return _decorator


def get_adapter_class(name: str) -> type[JobSourceAdapter]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"No adapter registered for {name!r}. Known adapters: {sorted(_REGISTRY)}. "
            "See docs/job_engine/SOURCE_ADAPTER_GUIDE.md."
        ) from exc


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)


def _ensure_builtin_adapters_imported() -> None:
    """Import side-effects register the built-in adapters exactly once."""
    from .generic.adapter import GenericAdapter  # noqa: F401
    from .instinctools.adapter import InstinctoolsAdapter  # noqa: F401
    from .linkedin.adapter import LinkedInAdapter  # noqa: F401


_ensure_builtin_adapters_imported()
