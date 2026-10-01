from __future__ import annotations

import pytest

from jobEngine.sources.base import JobSourceAdapter, SourceDiscoveryResult, SourceExecutionContext
from jobEngine.sources.registry import available_adapters, get_adapter_class, register_adapter


def test_builtin_adapters_are_registered():
    adapters = available_adapters()
    assert {"generic", "linkedin", "instinctools"} <= set(adapters)


def test_get_unknown_adapter_raises_with_helpful_message():
    with pytest.raises(KeyError, match="No adapter registered"):
        get_adapter_class("does_not_exist")


def test_register_new_adapter_is_discoverable():
    @register_adapter("test_registry_adapter")
    class _Adapter(JobSourceAdapter):
        async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
            return SourceDiscoveryResult(source_id=context.config.id, status="success")

    assert get_adapter_class("test_registry_adapter") is _Adapter


def test_registering_same_name_twice_with_different_class_raises():
    @register_adapter("test_registry_dup")
    class _First(JobSourceAdapter):
        async def discover_jobs(self, context):  # noqa: D401
            raise NotImplementedError

    with pytest.raises(ValueError, match="already registered"):
        @register_adapter("test_registry_dup")
        class _Second(JobSourceAdapter):
            async def discover_jobs(self, context):
                raise NotImplementedError
