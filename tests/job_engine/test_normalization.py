from __future__ import annotations

from jobEngine.graph.nodes.normalization import _infer_remote_type
from jobEngine.models.job import RemoteType


def test_adapter_hint_is_trusted_over_text_inference():
    # location text alone would infer ONSITE, but an adapter (e.g. Instinctools'
    # explicit "#Remote" card tag) knows better and must win.
    assert (
        _infer_remote_type("Poland", "no work-arrangement keywords here", adapter_hint="remote")
        == RemoteType.REMOTE
    )


def test_invalid_adapter_hint_falls_back_to_text_inference():
    assert _infer_remote_type("Poland", "some description", adapter_hint="not-a-real-type") == RemoteType.ONSITE


def test_no_adapter_hint_uses_text_inference_as_before():
    assert _infer_remote_type(None, "fully remote position", adapter_hint=None) == RemoteType.REMOTE
    assert _infer_remote_type("Berlin, Germany", "on-site office role", adapter_hint=None) == RemoteType.ONSITE
    assert _infer_remote_type(None, "", adapter_hint=None) == RemoteType.UNKNOWN
