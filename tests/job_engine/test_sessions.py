import json

from jobEngine.persistence import sessions

_STATE = {"cookies": [{"name": "li_at", "value": "x", "domain": ".linkedin.com", "path": "/"}], "origins": []}


def test_save_load_and_delete_session():
    assert sessions.load_storage_state("linkedin") is None
    assert sessions.has_storage_state("linkedin") is False

    sessions.save_storage_state("linkedin", _STATE)
    assert sessions.load_storage_state("linkedin") == _STATE
    assert sessions.has_storage_state("linkedin") is True

    assert sessions.delete_storage_state("linkedin") is True
    assert sessions.load_storage_state("linkedin") is None


def test_legacy_session_file_is_imported_once(tmp_path):
    legacy = tmp_path / "linkedin_storage_state.json"
    legacy.write_text(json.dumps(_STATE), encoding="utf-8")

    assert sessions.load_storage_state("linkedin", legacy_path=legacy) == _STATE
    legacy.unlink()
    assert sessions.load_storage_state("linkedin", legacy_path=legacy) == _STATE


def test_legacy_file_without_cookies_is_ignored(tmp_path):
    legacy = tmp_path / "linkedin_storage_state.json"
    legacy.write_text(json.dumps({"cookies": [], "origins": []}), encoding="utf-8")
    assert sessions.load_storage_state("linkedin", legacy_path=legacy) is None
