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


_GUEST = {"cookies": [{"name": "bcookie", "value": "g", "domain": ".linkedin.com", "path": "/"}], "origins": []}


def _state_with_li_at(expires):
    return {"cookies": [{"name": "li_at", "value": "x", "domain": ".linkedin.com", "path": "/", "expires": expires}],
            "origins": []}


def test_signed_out_state_never_overwrites_the_saved_session():
    sessions.save_storage_state("linkedin", _STATE)
    assert sessions.save_storage_state("linkedin", _GUEST) is False
    assert sessions.save_storage_state("linkedin", {"cookies": [], "origins": []}) is False
    assert sessions.load_storage_state("linkedin") == _STATE


def test_expired_auth_cookie_is_not_a_session():
    assert sessions.save_storage_state("linkedin", _state_with_li_at(1_000)) is False
    assert sessions.storage_state_info("linkedin") is None


def test_info_reports_auth_cookie_expiry():
    sessions.save_storage_state("linkedin", _state_with_li_at(4_102_444_800))  # 2100-01-01
    info = sessions.storage_state_info("linkedin")
    assert info["expires_at"].startswith("2100-01-01")


def test_signed_out_session_reads_as_disconnected_until_next_sign_in():
    sessions.save_storage_state("linkedin", _STATE)
    sessions.mark_signed_out("linkedin", "redirected to /login")
    assert sessions.storage_state_info("linkedin") is None
    assert sessions.load_storage_state("linkedin") is None
    assert sessions.has_storage_state("linkedin") is False

    sessions.save_storage_state("linkedin", _STATE)  # fresh sign-in
    assert sessions.storage_state_info("linkedin") is not None
    assert sessions.load_storage_state("linkedin") == _STATE
