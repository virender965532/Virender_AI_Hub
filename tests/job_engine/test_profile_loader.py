from __future__ import annotations

import textwrap
from pathlib import Path

from jobEngine.services.profile_loader import load_candidate_profile_from_path, parse_summary_text


def test_parses_real_summary_txt():
    from jobEngine.config.settings import PROFILE_SUMMARY_PATH

    profile = load_candidate_profile_from_path(PROFILE_SUMMARY_PATH)
    assert profile.personal_information.name == "Virender Kumar"
    assert profile.personal_information.phone  # present internally...
    assert profile.total_experience_years == 7.0
    assert profile.current_role is not None
    assert profile.current_role.company == "Impressico Business Solutions"
    assert "LangGraph" in profile.ai_skills
    assert len(profile.previous_roles) >= 2
    assert profile.education


def test_phone_never_appears_in_llm_context():
    from jobEngine.config.settings import PROFILE_SUMMARY_PATH

    profile = load_candidate_profile_from_path(PROFILE_SUMMARY_PATH)
    context_text = profile.to_llm_context()
    assert profile.personal_information.phone not in context_text
    assert profile.personal_information.email not in context_text


def test_parser_degrades_gracefully_on_minimal_input():
    minimal = "Jane Doe\nSoftware Engineer\n\nPROFILE SUMMARY\nGeneralist with 3 years of experience.\n"
    profile = parse_summary_text(minimal)
    assert profile.personal_information.name == "Jane Doe"
    assert profile.total_experience_years == 3.0
    assert profile.current_role is None
    assert profile.projects == []
    assert profile.education == []


def test_missing_file_raises(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_candidate_profile_from_path(tmp_path / "nope.txt")
