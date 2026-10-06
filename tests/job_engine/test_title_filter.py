from __future__ import annotations

import pytest

from jobEngine.sources.title_filter import title_matches_keywords

_KEYWORDS = ["developer", "full-stack", "ai", "nodejs", "cicd", "machine-learning", "react"]


@pytest.mark.parametrize(
    "title",
    [
        "Senior Software Developer",
        "Full Stack Engineer",
        "Fullstack Engineer",
        "Full-Stack Engineer",
        "AI/ML Specialist",
        "Node.js Expert",
        "CI/CD Specialist",
        "Machine Learning Scientist",
        "React.js Lead",
    ],
)
def test_matching_titles_are_kept(title):
    assert title_matches_keywords(title, _KEYWORDS)


@pytest.mark.parametrize(
    "title",
    ["Maintenance Supervisor", "Sales Executive", "Chartered Accountant", "Reactor Operator", "", None],
)
def test_non_matching_titles_are_dropped(title):
    assert not title_matches_keywords(title, _KEYWORDS)


def test_no_keywords_keeps_everything():
    assert title_matches_keywords("Sales Executive", [])
