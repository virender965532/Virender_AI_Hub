from __future__ import annotations

import pytest

from jobEngine.sources.title_filter import title_matches_keywords, title_passes

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


@pytest.mark.parametrize(
    "title, matches",
    [
        ("AI Engineer", True),
        ("Senior AI/ML Developer", True),
        ("Gen-AI Specialist", True),
        ("SOFTWARE ENGINEER", True),  # lowercase keyword stays case-insensitive
        ("Ai Specialist", False),  # "AI" is matched as written
        ("ai specialist", False),
        ("AIR Traffic Controller", False),
    ],
)
def test_uppercase_keyword_is_case_sensitive(title, matches):
    assert title_matches_keywords(title, ["AI", "engineer"]) is matches


_EXCLUDE = ["java", "python", "intern", "c#", ".net", "go", "manager"]


@pytest.mark.parametrize(
    "title, passes",
    [
        ("Senior Java Developer", False),
        ("Full Stack Developer (React/Python)", False),
        ("Software Engineering Intern", False),
        ("Senior C# Engineer", False),
        ("Senior .NET Developer", False),
        ("Go Backend Engineer", False),
        ("Engineering Manager", False),
        ("Senior JavaScript Developer", True),
        ("Backend Engineer, Internal Tools", True),
        ("Node.js Engineer", True),
    ],
)
def test_title_exclude_keywords(title, passes):
    assert title_passes(title, [], _EXCLUDE) is passes


def test_no_exclude_keywords_drops_nothing():
    assert title_passes("Senior Java Developer", []) is True
