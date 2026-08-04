"""Workflow state and job record types."""

from __future__ import annotations

from typing import Any, TypedDict


class JobRecord(TypedDict, total=False):
    title: str
    company: str
    experience: str
    location: str
    salary: str
    link: str
    description: str
    skills: list[str]
    is_remote: bool
    uploaded_at: str
    is_relevant: bool
    relevant_percentage: float
    posted: int


class WorkflowState(TypedDict, total=False):
    """LangGraph state; `session` holds live Playwright objects (not serializable)."""

    session: Any
    jobs: list[JobRecord]
    errors: list[str]
    login_complete: bool
    fetch_complete: bool
    display_complete: bool
    enrich_jd: bool
    job_keyword: str
    job_age: str
    ctc_filters: list[str]
    no_of_jobs: int
    max_pages: int
    relevance_min_pct: float


def initial_workflow_state() -> WorkflowState:
    return {
        "jobs": [],
        "errors": [],
        "login_complete": False,
        "fetch_complete": False,
        "display_complete": False,
        "enrich_jd": False,
        "job_keyword": "javascript",
        "job_age": "3",
        "ctc_filters": [],
        "no_of_jobs": 25,
        "max_pages": 100,
        "relevance_min_pct": 80.0,
    }
