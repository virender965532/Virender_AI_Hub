from __future__ import annotations

import os

import pytest

# LLM/embeddings are disabled for the whole test suite by default — tests
# must never depend on a live OpenAI call or a real API key. Tests for the
# LLM/embedding code paths mock the relevant client methods instead.
os.environ.setdefault("JOB_ENGINE_LLM_ENABLED", "false")
os.environ.setdefault("JOB_ENGINE_EMBEDDINGS_ENABLED", "false")
os.environ.setdefault("JOB_ENGINE_PERSIST_RESULTS", "false")
os.environ.setdefault("OPENAI_API_KEY", "")

from jobEngine.config.settings import get_settings  # noqa: E402
from jobEngine.models.profile import CandidateProfile, PersonalInfo, RoleInfo  # noqa: E402

get_settings.cache_clear()


@pytest.fixture()
def settings():
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture()
async def browser_page():
    """A single headless Chromium page for adapter DOM-extraction tests.

    Tests use ``page.set_content(fixture_html)`` rather than ``page.goto`` —
    they exercise selector/extraction logic only, never real navigation, so
    they never touch the network or any SSRF-guarded code path.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            yield page
        finally:
            await browser.close()


@pytest.fixture()
def candidate_profile() -> CandidateProfile:
    return CandidateProfile(
        personal_information=PersonalInfo(
            name="Test Candidate", location="Noida, India", email="test@example.com", phone="+91 0000000000"
        ),
        professional_summary="Full Stack Developer with 7 years of experience in Next.js, React, Node.js, Python and GEN AI.",
        total_experience_years=7.0,
        current_role=RoleInfo(company="Acme", title="AI Full Stack Senior Developer", date_range_raw="2024 - Present", is_current=True),
        previous_roles=[],
        technical_skills={},
        ai_skills=["GEN AI", "LLMs", "RAG", "Agentic AI", "LangGraph"],
        cloud_skills=["AWS Lambda", "AWS EC2", "Docker", "CI/CD"],
        database_skills=["PostgreSQL", "MongoDB"],
        frontend_skills=["Next.js", "React.js", "TypeScript", "JavaScript"],
        backend_skills=["Node.js", "Python", "Express", "REST APIs", "Microservices"],
        ai_tools=["GitHub Copilot", "Cursor"],
        projects=[],
        education=[],
        domain_experience=[],
    )
