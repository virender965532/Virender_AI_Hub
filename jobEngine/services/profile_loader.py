"""Parse ``Profile/Summary.txt`` into a ``CandidateProfile`` at runtime.

Nothing about the candidate is hard-coded in this module — every field comes
from the actual file content. If a section is missing or a line doesn't match
the expected shape, the corresponding field is left empty rather than guessed
or invented, per the engine's grounding requirement (docs/job_engine/
ARCHITECTURE.md "Matching must be grounded").

The section header names in Profile/Summary.txt happen to be ALL CAPS
("PROFILE SUMMARY", "TECHNICAL SKILLS", "WORK EXPERIENCE", "KEY PROJECTS",
"EDUCATION") — this is used as the *only* structural assumption; nothing
else about wording or ordering is assumed beyond that.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

from ..config.settings import PROFILE_SUMMARY_PATH, get_settings
from ..models.profile import CandidateProfile, EducationInfo, PersonalInfo, ProjectInfo, RoleInfo

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^[A-Z][A-Z \-&/]{3,60}$")
_CONTACT_RE = re.compile(r"^(phone|email|linkedin)\s*:\s*(.+)$", re.IGNORECASE)
_ROLE_HEADER_RE = re.compile(r"^(.*?)\s*[—–-]{1,2}\s*(.+)$")
_ROLE_LOCATION_DATES_RE = re.compile(r"^(.*?)\s*\|\s*(.+)$")
_PROJECT_BULLET_RE = re.compile(r"^[•\-]\s*(.+?)\s*\(([^)]+)\)\s*[—–-]{1,2}\s*(.+)$")
_EDUCATION_LINE_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*,\s*(.*)$")
_YEARS_EXPERIENCE_RE = re.compile(r"(\d+(?:\.\d+)?)\+?\s*years?\s+of\s+experience", re.IGNORECASE)

_CATEGORY_TO_FIELD = {
    "front": "frontend_skills",
    "back": "backend_skills",
    "cloud": "cloud_skills",
    "database": "database_skills",
    "ai tool": "ai_tools",
}


def _split_sections(lines: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Return (preamble_lines, {section_name: body_lines})."""
    preamble: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if stripped and _HEADER_RE.match(stripped) and stripped.upper() == stripped:
            current = stripped
            sections[current] = []
            continue
        if current is None:
            preamble.append(line)
        else:
            sections[current].append(line)
    return preamble, sections


def _parse_personal_info(preamble: list[str]) -> tuple[PersonalInfo, str]:
    """Returns (PersonalInfo, headline) from the lines before the first section."""
    contact_fields: dict[str, str] = {}
    plain_lines: list[str] = []
    for raw in preamble:
        stripped = raw.strip()
        if not stripped:
            continue
        m = _CONTACT_RE.match(stripped)
        if m:
            contact_fields[m.group(1).lower()] = m.group(2).strip()
        else:
            plain_lines.append(stripped)

    name = plain_lines[0] if len(plain_lines) >= 1 else ""
    headline = plain_lines[1] if len(plain_lines) >= 2 else ""
    location = plain_lines[2] if len(plain_lines) >= 3 else ""

    info = PersonalInfo(
        name=name,
        location=location,
        linkedin_url=contact_fields.get("linkedin", ""),
        email=contact_fields.get("email", ""),
        phone=contact_fields.get("phone", ""),
    )
    return info, headline


def _parse_summary(lines: list[str], *, headline: str) -> tuple[str, float | None]:
    text = " ".join(s.strip() for s in lines if s.strip())
    if not text:
        text = headline
    years: float | None = None
    m = _YEARS_EXPERIENCE_RE.search(text)
    if m:
        try:
            years = float(m.group(1))
        except ValueError:
            years = None
    return text, years


def _parse_technical_skills(lines: list[str]) -> dict[str, list[str]]:
    skills: dict[str, list[str]] = {}
    pending_category: str | None = None
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.endswith(":"):
            pending_category = stripped[:-1].strip()
            continue
        if pending_category:
            values = [v.strip() for v in stripped.split(",") if v.strip()]
            skills[pending_category] = values
            pending_category = None
    return skills


def _bucket_technical_skills(technical_skills: dict[str, list[str]]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {
        "frontend_skills": [],
        "backend_skills": [],
        "cloud_skills": [],
        "database_skills": [],
        "ai_skills": [],
        "ai_tools": [],
    }
    for category, values in technical_skills.items():
        lower = category.strip().lower()
        if lower == "ai":
            buckets["ai_skills"].extend(values)
            continue
        matched = False
        for needle, field_name in _CATEGORY_TO_FIELD.items():
            if needle in lower:
                buckets[field_name].extend(values)
                matched = True
                break
        if not matched:
            # Unclassified category (e.g. a future new section) still counts as an AI-adjacent
            # or general skill via CandidateProfile.all_skills()'s technical_skills fallback.
            continue
    return buckets


def _parse_work_experience(lines: list[str]) -> list[RoleInfo]:
    roles: list[RoleInfo] = []
    current: RoleInfo | None = None
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        role_match = _ROLE_HEADER_RE.match(stripped)
        loc_dates_match = _ROLE_LOCATION_DATES_RE.match(stripped) if current and not current.date_range_raw else None

        if stripped.startswith("•"):
            if current:
                current.achievements.append(stripped.lstrip("•").strip())
            continue

        if loc_dates_match and current and not current.date_range_raw:
            location, dates = loc_dates_match.groups()
            current.location = location.strip()
            current.date_range_raw = dates.strip()
            current.is_current = "present" in dates.lower()
            continue

        if role_match:
            company, title = role_match.groups()
            current = RoleInfo(company=company.strip(), title=title.strip())
            roles.append(current)
            continue

    return roles


def _parse_projects(lines: list[str]) -> list[ProjectInfo]:
    projects: list[ProjectInfo] = []
    current_org = ""
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        bullet_match = _PROJECT_BULLET_RE.match(stripped)
        if bullet_match:
            name, date_range, stack = bullet_match.groups()
            projects.append(
                ProjectInfo(
                    name=name.strip(),
                    organization=current_org,
                    date_range_raw=date_range.strip(),
                    tech_stack=[s.strip() for s in stack.split(",") if s.strip()],
                )
            )
            continue
        if stripped.startswith("•") or stripped.startswith("-"):
            projects.append(ProjectInfo(name=stripped.lstrip("•-").strip(), organization=current_org))
            continue
        current_org = stripped
    return projects


def _parse_education(lines: list[str]) -> list[EducationInfo]:
    entries: list[EducationInfo] = []
    pending_degree: str | None = None
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if pending_degree is None:
            pending_degree = stripped
            continue
        m = _EDUCATION_LINE_RE.match(stripped)
        if m:
            institution, date_range, location = m.groups()
            entries.append(
                EducationInfo(
                    degree=pending_degree,
                    institution=institution.strip(),
                    date_range_raw=date_range.strip(),
                    location=location.strip(),
                )
            )
        else:
            entries.append(EducationInfo(degree=pending_degree, institution=stripped))
        pending_degree = None
    if pending_degree:
        entries.append(EducationInfo(degree=pending_degree))
    return entries


def parse_summary_text(text: str) -> CandidateProfile:
    lines = text.splitlines()
    preamble, sections = _split_sections(lines)
    personal_info, headline = _parse_personal_info(preamble)

    summary_lines = sections.get("PROFILE SUMMARY", [])
    summary_text, parsed_years = _parse_summary(summary_lines, headline=headline)

    technical_skills = _parse_technical_skills(sections.get("TECHNICAL SKILLS", []))
    buckets = _bucket_technical_skills(technical_skills)

    roles = _parse_work_experience(sections.get("WORK EXPERIENCE", []))
    current_role = next((r for r in roles if r.is_current), roles[0] if roles else None)
    previous_roles = [r for r in roles if r is not current_role]

    projects = _parse_projects(sections.get("KEY PROJECTS", []))
    education = _parse_education(sections.get("EDUCATION", []))

    total_years = parsed_years if parsed_years is not None else get_settings().candidate_total_experience_years

    return CandidateProfile(
        personal_information=personal_info,
        professional_summary=summary_text,
        total_experience_years=total_years,
        current_role=current_role,
        previous_roles=previous_roles,
        technical_skills=technical_skills,
        ai_skills=buckets["ai_skills"],
        cloud_skills=buckets["cloud_skills"],
        database_skills=buckets["database_skills"],
        frontend_skills=buckets["frontend_skills"],
        backend_skills=buckets["backend_skills"],
        ai_tools=buckets["ai_tools"],
        projects=projects,
        education=education,
        domain_experience=[],
    )


def load_candidate_profile_from_path(path: Path) -> CandidateProfile:
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate profile source not found at {path}. Place a Summary.txt there "
            "(see docs/job_engine/CONFIGURATION.md)."
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    profile = parse_summary_text(text)
    logger.info(
        "Loaded candidate profile: %s skills, %s roles, %s projects",
        len(profile.all_skills()),
        1 + len(profile.previous_roles) if profile.current_role else len(profile.previous_roles),
        len(profile.projects),
    )
    return profile


@lru_cache(maxsize=1)
def load_candidate_profile() -> CandidateProfile:
    return load_candidate_profile_from_path(PROFILE_SUMMARY_PATH)
