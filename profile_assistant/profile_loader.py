"""Load, validate, cache and render the profile used as the only knowledge source.

Supports ``.json`` (validated against ``ProfileData``), ``.md``/``.txt``
(free text; the first non-empty line is taken as the person's name) and
``.pdf`` (e.g. a LinkedIn export). The primary profile can be combined with
supplementary sources (``PA_EXTRA_PROFILE_PATHS``); each is labeled in the
text so the agents know which one is primary. Phone numbers and email
addresses are redacted unless ``PA_REDACT_CONTACT_DETAILS=false``. The
loaded profile is cached in memory; ``reload_profile()`` re-reads it from disk
so the profile can be changed without restarting the app.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .config import get_settings

SUPPORTED_SUFFIXES = (".json", ".md", ".txt", ".pdf")


class ProfileLoadError(ValueError):
    """Raised when the profile file is missing, unsupported or invalid."""


class Experience(BaseModel):
    """One role in the person's work history."""

    model_config = ConfigDict(extra="allow")

    title: str
    company: str
    location: str = ""
    start: str = ""
    end: str = ""
    highlights: list[str] = Field(default_factory=list)


class Education(BaseModel):
    """One education entry."""

    model_config = ConfigDict(extra="allow")

    degree: str
    institution: str
    location: str = ""
    start: str = ""
    end: str = ""
    details: str = ""


class Project(BaseModel):
    """One project the person worked on."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    duration: str = ""
    technologies: list[str] = Field(default_factory=list)
    highlights: list[str] = Field(default_factory=list)


class ProfileData(BaseModel):
    """Schema for a JSON profile. Unknown top-level fields are kept and rendered too."""

    model_config = ConfigDict(extra="allow")

    name: str
    headline: str = ""
    summary: str = ""
    skills: dict[str, list[str]] | list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    contact_preferences: str | list[str] | dict[str, str] = ""

    @field_validator("name")
    @classmethod
    def _name_required(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must not be empty")
        return value.strip()


@dataclass(frozen=True)
class LoadedProfile:
    """A validated profile plus the text block the agents are grounded on."""

    name: str
    text: str
    source_path: Path


_lock = threading.Lock()
_cached: LoadedProfile | None = None


def _bullets(items: list[str], indent: str = "- ") -> list[str]:
    return [f"{indent}{item}" for item in items if str(item).strip()]


def _render_value(value: Any) -> list[str]:
    """Render an arbitrary JSON value (used for extra, unmodelled fields)."""
    if isinstance(value, dict):
        return [f"- {key}: {', '.join(map(str, v)) if isinstance(v, list) else v}" for key, v in value.items()]
    if isinstance(value, list):
        return _bullets([json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v) for v in value])
    return [str(value)]


def profile_to_text(profile: ProfileData) -> str:
    """Convert a ``ProfileData`` into a clean, labeled text block."""
    lines: list[str] = [f"NAME: {profile.name}"]
    if profile.headline:
        lines.append(f"HEADLINE: {profile.headline}")
    if profile.summary:
        lines += ["", "SUMMARY:", profile.summary.strip()]

    if profile.skills:
        lines += ["", "SKILLS:"]
        if isinstance(profile.skills, dict):
            lines += [f"- {category}: {', '.join(items)}" for category, items in profile.skills.items()]
        else:
            lines.append(f"- {', '.join(profile.skills)}")

    if profile.experience:
        lines += ["", "EXPERIENCE:"]
        for job in profile.experience:
            period = " - ".join(p for p in (job.start, job.end) if p)
            where = f" ({job.location})" if job.location else ""
            lines.append(f"- {job.title} at {job.company}{where}" + (f" | {period}" if period else ""))
            lines += _bullets(job.highlights, "  * ")

    if profile.projects:
        lines += ["", "PROJECTS:"]
        for project in profile.projects:
            lines.append(f"- {project.name}" + (f" | {project.duration}" if project.duration else ""))
            if project.description:
                lines.append(f"  Description: {project.description}")
            if project.technologies:
                lines.append(f"  Technologies: {', '.join(project.technologies)}")
            lines += _bullets(project.highlights, "  * ")

    if profile.education:
        lines += ["", "EDUCATION:"]
        for edu in profile.education:
            period = " - ".join(p for p in (edu.start, edu.end) if p)
            where = f", {edu.location}" if edu.location else ""
            lines.append(f"- {edu.degree}, {edu.institution}{where}" + (f" | {period}" if period else ""))
            if edu.details:
                lines.append(f"  {edu.details}")

    for label, items in (
        ("CERTIFICATIONS", profile.certifications),
        ("LANGUAGES", profile.languages),
        ("INTERESTS", profile.interests),
    ):
        if items:
            lines += ["", f"{label}:", *_bullets(items)]

    if profile.contact_preferences:
        lines += ["", "CONTACT PREFERENCES:", *_render_value(profile.contact_preferences)]

    for key, value in (profile.model_extra or {}).items():
        if value in (None, "", [], {}):
            continue
        lines += ["", f"{key.replace('_', ' ').upper()}:", *_render_value(value)]

    return "\n".join(lines).strip()


def _read_pdf_text(path: Path) -> str:
    """Extract the text of a PDF (e.g. a LinkedIn "Save to PDF" export)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ProfileLoadError("pypdf is required to read PDF profiles") from exc
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # noqa: BLE001 - any parser failure is a bad profile file
        raise ProfileLoadError(f"Could not read PDF profile {path}: {exc}") from exc


def clean_pdf_text(text: str) -> str:
    """Tidy extracted PDF text: page footers, HTML entities and mis-decoded glyphs."""
    text = text.replace("&amp;", "&").replace(" ", " ")
    text = re.sub(r"(?<=\w)�(?=\w)", "'", text)  # I�m -> I'm
    text = re.sub(r"\s*�-�\s*", " - ", text)  # date ranges
    text = text.replace("�", " ")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or re.fullmatch(r"Page \d+ of \d+", line):
            continue
        lines.append(line)
    return "\n".join(lines)


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"(?<![\w/])\+?\(?\d[\d\s().-]{7,}\d(?![\w/])")


def redact_contact_details(text: str) -> str:
    """Remove email addresses and phone numbers (10+ digits); URLs and years are kept."""
    text = _EMAIL_RE.sub("[email removed]", text)
    return _PHONE_RE.sub(
        lambda m: "[phone removed]" if sum(c.isdigit() for c in m.group()) >= 10 else m.group(), text
    )


def _load_source(path: Path) -> tuple[str | None, str]:
    """Load one profile file; returns (name if the file defines one, text)."""
    if not path.is_file():
        raise ProfileLoadError(f"Profile file not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ProfileLoadError(f"Unsupported profile format {suffix!r}; use one of {SUPPORTED_SUFFIXES}")

    if suffix == ".json":
        try:
            data = ProfileData.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ProfileLoadError(f"Profile {path} is not valid JSON: {exc}") from exc
        except ValidationError as exc:
            raise ProfileLoadError(f"Profile {path} failed validation:\n{exc}") from exc
        return data.name, profile_to_text(data)

    text = clean_pdf_text(_read_pdf_text(path)) if suffix == ".pdf" else path.read_text(encoding="utf-8").strip()
    if not text:
        raise ProfileLoadError(f"Profile {path} is empty")
    return None, text


def _load_from_disk(path: Path, extra_paths: tuple[Path, ...] = (), *, redact: bool = True) -> LoadedProfile:
    """Load the primary profile plus any supplementary sources into one text block."""
    name, text = _load_source(path)
    if name is None:
        first_line = next(line for line in text.splitlines() if line.strip())
        name = first_line.strip().lstrip("#").strip()

    if extra_paths:
        sections = [f"=== SOURCE 1 (primary): {path.name} ===\n{text}"]
        for index, extra in enumerate(extra_paths, start=2):
            _, extra_text = _load_source(extra)
            sections.append(
                f"=== SOURCE {index} (supplementary): {extra.name} ===\n"
                "(Use this source for additional details. Where it gives a different title, date, number or "
                "duration than SOURCE 1 for the same fact, SOURCE 1 is correct and this value must not be used.)\n"
                f"{extra_text}"
            )
        text = "\n\n".join(sections)
    if redact:
        text = redact_contact_details(text)
    return LoadedProfile(name=name, text=text, source_path=path)


def _load_configured() -> LoadedProfile:
    settings = get_settings()
    return _load_from_disk(
        settings.profile_path, settings.extra_profile_paths, redact=settings.redact_contact_details
    )


def get_profile() -> LoadedProfile:
    """Return the cached profile, loading it on first use."""
    global _cached
    if _cached is None:
        with _lock:
            if _cached is None:
                _cached = _load_configured()
    return _cached


def reload_profile() -> LoadedProfile:
    """Re-read the profile from disk and replace the cache (hot reload).

    On failure the previously cached profile is kept and the error is raised.
    """
    global _cached
    profile = _load_configured()
    with _lock:
        _cached = profile
    return profile


def clear_profile_cache() -> None:
    """Drop the cached profile (used by tests)."""
    global _cached
    with _lock:
        _cached = None
