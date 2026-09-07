"""Candidate profile schema — normalized representation of Profile/Summary.txt.

Nothing here is hand-authored candidate data; ``jobEngine/services/profile_loader.py``
parses the real ``Profile/Summary.txt`` file at runtime into these models. If a
section is missing/unparseable it is left empty rather than guessed, per the
"no invented skills/experience" requirement.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PersonalInfo(BaseModel):
    """``phone`` is loaded but must never be forwarded to an LLM prompt."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    location: str = ""
    linkedin_url: str = ""
    email: str = ""
    phone: str = ""


class RoleInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str
    title: str
    location: str = ""
    date_range_raw: str = ""
    is_current: bool = False
    achievements: list[str] = Field(default_factory=list)


class ProjectInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    organization: str = ""
    date_range_raw: str = ""
    tech_stack: list[str] = Field(default_factory=list)


class EducationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    degree: str
    institution: str = ""
    date_range_raw: str = ""
    location: str = ""


class CandidateProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    personal_information: PersonalInfo = Field(default_factory=PersonalInfo)
    professional_summary: str = ""
    total_experience_years: float = 0.0
    current_role: RoleInfo | None = None
    previous_roles: list[RoleInfo] = Field(default_factory=list)

    technical_skills: dict[str, list[str]] = Field(default_factory=dict)
    ai_skills: list[str] = Field(default_factory=list)
    cloud_skills: list[str] = Field(default_factory=list)
    database_skills: list[str] = Field(default_factory=list)
    frontend_skills: list[str] = Field(default_factory=list)
    backend_skills: list[str] = Field(default_factory=list)
    ai_tools: list[str] = Field(default_factory=list)

    projects: list[ProjectInfo] = Field(default_factory=list)
    education: list[EducationInfo] = Field(default_factory=list)
    domain_experience: list[str] = Field(default_factory=list)

    def all_skills(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        buckets = [
            self.ai_skills,
            self.cloud_skills,
            self.database_skills,
            self.frontend_skills,
            self.backend_skills,
            self.ai_tools,
            *self.technical_skills.values(),
        ]
        for bucket in buckets:
            for skill in bucket:
                key = skill.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(skill.strip())
        return out

    def to_llm_context(self) -> str:
        """Text block safe to send to an LLM: no phone number, no raw email."""

        lines: list[str] = []
        name = self.personal_information.name or "Candidate"
        lines.append(f"Name: {name}")
        if self.personal_information.location:
            lines.append(f"Location: {self.personal_information.location}")
        lines.append(f"Total experience: {self.total_experience_years} years")
        if self.professional_summary:
            lines.append(f"Summary: {self.professional_summary}")

        if self.current_role:
            r = self.current_role
            lines.append(f"Current role: {r.title} at {r.company} ({r.date_range_raw})")
            if r.achievements:
                lines.append("Current role highlights:")
                lines.extend(f"  - {a}" for a in r.achievements)

        if self.previous_roles:
            lines.append("Previous roles:")
            for r in self.previous_roles:
                lines.append(f"  - {r.title} at {r.company} ({r.date_range_raw})")

        lines.append("Skills: " + ", ".join(self.all_skills()))

        if self.projects:
            lines.append("Key projects:")
            for p in self.projects:
                stack = f" — {', '.join(p.tech_stack)}" if p.tech_stack else ""
                org = f" [{p.organization}]" if p.organization else ""
                lines.append(f"  - {p.name}{org}{stack}")

        if self.domain_experience:
            lines.append("Domain experience: " + ", ".join(self.domain_experience))

        if self.education:
            lines.append("Education:")
            for e in self.education:
                lines.append(f"  - {e.degree}, {e.institution} ({e.date_range_raw})")

        return "\n".join(lines)
