"""The match score, computed in code from the matcher's requirement checks.

The LLM never outputs a number (a free 0-100 number drifts between runs). It
lists the job's requirements as tier + verdict labels; code then corrects the
labels it can check and applies one formula:

    S = 100 × Σ(w × v) / Σ(w)        w: core 3, required 2, preferred 1
                                      v: met 1.0, partial 0.5, missing 0.0
    E = 100                           job states no years range, or Y within it
      = max(0,  100 − 25 × (min − Y)) under-qualified
      = max(70, 100 − 10 × (Y − max)) over-qualified
    K = rule_evaluation.rule_score    keyword/stack overlap (already deterministic)

    score = round(0.90·S + 0.05·E + 0.05·K), capped at 60 if a core requirement is missing

Corrections (they override the LLM, so a flipped label doesn't move the score):

- a requirement naming a skill from the profile's skills list is "met";
- one naming a sub-topic of a skill the profile shows (``config/implied_skills.yaml``,
  e.g. SSR under Next.js) is "met";
- one marked "nice to have" / "preferred" / "a plus" / "bonus" in its text is "preferred";
- a years-of-experience line naming no profile skill is dropped (E scores it).

Location / eligibility never move the score; they are shown separately.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ..config.settings import PROJECT_ROOT, EngineSettings
from ..models.job import Job, JobMatchAnalysis, RuleEvaluation, ScoreBreakdown, ScoredRequirement
from ..models.profile import CandidateProfile

logger = logging.getLogger(__name__)

IMPLIED_SKILLS_PATH = PROJECT_ROOT / "config" / "implied_skills.yaml"

TIER_WEIGHTS = {"core": 3.0, "required": 2.0, "preferred": 1.0}
VERDICT_VALUES = {"met": 1.0, "partial": 0.5, "missing": 0.0}
SKILLS_WEIGHT, EXPERIENCE_WEIGHT, KEYWORDS_WEIGHT = 0.90, 0.05, 0.05
MISSING_CORE_CAP = 60.0
UNDER_QUALIFIED_PENALTY_PER_YEAR = 25.0
OVER_QUALIFIED_PENALTY_PER_YEAR = 10.0
OVER_QUALIFIED_FLOOR = 70.0
# Bump when the formula changes; shown with the breakdown.
FORMULA_VERSION = "2"


# --- text helpers -------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _has_phrase(haystack: str, phrase: str) -> bool:
    """``phrase`` occurs in ``haystack`` as whole words (both already normalized)."""
    phrase = _norm(phrase)
    if not phrase:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", haystack) is not None


def profile_skill_named(text: str, profile_skills: list[str]) -> str | None:
    """The first skill from the profile's skills list that ``text`` names ("React.js" also matches "React")."""
    haystack = _norm(text)
    for skill in profile_skills:
        norm = _norm(skill)
        variants = {norm, re.sub(r"\.?js$", "", norm).strip()}
        if any(len(v) >= 2 and _has_phrase(haystack, v) for v in variants):
            return skill
    return None


def fingerprint(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def profile_fingerprint(profile: CandidateProfile) -> str:
    return fingerprint(profile.to_llm_context())


# --- implied-skills map -------------------------------------------------------


@dataclass(frozen=True)
class ImpliedSkills:
    """``config/implied_skills.yaml``: parent skill -> (aliases, implied sub-topics)."""

    entries: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ImpliedSkills":
        entries = {}
        for name, spec in (data or {}).items():
            spec = spec or {}
            aliases = tuple(str(a) for a in (spec.get("aliases") or [name]))
            implies = tuple(str(i) for i in (spec.get("implies") or []))
            entries[str(name)] = (aliases, implies)
        return cls(entries=entries)

    def present(self, profile_text: str) -> list[str]:
        """Parent skills the profile shows (any alias appears in it)."""
        haystack = _norm(profile_text)
        return [name for name, (aliases, _) in self.entries.items() if any(_has_phrase(haystack, a) for a in aliases)]

    def implied_by(self, text: str, present: list[str]) -> str | None:
        """The present parent skill whose sub-topics ``text`` names, if any."""
        haystack = _norm(text)
        for name in present:
            if any(_has_phrase(haystack, sub) for sub in self.entries[name][1]):
                return name
        return None


@lru_cache(maxsize=4)
def load_implied_skills(path: Path = IMPLIED_SKILLS_PATH) -> ImpliedSkills:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Implied-skills map not found at %s; nothing will count as implied", path)
        return ImpliedSkills(entries={})
    return ImpliedSkills.from_mapping(yaml.safe_load(text) or {})


# --- experience -----------------------------------------------------------------

_YEARS_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:(?:-|–|to)\s*(\d{1,2}(?:\.\d)?)\s*\+?\s*)?(?:years?|yrs?)\b", re.IGNORECASE
)
_PREFERRED_MARKERS = re.compile(r"\b(nice to have|preferred|a plus|bonus|good to have)\b", re.IGNORECASE)


def required_years_range(job: Job) -> tuple[float, float | None] | None:
    """(min, max or None) years the job asks for: its experience field first, then
    the requirements, then the description; the first plausible match wins."""
    for text in (job.experience_required or "", " ".join(job.requirements), job.description):
        for m in _YEARS_RE.finditer(text or ""):
            low = float(m.group(1))
            high = float(m.group(2)) if m.group(2) else None
            if 0 < low <= 30 and (high is None or low <= high <= 40):
                return low, high
    return None


def experience_score(candidate_years: float, years: tuple[float, float | None] | None) -> float:
    if years is None:
        return 100.0
    low, high = years
    if candidate_years < low:
        return max(0.0, 100.0 - UNDER_QUALIFIED_PENALTY_PER_YEAR * (low - candidate_years))
    if high is not None and candidate_years > high:
        return max(OVER_QUALIFIED_FLOOR, 100.0 - OVER_QUALIFIED_PENALTY_PER_YEAR * (candidate_years - high))
    return 100.0


def _years_label(years: tuple[float, float | None] | None) -> str:
    if years is None:
        return ""
    low, high = (f"{v:g}" if v is not None else None for v in years)
    return f"{low}-{high}" if high else f"{low}+"


# --- requirements ------------------------------------------------------------------


def corrected_requirements(
    analysis: JobMatchAnalysis, profile: CandidateProfile, implied: ImpliedSkills
) -> list[ScoredRequirement]:
    profile_skills = profile.all_skills()
    present = implied.present(profile.to_llm_context())
    seen: set[str] = set()
    out: list[ScoredRequirement] = []
    for check in analysis.requirements:
        text = check.requirement.strip()
        key = _norm(text)
        if not key or key in seen:
            continue
        seen.add(key)
        listed = profile_skill_named(text, profile_skills)
        if _YEARS_RE.search(text) and not listed:
            continue  # a years line: scored by E, not S
        tier, verdict, notes = check.tier, check.verdict, []
        if tier != "preferred" and _PREFERRED_MARKERS.search(text):
            tier = "preferred"
            notes.append("marked optional in the job")
        if verdict != "met":
            parent = None if listed else implied.implied_by(text, present)
            if listed:
                verdict = "met"
                notes.append(f"{listed} is listed in the profile")
            elif parent:
                verdict = "met"
                notes.append(f"implied by {parent}")
        out.append(
            ScoredRequirement(requirement=text[:500], tier=tier, verdict=verdict, adjusted="; ".join(notes))
        )
    return out


def candidate_years(profile: CandidateProfile, settings: EngineSettings) -> float:
    return profile.total_experience_years or settings.candidate_total_experience_years


def compute_fit_score(
    *,
    analysis: JobMatchAnalysis,
    job: Job,
    profile: CandidateProfile,
    rule_eval: RuleEvaluation,
    settings: EngineSettings,
    implied: ImpliedSkills | None = None,
) -> ScoreBreakdown:
    requirements = corrected_requirements(analysis, profile, implied or load_implied_skills())
    keywords = round(min(100.0, max(0.0, rule_eval.rule_score)), 1)
    total = sum(TIER_WEIGHTS[r.tier] for r in requirements)
    if total:
        skills = 100.0 * sum(TIER_WEIGHTS[r.tier] * VERDICT_VALUES[r.verdict] for r in requirements) / total
    else:
        skills = keywords  # no requirements listed: fall back to the keyword overlap
    years = required_years_range(job)
    y = candidate_years(profile, settings)
    experience = experience_score(y, years)

    score = float(round(SKILLS_WEIGHT * skills + EXPERIENCE_WEIGHT * experience + KEYWORDS_WEIGHT * keywords))
    core_missing = any(r.tier == "core" and r.verdict == "missing" for r in requirements)
    if core_missing:
        score = min(score, MISSING_CORE_CAP)
    return ScoreBreakdown(
        score=min(100.0, max(0.0, score)),
        skills=round(skills, 1),
        experience=round(experience, 1),
        keywords=keywords,
        core_missing=core_missing,
        required_years=_years_label(years),
        candidate_years=y,
        requirements=requirements,
        formula_version=FORMULA_VERSION,
    )
