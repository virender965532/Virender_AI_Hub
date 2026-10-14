"""Can this job be done remotely by someone living in India? Rule-based
classification with quoted evidence (``JOB_WORLDWIDE_REMOTE_ENABLED``).

Categories (``models.job.WorldwideCategory``):
  A worldwide_confirmed — explicit "work from anywhere in the world" / "contractors
    in any country" text, with nothing in the listing contradicting it.
  B india_eligible      — India explicitly named as allowed ("candidates based in
    India", "Remote — India"), or a remote job posted for an Indian location.
  C needs_verification  — remote, but no policy that settles India either way
    ("Remote", "Remote — US", "APAC"), or positive and restrictive text conflict.
  D country_restricted  — residence/location/work-authorization requirement for
    another country, India explicitly excluded, or on-site/hybrid abroad.
  E insufficient_info   — description missing/too short, or no remote signal at all.
  india_local           — posted in India and not remote (on-site/hybrid/unknown).

The description is split into sentences and every rule runs per sentence, so the
evidence stored is the exact sentence that matched. A sentence that names India
as allowed never counts as a restriction ("must be located in the US or India").
Jobs the rules leave at C can get an LLM second opinion (``worldwide_llm.py``).
"""

from __future__ import annotations

import re

from ..models.job import (
    EmploymentArrangement,
    RemoteType,
    WorldwideCategory,
    WorldwideClassification,
    WorldwideEvidence,
)
from .global_remote import is_india_location

_I = re.IGNORECASE

# Shorter than this and the posting can't be checked for location restrictions.
MIN_DESCRIPTION_CHARS = 200
_MAX_EVIDENCE_CHARS = 300

CATEGORY_LABELS = {
    WorldwideCategory.WORLDWIDE_CONFIRMED: "Worldwide Remote — Confirmed",
    WorldwideCategory.INDIA_ELIGIBLE: "International Remote — India Eligible",
    WorldwideCategory.NEEDS_VERIFICATION: "Potentially Worldwide — Needs Verification",
    WorldwideCategory.COUNTRY_RESTRICTED: "Country Restricted — Not Eligible from India",
    WorldwideCategory.INSUFFICIENT_INFO: "Insufficient Information",
    WorldwideCategory.INDIA_LOCAL: "India-based — Not Remote",
}

# Regions that do not include India. "US" is matched case-sensitively so the
# pronoun "us" ("join us", "contact us only") never reads as the country.
_EXCLUDING = (
    r"(?:(?-i:US|U\.S\.?A?\.?|USA)|united\s+states|america|north\s+america|americas|canada|mexico|"
    r"latam|latin\s+america|south\s+america|brazil|argentina|colombia|chile|"
    r"(?-i:UK)|u\.k\.|united\s+kingdom|great\s+britain|britain|england|scotland|ireland|"
    r"(?-i:EU|EEA)|e\.u\.|europe|european\s+union|european|emea|central\s+europe|eastern\s+europe|"
    r"western\s+europe|nordics?|dach|benelux|baltics?|germany|france|spain|portugal|italy|"
    r"netherlands|belgium|poland|romania|ukraine|switzerland|austria|sweden|norway|denmark|"
    r"finland|czechia|czech\s+republic|hungary|greece|turkey|serbia|croatia|bulgaria|estonia|"
    r"latvia|lithuania|israel|uae|united\s+arab\s+emirates|dubai|saudi\s+arabia|qatar|africa|"
    r"nigeria|kenya|egypt|south\s+africa|singapore|philippines|vietnam|indonesia|malaysia|"
    r"thailand|japan|korea|china|hong\s+kong|taiwan|pakistan|australia|new\s+zealand|"
    r"(?-i:ANZ|EST|PST|CST|CET|CEST|GMT|BST))"
)
# Regions that contain India: "APAC only" doesn't exclude an India-based applicant.
_INCLUSIVE = r"(?:apac|asia[\s-]pacific|asia|south\s+asia)"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+|\s+[•·▪●]\s+")

_WORLDWIDE_RES = [
    re.compile(p, _I)
    for p in (
        r"\b(?:worldwide|global(?:ly)?)[\s,-]+remote\b",
        r"\bremote\s*[-–—(,:/|]\s*(?:worldwide|global|anywhere)\b",
        r"\b(?:fully\s+)?remote,?\s+(?:anywhere|worldwide|globally)\b",
        r"\b(?:work|working|live|living|based|remotely)\s+(?:remotely\s+)?from\s+anywhere\b",
        r"\banywhere\s+in\s+the\s+world\b",
        r"\b(?:from|in)\s+any\s+(?:country|location|time\s*zone)\b",
        r"\b(?:open\s+to|accept\w*|welcom\w*|hir(?:e|es|ing)|recruit\w*|consider\w*)\s+(?:[\w-]+\s+){0,3}?"
        r"(?:from\s+)?(?:globally|worldwide|all\s+over\s+the\s+world|around\s+the\s+world|anywhere|"
        r"every\s+country|all\s+countries)\b",
        r"\b(?:candidates|applicants|talent|contractors|engineers|developers|people)\s+(?:from\s+|based\s+)?"
        r"(?:anywhere|worldwide|globally|around\s+the\s+world|all\s+over\s+the\s+world)\b",
        r"\blocation[\s-]+(?:independent|agnostic)\b",
        r"\bno\s+(?:location|geographic(?:al)?|country)\s+restrictions?\b",
    )
]
# "Work from anywhere in the US" is a restriction, not worldwide.
_SCOPED_ANYWHERE_RE = re.compile(
    r"\banywhere\s+(?:in|within|across|throughout|inside)\s+(?!(?:the\s+)?(?:world|globe)\b)[A-Za-z]", _I
)
_CONTRACTOR_WORD_RE = re.compile(r"\b(?:contractors?|freelanc\w*|b2b|independent)\b", _I)

_INDIA = r"\bindia\b|\bindian\s+(?:candidates|applicants|residents|engineers|developers|contractors|talent)"
_INDIA_NEGATIVE_RES = [
    re.compile(
        r"\b(?:except|excluding|exclude[sd]?|not\s+(?:open|available|eligible|accepting|considering|hiring)"
        r"|cannot\s+(?:hire|accept|consider)|can't\s+(?:hire|accept|consider)|unable\s+to\s+(?:hire|accept|consider)"
        r"|no\s+(?:candidates|applicants)\s+from)[^.;\n]{0,60}?\bindia\b",
        _I,
    ),
    re.compile(r"\bindia\b[^.;\n]{0,40}?\b(?:not\s+eligible|is\s+excluded|are\s+excluded|not\s+supported|not\s+accepted)\b", _I),
]
_INDIA_POSITIVE_RES = [
    re.compile(
        rf"\b(?:hir(?:e|es|ing)|recruit\w*|open\s+to|accept\w*|welcom\w*|eligib\w*|consider\w*)\b[^.;\n]{{0,60}}?(?:{_INDIA})",
        _I,
    ),
    re.compile(
        r"\b(?:candidates?|applicants?|contractors?|talent|engineers|developers)\s+(?:who\s+(?:are|live)\s+)?"
        rf"(?:from|in|based\s+in|located\s+in|residing\s+in|living\s+in|across)\b[^.;\n]{{0,60}}?(?:{_INDIA})",
        _I,
    ),
    re.compile(
        r"\b(?:(?:based|located|residing|reside|living|live)\s+in|remote(?:ly)?\s+(?:from|in|within)|"
        rf"work\s+from|apply\s+from)\s+(?:[\w,]+\s+){{0,6}}?(?:or\s+|and\s+)?(?:{_INDIA})",
        _I,
    ),
    re.compile(
        r"\bindia\b[^.;\n]{0,30}?\b(?:eligible|welcome|candidates|applicants|contractors|engineers|developers|is\s+ok|is\s+fine)\b",
        _I,
    ),
    re.compile(r"\bindia[\s-](?:based|only|remote)\b|\bpan[\s-]india\b|\bremote\s*[-–—(,:/|]\s*india\b", _I),
]
# Company facts ("offices in India", "clients in India") are not hiring policy.
_INDIA_COMPANY_FACT_RE = re.compile(
    r"\b(?:office|offices|headquarter\w*|hq|subsidiar\w*|clients?|customers?|market|presence)\b[^.;\n]{0,40}?\bindia\b", _I
)

_HARD_RESTRICTION_RES = [
    re.compile(p, _I)
    for p in (
        rf"\b{_EXCLUDING}[\s-]*(?:only|based\s+only|residents?\s+only|citizens?\s+only)\b",
        rf"\b{_EXCLUDING}[\s-]based\s+(?:candidates|applicants|employees|contractors|talent|only|individuals)\b",
        r"\b(?:only|exclusively)\s+(?:open\s+to|hiring|accepting|considering|available\s+(?:to|for)|for)\s+"
        rf"[^.;\n]{{0,60}}?\b(?:in|from|within|based\s+in|residing\s+in|located\s+in|residents\s+of)\s+(?:the\s+)?{_EXCLUDING}",
        r"\b(?:must|should|need\s+to|needs\s+to|required\s+to|have\s+to)\s+(?:be\s+)?(?:currently\s+)?"
        r"(?:located|based|residing|reside|live|living|a\s+resident|resident|physically\s+located)\s+(?:in|within|of)\s+"
        rf"(?:the\s+)?(?:one\s+of\s+)?[^.;\n]{{0,30}}?{_EXCLUDING}",
        rf"\b(?:within|in)\s+(?:the\s+)?{_EXCLUDING}\s+only\b",
        rf"\banywhere\s+(?:in|within|across|throughout)\s+(?:the\s+)?(?:continental\s+)?{_EXCLUDING}",
        rf"\bresidents?\s+of\s+(?:the\s+)?{_EXCLUDING}",
        r"\b(?:cannot|can\s*not|can't|unable\s+to|do\s+not|don't|won't|will\s+not)\s+(?:hire|consider|accept|employ)\s+"
        rf"[^.;\n]{{0,40}}?\boutside\s+(?:of\s+)?(?:the\s+)?{_EXCLUDING}",
        rf"\b{_EXCLUDING}\s+citizen(?:s|ship)?\b[^.;\n]{{0,20}}?\b(?:required|only|must)\b",
        r"\b(?:security|government|secret|top\s+secret)\s+clearance\b",
    )
]
_AUTH_RESTRICTION_RES = [
    re.compile(p, _I)
    for p in (
        r"\b(?:authori[sz]ed|authori[sz]ation|eligible|eligibility|legally\s+(?:able|permitted|allowed)|right)\s+"
        rf"to\s+work\s+(?:in|for)\s+(?:the\s+)?{_EXCLUDING}",
        rf"\b{_EXCLUDING}\s+work\s+(?:authori[sz]ation|permit|visa)\b",
        r"\bwithout\s+(?:the\s+need\s+for\s+)?(?:current\s+or\s+future\s+)?(?:visa\s+)?sponsorship\b",
        r"\b(?:green\s+card|(?-i:W-?2)\b)",
    )
]
_SCOPED_REMOTE_RES = [
    re.compile(p, _I)
    for p in (
        rf"\bremote\s*[-–—(,:/|]\s*(?:the\s+)?(?:{_EXCLUDING}|{_INCLUSIVE})\b",
        rf"\b(?:{_EXCLUDING}|{_INCLUSIVE})[\s-]+remote\b",
        rf"\bremote(?:ly)?\s+(?:in|within|across|from)\s+(?:the\s+)?(?:{_EXCLUDING}|{_INCLUSIVE})\b",
        rf"\b(?:{_EXCLUDING}|{_INCLUSIVE})\s*\((?:remote)\)",
    )
]
_REMOTE_ROLE_RE = re.compile(
    r"\b(?:fully|100\s*%|completely|entirely)[\s-]+remote\b|\bremote[\s-]+(?:role|position|job|first|work|opportunity)\b"
    r"|\bwork(?:ing)?\s+remotely\b|\bremote\b",
    _I,
)
_LOCATION_WORLDWIDE_RE = re.compile(r"\b(?:worldwide|anywhere|global)\b", _I)
_LOCATION_ONSITE_RE = re.compile(r"\((?:on[\s-]?site|hybrid)\)", _I)

# --- employment arrangement ---
_FREELANCE_RE = re.compile(r"\bfreelanc\w*|\bproject[\s-]based\b|\bper[\s-]project\b", _I)
_CONTRACT_RE = re.compile(
    r"\bcontract(?:ual)?\s+(?:role|position|basis|engagement|opportunity|job|work)\b|\bon\s+a\s+contract(?:ual)?\s+basis\b"
    r"|\bas\s+an?\s+(?:independent\s+)?contractor\b|\b(?:full|part)[\s-]time\s+contract(?:or)?\b"
    r"|\bcontract[\s-]to[\s-]hire\b|\bb2b\s+contract\b|\b1099\b|\bc2c\b|\bcorp[\s-]to[\s-]corp\b"
    r"|\bindependent\s+contractor\s+(?:role|position|agreement|basis)\b",
    _I,
)
_TEMPORARY_RE = re.compile(r"\btemporary\b|\bfixed[\s-]term\b|\b\d{1,2}\s*(?:-|to)?\s*\d{0,2}[\s-]*months?\s+contract\b", _I)
_PART_TIME_RE = re.compile(r"\bpart[\s-]time\b|\b(?:[5-9]|[12]\d|30)\s*(?:-\s*(?:[5-9]|[12]\d|30)\s*)?(?:hours|hrs)\s*(?:per|/|a)\s*week\b", _I)
_FULL_TIME_RE = re.compile(r"\bfull[\s-]time\b|\b(?:40)\s*(?:hours|hrs)\s*(?:per|/|a)\s*week\b", _I)


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_EVIDENCE_CHARS else text[: _MAX_EVIDENCE_CHARS - 1] + "…"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text or "") if s and s.strip()]


def _any(patterns: list[re.Pattern], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def _ev(text: str, found_in: str, signal: str, rule: str) -> WorldwideEvidence:
    return WorldwideEvidence(text=_clip(text), found_in=found_in, signal=signal, rule=rule)


def detect_arrangement(
    employment_type: str | None, title: str, description: str
) -> tuple[EmploymentArrangement, WorldwideEvidence | None]:
    """Employee vs contractor vs freelance, from LinkedIn's employment type first,
    then the title and description. "Contract" is about the arrangement only — it
    says nothing about where the contractor may live."""
    et = (employment_type or "").strip()
    text = f"{title}\n{description}"
    lowered_et = et.lower()

    def _first(pattern: re.Pattern) -> str | None:
        for s in _sentences(text):
            if pattern.search(s):
                return s
        return None

    is_freelance = "freelance" in lowered_et or bool(_FREELANCE_RE.search(text))
    is_contract = "contract" in lowered_et or bool(_CONTRACT_RE.search(text))
    is_temporary = "temporary" in lowered_et or bool(_TEMPORARY_RE.search(text))
    is_part_time = "part" in lowered_et or bool(_PART_TIME_RE.search(text))
    is_full_time = "full" in lowered_et or bool(_FULL_TIME_RE.search(text))

    if is_freelance:
        arrangement, pattern = EmploymentArrangement.FREELANCE, _FREELANCE_RE
    elif is_contract and is_temporary:
        arrangement, pattern = EmploymentArrangement.TEMPORARY_CONTRACT, _TEMPORARY_RE
    elif is_contract and is_part_time:
        arrangement, pattern = EmploymentArrangement.PART_TIME_CONTRACTOR, _PART_TIME_RE
    elif is_contract and is_full_time:
        arrangement, pattern = EmploymentArrangement.FULL_TIME_CONTRACTOR, _CONTRACT_RE
    elif is_contract:
        arrangement, pattern = EmploymentArrangement.CONTRACTOR, _CONTRACT_RE
    elif is_temporary:
        arrangement, pattern = EmploymentArrangement.TEMPORARY_CONTRACT, _TEMPORARY_RE
    elif is_part_time:
        arrangement, pattern = EmploymentArrangement.PART_TIME, _PART_TIME_RE
    elif is_full_time:
        arrangement, pattern = EmploymentArrangement.FULL_TIME_EMPLOYEE, _FULL_TIME_RE
    else:
        return EmploymentArrangement.OTHER, None

    if et:
        return arrangement, _ev(et, "employment_type", "neutral", "employment_type")
    quote = _first(pattern) or _first(_CONTRACT_RE)
    return arrangement, (_ev(quote, "description", "neutral", "arrangement") if quote else None)


def is_contractor_arrangement(arrangement: EmploymentArrangement) -> bool:
    return arrangement in (
        EmploymentArrangement.FULL_TIME_CONTRACTOR,
        EmploymentArrangement.PART_TIME_CONTRACTOR,
        EmploymentArrangement.CONTRACTOR,
        EmploymentArrangement.FREELANCE,
        EmploymentArrangement.TEMPORARY_CONTRACT,
    )


def _result(category, arrangement, evidence, reason) -> WorldwideClassification:
    return WorldwideClassification(
        category=category,
        india_eligible=category
        in (WorldwideCategory.WORLDWIDE_CONFIRMED, WorldwideCategory.INDIA_ELIGIBLE, WorldwideCategory.INDIA_LOCAL),
        arrangement=arrangement,
        evidence=evidence,
        reason=reason,
    )


def classify_worldwide(
    *,
    title: str,
    location: str | None,
    description: str,
    employment_type: str | None = None,
    workplace_type: str | None = None,
    remote_type: RemoteType | str | None = None,
) -> WorldwideClassification:
    """Classify one job. ``workplace_type`` is LinkedIn's explicit Remote/Hybrid/
    On-site chip (when read); ``remote_type`` is the pipeline's inferred value."""
    title = title or ""
    location = location or ""
    description = description or ""
    workplace = (workplace_type or "").strip().lower()
    remote_type = RemoteType(remote_type) if remote_type else RemoteType.UNKNOWN

    arrangement, arrangement_ev = detect_arrangement(employment_type, title, description)
    base_ev = [arrangement_ev] if arrangement_ev else []

    explicit_onsite = workplace in ("on-site", "onsite", "hybrid") or bool(_LOCATION_ONSITE_RE.search(location))
    remote_signal = (
        workplace == "remote"
        or remote_type == RemoteType.REMOTE
        or "remote" in location.lower()
        or "remote" in title.lower()
        or bool(_REMOTE_ROLE_RE.search(description))
    )
    workplace_ev = [_ev(workplace_type, "workplace_type", "neutral", "workplace_type")] if workplace_type else []

    # --- Posted for an Indian location ---
    if is_india_location(location):
        loc_ev = _ev(location, "location", "positive", "india_location")
        if remote_signal and not explicit_onsite:
            return _result(
                WorldwideCategory.INDIA_ELIGIBLE, arrangement, [loc_ev, *workplace_ev, *base_ev],
                "Remote job posted for an Indian location (the employer may be Indian or international).",
            )
        return _result(
            WorldwideCategory.INDIA_LOCAL, arrangement, [loc_ev, *workplace_ev, *base_ev],
            "Posted in India and not marked remote.",
        )

    result = _classify_listing(
        title=title, location=location, description=description, arrangement=arrangement,
        base_ev=base_ev, workplace_ev=workplace_ev, explicit_onsite=explicit_onsite, remote_signal=remote_signal,
    )
    if len(description.strip()) < MIN_DESCRIPTION_CHARS:
        # Too little text to rule out a restriction elsewhere: keep only a restriction,
        # or a positive verdict whose evidence is in the (short) description itself.
        decisive = result.category == WorldwideCategory.COUNTRY_RESTRICTED or (
            result.category in (WorldwideCategory.WORLDWIDE_CONFIRMED, WorldwideCategory.INDIA_ELIGIBLE)
            and any(e.signal == "positive" and e.found_in == "description" for e in result.evidence)
        )
        if not decisive:
            return _result(
                WorldwideCategory.INSUFFICIENT_INFO, arrangement,
                [*([_ev(location, "location", "neutral", "location")] if location else []), *workplace_ev, *base_ev],
                "Job description missing or too short to check location restrictions.",
            )
    return result


def _classify_listing(
    *, title, location, description, arrangement, base_ev, workplace_ev, explicit_onsite, remote_signal
) -> WorldwideClassification:
    positive: list[WorldwideEvidence] = []
    india_positive: list[WorldwideEvidence] = []
    india_negative: list[WorldwideEvidence] = []
    hard: list[WorldwideEvidence] = []
    auth: list[WorldwideEvidence] = []
    scoped: list[WorldwideEvidence] = []

    fields = [("title", title), ("location", location)] + [("description", s) for s in _sentences(description)]
    for found_in, text in fields:
        if not text:
            continue
        if found_in == "location" and _LOCATION_WORLDWIDE_RE.search(text):
            positive.append(_ev(text, found_in, "positive", "worldwide_location"))
            continue
        if _any(_INDIA_NEGATIVE_RES, text):
            india_negative.append(_ev(text, found_in, "negative", "india_excluded"))
            continue
        if _any(_INDIA_POSITIVE_RES, text) and not _INDIA_COMPANY_FACT_RE.search(text):
            india_positive.append(_ev(text, found_in, "positive", "india_named_eligible"))
            continue
        if _any(_WORLDWIDE_RES, text) and not _SCOPED_ANYWHERE_RE.search(text):
            positive.append(_ev(text, found_in, "positive", "worldwide"))
        if _any(_HARD_RESTRICTION_RES, text):
            hard.append(_ev(text, found_in, "negative", "residence_requirement"))
        elif _any(_AUTH_RESTRICTION_RES, text):
            auth.append(_ev(text, found_in, "negative", "work_authorization"))
        elif _any(_SCOPED_REMOTE_RES, text):
            scoped.append(_ev(text, found_in, "neutral", "region_scoped_remote"))

    extra = [*workplace_ev, *base_ev]
    contractor_role = is_contractor_arrangement(arrangement)

    if india_negative:
        return _result(WorldwideCategory.COUNTRY_RESTRICTED, arrangement, [*india_negative, *extra],
                       "The listing explicitly excludes India.")

    if india_positive:
        if hard:
            return _result(WorldwideCategory.NEEDS_VERIFICATION, arrangement, [*india_positive, *hard, *extra],
                           "India is named, but another sentence requires residence elsewhere — check which applies.")
        category = WorldwideCategory.WORLDWIDE_CONFIRMED if positive else WorldwideCategory.INDIA_ELIGIBLE
        return _result(category, arrangement, [*india_positive, *positive, *auth, *extra],
                       "The listing explicitly names India as an allowed location.")

    if positive:
        contractor_only = all(_CONTRACTOR_WORD_RE.search(e.text) for e in positive)
        if explicit_onsite:
            return _result(WorldwideCategory.NEEDS_VERIFICATION, arrangement, [*positive, *extra],
                           "Worldwide wording, but LinkedIn marks the job on-site/hybrid.")
        if hard:
            return _result(WorldwideCategory.NEEDS_VERIFICATION, arrangement, [*positive, *hard, *extra],
                           "Worldwide wording conflicts with a residence requirement elsewhere in the listing.")
        if contractor_only and not contractor_role:
            return _result(WorldwideCategory.NEEDS_VERIFICATION, arrangement, [*positive, *auth, *extra],
                           "The worldwide policy mentions contractors, but this role is not listed as a contract — "
                           "the policy may not apply to it.")
        if auth and not contractor_only:
            return _result(WorldwideCategory.NEEDS_VERIFICATION, arrangement, [*positive, *auth, *extra],
                           "Worldwide wording, but the listing also asks for work authorization in another country.")
        return _result(WorldwideCategory.WORLDWIDE_CONFIRMED, arrangement, [*positive, *auth, *extra],
                       "The listing explicitly allows working from anywhere.")

    if hard:
        return _result(WorldwideCategory.COUNTRY_RESTRICTED, arrangement, [*hard, *auth, *extra],
                       "The listing requires living in a country other than India.")
    if explicit_onsite:
        return _result(WorldwideCategory.COUNTRY_RESTRICTED, arrangement,
                       [_ev(location, "location", "negative", "onsite_abroad"), *extra] if location else extra,
                       "On-site/hybrid role outside India.")
    if auth:
        if contractor_role:
            return _result(WorldwideCategory.NEEDS_VERIFICATION, arrangement, [*auth, *scoped, *extra],
                           "Asks for work authorization abroad, but the role is a contract — the requirement "
                           "may not apply to contractors.")
        return _result(WorldwideCategory.COUNTRY_RESTRICTED, arrangement, [*auth, *scoped, *extra],
                       "Employee role that requires work authorization in another country.")
    if remote_signal:
        loc_ev = [_ev(location, "location", "neutral", "location")] if location else []
        return _result(WorldwideCategory.NEEDS_VERIFICATION, arrangement, [*scoped, *loc_ev, *extra],
                       "Remote, but the listing doesn't say whether India-based applicants are accepted.")
    return _result(WorldwideCategory.INSUFFICIENT_INFO, arrangement,
                   [*([_ev(location, "location", "neutral", "location")] if location else []), *extra],
                   "No remote or location-eligibility information found.")
