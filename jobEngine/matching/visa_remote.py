"""Visa-sponsorship and global-remote evidence in a job description
(``JOB_WORLDWIDE_REMOTE_ENABLED``). Discovery labels only: nothing here rejects a job.

Two independent claims, each labelled from quoted sentences:
  visa_status   POSITIVE_EVIDENCE ("we provide full visa sponsorship"),
                NEGATIVE_EVIDENCE ("we cannot sponsor work visas"),
                MENTIONED_UNCLEAR (a question, "visa requirements apply", relocation
                only, or positive and negative statements that conflict),
                NOT_MENTIONED.
  remote_status GLOBAL_REMOTE_EVIDENCE ("work from anywhere in the world"),
                RESTRICTED_REMOTE ("remote within the United States"),
                REMOTE_UNCLEAR ("remote-first", "100% remote" with no scope, or
                worldwide wording contradicted by a restriction / on-site requirement),
                NOT_MENTIONED.
Relocation support is a third, separate claim: relocation is never read as visa
sponsorship, nor the other way round.

These rules give every job a label up front. Jobs sent to the LLM matcher get a
second read from that same call (``JobMatchAnalysis.eligibility``; no extra LLM
call), merged by ``apply_llm_eligibility``: the LLM's label wins, but a decisive
label (positive/negative, global/restricted, confirmed/not offered) only counts
when one of its quotes occurs verbatim in the posting.

The description is split into sections (by heading lines such as "Benefits:" or
"About the company") and sentences; every rule runs per sentence, so the stored
evidence is the exact sentence plus the section it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models.job import (
    EvidenceSnippet,
    JobEligibility,
    RelocationSupport,
    RemoteStatus,
    VisaRemoteClassification,
    VisaStatus,
)
from .global_remote import GLOBAL_REMOTE_TAG, is_india_location
from .worldwide_remote import (
    _EXCLUDING,
    _HARD_RESTRICTION_RES,
    _SCOPED_ANYWHERE_RE,
    _SCOPED_REMOTE_RES,
    _WORLDWIDE_RES,
    _clip,
    _sentences,
)
from .worldwide_llm import verified_quotes

_I = re.IGNORECASE

# --- visa -------------------------------------------------------------------
_VISA_KEYWORD_RE = re.compile(
    r"\bvisas?\b|\bwork\s+permits?\b|\bimmigration\b|\bsponsorship\b|\bh-?1b\b|\bblue\s+card\b"
    r"|\bsponsor(?:s|ed|ing)?\b[^.;\n]{0,40}?\b(?:visa|work|candidates?|applicants?|employment|immigration|"
    r"international|foreign|eligible|you)\b"
    r"|\binternational\s+(?:candidates|applicants|talent|hires)\b|\bskilled\s+worker\b",
    _I,
)
_VISA_NEGATIVE_RES = [
    re.compile(p, _I)
    for p in (
        r"\b(?:do|does|will|can|are|is)\s*(?:not|n't)\s+(?:\w+\s+){0,3}?(?:provide|offer|sponsor|support)\w*"
        r"[^.;\n]{0,40}?\b(?:visas?|sponsorship|work\s+permits?|immigration)\b",
        r"\b(?:cannot|can\s*not|can't|won't|unable\s+to|not\s+able\s+to|never)\s+(?:\w+\s+){0,3}?sponsor",
        r"\b(?:cannot|can\s*not|can't|won't|unable\s+to|not\s+able\s+to)\s+(?:\w+\s+){0,3}?"
        r"(?:provide|offer|support)\w*[^.;\n]{0,30}?\b(?:visas?|sponsorship|work\s+permits?)\b",
        r"\bno\s+(?:visa\s+|work\s+permit\s+)?sponsorship\b",
        r"\bwithout\s+(?:the\s+need\s+(?:for|of)\s+)?(?:any\s+)?(?:current\s+or\s+future\s+|future\s+)?"
        r"(?:visa\s+|employer\s+|company\s+)?sponsorship\b",
        r"\b(?:visa\s+)?sponsorship\s+(?:is\s+|will\s+)?not\s+(?:be\s+)?(?:available|provided|offered|possible|an\s+option)\b",
        r"\bnot\s+(?:eligible|able)\s+(?:for|to\s+receive)\s+(?:visa\s+)?sponsorship\b",
        r"\b(?:require|requires|need|needs)\s+(?:\w+\s+){0,3}?sponsorship\b[^.;\n]{0,60}?"
        r"\b(?:not\s+be\s+considered|ineligible|not\s+eligible|cannot\s+be\s+considered)\b",
    )
]
_VISA_POSITIVE_RES = [
    re.compile(p, _I)
    for p in (
        r"\b(?:visa|work\s+permit|immigration)\s+(?:sponsorship|support|assistance|help|processing)\s+"
        r"(?:is\s+|will\s+be\s+)?(?:available|provided|offered|included|covered)\b",
        r"\b(?:sponsorship|visa\s+support)\s+(?:is\s+)?available\b",
        r"\b(?:full|complete|comprehensive)\s+(?:visa|work\s+permit|immigration)\s+(?:sponsorship|support)\b",
        r"\bwe\s+(?:\w+\s+){0,2}?(?:sponsor|provide|offer|cover|support|assist\s+with|help\s+with|handle|arrange)\w*\s+"
        r"[^.;\n]{0,50}?\b(?:visas?|work\s+permits?|sponsorship|immigration)\b",
        r"\b(?:will|can|happy\s+to|able\s+to)\s+sponsor\b",
        r"\bsponsor(?:s|ing)?\s+(?:\w+\s+){0,3}?(?:visas?|work\s+permits?|international\s+(?:candidates|applicants|hires))\b",
        r"\b(?:visa|immigration)\s+(?:sponsorship|support)\s+(?:for|to)\s+(?:eligible\s+)?(?:international|foreign|overseas)\b",
    )
]
# Anti-scam footers ("we will never solicit money to complete visa requirements") say
# nothing about sponsorship.
_VISA_BOILERPLATE_RE = re.compile(r"\bsolicit\w*|\bmoney\b|\bfees?\b|\bpayments?\b|\bscams?\b|\bfraud\w*", _I)
# A benefit bullet that is only "Visa sponsorship" / "Visa support & relocation".
_VISA_BENEFIT_LINE_RE = re.compile(
    r"^\W*(?:full\s+|complete\s+)?(?:visa|work\s+permit)\s+(?:sponsorship|support|assistance)"
    r"(?:\s+(?:and|&)\s+relocation(?:\s+\w+)?)?\W*$",
    _I,
)

# --- relocation -------------------------------------------------------------
_RELOCATION_KEYWORD_RE = re.compile(r"\breloca\w*", _I)
_RELOCATION_NEGATIVE_RE = re.compile(
    r"\bno\s+relocation\b|\brelocation\s+(?:\w+\s+)?(?:is\s+)?not\s+(?:\w+\s+)?(?:provided|offered|available|supported|covered)\b"
    r"|\b(?:cannot|can't|won't|unable\s+to|do\s+not|does\s+not|don't|doesn't)\s+(?:\w+\s+){0,2}?(?:offer|provide|support|cover)\s+"
    r"[^.;\n]{0,20}?relocation\b",
    _I,
)
_RELOCATION_POSITIVE_RE = re.compile(
    r"\breloca\w*\s+(?:assistance|support|package|bonus|allowance|budget|stipend|help|expenses|costs)\b"
    r"(?![^.;\n]{0,15}\?)"
    r"|\b(?:help|assist|support)\w*\s+(?:\w+\s+){0,2}?(?:with\s+)?(?:your\s+)?relocat\w*"
    r"|\bwe\s+(?:\w+\s+){0,2}?(?:offer|provide|cover|pay|support)\w*\s+[^.;\n]{0,30}?relocation\b",
    _I,
)

# --- remote -----------------------------------------------------------------
_REMOTE_KEYWORD_RE = re.compile(
    r"\bremote(?:ly)?\b|\bwork(?:ing)?\s+from\s+(?:home|anywhere)\b|\bwfh\b|\blocation[\s-]+(?:independent|agnostic)\b"
    r"|\bdistributed\s+(?:team|company|workforce)\b|\banywhere\s+in\s+the\s+world\b",
    _I,
)
_GEO_AUTH_RE = re.compile(
    r"\b(?:authori[sz]ed|authori[sz]ation|eligible|eligibility|legally\s+(?:able|permitted|allowed)|right)\s+"
    rf"to\s+work\s+(?:in|for)\s+(?:the\s+)?{_EXCLUDING}"
    rf"|\b{_EXCLUDING}\s+work\s+(?:authori[sz]ation|permit|eligibility)\b",
    _I,
)
# "Remote within approved European countries": a qualifier the shared patterns don't allow for.
_SCOPED_REMOTE_QUALIFIED_RE = re.compile(
    rf"\bremote(?:ly)?\s+(?:only\s+)?(?:in|within|across|from)\s+(?:the\s+)?(?:[\w-]+\s+){{1,2}}?{_EXCLUDING}", _I
)
_ONSITE_REQUIREMENT_RE = re.compile(
    r"\b(?:must|required\s+to|need\s+to|needs\s+to|expected\s+to|will\s+need\s+to)\s+(?:\w+\s+){0,3}?"
    r"(?:work\s+)?(?:on[\s-]?site|in[\s-](?:the\s+|our\s+)?office|from\s+(?:the\s+|our\s+)?office|commute|relocate)\b"
    r"|\b(?:on[\s-]?site|in[\s-]office)\s+(?:presence\s+)?(?:is\s+)?(?:required|mandatory)\b"
    r"|\b(?:\d|one|two|three|four|five)\s+days?\s+(?:a|per|each)\s+week\s+(?:in|at)\s+(?:the\s+|our\s+)?office\b"
    r"|\bthis\s+(?:is\s+an?\s+|role\s+is\s+)(?:on[\s-]?site|office[\s-]based|hybrid)\b",
    _I,
)

# Canonical names for remote_countries_allowed / job_country. Only names a sentence
# actually contains are ever reported.
_COUNTRY_NAMES = [
    (r"(?-i:US|USA|U\.S\.?A?)|united\s+states", "United States"), (r"north\s+america", "North America"),
    (r"(?-i:UK)|u\.k\.|united\s+kingdom|great\s+britain|britain|england|scotland", "United Kingdom"),
    (r"canada", "Canada"), (r"mexico", "Mexico"), (r"brazil", "Brazil"), (r"ireland", "Ireland"),
    (r"germany", "Germany"), (r"france", "France"), (r"spain", "Spain"), (r"portugal", "Portugal"),
    (r"italy", "Italy"), (r"netherlands", "Netherlands"), (r"belgium", "Belgium"), (r"poland", "Poland"),
    (r"romania", "Romania"), (r"switzerland", "Switzerland"), (r"austria", "Austria"), (r"sweden", "Sweden"),
    (r"norway", "Norway"), (r"denmark", "Denmark"), (r"finland", "Finland"), (r"israel", "Israel"),
    (r"uae|united\s+arab\s+emirates|dubai", "United Arab Emirates"), (r"saudi\s+arabia", "Saudi Arabia"),
    (r"qatar", "Qatar"), (r"singapore", "Singapore"), (r"japan", "Japan"), (r"australia", "Australia"),
    (r"new\s+zealand", "New Zealand"), (r"india", "India"),
    (r"(?-i:EU|EEA)|european|europe", "Europe"), (r"emea", "EMEA"), (r"latam|latin\s+america", "LATAM"),
    (r"apac|asia[\s-]pacific", "APAC"),
]
_COUNTRY_RES = [(re.compile(rf"\b(?:{p})\b", _I), name) for p, name in _COUNTRY_NAMES]
_REGIONS = {"North America", "Europe", "EMEA", "LATAM", "APAC"}

# A heading line: "Benefits:", "About the Company", "Why join us?", "Location".
_HEADING_RE = re.compile(
    r"^\W*(?:about\s+(?:the\s+)?(?:company|us|role|team|job|position)|about\s+\w+|why\s+join\s+us|why\s+\w+|benefits"
    r"|perks(?:\s+(?:and|&)\s+benefits)?|what\s+we\s+offer|what\s+you(?:'ll)?\s+get|compensation(?:\s+(?:and|&)\s+benefits)?"
    r"|work(?:ing)?\s+(?:arrangement|model|location|conditions)|location|requirements|qualifications"
    r"|responsibilities|what\s+you(?:'ll)?\s+do|who\s+you\s+are|nice\s+to\s+have|how\s+to\s+apply)\W*$",
    _I,
)
# "• Visa sponsorship" is a benefit, never a heading.
_BULLET_RE = re.compile(r"^\s*(?:[-*•·▪●–]|\d+[.)])\s*")
_MAX_HEADING_CHARS = 60


@dataclass
class JobFlags:
    """Raw signals found in one job description (see ``classify_job``)."""

    visa_keyword_found: bool
    positive_visa_evidence: bool
    negative_visa_evidence: bool
    remote_keyword_found: bool
    global_remote_evidence: bool
    remote_restrictions_found: bool
    onsite_requirement_found: bool = False


def classify_job(job: JobFlags) -> dict:
    """Labels from flags. Contradictory evidence is never resolved silently: it is
    reported as unclear with ``*_conflict`` set. ``has_discovery_signal`` is a
    discovery label only — a job without it is still kept."""
    visa_conflict = job.positive_visa_evidence and job.negative_visa_evidence
    remote_conflict = job.global_remote_evidence and (job.remote_restrictions_found or job.onsite_requirement_found)
    if visa_conflict:
        visa_status = VisaStatus.MENTIONED_UNCLEAR
    elif job.positive_visa_evidence:
        visa_status = VisaStatus.POSITIVE_EVIDENCE
    elif job.negative_visa_evidence:
        visa_status = VisaStatus.NEGATIVE_EVIDENCE
    elif job.visa_keyword_found:
        visa_status = VisaStatus.MENTIONED_UNCLEAR
    else:
        visa_status = VisaStatus.NOT_MENTIONED

    if remote_conflict:
        remote_status = RemoteStatus.REMOTE_UNCLEAR
    elif job.global_remote_evidence:
        remote_status = RemoteStatus.GLOBAL_REMOTE_EVIDENCE
    elif job.remote_restrictions_found and job.remote_keyword_found:
        remote_status = RemoteStatus.RESTRICTED_REMOTE
    elif job.remote_keyword_found:
        remote_status = RemoteStatus.REMOTE_UNCLEAR
    else:
        remote_status = RemoteStatus.NOT_MENTIONED

    return {
        "has_discovery_signal": job.visa_keyword_found or job.remote_keyword_found,
        "visa_status": visa_status,
        "remote_status": remote_status,
        "visa_conflict": visa_conflict,
        "remote_conflict": remote_conflict,
    }


# "Unrestricted U.S. work authorization" must stay one sentence.
_ABBREVIATION_END_RE = re.compile(r"\b(?:U\.S\.?A?|U\.K|E\.U|e\.g|i\.e|etc|vs|Inc|Ltd|Co|Sr|Jr|No)\.$", _I)


def _split(line: str) -> list[str]:
    out: list[str] = []
    for part in _sentences(line):
        if out and _ABBREVIATION_END_RE.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _sectioned_sentences(description: str) -> list[tuple[str, str]]:
    """(section, sentence) pairs; the section is the last heading line seen, else "description"."""
    out: list[tuple[str, str]] = []
    section = "description"
    for line in (description or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_bullet = bool(_BULLET_RE.match(stripped))
        if not is_bullet and len(stripped) <= _MAX_HEADING_CHARS and (
            _HEADING_RE.match(stripped) or (stripped.endswith(":") and len(stripped.split()) <= 6)
        ):
            section = stripped.strip(" :#*-•").strip() or section
            continue
        if is_bullet:
            stripped = _BULLET_RE.sub("", stripped)
        out.extend((section, s) for s in _split(stripped))
    return out


def _countries(text: str) -> list[str]:
    return [name for pattern, name in _COUNTRY_RES if pattern.search(text)]


def job_country(location: str | None) -> str | None:
    """Country of the posting location ("Berlin, Germany" -> "Germany"), or None if not a known country."""
    if not location:
        return None
    if is_india_location(location):
        return "India"
    for part in reversed(re.sub(r"\([^)]*\)", "", location).split(",")):
        names = [n for n in _countries(part) if n not in _REGIONS]
        if names:
            return names[0]
    return None


def _snippet(text: str, section: str, signal: str) -> EvidenceSnippet:
    return EvidenceSnippet(text=_clip(text), section=section, signal=signal)


def _visa_signal(sentence: str) -> str | None:
    """positive / negative / unclear for a sentence that mentions visas, else None."""
    if not _VISA_KEYWORD_RE.search(sentence) or _VISA_BOILERPLATE_RE.search(sentence):
        return None
    if any(p.search(sentence) for p in _VISA_NEGATIVE_RES):
        return "negative"
    if sentence.rstrip().endswith("?"):
        return "unclear"  # "Will you require visa sponsorship?"
    if any(p.search(sentence) for p in _VISA_POSITIVE_RES) or _VISA_BENEFIT_LINE_RE.match(sentence):
        return "positive"
    return "unclear"


def _relocation_signal(sentence: str) -> str | None:
    if not _RELOCATION_KEYWORD_RE.search(sentence):
        return None
    if _RELOCATION_NEGATIVE_RE.search(sentence):
        return "negative"
    if sentence.rstrip().endswith("?"):
        return "unclear"
    if _RELOCATION_POSITIVE_RE.search(sentence):
        return "positive"
    return "unclear"  # "must be willing to relocate" is a requirement, not support


def _confidence(decisive: bool, conflict: bool, mentioned: bool) -> float | None:
    """Rule-derived, not a model probability: explicit statement 0.9, vague mention 0.5, contradiction 0.3."""
    if not mentioned:
        return None
    if conflict:
        return 0.3
    return 0.9 if decisive else 0.5


def classify_visa_remote(
    *,
    title: str,
    location: str | None,
    description: str,
    workplace_type: str | None = None,
) -> VisaRemoteClassification:
    """Classify one job. ``workplace_type`` is LinkedIn's Remote/Hybrid/On-site chip, when read."""
    location = location or ""
    workplace = (workplace_type or "").strip().lower()

    visa_ev: list[EvidenceSnippet] = []
    relocation_ev: list[EvidenceSnippet] = []
    remote_ev: list[EvidenceSnippet] = []
    allowed: list[str] = []
    visa_signals: set[str] = set()
    relocation_signals: set[str] = set()
    remote_kw = global_ev = restricted = onsite = False

    if workplace == "remote":
        remote_kw = True
        remote_ev.append(_snippet(workplace_type, "workplace_type", "neutral"))
    elif workplace in ("on-site", "onsite", "hybrid"):
        onsite = True
        remote_ev.append(_snippet(workplace_type, "workplace_type", "negative"))
    if location and re.search(r"\b(?:worldwide|anywhere|global)\b", location, _I):
        remote_kw = global_ev = True
        remote_ev.append(_snippet(location, "location", "positive"))
    elif location and _REMOTE_KEYWORD_RE.search(location):
        remote_kw = True

    fields = [("title", title or "")] + _sectioned_sentences(description)
    for section, sentence in fields:
        if not sentence:
            continue
        visa = _visa_signal(sentence)
        if visa:
            visa_signals.add(visa)
            visa_ev.append(_snippet(sentence, section, visa))
        relocation = _relocation_signal(sentence)
        if relocation:
            relocation_signals.add(relocation)
            relocation_ev.append(_snippet(sentence, section, relocation))

        remote_here = bool(_REMOTE_KEYWORD_RE.search(sentence))
        remote_kw = remote_kw or remote_here
        is_global = any(p.search(sentence) for p in _WORLDWIDE_RES) and not _SCOPED_ANYWHERE_RE.search(sentence)
        is_restricted = (
            any(p.search(sentence) for p in _HARD_RESTRICTION_RES)
            or any(p.search(sentence) for p in _SCOPED_REMOTE_RES)
            or bool(_SCOPED_REMOTE_QUALIFIED_RE.search(sentence))
            or bool(_GEO_AUTH_RE.search(sentence))
            or bool(_SCOPED_ANYWHERE_RE.search(sentence) and remote_here)
        )
        is_onsite = bool(_ONSITE_REQUIREMENT_RE.search(sentence))
        if is_global:
            global_ev = remote_kw = True
            remote_ev.append(_snippet(sentence, section, "positive"))
        elif is_restricted:
            restricted = True
            allowed += [c for c in _countries(sentence) if c not in allowed]
            remote_ev.append(_snippet(sentence, section, "negative"))
        elif is_onsite:
            onsite = True
            remote_ev.append(_snippet(sentence, section, "negative"))
        elif remote_here:
            remote_ev.append(_snippet(sentence, section, "neutral"))

    flags = JobFlags(
        visa_keyword_found=bool(visa_signals or relocation_signals),
        positive_visa_evidence="positive" in visa_signals,
        negative_visa_evidence="negative" in visa_signals,
        remote_keyword_found=remote_kw,
        global_remote_evidence=global_ev,
        remote_restrictions_found=restricted,
        onsite_requirement_found=onsite,
    )
    labels = classify_job(flags)

    if relocation_signals == {"positive"}:
        relocation = RelocationSupport.CONFIRMED
    elif relocation_signals == {"negative"}:
        relocation = RelocationSupport.NOT_OFFERED
    elif relocation_signals:
        relocation = RelocationSupport.MENTIONED_UNCLEAR
    else:
        relocation = RelocationSupport.NOT_MENTIONED

    visa_status, remote_status = labels["visa_status"], labels["remote_status"]
    scores = [
        s
        for s in (
            _confidence(
                visa_status in (VisaStatus.POSITIVE_EVIDENCE, VisaStatus.NEGATIVE_EVIDENCE),
                labels["visa_conflict"], flags.visa_keyword_found,
            ),
            _confidence(
                remote_status in (RemoteStatus.GLOBAL_REMOTE_EVIDENCE, RemoteStatus.RESTRICTED_REMOTE),
                labels["remote_conflict"], flags.remote_keyword_found,
            ),
        )
        if s is not None
    ]

    return VisaRemoteClassification(
        visa_status=visa_status,
        remote_status=remote_status,
        visa_keyword_found=flags.visa_keyword_found,
        remote_keyword_found=flags.remote_keyword_found,
        has_discovery_signal=labels["has_discovery_signal"],
        visa_evidence=visa_ev,
        relocation_evidence=relocation_ev,
        remote_evidence=remote_ev if flags.remote_keyword_found or onsite else [],
        remote_countries_allowed=allowed if remote_status == RemoteStatus.RESTRICTED_REMOTE else [],
        job_country=job_country(location),
        relocation_support=relocation,
        onsite_requirement=onsite,
        visa_conflict=labels["visa_conflict"],
        remote_conflict=labels["remote_conflict"],
        conflicting_evidence=labels["visa_conflict"] or labels["remote_conflict"],
        classification_confidence=min(scores) if scores else None,
    )


_DECISIVE_VISA = {VisaStatus.POSITIVE_EVIDENCE: "positive", VisaStatus.NEGATIVE_EVIDENCE: "negative"}
_DECISIVE_REMOTE = {RemoteStatus.GLOBAL_REMOTE_EVIDENCE: "positive", RemoteStatus.RESTRICTED_REMOTE: "negative"}
_DECISIVE_RELOCATION = {RelocationSupport.CONFIRMED: "positive", RelocationSupport.NOT_OFFERED: "negative"}


def _is_global(sentence: str) -> bool:
    return any(p.search(sentence) for p in _WORLDWIDE_RES) and not _SCOPED_ANYWHERE_RE.search(sentence)


def _is_restriction(sentence: str) -> bool:
    return bool(
        any(p.search(sentence) for p in _HARD_RESTRICTION_RES)
        or any(p.search(sentence) for p in _SCOPED_REMOTE_RES)
        or _SCOPED_REMOTE_QUALIFIED_RE.search(sentence)
        or _GEO_AUTH_RE.search(sentence)
        or _countries(sentence)
    )


# Does a quote really support the decisive label? The LLM finds and reads the sentence;
# these checks keep it from over-reading one ("fully remote" is not "worldwide").
_QUOTE_SUPPORTS = {
    VisaStatus.POSITIVE_EVIDENCE: lambda q: _visa_signal(q) in ("positive", "unclear") and not q.rstrip().endswith("?"),
    VisaStatus.NEGATIVE_EVIDENCE: lambda q: _visa_signal(q) is not None,
    RemoteStatus.GLOBAL_REMOTE_EVIDENCE: _is_global,
    RemoteStatus.RESTRICTED_REMOTE: _is_restriction,
    RelocationSupport.CONFIRMED: lambda q: _relocation_signal(q) in ("positive", "unclear") and not q.rstrip().endswith("?"),
    RelocationSupport.NOT_OFFERED: lambda q: _relocation_signal(q) is not None,
}


def _merge_claim(status, quotes, job_text, decisive, current_status, current_ev, current_conflict):
    """(status, evidence, conflict) for one claim.

    * A decisive LLM label counts only with a verbatim quote that the check above accepts;
      if the rules decided the opposite, the claim becomes unclear with a conflict.
    * A non-decisive LLM label (unclear / not mentioned) replaces only a non-decisive rules
      label: the LLM missing a sentence never erases explicit rules evidence.
    """
    unclear = next(s for s in type(status) if s.value in ("MENTIONED_UNCLEAR", "REMOTE_UNCLEAR"))
    signal = decisive.get(status)
    if signal:
        supported = [q for q in verified_quotes(quotes, job_text) if _QUOTE_SUPPORTS[status](q)]
        if not supported:
            return current_status, current_ev, current_conflict
        evidence = [_snippet(q, "description", signal) for q in supported]
        if current_status in decisive and current_status != status:
            return unclear, [*current_ev, *evidence], True
        return status, evidence, current_conflict and status == unclear
    if current_status in decisive:
        return current_status, current_ev, current_conflict
    return status, [_snippet(q, "description", "unclear") for q in verified_quotes(quotes, job_text)] or (
        current_ev if status == current_status else []
    ), current_conflict and status == current_status


def india_quotes_support(category: str, quotes: list[str]) -> list[str]:
    """The matcher's India quotes that really back its category (same idea as ``_QUOTE_SUPPORTS``)."""
    if category == "worldwide_confirmed":
        return [q for q in quotes if _is_global(q)]
    if category == "india_eligible":
        return [q for q in quotes if re.search(r"\bindia\b", q, _I)]
    if category == "country_restricted":
        return [q for q in quotes if _is_restriction(q)]
    return quotes


def apply_llm_eligibility(
    current: VisaRemoteClassification, llm: JobEligibility, job_text: str
) -> VisaRemoteClassification:
    """Merge the matcher's eligibility read into the rules' labels; Python makes the final call."""
    visa, visa_ev, visa_conflict = _merge_claim(
        VisaStatus(llm.visa_status), llm.visa_quotes, job_text, _DECISIVE_VISA,
        current.visa_status, current.visa_evidence, current.visa_conflict,
    )
    remote, remote_ev, remote_conflict = _merge_claim(
        RemoteStatus(llm.remote_status), llm.remote_quotes, job_text, _DECISIVE_REMOTE,
        current.remote_status, current.remote_evidence, current.remote_conflict,
    )
    relocation, relocation_ev, _ = _merge_claim(
        RelocationSupport(llm.relocation_support), llm.relocation_quotes, job_text, _DECISIVE_RELOCATION,
        current.relocation_support, current.relocation_evidence, False,
    )
    if relocation == RelocationSupport.MENTIONED_UNCLEAR and current.relocation_support != relocation:
        relocation = current.relocation_support  # "unclear" is not a relocation verdict worth overriding with
        relocation_ev = current.relocation_evidence
    visa_mentioned = visa != VisaStatus.NOT_MENTIONED or relocation != RelocationSupport.NOT_MENTIONED
    remote_mentioned = remote != RemoteStatus.NOT_MENTIONED
    scores = [
        s
        for s in (
            _confidence(visa in _DECISIVE_VISA, visa_conflict, visa_mentioned),
            _confidence(remote in _DECISIVE_REMOTE, remote_conflict, remote_mentioned),
        )
        if s is not None
    ]
    # Only names the posting really contains.
    named = [c for c in llm.remote_countries_allowed if c and c.lower() in job_text.lower()]
    allowed = current.remote_countries_allowed or named
    return current.model_copy(
        update={
            "visa_status": visa,
            "visa_evidence": visa_ev,
            "remote_status": remote,
            "remote_evidence": remote_ev,
            "relocation_support": relocation,
            "relocation_evidence": relocation_ev,
            "visa_keyword_found": visa_mentioned,
            "remote_keyword_found": remote_mentioned,
            "has_discovery_signal": visa_mentioned or remote_mentioned,
            "remote_countries_allowed": allowed if remote == RemoteStatus.RESTRICTED_REMOTE else [],
            "visa_conflict": visa_conflict,
            "remote_conflict": remote_conflict,
            "conflicting_evidence": visa_conflict or remote_conflict,
            "classification_confidence": min(scores) if scores else None,
            "method": "llm",
        }
    )


def with_global_remote_tag(tags: list[str], classification: VisaRemoteClassification) -> list[str]:
    """``tags`` with the "Global Remote" tag set exactly when remote_status is global, so the
    tag and the label never disagree (the older ``is_global_remote`` heuristic also tags
    "fully remote" with no stated scope)."""
    out = [t for t in tags if t != GLOBAL_REMOTE_TAG]
    if classification.remote_status == RemoteStatus.GLOBAL_REMOTE_EVIDENCE:
        out.append(GLOBAL_REMOTE_TAG)
    return out
