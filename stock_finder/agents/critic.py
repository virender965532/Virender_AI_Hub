"""Critic (Generator-Critic pattern) for every CompanyAnalysis and HoldingReview.

Two layers:

1. **Deterministic checks** (always run, no LLM): every number in the output
   appears in the input data; cited ids exist and news claims are cited; no
   forbidden overconfident phrases; the verdict / action is consistent with the
   scores, data quality and concentration flags; at least one risk is listed.
   A failure here returns FAIL without spending an LLM call.
2. **LLM review** with ``SF_CRITIC_MODEL`` (a different model from the analyst)
   for what code cannot judge: fair presentation of risks, logic, tone.

``run_with_critic`` drives the loop: generate -> review -> (FAIL) one revision
with the feedback -> review -> (still FAIL) the caller downgrades the result.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar

from pydantic import BaseModel

from ..config import StockFinderSettings
from ..llm_client import LLMUnavailableError
from ..observability import BudgetExceededError, Trace
from ..prompts import CRITIC_SYSTEM, FORBIDDEN_PHRASES
from ..schemas import CriticStatus, CriticVerdict
from .common import input_message, to_json

logger = logging.getLogger(__name__)

D = TypeVar("D", bound=BaseModel)

_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d[\d,]*(?:\.\d+)?")
# Conventional parameters that are not data: indicator windows, horizons, small counts.
_CONVENTIONAL = {12.0, 14.0, 20.0, 21.0, 26.0, 50.0, 52.0, 63.0, 100.0, 126.0, 200.0, 252.0}
_NEGATIONS = ("not ", "no ", "never ", "isn't ", "aren't ", "without ", "nothing is ")
_NO_NEWS_RE = re.compile(
    r"\b(no|not any|lack of)\b[^.;]{0,40}?\b(news|articles|coverage|headlines)\b"
    r"|\bnews (is|was) (unavailable|not available|limited|scarce)\b",
    re.IGNORECASE,
)
_NEWS_CLAIM_RE = re.compile(r"\b(news|reported|reports|announced|according to|article|headline|media)\b", re.IGNORECASE)


# --- number grounding -----------------------------------------------------------------------------

def parse_number(token: str) -> float | None:
    try:
        return float(token.replace(",", "").replace("+", ""))
    except ValueError:
        return None


def extract_numbers(text: str) -> list[tuple[str, float]]:
    """``(token, value)`` for each number in ``text``; "5,00,000" and "1,234.5" are handled."""
    found = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group(0).rstrip(",")
        value = parse_number(token)
        if value is not None:
            found.append((token, value))
    return found


def collect_numbers(data: Any, out: set[float] | None = None) -> set[float]:
    """Every number in ``data`` (recursively, including numbers inside strings such as dates and notes)."""
    out = set() if out is None else out
    if isinstance(data, bool) or data is None:
        return out
    if isinstance(data, (int, float)):
        out.add(abs(float(data)))
    elif isinstance(data, str):
        for _, value in extract_numbers(data):
            out.add(abs(value))
    elif isinstance(data, dict):
        for value in data.values():
            collect_numbers(value, out)
    elif isinstance(data, (list, tuple, set)):
        for value in data:
            collect_numbers(value, out)
    elif isinstance(data, BaseModel):
        collect_numbers(data.model_dump(mode="json"), out)
    return out


def _ignorable(value: float) -> bool:
    value = abs(value)
    return (value <= 10 and value == int(value)) or value in _CONVENTIONAL or (1990 <= value <= 2100 and value == int(value))


def is_grounded(value: float, allowed: set[float]) -> bool:
    """Exact match, or a faithful rounding of an input value (to 1 decimal, or to an integer for values >= 10)."""
    value = abs(value)
    for ref in allowed:
        if abs(value - ref) <= 0.005 + abs(ref) * 1e-6:
            return True
        if abs(value - round(ref, 1)) < 1e-9:
            return True
        if value == int(value) and abs(ref) >= 10 and abs(value - ref) <= 0.5:
            return True
    return False


def ungrounded_numbers(texts: Iterable[str], allowed: set[float]) -> list[str]:
    bad: list[str] = []
    for text in texts:
        for token, value in extract_numbers(text):
            if not _ignorable(value) and not is_grounded(value, allowed) and token not in bad:
                bad.append(token)
    return bad


# --- language / citations -------------------------------------------------------------------------

def forbidden_phrases(texts: Iterable[str]) -> list[str]:
    """Overconfident phrases used affirmatively ("not guaranteed" is fine)."""
    hits: list[str] = []
    for text in texts:
        lowered = text.lower()
        for phrase in FORBIDDEN_PHRASES:
            for match in re.finditer(re.escape(phrase), lowered):
                before = lowered[max(0, match.start() - 14):match.start()]
                if phrase.startswith(("guarantee", "certain")) and any(n in before for n in _NEGATIONS):
                    continue
                if phrase not in hits:
                    hits.append(phrase)
    return hits


def text_fields(draft: BaseModel, exclude: tuple[str, ...] = ("citation_ids", "verdict", "action", "confidence")) -> list[str]:
    texts: list[str] = []
    for name, value in draft.model_dump().items():
        if name in exclude:
            continue
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list):
            texts.extend(str(v) for v in value)
    return texts


def makes_news_claim(text: str) -> bool:
    return bool(_NEWS_CLAIM_RE.search(text)) and not _NO_NEWS_RE.search(text)


# --- verdict --------------------------------------------------------------------------------------

@dataclass
class ReviewContext:
    """What the critic needs besides the draft."""

    kind: str  # "company" | "holding"
    label: str  # symbol, for logs
    input_data: dict[str, Any]
    allowed_citation_ids: set[str]
    news_text: str = ""  # article titles/summaries the analyst saw (numbers there are grounded too)
    news_fields: tuple[str, ...] = ("news_view",)
    logic_check: Callable[[BaseModel], tuple[bool, str]] = lambda d: (True, "")
    data_quality: str = "good"
    generator_model: str = ""  # the critic (and its fallback) must never be this model
    extra_allowed_numbers: set[float] = field(default_factory=set)


def deterministic_review(draft: BaseModel, ctx: ReviewContext) -> CriticVerdict:
    """Rule-based checks; returns FAIL with specific feedback for every rule broken."""
    problems: list[str] = []
    allowed = collect_numbers(ctx.input_data) | collect_numbers(ctx.news_text) | ctx.extra_allowed_numbers
    texts = text_fields(draft)
    bad_numbers = ungrounded_numbers(texts, allowed)
    if bad_numbers:
        problems.append(f"numbers not present in the input data: {', '.join(bad_numbers[:8])} — use only input values")

    ids = [str(i).strip().strip("[]") for i in getattr(draft, "citation_ids", [])]
    unknown = [i for i in ids if i not in ctx.allowed_citation_ids]
    claims_cited = not unknown
    if unknown:
        problems.append(f"citation ids not in the input: {', '.join(unknown[:5])}")
    news_texts = [str(getattr(draft, f, "")) for f in ctx.news_fields]
    if any(makes_news_claim(t) for t in news_texts) and not [i for i in ids if i in ctx.allowed_citation_ids]:
        claims_cited = False
        problems.append("news claims are made without citing any article id in citation_ids "
                        "(cite the supporting articles, or state that there is no relevant news)")

    phrases = forbidden_phrases(texts)
    overconfident = bool(phrases)
    if phrases:
        problems.append(f"overconfident language is not allowed: {', '.join(phrases)}")
    if getattr(draft, "confidence", "") == "high" and ctx.data_quality == "poor":
        overconfident = True
        problems.append("confidence cannot be 'high' when data quality is poor")

    logic_ok, logic_msg = ctx.logic_check(draft)
    if not logic_ok:
        problems.append(logic_msg)
    risks = getattr(draft, "risks", None)
    risks_balanced = bool(risks)
    if not risks_balanced:
        problems.append("list at least one concrete risk")

    passed = not problems
    return CriticVerdict(
        reasoning="Deterministic checks: " + ("all passed." if passed else f"{len(problems)} problem(s) found."),
        numbers_grounded=not bad_numbers, claims_cited=claims_cited, logic_consistent=logic_ok,
        risks_balanced=risks_balanced, overconfident=overconfident,
        verdict="PASS" if passed else "FAIL", feedback="; ".join(problems),
    )


class Critic:
    """Deterministic checks first, then an independent LLM review with ``SF_CRITIC_MODEL``."""

    def __init__(self, settings: StockFinderSettings, llm: Any) -> None:
        self.settings = settings
        self.llm = llm

    def fallback_for(self, generator_model: str) -> str:
        """A fallback critic model that is neither the critic itself nor the model that wrote the draft ("" = none)."""
        s = self.settings
        for candidate in (s.fallback_model, s.analyst_model, s.worker_model):
            if candidate and candidate not in (s.critic_model, generator_model):
                return candidate
        return ""

    def review(self, draft: BaseModel, ctx: ReviewContext, trace: Trace) -> tuple[CriticVerdict, str]:
        """Returns ``(verdict, reviewer_model)``; ``reviewer_model`` is "" when only code checks ran.

        ``BudgetExceededError`` propagates.
        """
        verdict = deterministic_review(draft, ctx)
        if verdict.verdict == "FAIL":
            return verdict, ""
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM},
            input_message({"input_data": ctx.input_data, "analyst_output": draft.model_dump()},
                          task="Review the analyst_output against the input_data (news article text omitted)."),
        ]
        try:
            result = self.llm.parse(model=self.settings.critic_model, messages=messages, schema=CriticVerdict,
                                    trace=trace, purpose=f"critic:{ctx.label}", temperature=0.0,
                                    fallback_model=self.fallback_for(ctx.generator_model))
        except LLMUnavailableError as exc:
            logger.warning("trace_id=%s critic unavailable for %s: %s", trace.trace_id, ctx.label, exc)
            return verdict.model_copy(update={"reasoning": "LLM critic unavailable; deterministic checks only."}), ""
        llm_verdict: CriticVerdict = result.output  # type: ignore[assignment]
        ok = (llm_verdict.verdict == "PASS" and llm_verdict.numbers_grounded and llm_verdict.claims_cited
              and llm_verdict.logic_consistent and llm_verdict.risks_balanced and not llm_verdict.overconfident)
        if llm_verdict.verdict == "PASS" and not ok:
            llm_verdict = llm_verdict.model_copy(update={
                "verdict": "FAIL", "feedback": llm_verdict.feedback or "Reviewer flags were not all satisfied."})
        return llm_verdict, result.model


def run_with_critic(
    generate: Callable[[str | None], D],
    critic: Critic,
    ctx_for: Callable[[D], ReviewContext],
    trace: Trace,
) -> tuple[D, CriticStatus]:
    """Generate -> review -> one revision on FAIL -> review. Returns the final draft and the critic status.

    ``status.downgraded`` tells the caller to soften the recommendation (still FAIL
    after revision, or the second review could not run). ``BudgetExceededError``
    raised by the *first* generation propagates; later ones are absorbed.
    """
    draft = generate(None)
    ctx = ctx_for(draft)
    try:
        verdict, reviewer = critic.review(draft, ctx, trace)
    except BudgetExceededError:
        return draft, CriticStatus(passed=False, downgraded=True, model="",
                                   concerns=["Not reviewed: request budget exhausted."])
    trace.log("critic_verdict", target=ctx.label, round=1, verdict=verdict.verdict, reviewer=reviewer or "code",
              feedback=verdict.feedback)
    if verdict.verdict == "PASS":
        concerns = [verdict.reasoning] if not reviewer and "unavailable" in verdict.reasoning else []
        return draft, CriticStatus(passed=True, model=reviewer or "deterministic checks", concerns=concerns)
    try:
        revised = generate(verdict.feedback)
        ctx = ctx_for(revised)
        second, reviewer = critic.review(revised, ctx, trace)
    except (BudgetExceededError, LLMUnavailableError) as exc:
        reason = "request budget exhausted" if isinstance(exc, BudgetExceededError) else "model unavailable"
        return draft, CriticStatus(passed=False, revised=False, downgraded=True, model=reviewer or "deterministic checks",
                                   concerns=[verdict.feedback, f"Revision not possible: {reason}."])
    trace.log("critic_verdict", target=ctx.label, round=2, verdict=second.verdict, reviewer=reviewer or "code",
              feedback=second.feedback)
    if second.verdict == "PASS":
        return revised, CriticStatus(passed=True, revised=True, model=reviewer or "deterministic checks")
    return revised, CriticStatus(passed=False, revised=True, downgraded=True, model=reviewer or "deterministic checks",
                                 concerns=[c for c in (second.feedback, second.reasoning) if c][:2])
