"""Visa-sponsorship / remote-scope labels (matching/visa_remote.py) and the
retention rules around them: labels never drop a relevant job."""

from __future__ import annotations

import asyncio

import pytest

from jobEngine.config.settings import get_settings
from jobEngine.graph.nodes.matching import _not_eligible_reason
from jobEngine.graph.nodes.normalization import normalization_node
from jobEngine.graph.nodes.validation import validation_node
from jobEngine.matching.visa_remote import JobFlags, classify_job, classify_visa_remote, job_country
from jobEngine.models.job import RawJob
from jobEngine.models.job import RelocationSupport as Reloc
from jobEngine.models.job import RemoteStatus as Remote
from jobEngine.models.job import VisaStatus as Visa

_PROFILE_TEXT = (
    "You will build Node.js, React and Next.js services in TypeScript on AWS, ship generative AI features "
    "with RAG and LangGraph, review pull requests and own features end to end."
)


def _classify(description, *, title="Senior Full Stack Engineer", location="Berlin, Germany", workplace_type=None):
    return classify_visa_remote(
        title=title, location=location, description=f"{_PROFILE_TEXT}\n{description}", workplace_type=workplace_type
    )


def _texts(snippets, signal=None):
    return [s.text for s in snippets if signal is None or s.signal == signal]


# --- 1-4, 11: visa sponsorship ---------------------------------------------


def test_explicit_positive_visa_sponsorship():
    r = _classify("Benefits:\nWe provide full visa sponsorship.\nGenerous learning budget.")
    assert r.visa_status == Visa.POSITIVE_EVIDENCE
    assert r.visa_keyword_found
    assert _texts(r.visa_evidence, "positive") == ["We provide full visa sponsorship."]
    assert r.visa_evidence[0].section == "Benefits"
    assert not r.conflicting_evidence


def test_benefit_bullets_are_evidence_not_headings():
    r = _classify("What we offer:\n• Visa sponsorship\n• Relocation package\n• Private health insurance")
    assert r.visa_status == Visa.POSITIVE_EVIDENCE
    assert r.relocation_support == Reloc.CONFIRMED
    assert [(s.text, s.section) for s in r.visa_evidence] == [("Visa sponsorship", "What we offer")]


def test_sponsoring_events_is_not_visa_sponsorship():
    assert _classify("We sponsor hackathons and local meetups.").visa_status == Visa.NOT_MENTIONED


@pytest.mark.parametrize(
    "sentence",
    [
        "We do not provide visa sponsorship.",
        "We cannot sponsor work visas.",
        "Must be eligible to work without sponsorship.",
    ],
)
def test_explicit_refusal_to_sponsor(sentence):
    r = _classify(sentence)
    assert r.visa_status == Visa.NEGATIVE_EVIDENCE
    assert _texts(r.visa_evidence, "negative") == [sentence]


@pytest.mark.parametrize("sentence", ["Will you require visa sponsorship?", "Visa requirements apply."])
def test_visa_mentioned_only_as_question_or_vaguely(sentence):
    r = _classify(sentence)
    assert r.visa_status == Visa.MENTIONED_UNCLEAR
    assert r.visa_keyword_found
    assert _texts(r.visa_evidence) == [sentence]


def test_no_visa_keywords():
    r = _classify("We value ownership and clean code.")
    assert r.visa_status == Visa.NOT_MENTIONED
    assert not r.visa_keyword_found
    assert r.visa_evidence == []
    assert r.relocation_support == Reloc.NOT_MENTIONED


def test_relocation_alone_is_not_visa_sponsorship():
    r = _classify("Relocation assistance available.")
    assert r.relocation_support == Reloc.CONFIRMED
    assert r.visa_status == Visa.MENTIONED_UNCLEAR
    assert r.visa_evidence == []
    assert _texts(r.relocation_evidence) == ["Relocation assistance available."]


def test_visa_sponsorship_does_not_confirm_relocation():
    r = _classify("Visa sponsorship available for international candidates.")
    assert r.visa_status == Visa.POSITIVE_EVIDENCE
    assert r.relocation_support == Reloc.NOT_MENTIONED


def test_positive_and_negative_sponsorship_statements_conflict():
    r = _classify(
        "We sponsor eligible international applicants for our Berlin office. "
        "We cannot sponsor work visas for roles in the United States."
    )
    assert r.visa_status == Visa.MENTIONED_UNCLEAR
    assert r.visa_conflict and r.conflicting_evidence
    assert len(_texts(r.visa_evidence, "positive")) == 1
    assert len(_texts(r.visa_evidence, "negative")) == 1
    assert r.classification_confidence == 0.3


# --- 5-8, 12: remote scope --------------------------------------------------


def test_explicit_worldwide_remote():
    r = _classify("Why join us:\nWork from anywhere in the world.", location="United States", workplace_type="Remote")
    assert r.remote_status == Remote.GLOBAL_REMOTE_EVIDENCE
    assert "Work from anywhere in the world." in _texts(r.remote_evidence, "positive")
    assert any(s.section == "Why join us" for s in r.remote_evidence)
    assert not r.remote_conflict


def test_remote_restricted_to_united_states():
    r = _classify("Remote within the United States.", location="United States", workplace_type="Remote")
    assert r.remote_status == Remote.RESTRICTED_REMOTE
    assert r.remote_countries_allowed == ["United States"]
    assert "Remote within the United States." in _texts(r.remote_evidence, "negative")


@pytest.mark.parametrize(
    "sentence, allowed",
    [("Remote, but must reside in the UK.", ["United Kingdom"]), ("Remote within approved European countries.", ["Europe"])],
)
def test_other_restricted_remote(sentence, allowed):
    r = _classify(sentence)
    assert r.remote_status == Remote.RESTRICTED_REMOTE
    assert r.remote_countries_allowed == allowed


@pytest.mark.parametrize("sentence", ["This is a 100% remote position.", "We are a remote-first company."])
def test_remote_without_geographic_scope_is_unclear(sentence):
    r = _classify(sentence)
    assert r.remote_status == Remote.REMOTE_UNCLEAR
    assert r.remote_countries_allowed == []
    assert r.classification_confidence == 0.5


def test_worldwide_contradicted_by_onsite_requirement():
    r = _classify(
        "About the company:\nOur team works remotely from anywhere in the world.\n"
        "Work arrangement:\nThis role requires you to be on-site in our Munich office. You must work on-site three days a week."
    )
    assert r.remote_status == Remote.REMOTE_UNCLEAR
    assert r.remote_conflict and r.onsite_requirement
    assert _texts(r.remote_evidence, "positive")
    assert _texts(r.remote_evidence, "negative")


def test_worldwide_contradicted_by_country_restriction():
    r = _classify("Fully remote, worldwide. Candidates must be located in the United States.")
    assert r.remote_status == Remote.REMOTE_UNCLEAR
    assert r.remote_conflict and r.conflicting_evidence
    assert _texts(r.remote_evidence, "negative") == ["Candidates must be located in the United States."]


def test_onsite_workplace_contradicts_worldwide_wording():
    r = _classify("Employees can work remotely from any country.", workplace_type="On-site")
    assert r.remote_conflict
    assert r.remote_status == Remote.REMOTE_UNCLEAR


# --- 9-10: combined ------------------------------------------------------------


def test_visa_sponsorship_for_onsite_role():
    r = _classify("We will support your work visa application. You must work on-site in our London office.",
                  location="London, England, United Kingdom", workplace_type="On-site")
    assert r.visa_status == Visa.POSITIVE_EVIDENCE
    assert r.remote_status == Remote.NOT_MENTIONED
    assert r.onsite_requirement
    assert r.job_country == "United Kingdom"


def test_job_with_neither_signal_is_labelled_not_dropped():
    r = _classify("Competitive salary and a great team.", location="Toronto, Ontario, Canada")
    assert r.visa_status == Visa.NOT_MENTIONED
    assert r.remote_status == Remote.NOT_MENTIONED
    assert not r.has_discovery_signal
    assert r.classification_confidence is None
    assert r.job_country == "Canada"


def test_classify_job_flags():
    labels = classify_job(JobFlags(True, True, False, True, False, True))
    assert labels["visa_status"] == Visa.POSITIVE_EVIDENCE
    assert labels["remote_status"] == Remote.RESTRICTED_REMOTE
    assert classify_job(JobFlags(False, False, False, False, False, False))["has_discovery_signal"] is False


@pytest.mark.parametrize(
    "location, country",
    [("Berlin, Berlin, Germany", "Germany"), ("United States (Remote)", "United States"), ("Bengaluru, Karnataka, India", "India"),
     ("EMEA", None), (None, None)],
)
def test_job_country(location, country):
    assert job_country(location) == country


# --- retention: classification never removes relevant jobs ----------------------


@pytest.fixture()
def worldwide_on(monkeypatch):
    monkeypatch.setenv("JOB_WORLDWIDE_REMOTE_ENABLED", "true")
    monkeypatch.setenv("JOB_REJECT_COUNTRY_RESTRICTED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _raw(job_id, description, *, location="London, England, United Kingdom", workplace_type="On-site"):
    return RawJob(
        source="linkedin",
        source_job_id=job_id,
        title="Senior Full Stack Engineer",
        company="Acme",
        location=location,
        description=f"{_PROFILE_TEXT} {description}",
        application_url=f"https://www.linkedin.com/jobs/view/{job_id}/",
        source_url=f"https://www.linkedin.com/jobs/view/{job_id}/",
        extra={"workplace_type": workplace_type},
    ).model_dump(mode="json")


def _pipeline(*raws, request):
    state = {"run_id": "r1", "discovered_jobs": list(raws), "request": request,
             "source_configs": [{"id": "linkedin", "allowed_domains": ["linkedin.com"]}]}
    normalized = asyncio.run(normalization_node(state))["normalized_jobs"]
    return asyncio.run(validation_node({**state, "normalized_jobs": normalized}))["validated_jobs"]


def test_normalization_attaches_labels_and_keeps_every_job(worldwide_on):
    jobs = _pipeline(
        _raw("1", "We provide full visa sponsorship."),
        _raw("2", "We do not provide visa sponsorship."),
        _raw("3", "Competitive salary."),
        request={},
    )
    assert [j["visa_remote"]["visa_status"] for j in jobs] == ["POSITIVE_EVIDENCE", "NEGATIVE_EVIDENCE", "NOT_MENTIONED"]


def test_remote_only_keeps_onsite_jobs_offering_sponsorship(worldwide_on):
    jobs = _pipeline(
        _raw("1", "We provide full visa sponsorship."),
        _raw("2", "We do not provide visa sponsorship."),
        request={"remote_only": True},
    )
    assert [j["source_job_id"] for j in jobs] == ["1"]


def _restricted_job(visa_status):
    return {"worldwide": {"category": "country_restricted", "reason": "US only"}, "visa_remote": {"visa_status": visa_status}}


def test_country_restricted_jobs_kept_by_default(worldwide_on):
    assert _not_eligible_reason(_restricted_job("NEGATIVE_EVIDENCE"), get_settings()) is None


def test_reject_setting_still_keeps_jobs_offering_sponsorship(worldwide_on, monkeypatch):
    monkeypatch.setenv("JOB_REJECT_COUNTRY_RESTRICTED", "true")
    get_settings.cache_clear()
    settings = get_settings()
    assert _not_eligible_reason(_restricted_job("POSITIVE_EVIDENCE"), settings) is None
    assert _not_eligible_reason(_restricted_job("NOT_MENTIONED"), settings).startswith("Not eligible from India")


# --- real postings that were mislabelled ---------------------------------------


def test_anti_scam_visa_footer_is_not_a_visa_mention():
    r = _classify(
        "Commerce Will Never\n"
        "attempt to solicit money from you as part of the hiring process or as part of an employment offer;\n"
        "solicit money to complete visa requirements as part of a job offer.\n"
        "Work Where You Thrive\nCommerce is a remote-first company.",
        location="Austin, TX (Remote)", workplace_type="Remote",
    )
    assert r.visa_status == Visa.NOT_MENTIONED
    assert r.visa_evidence == []
    assert r.remote_status == Remote.REMOTE_UNCLEAR


def test_fully_remote_with_us_work_authorization_is_restricted():
    r = _classify(
        "About the job\nThis is a Fully Remote Job\nRequirements:\n"
        "Unrestricted U.S. work authorization without visa sponsorship.",
        location="United States (Remote)", workplace_type="Remote",
    )
    assert r.visa_status == Visa.NEGATIVE_EVIDENCE
    assert r.remote_status == Remote.RESTRICTED_REMOTE
    assert r.remote_countries_allowed == ["United States"]


def test_fully_remote_without_scope_gets_no_global_remote_tag(worldwide_on):
    jobs = _pipeline(_raw("1", "This is a Fully Remote Job.", location="United States (Remote)", workplace_type="Remote"),
                     request={})
    assert jobs[0]["visa_remote"]["remote_status"] == "REMOTE_UNCLEAR"
    assert "Global Remote" not in jobs[0]["tags"]


def test_explicit_worldwide_gets_global_remote_tag(worldwide_on):
    jobs = _pipeline(_raw("1", "You can work from anywhere in the world.", location="Remote", workplace_type="Remote"),
                     request={})
    assert jobs[0]["visa_remote"]["remote_status"] == "GLOBAL_REMOTE_EVIDENCE"
    assert "Global Remote" in jobs[0]["tags"]


# --- merging the matcher's eligibility read ------------------------------------

from jobEngine.graph.nodes import matching  # noqa: E402
from jobEngine.matching.visa_remote import apply_llm_eligibility  # noqa: E402
from jobEngine.models.job import JobEligibility, JobMatchAnalysis  # noqa: E402
from jobEngine.models.profile import CandidateProfile  # noqa: E402


def _read(**kw):
    base = dict(visa_status="NOT_MENTIONED", relocation_support="NOT_MENTIONED", remote_status="NOT_MENTIONED",
                india_category="needs_verification")
    return JobEligibility(**{**base, **kw})


def test_llm_decisive_label_needs_a_verbatim_quote():
    text = f"{_PROFILE_TEXT} Applicants from any country are welcome; we sponsor work visas for this role."
    rules = classify_visa_remote(title="Engineer", location="Germany", description=text)
    grounded = apply_llm_eligibility(rules, _read(visa_status="POSITIVE_EVIDENCE",
                                                  visa_quotes=["we sponsor work visas for this role."]), text)
    assert grounded.visa_status == Visa.POSITIVE_EVIDENCE and grounded.method == "llm"
    invented = apply_llm_eligibility(rules, _read(remote_status="RESTRICTED_REMOTE",
                                                  remote_quotes=["Candidates must live in the US."]), text)
    assert invented.remote_status == rules.remote_status  # quote not in the posting: rules stand


def test_llm_can_clear_a_false_visa_mention():
    text = f"{_PROFILE_TEXT} Visa requirements apply."
    rules = classify_visa_remote(title="Engineer", location="Germany", description=text)
    assert rules.visa_status == Visa.MENTIONED_UNCLEAR
    assert apply_llm_eligibility(rules, _read(), text).visa_status == Visa.NOT_MENTIONED


def test_one_llm_call_per_job_and_no_separate_worldwide_review(worldwide_on, monkeypatch):
    monkeypatch.setenv("JOB_LLM_TOP_K", "-1")
    get_settings.cache_clear()
    description = "This is a fully remote role. Work from anywhere in the world. We provide full visa sponsorship."
    job = _pipeline(_raw("1", description, location="United States (Remote)", workplace_type="Remote"), request={})[0]
    job["worldwide"]["category"] = "needs_verification"  # what the separate review would otherwise look at
    calls = []

    def analyze(self, *, profile, job, metrics):
        calls.append(job.source_job_id)
        metrics.calls += 1
        return JobMatchAnalysis(
            recommendation="strong_match", explanation="fit",
            eligibility=_read(visa_status="POSITIVE_EVIDENCE", visa_quotes=["We provide full visa sponsorship."],
                              remote_status="GLOBAL_REMOTE_EVIDENCE", remote_quotes=["Work from anywhere in the world."],
                              india_category="worldwide_confirmed", india_quotes=["Work from anywhere in the world."]),
        )

    def no_second_call(self, *a, **kw):
        raise AssertionError("worldwide review must not run after the matcher's read")

    monkeypatch.setattr(matching.LLMMatcher, "available", lambda self: True)
    monkeypatch.setattr(matching.LLMMatcher, "_get_structured_llm", lambda self: None)
    monkeypatch.setattr(matching.LLMMatcher, "analyze", analyze)
    monkeypatch.setattr(matching.WorldwideReviewer, "available", lambda self: True)
    monkeypatch.setattr(matching.WorldwideReviewer, "review", no_second_call)
    state = {"validated_jobs": [job], "request": {"minimum_match_score": 50},
             "candidate_profile": CandidateProfile().model_dump(mode="json")}
    out = asyncio.run(matching.matching_node(state))
    (matched,) = out["matched_jobs"]
    assert calls == ["1"] and out["metrics"]["llm_calls"] == 1
    assert matched["job"]["visa_remote"]["method"] == "llm"
    assert matched["job"]["visa_remote"]["visa_status"] == "POSITIVE_EVIDENCE"
    assert matched["job"]["worldwide"]["category"] == "worldwide_confirmed"


def test_llm_cannot_upgrade_fully_remote_to_global():
    """gpt-4o-mini labelled these GLOBAL_REMOTE_EVIDENCE in a live run, quoting them verbatim."""
    text = f"{_PROFILE_TEXT}\nThis is a Fully Remote Job\nUnrestricted U.S. work authorization without visa sponsorship."
    rules = classify_visa_remote(title="Engineer", location="United States (Remote)", description=text, workplace_type="Remote")
    assert rules.remote_status == Remote.RESTRICTED_REMOTE
    merged = apply_llm_eligibility(
        rules, _read(remote_status="GLOBAL_REMOTE_EVIDENCE", remote_quotes=["This is a Fully Remote Job"]), text
    )
    assert merged.remote_status == Remote.RESTRICTED_REMOTE
    assert merged.remote_countries_allowed == ["United States"]

    unscoped = f"{_PROFILE_TEXT}\nCommerce is a remote-first company."
    rules = classify_visa_remote(title="Engineer", location="Austin, TX (Remote)", description=unscoped)
    merged = apply_llm_eligibility(
        rules, _read(remote_status="GLOBAL_REMOTE_EVIDENCE", remote_quotes=["Commerce is a remote-first company."]), unscoped
    )
    assert merged.remote_status == Remote.REMOTE_UNCLEAR


def test_llm_global_against_rules_restriction_is_a_conflict():
    text = f"{_PROFILE_TEXT} Candidates must be located in the United States. You can work from anywhere in the world."
    rules = classify_visa_remote(title="Engineer", location="Germany", description=text)
    assert rules.remote_conflict  # the rules already see both
    merged = apply_llm_eligibility(rules, _read(remote_status="RESTRICTED_REMOTE",
                                                remote_quotes=["Candidates must be located in the United States."]), text)
    assert merged.remote_status == Remote.RESTRICTED_REMOTE and not merged.remote_conflict


def test_llm_missing_a_sentence_never_erases_rules_evidence():
    text = f"{_PROFILE_TEXT} We provide full visa sponsorship."
    rules = classify_visa_remote(title="Engineer", location="Germany", description=text)
    assert apply_llm_eligibility(rules, _read(), text).visa_status == Visa.POSITIVE_EVIDENCE
