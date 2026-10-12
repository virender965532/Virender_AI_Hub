from __future__ import annotations

import pytest

from jobEngine.matching.worldwide_remote import classify_worldwide, detect_arrangement
from jobEngine.models.job import EmploymentArrangement as Arr
from jobEngine.models.job import WorldwideCategory as Cat

_FILLER = (
    " You will build Node.js and React services on AWS, own features end to end, review pull requests,"
    " and work closely with product and design to ship a TypeScript and Next.js platform used by thousands."
)


def _classify(description, *, title="Senior Node.js Engineer", location="United States (Remote)", **kw):
    return classify_worldwide(title=title, location=location, description=description + _FILLER, **kw)


def _positive_texts(result):
    return [e.text for e in result.evidence if e.signal == "positive"]


def test_worldwide_contractors_confirmed_with_evidence():
    r = _classify(
        "This is a contract role. We work with independent contractors worldwide and you can work from any country.",
        employment_type="Contract",
    )
    assert r.category == Cat.WORLDWIDE_CONFIRMED
    assert r.india_eligible
    assert r.arrangement == Arr.CONTRACTOR
    assert any("from any country" in t for t in _positive_texts(r))
    assert all(e.found_in for e in r.evidence)


def test_india_based_remote_employees_eligible():
    r = _classify(
        "Full-time remote position. We are hiring engineers based in the US, Poland or India for this team.",
        location="Poland (Remote)",
        employment_type="Full-time",
    )
    assert r.category == Cat.INDIA_ELIGIBLE
    assert r.india_eligible
    assert r.arrangement == Arr.FULL_TIME_EMPLOYEE
    assert any("India" in t and e.found_in == "description" for e in r.evidence for t in [e.text])


def test_us_only_remote_employee_restricted():
    r = _classify(
        "This is a fully remote position, US only. Candidates must be authorized to work in the United States.",
        employment_type="Full-time",
    )
    assert r.category == Cat.COUNTRY_RESTRICTED
    assert not r.india_eligible
    assert any(e.signal == "negative" and "US only" in e.text for e in r.evidence)


def test_remote_us_only_in_title_is_not_worldwide():
    r = _classify("Fully remote role on our platform team.", title="Senior React Developer (Remote — US only)")
    assert r.category == Cat.COUNTRY_RESTRICTED


def test_europe_only_not_eligible():
    r = _classify(
        "Remote within Europe. You must be based in the EU to be considered for this role.",
        location="European Union (Remote)",
    )
    assert r.category == Cat.COUNTRY_RESTRICTED
    assert not r.india_eligible


def test_remote_europe_without_rule_needs_verification():
    r = _classify("Remote — Europe. Join our product team.", location="Germany (Remote)")
    assert r.category == Cat.NEEDS_VERIFICATION


def test_remote_without_location_policy_needs_verification():
    r = _classify("This is a remote position on our payments team.", location="United States (Remote)")
    assert r.category == Cat.NEEDS_VERIFICATION
    assert not r.india_eligible


def test_part_time_worldwide_contractor():
    r = _classify(
        "Part-time contract, about 20 hours per week. Contractors can be located anywhere in the world.",
        employment_type="Part-time",
        location="Worldwide",
    )
    assert r.category == Cat.WORLDWIDE_CONFIRMED
    assert r.arrangement == Arr.PART_TIME_CONTRACTOR


@pytest.mark.parametrize("description", ["", "   ", "Apply on our site."])
def test_missing_description_is_insufficient(description):
    r = classify_worldwide(title="Senior Engineer", location="United States (Remote)", description=description)
    assert r.category == Cat.INSUFFICIENT_INFO
    assert not r.india_eligible


def test_short_description_keeps_only_decisive_evidence():
    kw = {"title": "Engineer", "location": "United States (Remote)"}
    assert classify_worldwide(description="Remote, US only.", **kw).category == Cat.COUNTRY_RESTRICTED
    assert classify_worldwide(description="Remote. Open to applicants worldwide.", **kw).category == Cat.WORLDWIDE_CONFIRMED
    # A "Worldwide" location alone can't be checked against a description that isn't there.
    assert classify_worldwide(title="Engineer", location="Worldwide", description="").category == Cat.INSUFFICIENT_INFO
    assert classify_worldwide(description="Remote role.", **kw).category == Cat.INSUFFICIENT_INFO


def test_work_from_anywhere_in_the_us_is_not_worldwide():
    r = _classify("Work from anywhere in the US. We are a remote-first company.")
    assert r.category == Cat.COUNTRY_RESTRICTED


def test_contractor_policy_does_not_apply_to_employee_role():
    r = _classify(
        "We hire contractors worldwide across many projects. This position is a full-time employee role with benefits.",
        employment_type="Full-time",
    )
    assert r.category == Cat.NEEDS_VERIFICATION
    assert "contractor" in r.reason.lower()


def test_work_authorization_with_contract_role_needs_verification():
    r = _classify(
        "This is a contract role. Candidates must be authorized to work in the United States.",
        employment_type="Contract",
    )
    assert r.category == Cat.NEEDS_VERIFICATION


def test_worldwide_text_conflicting_with_residence_requirement():
    r = _classify("Work from anywhere in the world! Note: you must be located in Canada for payroll reasons.")
    assert r.category == Cat.NEEDS_VERIFICATION
    assert {e.signal for e in r.evidence} >= {"positive", "negative"}


def test_india_explicitly_excluded():
    r = _classify("Remote worldwide, except India and China due to payroll limitations.", location="Worldwide")
    assert r.category == Cat.COUNTRY_RESTRICTED


def test_company_offices_in_india_are_not_eligibility():
    r = _classify("We have offices in Bangalore, India and London. This is a remote position.")
    assert r.category == Cat.NEEDS_VERIFICATION


def test_us_based_company_is_not_a_restriction():
    r = _classify("We are a US-based company building developer tools. The role is remote.")
    assert r.category == Cat.NEEDS_VERIFICATION


def test_pronoun_us_is_not_the_country():
    r = _classify("Join us only if you love TypeScript. Fully remote, open to applicants worldwide.")
    assert r.category == Cat.WORLDWIDE_CONFIRMED


def test_onsite_abroad_is_restricted():
    r = _classify("Join our Berlin office team.", location="Berlin, Germany (On-site)")
    assert r.category == Cat.COUNTRY_RESTRICTED


def test_india_remote_posting_is_india_eligible():
    r = classify_worldwide(title="Senior Node.js Developer", location="India (Remote)", description="")
    assert r.category == Cat.INDIA_ELIGIBLE
    assert r.india_eligible


def test_india_onsite_posting_is_local():
    r = classify_worldwide(title="Senior Node.js Developer", location="Noida, Uttar Pradesh, India (On-site)", description="x" * 300)
    assert r.category == Cat.INDIA_LOCAL


@pytest.mark.parametrize(
    ("employment_type", "text", "expected"),
    [
        ("Full-time", "Great team.", Arr.FULL_TIME_EMPLOYEE),
        ("Contract", "Full-time hours.", Arr.FULL_TIME_CONTRACTOR),
        ("Contract", "Part-time, 15 hours per week.", Arr.PART_TIME_CONTRACTOR),
        ("Contract", "Great team.", Arr.CONTRACTOR),
        (None, "This is a 6 month contract.", Arr.TEMPORARY_CONTRACT),
        (None, "Freelance, project-based work.", Arr.FREELANCE),
        ("Part-time", "Great team.", Arr.PART_TIME),
        (None, "Great team.", Arr.OTHER),
        (None, "We use smart contracts on Ethereum.", Arr.OTHER),
    ],
)
def test_detect_arrangement(employment_type, text, expected):
    assert detect_arrangement(employment_type, "Engineer", text)[0] == expected
