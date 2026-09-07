from __future__ import annotations

from jobEngine.graph.nodes.deduplication import deduplication_node
from jobEngine.models.job import Job


def _job_dict(**overrides) -> dict:
    job = Job(
        job_id=overrides.pop("job_id", "j1"),
        source="test",
        title=overrides.pop("title", "Senior AI Engineer"),
        company=overrides.pop("company", "Acme"),
        location=overrides.pop("location", "Remote"),
        description="Some job description text that is long enough.",
        application_url=overrides.pop("application_url", "https://example.com/apply/1"),
        source_url="https://example.com/job/1",
        raw_text_hash=overrides.pop("raw_text_hash", "hash-1"),
    )
    return job.model_dump(mode="json")


async def test_exact_duplicate_by_content_hash_is_removed():
    jobs = [_job_dict(raw_text_hash="same-hash"), _job_dict(raw_text_hash="same-hash", application_url="https://example.com/apply/2")]
    result = await deduplication_node({"validated_jobs": jobs})
    assert len(result["validated_jobs"]) == 1
    assert len(result["duplicate_jobs"]) == 1


async def test_duplicate_by_application_url_is_removed():
    jobs = [
        _job_dict(raw_text_hash="hash-a", application_url="https://example.com/apply/x"),
        _job_dict(raw_text_hash="hash-b", application_url="https://example.com/apply/x"),
    ]
    result = await deduplication_node({"validated_jobs": jobs})
    assert len(result["validated_jobs"]) == 1


async def test_duplicate_by_company_title_location_combo():
    jobs = [
        _job_dict(raw_text_hash="hash-a", application_url="https://a.com/1", title="Senior AI Engineer", company="Acme", location="Remote"),
        _job_dict(raw_text_hash="hash-b", application_url="https://b.com/1", title="senior ai engineer", company="ACME", location="remote"),
    ]
    result = await deduplication_node({"validated_jobs": jobs})
    assert len(result["validated_jobs"]) == 1


async def test_distinct_jobs_are_kept():
    jobs = [
        _job_dict(raw_text_hash="hash-a", application_url="https://a.com/1", title="Senior AI Engineer", company="Acme"),
        _job_dict(raw_text_hash="hash-b", application_url="https://b.com/1", title="Backend Engineer", company="Other Co"),
    ]
    result = await deduplication_node({"validated_jobs": jobs})
    assert len(result["validated_jobs"]) == 2
    assert result["duplicate_jobs"] == []
