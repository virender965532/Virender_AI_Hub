from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from jobEngine.config.sources import JobSourceConfig, load_job_sources


def test_real_job_sources_yaml_loads_and_validates():
    sources = load_job_sources()
    ids = {s.id for s in sources}
    assert "linkedin" in ids
    assert "instinctools" in ids
    linkedin = next(s for s in sources if s.id == "linkedin")
    assert linkedin.requires_auth is True
    assert linkedin.allowed_domains == ["www.linkedin.com"]


def test_rejects_non_http_scheme():
    with pytest.raises(Exception):
        JobSourceConfig(
            id="bad",
            name="Bad",
            base_url="ftp://example.com",
            search_urls=["ftp://example.com/jobs"],
            adapter="generic",
        )


def test_rejects_duplicate_ids(tmp_path: Path):
    cfg_path = tmp_path / "job_sources.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            job_sources:
              - id: dup
                name: Dup 1
                base_url: "https://example.com"
                search_urls: ["https://example.com/jobs"]
                adapter: generic
              - id: dup
                name: Dup 2
                base_url: "https://example.com"
                search_urls: ["https://example.com/jobs"]
                adapter: generic
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate source id"):
        load_job_sources(cfg_path)


def test_enabled_source_requires_search_urls(tmp_path: Path):
    cfg_path = tmp_path / "job_sources.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            job_sources:
              - id: nourls
                name: No URLs
                enabled: true
                base_url: "https://example.com"
                adapter: generic
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no search_urls"):
        load_job_sources(cfg_path)


def test_missing_config_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_job_sources(tmp_path / "does_not_exist.yaml")
