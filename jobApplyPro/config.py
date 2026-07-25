from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PROFILE_DIR = ROOT / os.environ.get("JOB_APPLY_PROFILE_DIR", "Profile")
OUTPUT_DIR = ROOT / os.environ.get("JOB_APPLY_OUTPUT_DIR", "output") / "job_apply"
HISTORY_DIR = ROOT / os.environ.get("JOB_APPLY_HISTORY_DIR", "history") / "job_apply"
RESUME_FILENAME = os.environ.get("JOB_APPLY_RESUME_FILE", "VirenderFullstackResume.pdf")

OPENAI_AGENT_MODEL = os.environ.get("JOB_APPLY_OPENAI_MODEL") or os.environ.get(
    "OPENAI_AGENT_MODEL", "gpt-4o-mini"
)

JD_ANALYSIS_PATH = OUTPUT_DIR / "JobRequirementAnalysis" / "JobRequirementAnalysis.txt"
PROFILE_ANALYSIS_PATH = (
    OUTPUT_DIR / "CandidateProfileAnalysis" / "CandidateProfileAnalysis.txt"
)
EMAIL_DRAFT_PATH = OUTPUT_DIR / "EmailDraftsResponse" / "EmailDraftsResponse.json"
HISTORY_FILE = HISTORY_DIR / "applications.json"
