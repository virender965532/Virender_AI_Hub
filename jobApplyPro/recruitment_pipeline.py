from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.parse
import webbrowser
from typing import Any

from dotenv import load_dotenv
from pypdf import PdfReader

from agents import Agent, Runner, trace

from .config import (
    EMAIL_DRAFT_PATH,
    JD_ANALYSIS_PATH,
    OPENAI_AGENT_MODEL,
    PROFILE_ANALYSIS_PATH,
    PROFILE_DIR,
    RESUME_FILENAME,
)
from .history import save_application_history

load_dotenv(override=True)
logger = logging.getLogger(__name__)

PROMPTS = {
    "JobRequirementAnalyzer": (
        "Understand the job requirement provided in the input text.\n"
        "Return a detailed breakdown of: key responsibilities, required skills (hard and soft),"
        " seniority level, mandatory vs nice-to-have qualifications, and suggested interview topics."
    ),
    "CandidateProfileAnalyzer": (
        "Given the candidate resume text provided in the input, analyze and summarize the candidate's "
        "skills, experience, achievements, and gaps. Output a structured profile with skill tags, "
        "years of experience per area, notable projects, and suggested strengths to highlight."
    ),
    "HiringManagerEmailDraftsman": (
        "Using the finalized job description, and extracted resume profile, draft a highly professional, "
        "concise, and persuasive email to the hiring manager expressing interest in the role. The email "
        "must open courteously, highlight the strongest skills, achievements, and experience that "
        "directly match the job description, and clearly show why the candidate is an excellent fit for "
        "the position. Incorporate important keywords from both the resume profile and job description to "
        "improve relevance and impact. Ensure the tone is polite, confident, and value-driven. Close with "
        "a polite call to action, include contact details, and mention attached documents such as the "
        "resume and fitment report. Output only the email body."
    ),
    "EmailSubjectLineCreator": (
        "Generate a professional, concise, and attention-grabbing subject line for a job application "
        "email, using the candidate's top strengths and the job title as context. The subject line should "
        "be clear, specific, and appealing to a hiring manager, reflecting the role, years of experience "
        "if applicable, and top relevant skills or technologies extracted from the resume profile and job "
        'description. Format it in a clean, industry-standard way similar to: "Experienced Frontend '
        'Engineer — 6 yrs, React & TypeScript".'
    ),
    "EmailReviewerEvaluator": (
        """Review the one generated subject line and the two generated email bodies together for quality, clarity, professionalism, tone, grammar, persuasiveness, and alignment with both the job description and the candidate profile. First, select the best email body from the two drafts based on clarity, structure, personalization, alignment with job requirements, keyword usage, achievements, tone, conciseness, and persuasive impact. Then evaluate this chosen email together with the subject line to ensure the subject is sharp, role-specific, and attention-grabbing, and the email body is well-structured, personalized, keyword-rich, achievement-focused, and clearly demonstrates the candidate's fit. If any issue is found—such as weak phrasing, missing keywords, poor tone, insufficient personalization, or unclear messaging—you must request a regeneration. Only one regeneration cycle is allowed. If regeneration is required, you must call EmailSubjectLineCreator once and HiringManagerEmailDraftsman twice simultaneously, then select the best regenerated email body and evaluate it again with the regenerated subject line. If the regenerated versions still fall short, return the best possible result and explain its limitations. Your final output must be exactly one JSON object, using the following structure when fully satisfied: { "is_satisfied": true, "reason": "<explanation>", "final_subject": "<best subject line>", "final_email_body": "<best email body>" }. Do not include any text outside the JSON, and do not use Markdown."""
    ),
}


def open_email_default(to_email: str, subject: str, html_content: str) -> dict[str, Any]:
    logger.info("open_email_default started")
    try:
        subject_encoded = urllib.parse.quote(subject)
        body_encoded = urllib.parse.quote(html_content)
        gmail_url = (
            f"https://mail.google.com/mail/?view=cm&fs=1&tf=1&to={to_email}"
            f"&su={subject_encoded}&body={body_encoded}"
        )
        webbrowser.open(gmail_url)
        logger.info("open_email_default completed successfully")
        return {"status": "success", "message": "Email client opened successfully."}
    except Exception as e:
        logger.error("open_email_default failed: %s", e)
        return {"status": "failure", "error": str(e)}


def extract_domain(email: str) -> str:
    try:
        return email.split("@")[1]
    except IndexError:
        return ""


def extract_resume_data() -> str:
    resume_path = PROFILE_DIR / RESUME_FILENAME
    if not resume_path.exists():
        raise FileNotFoundError(
            f"Resume not found at {resume_path}. "
            f"Place {RESUME_FILENAME} in {PROFILE_DIR}."
        )
    reader = PdfReader(str(resume_path))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def save_data(data: str, path: str) -> bool:
    try:
        abs_path = os.path.abspath(path)
        folder = os.path.dirname(abs_path)
        os.makedirs(folder, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(data)
        return True
    except Exception as e:
        logger.error("save_data failed: %s", e)
        return False


def save_data_json(data: dict[str, Any], path: str) -> bool:
    try:
        abs_path = os.path.abspath(path)
        folder = os.path.dirname(abs_path)
        os.makedirs(folder, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error("save_data_json failed: %s", e)
        return False


def extract_profile_data(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


async def run_recruitment_pipeline(
    job_requirement_text: str,
    hiring_manager_email: str,
    *,
    reparse: bool = False,
) -> dict[str, Any] | None:
    logger.info("run_recruitment_pipeline started")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    domain = extract_domain(hiring_manager_email)
    try:
        resume_data = extract_resume_data()
        job_analyser = Agent(
            name="Job Requirement Analyzer",
            instructions=PROMPTS["JobRequirementAnalyzer"],
            model=OPENAI_AGENT_MODEL,
        )
        candidate_profile_agent = Agent(
            name="Candidate Profile Analyzer",
            instructions=PROMPTS["CandidateProfileAnalyzer"],
            model=OPENAI_AGENT_MODEL,
        )

        with trace(f"{domain} Profile & JD Tracking for Agentic Apply Pro AI"):
            if reparse or not PROFILE_ANALYSIS_PATH.exists():
                results = await asyncio.gather(
                    Runner.run(job_analyser, job_requirement_text),
                    Runner.run(candidate_profile_agent, resume_data),
                )
                jd_data = results[0].final_output
                profile_data = results[1].final_output
            else:
                results = await asyncio.gather(
                    Runner.run(job_analyser, job_requirement_text),
                )
                jd_data = results[0].final_output
                profile_data = extract_profile_data(str(PROFILE_ANALYSIS_PATH))

        save_data(jd_data, str(JD_ANALYSIS_PATH))
        save_data(profile_data, str(PROFILE_ANALYSIS_PATH))

        subject_creator = Agent(
            name="Email Subject Line Creator",
            instructions=(
                f"{PROMPTS['EmailSubjectLineCreator']} Job Requirement Analysis:\n{jd_data}\n\n"
                f"Candidate Profile Analysis:\n{profile_data}"
            ),
            model=OPENAI_AGENT_MODEL,
        )
        email_draftsman = Agent(
            name="Hiring Manager Email Draftsman",
            instructions=(
                f"{PROMPTS['HiringManagerEmailDraftsman']} Job Requirement Analysis:\n{jd_data}\n\n"
                f"Candidate Profile Analysis:\n{profile_data}"
            ),
            model=OPENAI_AGENT_MODEL,
        )
        subject_tool = subject_creator.as_tool(
            tool_name="subject_creator",
            tool_description=PROMPTS["EmailSubjectLineCreator"],
        )
        draftsman_tool = email_draftsman.as_tool(
            tool_name="email_draftsman",
            tool_description=PROMPTS["HiringManagerEmailDraftsman"],
        )
        reviewer = Agent(
            name="Email Reviewer Evaluator",
            instructions=(
                f"{PROMPTS['EmailReviewerEvaluator']} Job Requirement Analysis:\n{jd_data}\n\n"
                f"Candidate Profile Analysis:\n{profile_data}"
            ),
            model=OPENAI_AGENT_MODEL,
            tools=[draftsman_tool, subject_tool],
        )

        with trace(f"{domain} Email Drafting & Reviewing for Agentic Apply Pro AI"):
            response = await Runner.run(
                reviewer,
                (
                    f"{PROMPTS['EmailReviewerEvaluator']} Job Requirement Analysis:\n{jd_data}\n\n"
                    f"Candidate Profile Analysis:\n{profile_data}"
                ),
            )

        final_output = response.final_output
        if isinstance(final_output, str):
            final_output = json.loads(final_output)

        email_payload = {
            "subject": final_output["final_subject"],
            "body": final_output["final_email_body"],
        }
        save_data_json(email_payload, str(EMAIL_DRAFT_PATH))
        email_status = open_email_default(
            hiring_manager_email,
            email_payload["subject"],
            email_payload["body"],
        )

        result = {
            "domain": domain,
            "hiring_manager_email": hiring_manager_email,
            "subject": email_payload["subject"],
            "body": email_payload["body"],
            "email_status": email_status,
            "review_reason": final_output.get("reason", ""),
        }
        save_application_history(
            {
                "domain": domain,
                "hiring_manager_email": hiring_manager_email,
                "job_description_preview": job_requirement_text[:500],
                "subject": email_payload["subject"],
                "body": email_payload["body"],
                "email_status": email_status,
            }
        )
        logger.info("run_recruitment_pipeline completed")
        return result
    except Exception as e:
        logger.exception("run_recruitment_pipeline failed")
        raise RuntimeError(str(e)) from e
