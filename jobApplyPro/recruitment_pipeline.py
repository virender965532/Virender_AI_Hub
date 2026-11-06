from __future__ import annotations

import json
import logging
import os
import urllib.parse
import webbrowser
from typing import Any

from dotenv import load_dotenv

from agents import Agent, Runner, trace

from llm_provider import get_agents_model, llm_configured, missing_credentials_message

from .config import (
    EMAIL_DRAFT_PATH,
    JD_ANALYSIS_PATH,
    OPENAI_AGENT_MODEL,
    PROFILE_DIR,
    RESUME_FILENAME,
    SUMMARY_FILENAME,
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
    "HiringManagerEmailDraftsman": (
        "Using the finalized job description, and the candidate's summary, draft a highly professional, "
        "concise, and persuasive email to the hiring manager expressing interest in the role. The email "
        "must open courteously, highlight the strongest skills, achievements, and experience that "
        "directly match the job description, and clearly show why the candidate is an excellent fit for "
        "the position. Incorporate important keywords from both the candidate summary and job description to "
        "improve relevance and impact. Ensure the tone is polite, confident, and value-driven. Close with "
        "a polite call to action, include contact details, and mention attached documents such as the "
        "resume and fitment report. Output only the email body."
    ),
    "EmailSubjectLineCreator": (
        "Generate a professional, concise, and attention-grabbing subject line for a job application "
        "email, using the candidate's top strengths and the job title as context. The subject line should "
        "be clear, specific, and appealing to a hiring manager, reflecting the role, years of experience "
        "if applicable, and top relevant skills or technologies extracted from the candidate summary and job "
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


def ensure_resume_exists() -> None:
    resume_path = PROFILE_DIR / RESUME_FILENAME
    if not resume_path.exists():
        raise FileNotFoundError(
            f"Resume not found at {resume_path}. "
            f"Place {RESUME_FILENAME} in {PROFILE_DIR}."
        )


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
    open_email_client: bool = True,
) -> dict[str, Any] | None:
    """``open_email_client=False`` skips the Gmail compose tab (for unattended runs
    such as the job engine's HR drafts, which save a mailbox draft instead)."""
    logger.info("run_recruitment_pipeline started")
    if not llm_configured(os.environ.get("OPENAI_API_KEY")):
        raise ValueError(missing_credentials_message())
    # The model name on OpenAI; a local Chat Completions model on LLM_PROVIDER=ollama.
    agent_model = get_agents_model(OPENAI_AGENT_MODEL)

    domain = extract_domain(hiring_manager_email)
    try:
        ensure_resume_exists()
        summary_path = PROFILE_DIR / SUMMARY_FILENAME
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Candidate summary not found at {summary_path}. "
                f"Place {SUMMARY_FILENAME} in {PROFILE_DIR}."
            )
        profile_data = extract_profile_data(str(summary_path))

        job_analyser = Agent(
            name="Job Requirement Analyzer",
            instructions=PROMPTS["JobRequirementAnalyzer"],
            model=agent_model,
        )

        with trace(f"{domain} JD Tracking for Autonomous Application Agent"):
            jd_result = await Runner.run(job_analyser, job_requirement_text)
            jd_data = jd_result.final_output

        save_data(jd_data, str(JD_ANALYSIS_PATH))

        subject_creator = Agent(
            name="Email Subject Line Creator",
            instructions=(
                f"{PROMPTS['EmailSubjectLineCreator']} Job Requirement Analysis:\n{jd_data}\n\n"
                f"Candidate Summary:\n{profile_data}"
            ),
            model=agent_model,
        )
        email_draftsman = Agent(
            name="Hiring Manager Email Draftsman",
            instructions=(
                f"{PROMPTS['HiringManagerEmailDraftsman']} Job Requirement Analysis:\n{jd_data}\n\n"
                f"Candidate Summary:\n{profile_data}"
            ),
            model=agent_model,
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
                f"Candidate Summary:\n{profile_data}"
            ),
            model=agent_model,
            tools=[draftsman_tool, subject_tool],
        )

        with trace(f"{domain} Email Drafting & Reviewing for Autonomous Application Agent"):
            response = await Runner.run(
                reviewer,
                (
                    f"{PROMPTS['EmailReviewerEvaluator']} Job Requirement Analysis:\n{jd_data}\n\n"
                    f"Candidate Summary:\n{profile_data}"
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
        if open_email_client:
            email_status = open_email_default(
                hiring_manager_email,
                email_payload["subject"],
                email_payload["body"],
            )
        else:
            email_status = {"status": "skipped", "message": "Email client not opened (unattended run)."}

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
