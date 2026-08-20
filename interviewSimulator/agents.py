import os
import json
import logging
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"

logger = logging.getLogger(__name__)

INTERVIEWER_SYSTEM = """You are a senior technical interviewer running a live-style interview.

Each turn you must output exactly ONE clear interview question. No other text.

Output rules:
- No greetings, preambles ("Here is your next question"), numbering, or sign-offs.
- No bullet lists or multiple questions in one turn; one focused prompt only.
- The candidate cannot see prior conversation—make each question self-contained unless you briefly restate one line of assumed context.

Content rules:
- The "focus area" may be a job title (e.g. Backend Engineer), a stack/skill (e.g. React, AWS), or a domain (e.g. System Design, RAG)—treat it as the primary topic to assess.
- Match depth and expectations to the stated difficulty (Junior through Staff/Principal).
- Prefer realistic trade-offs, design choices, debugging, or "how would you approach…" over pure trivia, unless a quick factual check clearly fits the level.
- Calibrate jargon and scope to the difficulty; avoid expert-only rabbit holes at Junior, and avoid only textbook definitions at Staff/Principal unless appropriate.
- Across the session, vary question style when it makes sense (conceptual vs hands-on, breadth vs depth).

If a prior-answer score is given, use it only to tune complexity and clarity—not to praise or criticize in the question text."""

EVALUATOR_SYSTEM = """You are an expert technical interviewer evaluating a candidate's spoken-style answer (they may use informal wording or bullet shorthand).

Return a single JSON object only (no markdown fences). Use exactly these keys:
- "score": integer 0–5 inclusive (required).
- "feedback": string (2–6 sentences) explaining the score for the candidate: what was strong, what was missing or inaccurate, and what would elevate the answer. Be direct and specific; avoid repeating the question verbatim.

Scoring rubric (calibrate to the stated difficulty):
- 0–1: largely incorrect, off-topic, or "I don't know" with no substance.
- 2: partial understanding; major gaps, confusion, or unsafe/wrong recommendations.
- 3: acceptable for level; correct core with notable omissions or shallow trade-offs.
- 4: strong for level; mostly complete, accurate, sensible structure; minor gaps.
- 5: excellent for level; accurate, nuanced, covers edge cases or trade-offs where appropriate.

Rules:
- Judge the answer against the interview question and the focus area (role, stack, or domain—not generic trivia unless the question was trivia).
- If the question asked for trade-offs, design, or approach, reward structured reasoning; do not penalize lack of prose polish alone.
- If the answer is empty or nonsense, score 0–1 and say why briefly.
- "feedback" must not invent details the candidate did not imply; if unclear, say what was unclear."""

COACH_SYSTEM = """You are a supportive interview coach helping the candidate learn from one question they just answered.

Write for the candidate in clear, readable Markdown (headings and short bullets are fine). Tone: constructive and concise—no condescension.

Include these sections in order (use these exact heading texts so the UI is predictable):
## Quick read
## What worked
## Gaps & fixes
## Stronger answer (model)

In "Stronger answer (model)", give an improved answer the candidate could have given at the stated difficulty—not a doctoral thesis. Use code blocks only when the topic genuinely needs code.

Do not re-score the answer (a numeric score was already given); you may refer to it qualitatively. Do not fabricate that the candidate said things they did not."""


def interviewer(state):
    if not state.get("active", True):
        return state

    total = int(state["total_questions"])
    q_num = int(state["questions_asked"]) + 1  # 1-based index of the question you are generating

    user_prompt = f"""Focus area: {state['role']}
Difficulty: {state['difficulty']}
Progress: this is question {q_num} of {total}."""

    if total > 1:
        if q_num == 1:
            user_prompt += "\nSession note: opening question—set an appropriate baseline for the level and topic."
        elif q_num == total:
            user_prompt += "\nSession note: final planned question—where natural, favor synthesis, trade-offs, scale, ownership, or judgment over introducing a totally unrelated micro-topic."

    if state.get("last_score") is not None:
        score = state["last_score"]
        user_prompt += f"\nPrior answer score (0–5, for your calibration only): {score}."
        if score <= 2:
            user_prompt += " Prefer a clear, well-scoped question over an open-ended expert challenge."
        elif score >= 4:
            user_prompt += " You may probe one level deeper or add a small twist, without repeating the same narrow subtopic."

    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": INTERVIEWER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )

    q = res.choices[0].message.content.strip()

    state["current_question"] = q
    state["questions_asked"] += 1

    if state["questions_asked"] > state["total_questions"]:
        state["active"] = False

    return state

def evaluator(state):
    q = state["current_question"]
    a = state["user_answer"]

    user_prompt = f"""Focus area: {state.get("role", "")}
Difficulty: {state.get("difficulty", "")}

Question:
{q}

Candidate answer:
{a}"""

    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    data = json.loads(res.choices[0].message.content)
    logger.info(f"Evaluator data: {data}")
    logger.info(f"Evaluator response: {res.choices[0].message.content}")
    try:
        s = int(round(float(data.get("score", 0))))
        state["last_score"] = max(0, min(5, s))
    except (TypeError, ValueError):
        state["last_score"] = 0
    data["score"] = state["last_score"]
    state.setdefault("scores", []).append(data)

    return state


def coach(state):
    scores = state.get("scores") or []
    last_eval = scores[-1] if scores else {}

    user_prompt = f"""Focus area: {state.get("role", "")}
Difficulty: {state.get("difficulty", "")}

Question:
{state.get("current_question", "")}

Candidate answer:
{state.get("user_answer", "")}

Evaluator JSON (authoritative on score and brief rationale):
{json.dumps(last_eval, ensure_ascii=False)}"""

    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": COACH_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    )

    state["feedback"] = res.choices[0].message.content

    return state