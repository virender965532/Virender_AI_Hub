import os
import json
from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4.1-mini"


def interviewer(state):
    prompt = f"""
    Role: {state['role']}
    Difficulty: {state['difficulty']}
    Questions asked: {state['questions_asked']} / {state['total_questions']}
    """

    if state.get("last_score"):
        prompt += f"\nLast Score: {state['last_score']}"

    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Ask ONE interview question."},
            {"role": "user", "content": prompt}
        ]
    )

    q = res.choices[0].message.content.strip()

    state["current_question"] = q
    state["questions_asked"] += 1

    if state["questions_asked"] >= state["total_questions"]:
        state["active"] = False

    return state


def evaluator(state):
    q = state["current_question"]
    a = state["user_answer"]

    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Evaluate answer in JSON with score 0-5"},
            {"role": "user", "content": f"{q}\n{a}"}
        ],
        response_format={"type": "json_object"}
    )

    data = json.loads(res.choices[0].message.content)

    state["last_score"] = data.get("overall_score", 0)
    state.setdefault("scores", []).append(data)

    return state


def coach(state):
    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Give feedback + model answer"},
            {"role": "user", "content": str(state)}
        ]
    )

    state["feedback"] = res.choices[0].message.content

    return state