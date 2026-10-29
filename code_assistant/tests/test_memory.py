"""Session memory (summarization, changes) and context-window management."""

from __future__ import annotations

import pytest

from code_assistant.memory.context_manager import ContextManager, count_message_tokens
from code_assistant.memory.session import Session, SessionStore
from code_assistant.schemas import ProposedChange


def test_old_turns_are_summarized_not_dropped():
    session = Session("s", "w")
    for i in range(10):
        session.add_turn("user", f"question {i}")
        session.add_turn("assistant", f"answer {i}")
    seen = []
    assert session.compact(4, lambda text: seen.append(text) or "- asked about calc.py")
    assert len(session.turns) == 4 and session.summary == "- asked about calc.py"
    assert "question 0" in seen[0]
    assert "asked about calc.py" in session.history_text()


def test_summarizer_failure_keeps_a_digest():
    session = Session("s", "w")
    for i in range(10):
        session.add_turn("user", f"question {i}")

    def boom(_text):
        raise RuntimeError("down")

    session.compact(2, boom)
    assert "question 0" in session.summary and len(session.turns) == 2


def test_store_isolation_and_change_lookup():
    store = SessionStore()
    session = store.get_or_create(None, "w1")
    assert store.get_or_create(session.id, "w1") is session
    with pytest.raises(ValueError):
        store.get_or_create(session.id, "w2")
    change = ProposedChange(change_id="abc", path="a.py", unified_diff="", explanation="")
    session.add_change(change)
    assert store.find_change("abc") == (session, change)
    with pytest.raises(KeyError):
        store.find_change("nope")


def test_context_manager_compresses_old_tool_output():
    big = "\n".join(f"line {i} " + "x" * 50 for i in range(400))
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(8):
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": big})
    manager = ContextManager(budget_tokens=20_000)
    fitted = manager.fit(messages)
    assert count_message_tokens(fitted) <= 20_000
    assert fitted[0] == messages[0] and fitted[1] == messages[1]
    assert "[summarized:" in fitted[3]["content"]
    assert len(fitted) == len(messages)  # nothing dropped; tool-call pairing preserved
    assert messages[3]["content"] == big  # original untouched


def test_overflow_trim_shrinks_task_context():
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "x" * 20000}]
    trimmed = ContextManager(budget_tokens=100_000).overflow_trim(messages)
    assert len(trimmed[1]["content"]) < 12000 and "context trimmed" in trimmed[1]["content"]
