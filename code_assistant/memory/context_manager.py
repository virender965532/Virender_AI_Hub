"""Token accounting and context-window management for the agent loop.

The agent's message list grows with every tool result. Before each LLM call
``ContextManager.fit`` keeps it under ``CA_CONTEXT_TOKENS``: the system prompt,
the task message and the most recent tool results stay intact; older tool
outputs are replaced by a short digest that says what was omitted (never
dropped silently). ``overflow_trim`` is the aggressive variant used once when
the API still reports a context-length error.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

Messages = list[dict[str, Any]]

_PER_MESSAGE_OVERHEAD = 4


@lru_cache(maxsize=8)
def _encoding(model: str):
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - tiktoken is in requirements.txt
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001 - offline without cached encodings
            return None
    except Exception:  # noqa: BLE001 - tiktoken may need a download on first use
        return None


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = _encoding(model)
    if encoding is None:
        return max(1, len(text) // 4)
    return len(encoding.encode(text, disallowed_special=()))


def message_tokens(message: dict[str, Any], model: str = "gpt-4o") -> int:
    total = _PER_MESSAGE_OVERHEAD + count_tokens(str(message.get("content") or ""), model)
    for call in message.get("tool_calls") or []:
        total += count_tokens(json.dumps(call), model)
    return total


def count_message_tokens(messages: Messages, model: str = "gpt-4o") -> int:
    return sum(message_tokens(m, model) for m in messages)


def digest(content: str, keep_lines: int) -> str:
    """Keep the first ``keep_lines`` lines and state how much was omitted."""
    lines = content.splitlines()
    if len(lines) <= keep_lines:
        return content
    head = "\n".join(lines[:keep_lines])
    return f"{head}\n[summarized: {len(lines) - keep_lines} more lines of this earlier tool output omitted; re-run the tool if needed]"


class ContextManager:
    """Keeps an agent conversation under a token budget."""

    def __init__(self, budget_tokens: int, model: str = "gpt-4o", *, keep_recent_tool_results: int = 6) -> None:
        self.budget = budget_tokens
        self.model = model
        self.keep_recent = keep_recent_tool_results

    def tokens(self, messages: Messages) -> int:
        return count_message_tokens(messages, self.model)

    def _compress(self, messages: Messages, keep_recent: int, keep_lines: int) -> Messages:
        tool_positions = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        old = set(tool_positions[:-keep_recent] if keep_recent else tool_positions)
        result = []
        for i, message in enumerate(messages):
            if i in old:
                message = {**message, "content": digest(str(message.get("content") or ""), keep_lines)}
            result.append(message)
        return result

    def fit(self, messages: Messages) -> Messages:
        """Return a copy of ``messages`` under budget (progressively compressing older tool output)."""
        if self.tokens(messages) <= self.budget:
            return list(messages)
        for keep_recent, keep_lines in ((self.keep_recent, 20), (3, 8), (1, 3), (0, 2)):
            compressed = self._compress(messages, keep_recent, keep_lines)
            if self.tokens(compressed) <= self.budget:
                return compressed
        return self._shrink_task(compressed)

    def overflow_trim(self, messages: Messages) -> Messages:
        """Aggressive trim after a context-length error from the API."""
        return self._shrink_task(self._compress(messages, 1, 3), factor=0.5)

    def _shrink_task(self, messages: Messages, factor: float = 0.7) -> Messages:
        """Last resort: shorten the large task-context message (file tree + snippets), never the system prompt."""
        result = list(messages)
        for i, message in enumerate(result):
            if i > 0 and message.get("role") == "user":
                content = str(message.get("content") or "")
                if len(content) > 4000:
                    keep = int(len(content) * factor)
                    result[i] = {**message, "content": content[:keep // 2] + "\n...[context trimmed to fit the model]...\n" + content[-keep // 2:]}
                break
        return result
