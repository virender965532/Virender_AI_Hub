"""``python -m llm_provider check`` — verify the configured provider (no downloads).

``python -m llm_provider smoke`` — also send one small classification prompt
through the same path the app uses and print the answer.
"""

from __future__ import annotations

import json
import sys

from .errors import LLMProviderError
from .factory import describe, get_chat_model, get_openai_client, resolve_model
from .health import check_ollama
from .settings import get_provider_settings

_SMOKE_PROMPT = (
    "A company offers visa sponsorship and allows employees to work remotely worldwide. "
    "Classify the visa and remote-work evidence in one short sentence."
)


def main(argv: list[str]) -> int:
    command = argv[0] if argv else "check"
    if command not in ("check", "smoke"):
        print(__doc__)
        return 2
    try:
        s = get_provider_settings()
        print(json.dumps(describe(), indent=2))
        if s.is_ollama or s.embeddings_on_ollama:
            print("Ollama:", json.dumps(check_ollama(s)))
        if command == "smoke":
            model = resolve_model("gpt-4o-mini")
            answer = get_chat_model("gpt-4o-mini", temperature=0).invoke(_SMOKE_PROMPT)
            print(f"[langchain {s.provider}/{model}]", answer.content.strip())
            completion = get_openai_client(max_retries=0).chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": _SMOKE_PROMPT}], temperature=0,
            )
            print(f"[openai-sdk {s.provider}/{completion.model}]", (completion.choices[0].message.content or "").strip())
    except LLMProviderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
