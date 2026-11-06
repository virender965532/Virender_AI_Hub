"""Unit tests for the global LLM provider switch (no network: HTTP is mocked)."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel

import llm_provider
from llm_provider import (
    LLMProviderConfigError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
    check_ollama,
    get_agents_model,
    get_chat_model,
    get_embeddings,
    get_openai_client,
    get_provider_settings,
    llm_configured,
    provider_health,
    reset_caches,
    resolve_embedding_model,
    resolve_model,
)
from llm_provider.factory import ProviderClient
from llm_provider.settings import parse_model_map


def use(monkeypatch, **env: str) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    reset_caches()


class Verdict(BaseModel):
    visa_status: str
    remote_status: str


# --- configuration -----------------------------------------------------------------------


def test_default_provider_is_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    reset_caches()
    s = get_provider_settings()
    assert s.provider == "openai" and s.embedding_provider == "openai"
    assert s.ollama_model == "qwen3:8b" and s.ollama_host == "http://localhost:11434"
    assert s.ollama_think is False


def test_invalid_provider_is_rejected_with_clear_message(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="anthropic")
    with pytest.raises(LLMProviderConfigError, match="LLM_PROVIDER='anthropic' is not supported; use one of: openai, ollama"):
        get_provider_settings()


def test_provider_value_is_case_insensitive(monkeypatch):
    use(monkeypatch, LLM_PROVIDER=" Ollama ")
    assert get_provider_settings().provider == "ollama"


def test_invalid_host_and_numbers_are_rejected(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="ollama", OLLAMA_HOST="localhost:11434")
    with pytest.raises(LLMProviderConfigError, match="OLLAMA_HOST"):
        get_provider_settings()
    use(monkeypatch, OLLAMA_HOST="http://localhost:11434", OLLAMA_NUM_CTX="big")
    with pytest.raises(LLMProviderConfigError, match="OLLAMA_NUM_CTX"):
        get_provider_settings()


def test_model_map_parsing():
    assert parse_model_map(" gpt-4o = qwen3:14b , GPT-4.1=llama3.1:8b ,") == {
        "gpt-4o": "qwen3:14b", "gpt-4.1": "llama3.1:8b",
    }
    with pytest.raises(LLMProviderConfigError, match="OLLAMA_MODEL_MAP entry 'gpt-4o'"):
        parse_model_map("gpt-4o")


def test_embedding_provider_defaults_to_llm_provider(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="ollama")
    assert get_provider_settings().embedding_provider == "ollama"
    use(monkeypatch, EMBEDDING_PROVIDER="openai")
    assert get_provider_settings().embedding_provider == "openai"


# --- model naming ------------------------------------------------------------------------


def test_openai_names_are_untouched_on_openai():
    assert resolve_model("gpt-4o") == "gpt-4o"
    assert resolve_embedding_model("text-embedding-3-small") == "text-embedding-3-small"


def test_ollama_mapping_rules(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="ollama", OLLAMA_MODEL_MAP="gpt-4o=qwen3:14b")
    assert resolve_model("gpt-4o") == "qwen3:14b"  # explicit map
    assert resolve_model("GPT-4o") == "qwen3:14b"  # case-insensitive
    assert resolve_model("gpt-4o-mini") == "qwen3:8b"  # default model
    assert resolve_model("llama3.1:8b") == "llama3.1:8b"  # already an Ollama tag
    assert resolve_embedding_model("text-embedding-3-small") == "nomic-embed-text"


def test_credentials_needed_only_for_openai(monkeypatch):
    assert llm_configured("") is False
    assert llm_configured("sk-x") is True
    use(monkeypatch, LLM_PROVIDER="ollama")
    assert llm_configured("") is True
    assert llm_provider.embeddings_configured("") is True


def test_shared_critic_model_warns_only_on_ollama(monkeypatch, caplog):
    roles = {"PA_ANSWER_MODEL": "gpt-4o-mini", "PA_CRITIC_MODEL": "gpt-4o"}
    with caplog.at_level(logging.WARNING, logger="llm_provider.factory"):
        llm_provider.warn_if_shared_model("Profile Assistant", roles)
        assert not caplog.records
        use(monkeypatch, LLM_PROVIDER="ollama")
        llm_provider.warn_if_shared_model("Profile Assistant", roles)
    assert "not an independent model" in caplog.text
    caplog.clear()
    use(monkeypatch, OLLAMA_MODEL_MAP="gpt-4o=llama3.1:8b")
    with caplog.at_level(logging.WARNING, logger="llm_provider.factory"):
        llm_provider.warn_if_shared_model("Profile Assistant", roles)
    assert not caplog.records


# --- LangChain models --------------------------------------------------------------------


def test_chat_model_is_chatopenai_on_openai():
    from langchain_openai import ChatOpenAI

    llm = get_chat_model("gpt-4o-mini", openai_api_key="sk-test", timeout=12, temperature=0, seed=7)
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini" and llm.temperature == 0 and llm.seed == 7
    assert get_chat_model("gpt-4o-mini", openai_api_key="sk-test", timeout=12, temperature=0, seed=7) is llm


def test_chat_model_is_chatollama_on_ollama(monkeypatch):
    from langchain_ollama import ChatOllama

    use(monkeypatch, LLM_PROVIDER="ollama", OLLAMA_NUM_CTX="8192", OLLAMA_HOST="http://127.0.0.1:11434/")
    llm = get_chat_model("gpt-4o-mini", temperature=0, seed=7)  # no OpenAI key needed
    assert isinstance(llm, ChatOllama)
    assert llm.model == "qwen3:8b" and llm.base_url == "http://127.0.0.1:11434"
    assert llm.num_ctx == 8192 and llm.reasoning is False and llm.temperature == 0 and llm.seed == 7
    assert get_chat_model("gpt-4o-mini", temperature=0, seed=7) is llm  # reused
    assert get_chat_model("gpt-4o-mini", temperature=0.2) is not llm


def test_think_flag_reaches_chatollama(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="ollama", OLLAMA_THINK="true")
    assert get_chat_model("gpt-4o").reasoning is True


def test_embeddings_follow_embedding_provider(monkeypatch):
    from langchain_ollama import OllamaEmbeddings
    from langchain_openai import OpenAIEmbeddings

    assert isinstance(get_embeddings("text-embedding-3-small", openai_api_key="sk-test"), OpenAIEmbeddings)
    use(monkeypatch, LLM_PROVIDER="ollama")
    emb = get_embeddings("text-embedding-3-small")
    assert isinstance(emb, OllamaEmbeddings) and emb.model == "nomic-embed-text"


def _ollama_chat_response(content: str) -> httpx.Response:
    body = {
        "model": "qwen3:8b", "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content}, "done": True, "done_reason": "stop",
        "prompt_eval_count": 40, "eval_count": 12,
    }
    return httpx.Response(200, content=(json.dumps(body) + "\n").encode())


def mock_chatollama(llm, handler) -> None:
    """Point a ChatOllama at an in-memory transport (same request/response path as a real server)."""
    import ollama

    llm._client = ollama.Client(host=llm.base_url, transport=httpx.MockTransport(handler))


def test_chatollama_structured_output_contract(monkeypatch):
    """``with_structured_output(include_raw=True)`` returns raw/parsed/parsing_error, as consumers expect."""
    use(monkeypatch, LLM_PROVIDER="ollama")
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _ollama_chat_response('{"visa_status": "POSITIVE_EVIDENCE", "remote_status": "GLOBAL_REMOTE_EVIDENCE"}')

    llm = get_chat_model("gpt-4o-mini", temperature=0)
    mock_chatollama(llm, handler)
    result = llm.with_structured_output(Verdict, include_raw=True).invoke([{"role": "user", "content": "classify"}])

    assert result["parsing_error"] is None
    assert result["parsed"] == Verdict(visa_status="POSITIVE_EVIDENCE", remote_status="GLOBAL_REMOTE_EVIDENCE")
    assert result["raw"].usage_metadata["total_tokens"] == 52
    request = seen[0]
    assert request["model"] == "qwen3:8b" and request["think"] is False
    assert request["format"]["properties"].keys() == {"visa_status", "remote_status"}  # JSON-schema constrained
    assert request["options"]["num_ctx"] == 16384 and request["options"]["temperature"] == 0


def test_chatollama_invalid_json_is_a_parsing_error_not_a_crash(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="ollama")
    llm = get_chat_model("gpt-4o-mini")
    mock_chatollama(llm, lambda request: _ollama_chat_response("not json"))
    result = llm.with_structured_output(Verdict, include_raw=True).invoke("classify")
    assert result["parsed"] is None and result["parsing_error"] is not None


# --- OpenAI-SDK client -------------------------------------------------------------------


def test_openai_client_is_plain_sdk_client_on_openai():
    client = get_openai_client(api_key="sk-test", timeout=30, max_retries=0)
    assert type(client) is openai.OpenAI
    assert str(client.base_url).startswith("https://api.openai.com")


def test_openai_client_points_at_ollama_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    use(monkeypatch, LLM_PROVIDER="ollama", OLLAMA_TIMEOUT="123")
    client = get_openai_client(api_key="", timeout=30, max_retries=0)
    assert isinstance(client, ProviderClient)
    assert str(client.base_url) == "http://localhost:11434/v1/"
    assert client.timeout == 123 and client.max_retries == 0
    async_client = llm_provider.get_async_openai_client()
    assert isinstance(async_client._chat_client, openai.AsyncOpenAI)


def _sdk_over_mock(handler) -> openai.OpenAI:
    return openai.OpenAI(
        base_url="http://localhost:11434/v1", api_key="ollama", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _completion(content: str, tool_calls=None) -> dict:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "c1", "object": "chat.completion", "created": 0, "model": "qwen3:8b",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if tool_calls else "stop", "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_sdk_parse_request_is_adapted_for_ollama(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="ollama", OLLAMA_MODEL_MAP="gpt-4o=qwen3:14b")
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_completion('{"visa_status": "NOT_MENTIONED", "remote_status": "UNCLEAR"}'))

    sdk = _sdk_over_mock(handler)
    client = ProviderClient(sdk, lambda: sdk, get_provider_settings())
    completion = client.chat.completions.parse(
        model="gpt-4o", messages=[{"role": "user", "content": "x"}], response_format=Verdict,
        temperature=0, store=True, metadata={"stage": "s"}, timeout=30,
    )
    assert completion.choices[0].message.parsed == Verdict(visa_status="NOT_MENTIONED", remote_status="UNCLEAR")
    body = seen[0]
    assert body["model"] == "qwen3:14b"
    assert body["reasoning_effort"] == "none"  # qwen3 thinking off
    assert "store" not in body and "metadata" not in body
    assert body["response_format"]["type"] == "json_schema"


def test_sdk_tool_calls_and_explicit_omit(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="ollama")
    seen: list[dict] = []
    tool_call = [{"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_completion("", tool_call))

    sdk = _sdk_over_mock(handler)
    client = ProviderClient(sdk, lambda: sdk, get_provider_settings())
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {}}}}]
    completion = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "x"}], tools=tools, tool_choice="auto",
        reasoning_effort=openai.omit,  # what the Agents SDK passes when unset
    )
    assert completion.choices[0].message.tool_calls[0].function.name == "read_file"
    assert seen[0]["tools"] == tools and seen[0]["reasoning_effort"] == "none"


def test_sdk_thinking_enabled_leaves_reasoning_unset(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="ollama", OLLAMA_THINK="true")
    seen: list[dict] = []
    sdk = _sdk_over_mock(lambda r: (seen.append(json.loads(r.content)), httpx.Response(200, json=_completion("ok")))[1])
    ProviderClient(sdk, lambda: sdk, get_provider_settings()).chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "x"}],
    )
    assert "reasoning_effort" not in seen[0]


def test_sdk_embeddings_map_model(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="ollama")
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"object": "list", "model": "nomic-embed-text",
                                         "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                                         "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    sdk = _sdk_over_mock(handler)
    response = ProviderClient(sdk, lambda: sdk, get_provider_settings()).embeddings.create(
        model="text-embedding-3-small", input=["hi"],
    )
    assert response.data[0].embedding == [0.1, 0.2] and seen[0]["model"] == "nomic-embed-text"


def test_split_providers_route_embeddings_to_openai(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="ollama", EMBEDDING_PROVIDER="openai")
    client = get_openai_client(api_key="sk-test")
    assert str(client.base_url) == "http://localhost:11434/v1/"
    embed_inner = client.embeddings._inner
    assert str(embed_inner._client.base_url).startswith("https://api.openai.com")


def test_missing_model_and_unreachable_service_keep_openai_error_types(monkeypatch, caplog):
    use(monkeypatch, LLM_PROVIDER="ollama")

    def not_found(request):
        return httpx.Response(404, json={"error": {"message": "model 'qwen3:8b' not found"}})

    sdk = _sdk_over_mock(not_found)
    client = ProviderClient(sdk, lambda: sdk, get_provider_settings())
    with caplog.at_level(logging.ERROR, logger="llm_provider.factory"), pytest.raises(openai.NotFoundError):
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    assert "ollama pull qwen3:8b" in caplog.text

    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    sdk = _sdk_over_mock(refuse)
    client = ProviderClient(sdk, lambda: sdk, get_provider_settings())
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="llm_provider.factory"), pytest.raises(openai.APIConnectionError):
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "x"}])
    assert "Ollama is not reachable at http://localhost:11434" in caplog.text


def test_existing_module_retry_logic_treats_ollama_outage_as_transient(monkeypatch):
    """profile_assistant retries APIConnectionError then raises its own LLMUnavailableError."""
    from profile_assistant.config import get_settings
    from profile_assistant.llm_client import LLMClient, LLMUnavailableError

    use(monkeypatch, LLM_PROVIDER="ollama")
    calls = []

    def refuse(request):
        calls.append(1)
        raise httpx.ConnectError("refused", request=request)

    sdk = _sdk_over_mock(refuse)
    llm = LLMClient(get_settings(), client=ProviderClient(sdk, lambda: sdk, get_provider_settings()), sleep=lambda s: None)
    with pytest.raises(LLMUnavailableError):
        llm.parse(model="gpt-4o-mini", messages=[{"role": "user", "content": "x"}], schema=Verdict, trace_id="t")
    assert len(calls) == 1 + get_settings().max_retries


# --- Agents SDK --------------------------------------------------------------------------


def test_agents_model(monkeypatch):
    assert get_agents_model("gpt-4o-mini") == "gpt-4o-mini"
    from agents import OpenAIChatCompletionsModel

    use(monkeypatch, LLM_PROVIDER="ollama")
    model = get_agents_model("gpt-4o-mini")
    assert isinstance(model, OpenAIChatCompletionsModel) and model.model == "qwen3:8b"
    assert str(model._client.base_url) == "http://localhost:11434/v1/"


# --- Ollama health checks ----------------------------------------------------------------


class _FakeOllama:
    def __init__(self, names):
        self.names = names

    def list(self):
        return SimpleNamespace(models=[SimpleNamespace(model=n) for n in self.names])


def test_check_ollama_ok_and_latest_tag(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="ollama")
    status = check_ollama(client=_FakeOllama(["qwen3:8b", "nomic-embed-text:latest"]))
    assert status == {"ok": True, "host": "http://localhost:11434", "models": ["qwen3:8b", "nomic-embed-text"]}


def test_check_ollama_reports_missing_models_with_pull_command(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="ollama", OLLAMA_MODEL_MAP="gpt-4o=qwen3:14b")
    with pytest.raises(OllamaModelNotFoundError) as info:
        check_ollama(client=_FakeOllama(["qwen3:8b"]))
    message = str(info.value)
    assert "ollama pull qwen3:14b" in message and "ollama pull nomic-embed-text" in message
    assert check_ollama(client=_FakeOllama(["qwen3:8b", "qwen3:14b"]), include_embeddings=False)["ok"]


def test_check_ollama_unreachable_service(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="ollama", OLLAMA_HOST="http://127.0.0.1:9")  # nothing listens on port 9
    with pytest.raises(OllamaUnavailableError, match="Ollama is not reachable at http://127.0.0.1:9"):
        check_ollama()
    status = provider_health()
    assert status["ok"] is False and "not reachable" in status["error"]


def test_provider_health_on_openai_needs_no_ollama():
    assert provider_health() == {"ok": True, "provider": "openai"}


def test_invalid_provider_fails_consumers_loudly(monkeypatch):
    use(monkeypatch, LLM_PROVIDER="gemini")
    with pytest.raises(LLMProviderConfigError):
        get_openai_client(api_key="sk-test")
    with pytest.raises(LLMProviderConfigError):
        get_chat_model("gpt-4o")
