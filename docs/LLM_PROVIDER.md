# LLM provider switch (OpenAI ↔ Ollama)

Every LLM call in the hub goes through `llm_provider/`. One env var picks the backend:

```env
LLM_PROVIDER=openai   # default: hosted OpenAI, each module's *_MODEL settings, OPENAI_API_KEY
LLM_PROVIDER=ollama   # local Ollama server; OPENAI_API_KEY not needed
```

No module chooses a provider itself. Restart the app after changing `.env`.

## What is covered

| Module | Calls | Entry point used |
|---|---|---|
| jobEngine: job matcher (incl. visa / relocation / remote-scope / India eligibility), worldwide-remote reviewer, rejection analyzer | LangChain `with_structured_output` | `get_chat_model` |
| resumeIntel gap analysis | LangChain `with_structured_output` | `get_chat_model` |
| profile_assistant, code_assistant, stock_finder | OpenAI SDK `chat.completions.parse/create` (structured output, tool calls), `embeddings.create` | `get_openai_client` |
| enterpriseRAG | SDK chat (`json_object` mode) + embeddings, LangChain embeddings for FAISS | `get_openai_client`, `get_embeddings` |
| simpleRAG, dynamicRAG, AdvanceRAG, pageIndex | SDK chat + embeddings, LangChain embeddings | `get_openai_client`, `get_embeddings` |
| interviewSimulator | SDK chat | `get_openai_client` |
| jobApplyPro (Autonomous Application Agent, also used for HR drafts) | OpenAI Agents SDK (agents as tools) | `get_agents_model` |

On `openai`, the factory returns the same objects as before (`ChatOpenAI`, `OpenAIEmbeddings`, a plain `openai.OpenAI`, the model name for agents), so behavior is unchanged.

On `ollama`:

- LangChain consumers get `ChatOllama`. Structured output is constrained to the pydantic JSON schema, and `include_raw=True` still returns `raw` / `parsed` / `parsing_error`, so the existing parse → repair → reject loops work as before.
- SDK consumers get an OpenAI client pointed at Ollama's OpenAI-compatible `/v1` endpoint, behind an adapter that:
  - maps model names,
  - sends `reasoning_effort="none"` (qwen3 thinking off),
  - drops OpenAI-only fields (`store`, `metadata`).

  Errors keep their `openai.*` types, so each module's retry, fallback and circuit-breaker logic still applies. A connection error is retried as transient, and a missing model (404) fails fast. Each also logs a clear message.
- The Agents SDK gets `OpenAIChatCompletionsModel` over the same endpoint. Trace upload to OpenAI is disabled because it needs an OpenAI key.

### Model mapping

Modules keep their OpenAI model settings (`JOB_ENGINE_OPENAI_MODEL=gpt-4o-mini`, `PA_CRITIC_MODEL=gpt-4o`…). On Ollama each name resolves, in this order:

1. `OLLAMA_MODEL_MAP` entry (`gpt-4o=qwen3:14b,gpt-4.1=llama3.1:8b`, case-insensitive)
2. a name that is already an Ollama tag (contains `:`, e.g. `CA_CODER_MODEL=qwen3:14b`) is used as-is
3. otherwise `OLLAMA_MODEL` (default `qwen3:8b`)

Embedding models resolve the same way, falling back to `OLLAMA_EMBEDDING_MODEL` (default `nomic-embed-text`). Stock Finder reports and the job engine's rejection analyses record the model that actually ran. The job engine's analysis-cache version includes it too, so an OpenAI analysis is never reused as an Ollama one.

## Settings

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `ollama`. Anything else stops startup with a clear error. |
| `EMBEDDING_PROVIDER` | `LLM_PROVIDER` | e.g. `openai` to keep existing embeddings while chat runs locally (needs `OPENAI_API_KEY`). |
| `OLLAMA_HOST` | `http://localhost:11434` | |
| `OLLAMA_MODEL` | `qwen3:8b` | |
| `OLLAMA_MODEL_MAP` | empty | `openai-name=ollama-model,...` |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | qwen3:8b cannot embed (the server returns 501). |
| `OLLAMA_THINK` | `false` | qwen3 thinking. In one measured classification: 177 s with it vs 11 s without. |
| `OLLAMA_NUM_CTX` | `16384` | Context window for `ChatOllama` (job engine, resumeIntel). |
| `OLLAMA_TIMEOUT` | `300` | Seconds per call on Ollama. Replaces the 30–60 s OpenAI timeouts. |

**Context length.** Ollama's OpenAI-compatible endpoint ignores a per-request context size, so modules on the SDK path use the server's setting (4096 by default). Set the server's value to match `OLLAMA_NUM_CTX`. That keeps long prompts from being truncated, and avoids model reloads when both paths are used.

## Windows setup

The Ollama app, the Python packages and the model weights are three separate installs. `pip` installs none of the others, and the app never downloads a model by itself.

```powershell
# 1. Ollama app (official installer via winget; or download OllamaSetup.exe from https://ollama.com/download/windows)
winget install --id Ollama.Ollama -e
ollama --version

# 2. Match the server context to OLLAMA_NUM_CTX, then restart Ollama (quit it from the tray icon and start it again)
setx OLLAMA_CONTEXT_LENGTH 16384

# 3. Models: chat, plus embeddings for the RAG pages, Code Assistant index and Stock Finder news search
ollama pull qwen3:8b
ollama pull nomic-embed-text
ollama list

# 4. Is the service up?
Invoke-RestMethod http://localhost:11434/api/version
ollama run qwen3:8b "Say OK" --think=false

# 5. Python packages, inside the project's virtual environment
cd D:\Project\AI\Viru_AI_Hub
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt      # adds ollama + langchain-ollama

# 6. Check the configuration (no downloads), then one real call through both app paths
$env:LLM_PROVIDER = "ollama"
python -m llm_provider check
python -m llm_provider smoke
```

Then set `LLM_PROVIDER=ollama` in `.env` and start the app (`python app.py`). The startup log names the active provider and warns if Ollama or a model is missing. The job engine's health endpoint reports the same under `llm_provider`.

To switch back, set `LLM_PROVIDER=openai` (and keep `OPENAI_API_KEY`).

## Tests

```powershell
python -m pytest tests code_assistant/tests profile_assistant/tests stock_finder/tests -q   # mocked, no network

$env:RUN_OLLAMA_TESTS = "1"                         # live: real inference on the local model
python -m pytest tests/job_engine/test_ollama_live.py -v -s
```

The repo-level `conftest.py` pins every suite to `LLM_PROVIDER=openai`, so a local `.env` with `ollama` does not change unrelated tests.

## Known differences on Ollama

- **Answer quality.** qwen3:8b is much smaller than gpt-4o / gpt-4o-mini. In a direct probe it labeled "We sponsor H1B visas" as *not offered*. On the real job-matcher prompt it got the visa and remote labels right. The job engine's grounding rules still apply unchanged: decisive visa and remote labels need a verbatim quote from the posting. Spot-check results before relying on them. Use a larger model via `OLLAMA_MODEL` / `OLLAMA_MODEL_MAP` if your hardware allows.
- **Independent critics.** Profile Assistant (answer vs critic), Code Assistant (coder vs reviewer) and Stock Finder (analyst vs critic) require two different models. That check still runs on the configured names. If both map to one local model, startup logs a warning and the critic becomes a second pass of the same model. Map one role to another pulled model to restore independence, e.g. `OLLAMA_MODEL_MAP=gpt-4o=llama3.1:8b`.
- **Embeddings change vector spaces.** RAG indexes are built in memory per process, so they rebuild automatically. The Code Assistant's on-disk index records its embedding model and is rebuilt when that model changes.
- **Speed.** CPU-only inference took 10–60 s per call here. Lower `JOB_LLM_CONCURRENCY` (e.g. 2) if calls queue up and time out. The job engine's `JOB_LLM_PRICE_*` cost estimates do not apply to local models.
- **Not applicable on Ollama:** OpenAI stored completions (`store=True` / `metadata` in enterpriseRAG, dropped) and Agents SDK trace upload (disabled). OpenTelemetry instrumentation in the RAG demos still wraps the SDK client.
