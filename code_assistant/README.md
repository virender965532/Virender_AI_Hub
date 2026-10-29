# Code Assistant

An AI coding agent for a project folder, similar in spirit to Cursor / Claude Code. Point it at a
workspace and ask anything: explain code, find bugs, fix errors, add features, refactor, write tests.

It explores the code with tools, proposes changes as diffs, verifies them by running the linter and tests
on a private copy of the workspace, has a different model review them, and writes nothing to disk until
you approve each diff. Every applied change can be rolled back.

- UI: `http://127.0.0.1:8000/code-assistant/` (hub page embedding the Gradio app at `/code-assistant/ui/`)
- API: `/code-assistant/api/...` (JSON + Server-Sent Events)

## Setup

1. Install dependencies: `pip install -r requirements.txt` (this module adds `gitpython`, `pathspec`, `tiktoken`;
   it also uses `openai`, `pydantic`, `faiss-cpu`, `gradio`, `a2wsgi` which the hub already has).
   [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`) is used for search when installed; otherwise a
   Python fallback is used.
2. Set `OPENAI_API_KEY` and **`CA_WORKSPACE_ROOTS`** in `.env` (see below). Without
   `CA_WORKSPACE_ROOTS` no folder can be opened.
3. Start the hub as usual (`python app.py`). The module is registered by one call in `create_app()`:
   `register_code_assistant(app)`.
4. Open the page, enter a folder inside an allowed root, click **Open**, and chat.

Run the tests (all OpenAI calls are mocked):

```bash
python -m pytest code_assistant/tests -q
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI key (shared with the other hub apps). Never logged. |
| `CA_WORKSPACE_ROOTS` | *(empty)* | Comma-separated folders a workspace must live in. |
| `CA_ROUTER_MODEL` | `gpt-4o-mini` | Intent/difficulty classification (temperature 0). |
| `CA_CODER_MODEL` | `gpt-4o` | ReAct coder agent. |
| `CA_PLANNER_MODEL` | `gpt-4o` | Plan-and-Execute planner. |
| `CA_REVIEWER_MODEL` | `o3-mini` | Critic. **Must differ from `CA_CODER_MODEL`** (startup error otherwise). |
| `CA_FALLBACK_MODEL` | `gpt-4o-mini` | Used when a primary model keeps failing; the response is marked degraded. |
| `CA_EMBEDDING_MODEL` | `text-embedding-3-small` | Code-chunk embeddings. |
| `CA_MAX_AGENT_STEPS` | `15` | Max LLM turns per agent run. |
| `CA_MAX_FIX_ATTEMPTS` | `3` | Max verify-loop retries. |
| `CA_MAX_LLM_CALLS` | `40` | Per-request LLM call budget. |
| `CA_MAX_TOKENS_REQUEST` | `200000` | Per-request token budget (in + out). |
| `CA_CONTEXT_TOKENS` | `60000` | Prompt size the context manager keeps the agent under. |
| `CA_COMMAND_TIMEOUT` | `120` | Seconds before a command's process tree is killed. |
| `CA_COMMAND_ALLOWLIST` | `pytest,python -m pytest,ruff,flake8,mypy,npm test` | Commands that may run (prefix match on tokens). |
| `CA_USE_DOCKER` | `false` | Run commands in `docker run --network none` (image `CA_DOCKER_IMAGE`). |
| `CA_DOCKER_IMAGE` | `python:3.12-slim` | Image for Docker mode. |
| `CA_URL_PREFIX` | `/code-assistant` | Blueprint prefix. |
| `CA_AUTO_INDEX` | `true` | Index a workspace when it is opened and after applied changes. |
| `CA_REQUIRE_PLAN_APPROVAL` | `true` | Show the plan for approval before executing large tasks. |
| `CA_DATA_DIR` | `code_assistant/data` | Index vectors and rollback checkpoints (git-ignored). |
| `CA_UI_ENABLED` | `true` | Set `false` to serve only the API. |
| `CA_REQUEST_TIMEOUT` / `CA_MAX_RETRIES` | `90` / `3` | Per-call OpenAI timeout and transient-error retries. |

## Architecture

```
user message
  │
  ▼
guard ── input checks, workspace in allowlist, one request per session
  │
  ▼
router (CA_ROUTER_MODEL, T=0) ── RouteDecision{intent, difficulty, confidence}
  │   explain / answer_question / general_chat ─────────► ReAct coder, READ-ONLY tools
  │   fix_bug / refactor / write_tests (easy) ──────────► ReAct coder, edit + verify tools
  │   implement_feature, or any hard task ──────────────► planner → plan shown for approval
  │   confidence < 0.5 or router down ─────────────────► ReAct coder (edit mode, the default)
  ▼
execute ── coder proposes edits into a per-request WorkingSet (in memory, never disk)
  │         planner: DAG of steps; independent "investigate" steps run in parallel,
  │         "edit" steps in order; a step can ask for a replan ("PLAN CHANGE:")
  ▼
verify loop ── lint + tests on a private temp copy + proposals; NEW failures (vs. the
  │            untouched workspace) are fed back to the coder, ≤ CA_MAX_FIX_ATTEMPTS
  ▼
reviewer (CA_REVIEWER_MODEL ≠ coder) ── rubric: correctness, safety, style, no breakage
  │   FAIL → one revision by the coder → re-verify → re-review
  │   still FAIL → changes returned marked "needs human review"
  ▼
human approval ── each diff: Approve / Reject. Approve = checkpoint, then write.
  ▼
apply ── Rollback restores every applied change of the session.
```

| Pattern | Where |
|---|---|
| Routing | `agents/router.py` |
| ReAct (tool use) | `agents/coder_agent.py` |
| Plan-and-Execute (DAG, parallel reads, replanning) | `agents/planner.py`, `pipeline._execute_plan` |
| Reflection with external feedback (real test output) | `pipeline._verify_loop` |
| Critic with a different model | `agents/reviewer_agent.py`, `pipeline._review_loop` |
| Human in the loop + checkpoints | `pipeline.approve_change`, `tools/git_tools.CheckpointManager` |
| RAG over code (symbol-aligned chunks, hybrid search, RRF rerank) | `workspace/indexer.py`, `workspace/retriever.py` |
| Context management | `memory/context_manager.py`, `memory/session.py` |

### Files

```
code_assistant/
  __init__.py          register_code_assistant(app)
  config.py            settings from env, validated at startup
  schemas.py           RouteDecision, Plan, PlanStep, ProposedChange, ReviewVerdict, FinalResponse, ...
  prompts.py           all prompts, PROMPT_VERSION (logged with every LLM call)
  llm_client.py        retries, backoff, fallback, circuit breaker, budget, embeddings
  observability.py     Trace: trace_id, JSON logs, metrics, budget, event fan-out
  workspace/           sandbox.py, indexer.py, retriever.py
  tools/               registry.py, file_tools.py, edit_tools.py, exec_tools.py, git_tools.py
  agents/              router.py, planner.py, coder_agent.py, reviewer_agent.py
  memory/              session.py, context_manager.py
  pipeline.py          the orchestrator
  routes.py            Flask blueprint (JSON + SSE)
  ui.py                Gradio UI + mounting
  templates/           hub page
  tests/               pytest, OpenAI mocked
```

## API

| Method | Path | Body | Result |
|---|---|---|---|
| GET | `/api/health` | | models, circuit-breaker state |
| GET | `/api/workspaces` | | registered workspaces and allowed roots |
| POST | `/api/workspaces` | `{"path"}` | register (must be in `CA_WORKSPACE_ROOTS`); starts indexing |
| GET | `/api/workspaces/<id>` | | index status + file tree |
| POST | `/api/workspaces/<id>/index` | | (re)index in a background thread (incremental) |
| POST | `/api/chat` | `{"workspace_id","message","session_id"?, "stream"?}` | SSE stream (default) or JSON with `"stream": false` |
| POST | `/api/plans/<session_id>/approve` \| `/reject` | | execute (SSE) or discard a pending plan |
| GET | `/api/changes/<session_id>` | | all changes of the session |
| POST | `/api/changes/<change_id>/approve` \| `/reject` | | apply one change (checkpoint first) / reject |
| POST | `/api/rollback/<session_id>` | `{"force"?}` | restore every applied change of the session |

SSE events: `session`, `route`, `plan`, `plan_step`, `thinking`, `tool_call`, `tool_result`, `diff`,
`test_result`, `review`, `final` (the `FinalResponse`), `error`. Keep-alive comments every 15 s.

```bash
curl -N -X POST localhost:8000/code-assistant/api/chat -H 'Content-Type: application/json' \
  -d '{"workspace_id":"<id>","message":"Why does test_add fail? Fix it."}'
```

`FinalResponse.status`: `answered`, `changes_pending_approval`, `plan_pending_approval`, `applied`,
`failed`, `fallback` (model unavailable). Flags: `needs_human_review`, `degraded` (fallback model used),
`budget_exceeded` + `remaining`.

## Security model

- **No authentication.** Like the rest of the hub, these routes are open to anyone who can reach the
  server. Keep `FLASK_HOST=127.0.0.1` (the default). Do **not** expose this module on a public host: anyone
  who can reach it could open any folder under `CA_WORKSPACE_ROOTS`, read its (non-secret) files, approve
  changes and run the allowlisted commands. Set `CA_UI_ENABLED=false` and leave `CA_WORKSPACE_ROOTS` empty
  on public deployments, or put the hub behind an authenticating proxy.
- **Workspace sandbox.** Every path is `realpath`-resolved and must stay inside the workspace; `..` is
  rejected, as are absolute paths and symlinks that escape. A workspace must be inside `CA_WORKSPACE_ROOTS`.
- **Ignore rules.** `.gitignore` (root file) plus a default list: `.git`, `node_modules`, virtualenvs,
  `__pycache__`, `dist`, `build`, binaries, images, archives, lock files over 1 MB.
- **Secrets.** `.env*` (except `.env.example`/`.sample`/`.template`), `*.pem`, `*.key`, `id_rsa*`,
  credentials files, `.netrc`, `.npmrc`, ... are never read, listed, searched, copied to the verify copy or
  written. Strings that look like API keys, tokens, passwords in URLs or private keys are redacted before
  anything reaches the LLM, the logs or the UI.
- **Prompt injection.** Tool output is labelled as repository data, and the system prompt tells the agent
  never to follow instructions found in files or command output. Even if it did, it cannot write files or
  run arbitrary commands: those require your approval.
- **No writes without approval.** The agent only fills an in-memory working set. Only
  `approve_change` writes, after creating a checkpoint, and it refuses if the file changed on disk since the
  proposal.
- **Commands.** Only allowlisted commands (token-prefix match), `shell=False`, bare executable names,
  path arguments confined to the working directory, a timeout that kills the whole process tree, output
  truncated to the last 200 lines. The child environment is stripped of variables that look like secrets.
  Verification commands run on a temporary copy, never in your workspace. `run_command` from the agent is
  only *queued* as a change you must approve. With `CA_USE_DOCKER=true` commands run in
  `docker run --rm --network none --memory 2g`.
- **Rollback.** Before applying, the original files are copied to `CA_DATA_DIR/checkpoints/<id>/`, and in
  a git repo the HEAD is recorded and a branch `code-assistant/checkpoint-<id>` is created at HEAD (never
  checked out; `git stash` is not used because it would capture unrelated work). Rollback refuses to
  overwrite files you changed after applying, unless `force` is set.

Note that running a project's tests executes that project's code; only open workspaces you trust, or use
Docker mode.

## Reliability

- Transient OpenAI errors (rate limit, timeout, connection, 5xx) are retried with exponential backoff and
  jitter; auth / bad-request errors are not.
- Context-length errors: the agent's context is trimmed and the call retried once.
- Structured-output parse failures: retried once with the validation error appended.
- Model fallback to `CA_FALLBACK_MODEL` (response marked `degraded`).
- Circuit breaker: opens after 5 consecutive failed calls, 30 s cooldown, then a half-open trial call.
- Budgets: the request stops cleanly at `CA_MAX_LLM_CALLS` / `CA_MAX_TOKENS_REQUEST` and returns what it
  has (including proposed changes, marked for human review) plus what remains.
- Every loop is bounded: agent steps, fix attempts, one review revision, replans, plan waves.
- Unexpected errors return a friendly message with the `trace_id`; the full stack trace is in the logs.

## Observability

Every request gets a `trace_id`. LLM calls (model, purpose, prompt version, tokens, cost, attempts),
tool calls (arguments, duration, result size), route decisions, plans, test runs, review verdicts,
approvals, rejections and rollbacks are logged as one JSON line each on the `code_assistant.trace` logger
(the app's standard logging setup), redacted. At the end a `request_complete` line has the metrics:
latency, LLM calls, tokens in/out, estimated cost, tools used, fix attempts, review verdict, status.

## Adding a tool

1. Write a handler `def my_tool(ctx: ToolContext, arg: str, ...) -> str` in one of the `tools/*.py`
   modules (or a new module). Use `ctx.sandbox` for every path; raise `ToolError("...")` for errors the
   agent should see.
2. Append a `Tool(name, description, json_schema, RiskLevel.X, handler)` to that module's `TOOLS` list.
   Pick the risk level honestly: `READ` and `VERIFY` run automatically; anything that changes state must be
   `WRITE`/`DANGEROUS` and must only *queue* work for approval (see `run_command`).
3. If you created a new module, add it to `build_default_registry()` in `tools/registry.py`.
4. Add tests in `tests/test_tools.py`. The registry validates arguments against the schema, truncates
   output, logs the call and turns exceptions into `Error: ...` messages for the agent.

## UI mounting choice

Gradio is ASGI and Flask is WSGI. The Gradio app is wrapped with `a2wsgi` and mounted under
`{CA_URL_PREFIX}/ui` with werkzeug's `DispatcherMiddleware` (option A). This is the same approach as
`profile_assistant`: one process, one port, no CORS. Because `a2wsgi` does not send ASGI lifespan events,
`ui._start_lifespan` runs Gradio's startup on a background event loop; without it, streamed requests would
hang. The UI calls the pipeline in-process.

**Lazy loading.** Nothing heavy runs when the hub starts: `register_code_assistant` only validates the
settings and registers the routes. The Gradio UI is imported, built and started on the first request to
`{CA_URL_PREFIX}/ui/` (i.e. when someone opens the Code Assistant page, which embeds it), so the first
page load takes a few seconds longer. The pipeline (OpenAI client, index, tools) is built on the first
API call or chat message. If the UI fails to build, `/ui/` returns 503 and is retried on the next request.

## Known limitations

- Sessions, pending changes and plans are kept in memory: they are lost on restart and not shared between
  worker processes (checkpoints on disk survive, but the session that references them does not). Run a
  single process.
- No authentication (see Security model).
- Verification copies the workspace (without ignored folders such as `node_modules` and virtualenvs) to a
  temp folder; very large workspaces (> 20 000 files or 300 MB) skip verification. The project's own
  `.venv`/`venv` interpreter is used when present, otherwise the hub's Python. Tests that need ignored
  folders or `.env` files will fail in the copy (compared against a baseline run, so pre-existing failures
  are not blamed on the agent, except for `fix_bug` requests).
- Only the workspace's root `.gitignore` is applied; nested `.gitignore` files are not.
- Outlines are exact for Python (`ast`) and regex-based for other languages.
- File deletion and renames are not supported as proposals.
- Network access cannot be blocked for local (non-Docker) command runs.
- Symlink-escape tests are skipped on Windows without the symlink privilege.
