# Project Analysis — AI Job Discovery & Intelligent Job Matching Engine

## 1. Existing architecture (as found)

The repository is a single Flask app (`app.py`) with **one blueprint** (`routes/main_routes.py`,
`main_bp`) that fronts a set of independent, top-level feature packages. There is no `app/`
package wrapper — every feature lives in its own PascalCase/camelCase top-level directory:

| Package | Purpose | Orchestration style |
|---|---|---|
| `jobSearch/` | Naukri.com scraping via **Playwright (async)** + **LangGraph** (`login_node → fetch_jobs_node → display_jobs_node`) | LangGraph `StateGraph`, `TypedDict` state, `ainvoke` |
| `jobApplyPro/` | Resume/JD analysis + application email drafting | **openai-agents** SDK (`Agent`, `Runner`, `trace`) — a *different* LLM stack |
| `enterpriseRAG/` | Multi-agent RAG (18 agents) | LangGraph `StateGraph`, `TypedDict` state, sync `invoke`, custom `LLMService`/`ObservabilityService`/`GuardrailPipeline` |
| `simpleRAG/`, `dynamicRAG/`, `pageIndex/` | Simpler RAG variants | plain functions, FAISS |
| `interviewSimulator/` | Interview practice | own graph + TTS |
| `services/naukri_service.py` | **Legacy** Selenium/undetected-chromedriver Naukri scraper | superseded by `jobSearch/` (its route is commented out in `main_routes.py`); still imported for `run_job_detail`, which is also dead-routed |

Common conventions observed everywhere:

- **State**: `TypedDict(total=False)` + an `initial_state()`/`initial_workflow_state()` factory.
- **Config**: a module-level `Settings`/constants block reading `os.getenv(...)` with small
  `_env_int/_env_float/_env_bool` helpers, often `@lru_cache` (`enterpriseRAG/config/settings.py`)
  or a defaults function (`jobSearch/nodes/fetch_jobs_node.py:get_naukri_search_defaults`).
- **Logging**: root logger configured once in `app.py` (`setup_logging()`); every module does
  `logger = logging.getLogger(__name__)` and uses `logger.info/warning/exception`. No JSON
  structured logging exists yet, and no run/trace-id correlation exists outside `enterpriseRAG`.
- **Observability**: only `enterpriseRAG/services/observability.py` has a real service
  (`ObservabilityService`: per-workflow trace id, agent traces, token usage, guardrail events).
  Nothing wired to LangSmith/OpenTelemetry despite `langsmith` being an installed transitive
  dependency of `langchain` — no `LANGCHAIN_TRACING_V2` usage anywhere in the repo.
- **Guardrails**: only `enterpriseRAG/guardrails/pipeline.py` — regex-based prompt-injection /
  PII / hallucination-risk checks, recorded through `ObservabilityService.record_guardrail`.
- **Browser automation**: `jobSearch/utils/playwright_manager.py` — async Playwright, single
  `Browser`/`BrowserContext`/`Page` per run, heavy-asset request interception, headless controlled
  by `NAUKRI_HEADLESS` env var. This is the *only* Playwright usage in the repo and is a solid,
  directly reusable pattern (dataclass session, explicit `close_session`, timeouts on the page).
- **LLM access**: two parallel stacks exist — `openai-agents` (jobApplyPro) and a hand-rolled
  `LLMService` over the raw `openai` SDK (enterpriseRAG, `chat`/`chat_json`/`embed`, JSON mode,
  per-call token/latency logging). **`langchain-openai` is installed but nothing in the repo uses
  `ChatOpenAI.with_structured_output`** — there is no existing precedent to conflict with.
- **Persistence**: **there is no database layer anywhere in this project.** Persistence is done
  with flat JSON files (`jobApplyPro/history.py` → `history/job_apply/applications.json`) or an
  in-memory dict with TTL pruning (`jobSearch/utils/scrape_progress.py`, used for scrape-progress
  polling). No SQLAlchemy, no migrations, no ORM models exist.
- **Matching/scoring**: `jobSearch/utils/job_match_scoring.py` — a hand-tuned, hard-coded
  `SKILL_WEIGHTS`/alias/regex scorer against *this specific candidate's* stack, with an excluded
  -tech penalty (Java/C#/.NET/Golang). It is Naukri-specific and not resume-driven (weights are
  literals in code, not derived from `Profile/Summary.txt`), but its *shape* (normalize → alias →
  weight → penalty → category) is a good reference for the new rule-based pre-filter stage.
- **Flask surface**: one big `main_bp` blueprint registers every route for every module directly
  in `routes/main_routes.py`; templates live in `templates/`, JS in `static/js/<feature>.js`, one
  `<details>` dropdown nav in `templates/base.html` lists every tool.
- **Docker**: single `Dockerfile` (Python 3.11-slim + Google Chrome + `playwright install
  --with-deps chromium`), `docker-compose.yml` bind-mounts `Profile/`, `output/`, `scraper_debug/`,
  and the RAG upload/data dirs. `NAUKRI_HEADLESS=1` is set for the container.
- **Secrets**: `.env` (gitignored) + `.env.example` as a documented template; `python-dotenv`
  loaded in `app.py` and redundantly in a couple of leaf modules (`playwright_manager.py`,
  `enterpriseRAG/config/settings.py`) via `load_dotenv()`.
- **Tests**: **none exist.** No `tests/` directory, no `pytest` in `requirements.txt` (checked:
  not installed in the project's `.venv`), `.pytest_cache/` is gitignored defensively but unused.
- **Dependencies already available** that this module can reuse without adding anything new:
  `langgraph`, `langchain-core`, `langchain-openai`, `langchain-community`, `playwright`,
  `pydantic` (2.13.3, transitive via langchain), `PyYAML` (transitive), `openai`, `faiss-cpu`,
  `python-dotenv`.

## 2. Candidate profile source of truth

`Profile/Summary.txt` is a clean, section-headed plain-text resume (profile summary, technical
skills grouped by category, work experience with bullet achievements, key projects, education).
`Profile/VirenderFullstackAIResume.pdf` is the same content as a PDF (used as a secondary/raw-text
cross-check only, via `pypdf`, mirroring `jobApplyPro.recruitment_pipeline.extract_resume_data`).
The engine parses `Summary.txt` structurally (it is authored consistently enough to parse by
section headers) into a `CandidateProfile` Pydantic model; nothing is hard-coded/invented — if a
section is missing or a field can't be parsed, it is left empty rather than guessed.

## 3. Integration points

- **New Flask blueprint** `job_engine_bp`, registered in `app.py` alongside `main_bp` (additive,
  no existing route touched). Namespaced under `/api/jobs/...` and `/job-engine/health` to avoid
  colliding with the existing (already-registered) `/api/jobs/naukri*` routes owned by `jobSearch`.
- **Playwright**: new `jobEngine/browser/manager.py` reuses the exact
  `PlaywrightSession`/`launch_session`/`close_session` shape from `jobSearch/utils/
  playwright_manager.py`, generalized with a concurrency semaphore, per-source timeouts, and
  screenshot-on-failure — it does not touch the original file (Naukri flow keeps its own manager).
- **LangGraph**: new, independent `StateGraph` (`jobEngine/graph/builder.py`). Source nodes are
  created dynamically from the adapter registry + `config/job_sources.yaml`, unlike the two
  existing graphs which hard-code their (2-3) nodes — this is a deliberate improvement matching
  the spec's extensibility requirement, not a rewrite of the existing graphs.
- **LLM**: uses `langchain-openai`'s `ChatOpenAI.with_structured_output(JobMatchAnalysis)` for the
  grounded matching step — introduces no new dependency, and keeps `jobApplyPro`'s `openai-agents`
  stack and `enterpriseRAG`'s raw-`openai` `LLMService` untouched.
- **Persistence**: follows the existing flat-file convention (`jobApplyPro/history.py`) rather
  than introducing SQLAlchemy/Postgres — a new dependency-heavy DB layer would be duplicate
  infrastructure the rest of the app doesn't have or need yet. `jobEngine/persistence/
  repository.py` is written behind a small repository interface so a real DB can be swapped in
  later without touching callers.
- **Config**: new `config/job_sources.yaml` at repo root (new top-level `config/` dir — does not
  exist yet) plus `jobEngine/config/settings.py` for everything else (weights, timeouts, guardrail
  limits), following the `enterpriseRAG` `Settings` dataclass + env-var pattern.
- **Docker**: no Dockerfile changes needed — Chromium + `playwright install --with-deps chromium`
  is already installed; `docker-compose.yml` gets one added bind mount for `config/` so
  `job_sources.yaml` is editable without a rebuild.

## 4. Potential conflicts

- `/api/jobs/*` is already owned by `jobSearch`'s Naukri routes (`/api/jobs/naukri`,
  `/api/jobs/naukri/progress/<id>`). The new module's routes are added under the same `/api/jobs/`
  prefix per the spec (`/api/jobs/search`, `/api/jobs/sources`, ...) — verified there is no exact
  path collision, but this is a shared namespace going forward; documented in `CONFIGURATION.md`.
- Naming convention collision risk: the spec's example structure uses `app/modules/job_engine/`.
  This repo has no `app/` package, so the module is created as a top-level `jobEngine/` package to
  match the existing `jobSearch`/`jobApplyPro`/`enterpriseRAG` convention instead.
- `services/naukri_service.py` (legacy Selenium) is untouched; it is dead code behind commented
  routes and out of scope for this change.

## 5. Recommended integration strategy

1. Add `jobEngine/` as a new, self-contained top-level package (mirrors `enterpriseRAG/`'s
   internal layout: `config/`, `services/`, `graph/` instead of `workflows/`, plus
   `sources/`, `matching/`, `guardrails/`, `browser/`, `persistence/`, `observability/`, `api/`).
2. Add `config/job_sources.yaml` at repo root for source configuration.
3. Register one new blueprint in `app.py`; do not modify `routes/main_routes.py`.
4. Reuse, don't duplicate: Playwright session lifecycle pattern, `Settings` dataclass pattern,
   guardrail/observability *shape* from `enterpriseRAG`, JSON-file persistence pattern from
   `jobApplyPro`.
5. Add `tests/job_engine/` (new — first tests in the repo) with fixture HTML, no live-site
   dependency; add `pytest`/`pytest-asyncio` to `requirements.txt` (dev-only, nothing else new).
6. Document everything under `docs/job_engine/`.
