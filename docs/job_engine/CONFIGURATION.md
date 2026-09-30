# Configuration

## `config/job_sources.yaml`

See the fully-commented file itself — every field is documented inline. In
short: one YAML mapping under `job_sources:`, one entry per source, validated
by `jobEngine/config/sources.py:JobSourceConfig` (Pydantic — bad URLs, bad
scheme, duplicate ids, or an enabled source with no `search_urls` all fail
fast at load time, not mid-run).

## Environment variables (`jobEngine/config/settings.py`)

All have defaults; nothing is required except `OPENAI_API_KEY` for LLM
matching (the engine still runs and returns rule-based-only scores without
it — see `degrade_on_llm_failure`).

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Shared with the rest of the app; required for LLM matching |
| `JOB_ENGINE_OPENAI_MODEL` | `gpt-4o-mini` | Chat model for matching analysis |
| `OPENAI_TIMEOUT` | `60` | Seconds, applies to chat calls |
| `JOB_LLM_MAX_RETRIES` | `2` | Extra attempts after a malformed/invalid LLM response |
| `JOB_ENGINE_LLM_ENABLED` | `true` | Set `false` to force rule-only scoring |
| `JOB_MATCH_PROMPT_VERSION` | `v1` | Included in the system prompt; bump when you edit the prompt files |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | unset | Standard LangSmith tracing, opt-in |
| `JOB_BROWSER_HEADLESS` | `true` | Chromium headless mode |
| `JOB_BROWSER_TIMEOUT` | `60000` ms | Default Playwright action timeout |
| `JOB_BROWSER_NAV_TIMEOUT` | `45000` ms | Default Playwright navigation timeout |
| `JOB_BROWSER_MAX_RETRIES` | `3` | Whole-adapter-call retries on retryable `SourceError`s |
| `JOB_BROWSER_MAX_CONCURRENCY` | `3` | Max simultaneous open `BrowserContext`s |
| `JOB_BROWSER_RETRY_BACKOFF_SECONDS` | `2.0` | Linear backoff multiplier between retries |
| `JOB_CIRCUIT_BREAKER_THRESHOLD` | `3` | Consecutive failures before a source is skipped |
| `JOB_CIRCUIT_BREAKER_COOLDOWN_SECONDS` | `900` | How long a tripped source stays skipped |
| `JOB_MAX_SOURCES` | `10` | Hard cap on sources processed per run |
| `JOB_MAX_JOBS_PER_SOURCE` | `100` | Hard cap regardless of a source's own `max_jobs` |
| `JOB_MAX_PAGINATION_DEPTH` | `20` | Hard cap regardless of a source's own `max_pages` |
| `JOB_MAX_DESCRIPTION_CHARS` | `20000` | Per-job description truncation limit |
| `JOB_MAX_HTML_BYTES` | `5000000` | Raw HTML size guard (defense in depth) |
| `JOB_MATCH_WEIGHT_*` (6 vars) | 0.30/0.20/0.15/0.20/0.10/0.05 | Must sum to ~1.0 (validated at startup) |
| `JOB_MIN_MATCH_SCORE` | `70` | Engine-wide default; overridable per-request |
| `JOB_LLM_TOP_K` | `50` | Jobs sent to the (expensive) LLM stage, ranked by rule score |
| `JOB_RULE_PREFILTER_MIN_SCORE` | `15` | Defined but currently unused (no prefilter stage runs) — left as a settings no-op rather than a breaking removal |
| `JOB_CANDIDATE_EXPERIENCE_YEARS` | `7` | Fallback if the profile's summary text doesn't state years explicitly |
| `JOB_MAX_EXPERIENCE_GAP_YEARS` | `3` | How far a job's stated required-years can exceed the candidate's before hard rejection |
| `JOB_ENGINE_DRY_RUN` | `false` | Navigate/extract/score but skip persistence |
| `JOB_ENGINE_DEBUG` | `false` | Save HTML snapshots on failure (screenshots are always saved) |
| `JOB_ENGINE_PERSIST_RESULTS` | `true` | Write results to `output/job_engine/` |
| `JOB_ENGINE_DEGRADE_ON_LLM_FAILURE` | `true` | Mark result `degraded` instead of failing when LLM is unavailable |
| `LINKEDIN_STORAGE_STATE_PATH` | unset | Path to a pre-authenticated Playwright storage_state JSON |
| `JOB_LLM_PRICE_PER_1K_PROMPT_USD` / `_COMPLETION_USD` | gpt-4o-mini rates | For cost-estimate observability only |

## Request-level overrides (`POST /api/jobs/search` body)

`sources`, `keywords`, `location`, `remote`, `country`, `city`,
`employment_type`, `minimum_match_score`, `max_results` — see
`jobEngine/api/schemas.py:JobSearchRequest`. `minimum_match_score` overrides
`JOB_MIN_MATCH_SCORE` for that run only.

## Known limitation: request filters vs. live search

Source adapters navigate to the static `search_urls` configured in
`config/job_sources.yaml` — a request's `keywords`/`location` do **not**
reformulate LinkedIn's or a career page's own search query string. They are
instead applied as a post-discovery filter in `graph/nodes/validation.py`.
To search a different keyword set on a given source, add another
`search_urls` entry (or another source config entirely) with the query
baked into the URL, e.g. LinkedIn's
`?keywords=Generative%20AI%20Engineer`.
