# Security Review — AI Job Discovery & Matching Engine

Reviewed against section 54 of the implementation spec. Every item below is
enforced in code (file/function cited), not just documented intent.

## SSRF

`jobEngine/guardrails/url_guardrails.py` — every URL the engine ever
navigates to passes `check_url`/`validate_navigation_url`:
- scheme must be `http`/`https` (blocks `file://`, `javascript:`, `data:`, ...)
- hostname blocklist (`localhost`, cloud metadata hostnames)
- literal-IP blocklist (private/loopback/link-local/reserved/multicast/
  unspecified ranges, plus the AWS/GCP/Azure metadata IP `169.254.169.254`)
- DNS-resolved-IP check (a hostname that *resolves to* a private IP is
  blocked too, not just literal IPs in the URL)
- per-source domain allowlist (`allowed_domains`, auto-derived from
  `base_url`/`search_urls` in `config/job_sources.yaml`)

Enforced at three layers (defense in depth): (1) adapters call
`validate_navigation_url` before every `page.goto`; (2)
`browser/manager.py:open_page`'s Playwright request-routing guard aborts any
request — including a same-page redirect a malicious site triggers after
load — to a host outside the allowlist; (3) `normalization_node` re-checks
every extracted job's `application_url`/`source_url` against the same
allowlist before a `Job` is ever constructed.

Tests: `tests/job_engine/test_url_guardrails.py`.

## Prompt injection

Job descriptions are untrusted external content. Defense is architectural,
not pattern-matching alone: `guardrails/prompt_injection.py:wrap_untrusted_content`
delimits job text with explicit markers, and the system prompt
(`prompts/job_matching_system.txt`) instructs the model to treat everything
between them as data, never instructions, "no matter what it says." The
same function strips any embedded fake delimiter tokens first, so a job
description can't inject its own fake `<<<UNTRUSTED_JOB_CONTENT_END>>>` to
"escape" the wrapper early. `detect_suspicious_patterns` additionally flags
likely injection phrasing for logging/metrics (never used to block a job by
itself — see `matching/llm_matcher.py`).

Tests: `tests/job_engine/test_guardrails.py`,
`tests/job_engine/test_llm_matcher.py::test_prompt_wraps_job_content_as_untrusted`.

## Secret leakage

- No credentials are ever hard-coded; `OPENAI_API_KEY` and
  `LINKEDIN_STORAGE_STATE_PATH` are read from the environment only
  (`config/settings.py`).
- LinkedIn support is storage-state-file-based specifically so no password
  ever passes through this codebase (`sources/linkedin/adapter.py`).
- `guardrails/output_guardrails.py:_looks_like_secret` scans final LLM
  explanation/strengths/concerns text for API-key-shaped strings
  (`sk-...`, `AKIA...`, `api_key: ...`) before a job is returned; a match
  drops the job from the result.

## Logging leakage

`observability/logging.py`'s docstring and every call site in `jobEngine`
never pass candidate phone/email, cookies, or tokens. `CandidateProfile.
to_llm_context()` (`models/profile.py`) deliberately omits phone and raw
email from the text sent to the LLM in the first place — the leak surface
doesn't exist upstream of logging either. `guardrails/output_guardrails.py:
_contains_candidate_pii` double-checks the *output* side too (explanation
text containing the candidate's actual phone/email string is rejected).

## Malicious HTML / oversized responses

`guardrails/content_limits.py`: `enforce_html_size` truncates any page HTML
over `JOB_MAX_HTML_BYTES`; `enforce_description_size` truncates any
extracted description over `JOB_MAX_DESCRIPTION_CHARS`; both are applied at
the `RawJob` Pydantic model boundary too (field validators clip title/
company/description/list-items independently of whether an adapter
remembered to call the guardrail function) — two independent enforcement
points for the same limit.

Adapters never execute page-provided JavaScript outside the Playwright
browser context (no `eval`, no `exec` of scraped content anywhere in this
codebase) — extraction is entirely `query_selector`/`inner_text`/
`get_attribute` calls, not script injection.

## Malicious / attacker-controlled URLs and redirects

Covered under SSRF above — the Playwright request-routing guard
(`browser/manager.py`) fires for *every* request in a context, including
ones initiated by a redirect or client-side navigation after the initial
`goto` succeeded, so a page that loads clean and then redirects to an
internal address is still blocked.

## Denial-of-service via pagination / excessive concurrency

- `guardrails/content_limits.py:pagination_depth_exceeded` hard-caps
  pagination at `min(source.max_pages, JOB_MAX_PAGINATION_DEPTH)`,
  independent of what a source's own config claims.
- `browser/manager.py:BrowserManager` uses an `asyncio.Semaphore(JOB_BROWSER_
  MAX_CONCURRENCY)` — only that many `BrowserContext`s can be open at once
  across the whole run, regardless of how many sources are enabled.
- `config/settings.py`: `JOB_MAX_SOURCES`, `JOB_MAX_JOBS_PER_SOURCE` cap the
  blast radius of a single request.
- Per-source `rate_limit_delay_seconds` (config) is honored between
  paginated requests (`sources/generic/adapter.py`) — this project scrapes
  responsibly, not aggressively, by design (section 9).

## LLM abuse / cost control

- Multi-stage funnel (rules -> LLM) means the expensive stage only ever sees
  `JOB_LLM_TOP_K` (default 50) jobs per run, ranked by rule score, never every
  scraped job — see `ARCHITECTURE.md` "Matching pipeline".
- `JOB_LLM_MAX_RETRIES` bounds retries on malformed output; there is no
  unbounded retry loop anywhere in `matching/llm_matcher.py`.
- Token/cost is tracked per run (`ExecutionMetrics`) so runaway usage is
  visible, not silent.

## Arbitrary navigation

Every `page.goto` call in every adapter goes through
`validate_navigation_url` first; there is no code path that navigates to a
URL sourced from anywhere other than `config/job_sources.yaml`'s
`search_urls` or an `href` extracted from an already-domain-restricted page
(and that `href` is itself re-validated by the request-routing guard before
the browser actually follows it).

## Residual risk / known limitations (honest, not hidden)

- **In-memory circuit breaker and run-status registry are process-local**
  (`sources/circuit_breaker.py`, `api/run_status.py`) — acceptable for this
  app's current single-process deployment; would need a shared store for
  multi-process/multi-instance.
- **DNS-resolution SSRF check has a narrow TOCTOU window**: `check_url`
  resolves and validates a hostname's IP at *validation* time; Playwright
  itself resolves again at *connection* time. A DNS record that changes
  between those two resolutions (attacker-controlled "rebinding" DNS) is a
  known, generally-accepted limitation of application-level SSRF guards
  that don't control the underlying TCP connection — a network-level
  egress policy (firewall/proxy) is the complete mitigation for this class
  of attack and is outside this application's scope.
- **The generic/Instinctools/LinkedIn adapters' structured CSS-selector
  tiers are best-effort** (see `SOURCE_ADAPTER_GUIDE.md`) — a wrong selector
  degrades to "no jobs found" (safe failure), never to unsafe behavior.
