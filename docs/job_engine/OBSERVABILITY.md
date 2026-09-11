# Observability

## Structured logging

`jobEngine/observability/logging.py:log_event` emits one JSON object per log
line through the app's existing root logger (configured once in `app.py`;
this module never changes that global config, so every other module keeps
logging exactly as before). Fields, when present: `timestamp`, `level`,
`event`, `run_id`, `node`, `source`, `job_id`, `duration_ms`, `status`,
`error_type`, `trace_id`, plus any extra keyword arguments the caller passes.

Never logged (section 26): passwords, cookies, access tokens, API keys,
session ids, or candidate phone/email — `CandidateProfile.to_llm_context()`
and every log call site were written to exclude them; `guardrails/
output_guardrails.py` additionally scans final output for PII/secret-shaped
strings as a second line of defense.

Every important stage logs at least once via the node's own `logger.info`
calls (`normalization_node`, `validation_node`, `matching_node`,
`guardrail_check_node`, `ranking_node`, `final_output_node`)
— correlated by `run_id` since it's threaded through the whole graph state.

## Tracing

`ChatOpenAI` (via `langchain-openai`) automatically emits
LangSmith traces when the standard `LANGCHAIN_TRACING_V2`/`LANGCHAIN_API_KEY`/
`LANGCHAIN_PROJECT` env vars are set — this is opt-in and costs nothing to
leave unset (matches this repo's existing pattern of not having LangSmith
wired up anywhere yet; `jobEngine` is the first module to *support* it,
piggy-backing on `langchain-openai`'s built-in instrumentation rather than
hand-rolling a tracer).

## Cost / token tracking

`matching/llm_matcher.py:LLMCallMetrics` accumulates, per run:
`calls`, `retries`, `prompt_tokens`, `completion_tokens`, `total_tokens`,
`estimated_cost_usd` (using `JOB_LLM_PRICE_PER_1K_PROMPT_USD`/
`_COMPLETION_USD` — operator-configurable, approximate, observability-only,
never used for a billing decision). These land in
`JobSearchResult.metrics` (`ExecutionMetrics`, section 50):

```json
{
  "jobs_discovered": 184, "jobs_validated": 171,
  "jobs_rejected_by_rules": 41,
  "jobs_sent_to_llm": 50, "jobs_recommended": 37,
  "llm_calls": 50, "llm_retries": 3,
  "llm_prompt_tokens": 42000, "llm_completion_tokens": 9000,
  "llm_estimated_cost_usd": 0.0117,
  "source_failures": 1, "browser_failures": 2,
  "match_distribution": {"strong_match": 12, "good_match": 25, "reject": 9},
  "workflow_duration_ms": 48213.4
}
```

## Health check

`GET /api/jobs/health` (`jobEngine/api/health.py`) checks: config file loads
and parses, the output directory is writable (the "database" — flat files),
LLM provider configured+enabled (no live network call, to keep the
check fast and free), and Playwright is importable. Returns `200` when every
check passes, `503` otherwise — never returns secrets, only booleans/model
names.

## Metrics -> future dashboard

Every field in `ExecutionMetrics` is already structured and JSON-serializable
specifically so it can be dropped into a Flask template/chart later without
re-deriving anything — see section 50 of the original spec. No UI was built
for this in this pass (see the final implementation report's "Known
limitations").
