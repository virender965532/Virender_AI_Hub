# Adding a New Job Source

Adding a source never requires touching `graph/builder.py` or any other core
node — only the files below.

## The 9 steps (section 56)

1. **Add source configuration** to `config/job_sources.yaml`:
   ```yaml
   - id: company_a
     name: Company A
     enabled: true
     base_url: "https://company-a.com"
     search_urls:
       - "https://company-a.com/careers"
     adapter: generic          # or a new dedicated adapter name
     requires_auth: false
     timeout_seconds: 60
     max_jobs: 100
     max_pages: 5
     selectors:                 # only needed for the "generic" adapter
       job_card: ".job-listing"
       title: ".job-listing__title"
       company: ".job-listing__company"
       location: ".job-listing__location"
       url: "a"
       description: ".job-listing__description"
   ```
2. **Try the generic adapter first.** If `company-a.com/careers` is a plain
   list of job cards with CSS classes, you are done — no code needed. Test
   it immediately:
   ```bash
   curl -X POST http://localhost:8000/api/jobs/sources/test \
        -H "Content-Type: application/json" -d '{"source_id": "company_a"}'
   ```
3. **If the site needs custom logic** (infinite scroll, an internal JSON
   API, auth, a non-card layout), create an adapter:
   ```
   jobEngine/sources/company_a/adapter.py
   jobEngine/sources/company_a/__init__.py
   ```
4. **Implement discovery** — subclass `JobSourceAdapter`
   (`jobEngine/sources/base.py`) and implement `discover_jobs(context)`.
   Use `context.browser_manager.open_page(source_id=..., allowed_domains=...)`
   for an isolated, guardrailed `BrowserContext`+`Page`. Always navigate
   through `guardrails.url_guardrails.validate_navigation_url`.
5. **Implement the parser** — extract fields into `RawJob`
   (`jobEngine/models/job.py`). Reuse `jobEngine/sources/dom_utils.py`'s
   `text_or_none`/`href_or_none`/`extract_link_cards` helpers rather than
   hand-rolling Playwright selector calls.
6. **Define the Pydantic mapping** — you're constructing `RawJob` directly,
   so this is step 5, not a separate step; just make sure every required
   field (`title`, `company`, `application_url`) is populated or the job is
   dropped in `normalization`.
7. **Register the adapter**:
   ```python
   from ..registry import register_adapter
   from ..base import JobSourceAdapter, SourceDiscoveryResult, SourceExecutionContext

   @register_adapter("company_a")
   class CompanyAAdapter(JobSourceAdapter):
       async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
           ...
   ```
   Then import it in `jobEngine/sources/registry.py:_ensure_builtin_adapters_imported`
   (one line) so it self-registers on import.
8. **Add a fixture** — save a representative HTML page under
   `tests/job_engine/fixtures/company_a/listing.html` (view-source, save,
   redact anything sensitive).
9. **Add tests** — load the fixture with `page.set_content(html)` (never
   `page.goto` — no network, no SSRF surface) and assert on the parsed
   `RawJob` list, following `tests/job_engine/test_generic_adapter.py` or
   `test_instinctools_adapter.py` as a template. Then flip `enabled: true`.

## Adapter contract (recap of `sources/base.py`)

```python
class JobSourceAdapter(ABC):
    async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
        ...
```

- **Never raise** out of `discover_jobs` for a site-specific problem — catch
  it and add a `SourceError` (`jobEngine/models/errors.py`) to the result
  instead. An adapter that raises is still contained by
  `graph/nodes/source_discovery.py`'s own try/except, but a clean
  `SourceError` gives the caller a real reason, not a stack trace.
- Set `retryable=True` only for transient failures (timeout, navigation
  error, 5xx). `SELECTOR_MISSING`, `AUTH_REQUIRED`, and `ANTI_BOT_BLOCKED`
  are never retried automatically — retrying a stale selector or a blocked
  session just wastes a browser context.
- Respect `cfg.max_jobs`/`cfg.max_pages`/`cfg.rate_limit_delay_seconds` —
  these are enforced by guardrails too (defense in depth), but an adapter
  that ignores them entirely will still hammer the target site before the
  guardrail catches it.

## LinkedIn (special case)

No password automation or CAPTCHA/anti-bot bypass exists anywhere in this
engine. `LinkedInAdapter` only works with a Playwright `storage_state` file
the candidate generates *out of band* (log in manually once, then
`await context.storage_state(path="linkedin_state.json")`), pointed to by
`LINKEDIN_STORAGE_STATE_PATH`. Without it, or if LinkedIn redirects to a
login/checkpoint page anyway, the adapter reports a structured `SourceError`
(`AUTH_REQUIRED` or `ANTI_BOT_BLOCKED`) and the run continues with other
sources.

## Known limitations

- **Instinctools and LinkedIn selectors are best-effort.** This environment
  could not fetch either live page during development (LinkedIn requires
  auth by design; the Instinctools fetch tool used to inspect the page
  returned a transport error). Both adapters have a structured CSS-selector
  tier *and* a link-scan fallback tier that only depends on URL patterns
  (`/vacancies/...`, `/jobs/view/...`), so a markup change degrades to the
  fallback rather than silently returning zero jobs — but the structured
  tier's exact class names should be verified against the live site before
  relying on it for volume.
- **The circuit breaker is in-memory and process-local** — restarting the
  Flask process, or running multiple worker processes, resets/duplicates
  breaker state. Fine for a single-process deployment (this app's current
  Docker setup); would need a shared store (Redis, etc.) for multi-process.
- **Request-level `keywords`/`location`/`remote` filters are applied
  post-discovery**, not by re-templating each source's `search_urls` per
  request — see `CONFIGURATION.md`.
