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
| `JOB_LLM_TOP_K` | `50` | Jobs sent to the (expensive) LLM stage, ranked by rule score; `<= 0` (e.g. `-1`) sends every job |
| `JOB_LLM_CONCURRENCY` | `12` | LLM calls (matching + worldwide review) running at the same time; every job is analyzed independently. Raise for speed, lower if OpenAI returns 429 rate-limit errors |
| `LINKEDIN_MAX_TABS` | `3` | Browser tabs scraping LinkedIn at once, shared by every search URL: a search URL's results pages are scraped this many at a time (each tab has its own detail pane). `1` = one page at a time. Higher is faster but makes LinkedIn security checks more likely. Already-saved jobs are never re-opened |
| `JOB_RULE_PREFILTER_MIN_SCORE` | `15` | Defined but currently unused (no prefilter stage runs) — left as a settings no-op rather than a breaking removal. Jobs the hard-rejection rules reject are never sent to the LLM |
| `REJECTED_JOBS_MAX_AGE_HOURS` | `168` | How long a rejected job is remembered (MongoDB `rejected_jobs`, TTL index). While remembered, LinkedIn doesn't re-open it and matching reuses its stored analysis instead of calling the LLM, unless the job text or `Profile/Summary.txt` changed; the run's minimum score is re-applied |
| `JOB_CANDIDATE_EXPERIENCE_YEARS` | `7` | Fallback if the profile's summary text doesn't state years explicitly |
| `JOB_MAX_EXPERIENCE_GAP_YEARS` | `3` | How far a job's stated required-years can exceed the candidate's before hard rejection |
| `JOB_ENGINE_DRY_RUN` | `false` | Navigate/extract/score but skip persistence |
| `JOB_ENGINE_DEBUG` | `false` | Save HTML snapshots on failure (screenshots are always saved) |
| `JOB_ENGINE_PERSIST_RESULTS` | `true` | Write results to `output/job_engine/` |
| `JOB_ENGINE_DEGRADE_ON_LLM_FAILURE` | `true` | Mark result `degraded` instead of failing when LLM is unavailable |
| `LINKEDIN_STORAGE_STATE_PATH` | unset | Path to a pre-authenticated Playwright storage_state JSON |
| `JOB_LLM_PRICE_PER_1K_PROMPT_USD` / `_COMPLETION_USD` | gpt-4o-mini rates | For cost-estimate observability only |
| `JOB_WORLDWIDE_REMOTE_ENABLED` | `false` | Classify every job's eligibility from India (worldwide / India-eligible / needs verification / restricted / insufficient info) with quoted evidence, read LinkedIn's workplace + employment chips, and show badges, filters and an eligibility sort in the UI. Country-restricted jobs (by the rules, or by the LLM review below) are rejected without an LLM match, remembered in `rejected_jobs`, and left out of the Saved Jobs page. Off = previous behaviour |
| `JOB_WORLDWIDE_LLM_REVIEW` | `true` | With the flag above on and the LLM available, send shown "needs verification" jobs to the existing OpenAI model for a second opinion; its quotes must appear verbatim in the posting or the verdict is ignored |
| `JOB_WORLDWIDE_LLM_MAX_JOBS` | `25` | Cap on those review calls per run (a saved review is reused while the description is unchanged) |
| `JOB_REJECTION_MODEL` | `gpt-4o` | Model for the rejection analysis on the Applied Jobs page (`jobEngine/matching/rejection_analyzer.py`). Called only when a job is marked rejected and has no stored analysis, or on Re-evaluate; the result is stored on the applied job. Candidate-side facts the resume lacks (timezone, expected pay, job types, work authorization) come from `config/candidate_preferences.yaml` |

| `ANALYSIS_CACHE_MAX_AGE_HOURS` | `720` | LLM match analyses are cached in MongoDB `job_analysis_cache`, keyed by job text + `Profile/Summary.txt` + matcher version (prompts, model, schema). A job seen again, even after it was removed from Saved Jobs, reuses its analysis instead of a new call, so its score does not change between runs |

**Match score** (`jobEngine/matching/fit_score.py`) — the LLM gives no number; it lists the job's requirements as tier (core / required / preferred) × verdict (met / partial / missing). Code corrects the labels it can check: a requirement naming a skill from the profile's skills list, or a sub-topic in `config/implied_skills.yaml` of a skill the profile shows, is *met*; one the job marks "nice to have" / "preferred" / "a plus" is *preferred*; a years-of-experience line is dropped (scored by E). Then:

- S = 100 × Σ(w × v) / Σ(w), with w core 3 / required 2 / preferred 1 and v met 1 / partial 0.5 / missing 0 (no requirements: S = K)
- E = 100 when the job states no years range or the candidate is within it; under-qualified 100 − 25 per missing year (min 0); over-qualified 100 − 10 per extra year (min 70)
- K = the rule evaluation's keyword/stack overlap
- score = round(0.90·S + 0.05·E + 0.05·K), capped at 60 when a core requirement is missing

Location / eligibility never change the score; they are shown separately. Saved and rejected jobs have their score recomputed from the stored analysis on every run, so a formula change applies without new LLM calls.

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

## Scheduled search + Pushover notifications

`python -m jobEngine.cron` runs one search with the Remote Job Finder's default
sources and the keywords already in `config/job_sources.yaml`, saves the results
like a manual search, and sends a [Pushover](https://pushover.net/api#messages)
push listing the jobs this run added (a failed run is pushed too).

`python app.py` runs it on a schedule by itself (`jobEngine/scheduler.py`), so the
EC2 container needs no crontab. A scheduled time that passes while the app is
down, restarting, or still busy with the previous run is skipped, not caught up;
the next run is the next scheduled time. Set `JOB_CRON_ENABLED=false` on any
machine that should not run it (e.g. a local dev box).

| Variable | Default | Purpose |
|---|---|---|
| `JOB_CRON_ENABLED` | `true` | Run the schedule inside `python app.py` |
| `JOB_CRON_SCHEDULE` | `0 * * * *` | Crontab expression (minute hour day month weekday); default every hour on the hour |
| `JOB_CRON_TIMEZONE` | `UTC` | Timezone the schedule is read in |
| `PUSHOVER_TOKEN` / `PUSHOVER_USER` | — | Pushover application token / user key; no push is sent if missing |
| `JOB_CRON_MIN_MATCH_SCORE` | `30` | Minimum match score |
| `JOB_CRON_MAX_RESULTS` | `-1` | Jobs per search URL, `-1` = all |
| `JOB_CRON_REMOTE_ONLY` | `true` | Keep only remote/hybrid roles |
| `JOB_CRON_SOURCES` | page defaults | Comma-separated source ids |
| `JOB_CRON_SAVED_JOBS_URL` | `https://ai.ritishka.in/saved-jobs` | Link opened from the push and the job email |

**Job email** (`jobEngine/notifications/job_alerts.py`) — after every search run (manual or scheduled) whose jobs are saved, jobs scoring strictly above the threshold are emailed in one message. Each job is mailed once (`mailed_at` on its saved-jobs document); a failed send is retried by the next run.

| Variable | Default | Effect |
|---|---|---|
| `JOB_MAIL_THRESHOLD_SCORE` | unset (off) | Mail jobs whose match score is above this |
| `JOB_MAIL_TO` | `virenderfullstack@gmail.com` | Recipient |
| `EMAIL_SENDER` / `EMAIL_PASSWORD` | — | SMTP login and From address; for Gmail an App Password. No email is sent if missing |
| `SMTP_HOST` / `SMTP_PORT` | `smtp.gmail.com` / `587` | SMTP server; port 465 uses implicit TLS, others STARTTLS |

**HR email + drafts** (`jobEngine/notifications/hr_drafts.py`) — when a job in that email has an email address in its posting, the job engine's LLM reads it into `HREmail` (`email_id`, `is_hr_mail`, `reason` = the posting's sentence containing the address; an address not really in the posting is discarded, and without an LLM the address pattern decides). It is stored on the saved job (`hr_contact`). For an HR address the alert shows that sentence, highlighted, and a **Create draft email** link to `/hr-draft/<saved key>`. Opening the link runs the Autonomous Application Agent (`jobApplyPro`) with the job description, the address and your profile, as if pasted into its page; the generated subject and body are saved over IMAP as a draft to the HR address (resume attached) in the `EMAIL_SENDER` mailbox, logging in with `EMAIL_PASSWORD`. If that login fails, the page offers a prefilled Gmail compose link (Gmail keeps it in Drafts, without the attachment). A saved draft is recorded on the saved job (`hr_draft`) and shown again on later visits; "Create again" makes a new one.

| Variable | Default | Effect |
|---|---|---|
| `JOB_HR_DRAFT_URL` | `https://ai.ritishka.in/hr-draft` | Base of the "Create draft email" link |
| `IMAP_HOST` | `imap.gmail.com` | IMAP server |
