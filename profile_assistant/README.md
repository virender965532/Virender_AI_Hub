# Profile Assistant ("Ask Virender - Career Consultant")

The UI title is `Ask <first name> - Career Consultant`, built from the
profile's `name`. A career-consultant chatbot about one person. Each question
is classified into one of three types:

| Type | Example | Behaviour |
|---|---|---|
| `about_person` | "Why are you switching companies so often?", "Is he a fit for a GenAI role?" | Answered **only from profile facts**. Reasoning from those facts is allowed (tenures, career progression, fit for a role); invented facts or motives are not. If the specific detail isn't in the profile, the answer says so and gives the relevant facts the profile does show. `I don't know.` only when the profile has nothing relevant. |
| `domain` | "What is RAG?", "SSG vs SSR in Next.js?" | Answered with accurate general professional knowledge, connected to the person's experience where the profile supports it. |
| `out_of_scope` | "Best stock to buy?", "Capital of France?" | Exactly `I don't know.` |

The knowledge source is the primary profile (`PA_PROFILE_PATH`) plus any
supplementary sources (`PA_EXTRA_PROFILE_PATHS`, by default the LinkedIn
export `Profile/Profile.pdf`). Sources are combined; when they contradict each
other on a fact, the primary one wins.

Every answer is checked by a separate **critic agent** that runs on a
different OpenAI model before the user sees it. When the critic rejects an
answer, the answer agent revises it using the critic's feedback. After
`PA_MAX_REVISIONS` failed revisions, the assistant falls back to
`I don't know.`

## Architecture

```
question
   │
   ▼
input guard ── empty / > 2000 chars / prompt injection ──► polite refusal (status: blocked)
   │
   ▼
answer agent (PA_ANSWER_MODEL, temp 0, PROFILE + last 4 turns)
   │
   ▼
critic agent (PA_CRITIC_MODEL, temp 0) ──PASS──► output guard ──► answer (passed / revised)
   │ FAIL
   ▼
revise with critic feedback ── loop, at most PA_MAX_REVISIONS times
   │ still FAIL
   ▼
fallback: "I don't know." (status: fallback)

output guard ── leaked system prompt text / secret ──► "I don't know." (status: blocked)
OpenAI unavailable after retries ──► "Sorry, I'm unable to answer right now. Please try again." (status: error)
```

- An `I don't know.` answer also goes to the critic, which catches false
  `I don't know.` answers when the profile does contain the answer.
- The PASS rule (`groundedness >= 4 AND relevance >= 4 AND scope_correct`) is
  also enforced in code ([critic_agent.py](critic_agent.py)), so a lenient
  verdict from the model can't get through.
- The answer agent's output is normalised: if it classifies a question as
  `out_of_scope`, the answer becomes exactly `I don't know.` with no evidence.
- The critic also fails vague answers (for example a job-change answer that
  doesn't list the roles and dates, or a bare "Yes" to a fit question).
- Chat history only helps resolve references like "he" or "that project".
  Messages with any role other than `user`/`assistant` are dropped, and the
  profile is still the only source of facts.

| File | Responsibility |
|---|---|
| `config.py` | Env-var settings; fails at startup if the answer and critic models are the same |
| `profile_loader.py` | Loads `.json` (validated by Pydantic), `.md`/`.txt` and `.pdf` profiles, combines the primary and supplementary sources, redacts phone/email, caches in memory, `reload_profile()` |
| `schemas.py` | `AnswerOutput` and `CriticVerdict` structured outputs |
| `prompts.py` | All prompts and fixed messages, versioned with `PROMPT_VERSION` |
| `llm_client.py` | One shared OpenAI client: `chat.completions.parse`, timeout, transient-only retries with backoff and jitter, one repair retry for a bad parse |
| `guardrails.py` | Input checks (empty, length, injection) and output checks (prompt leak, secrets) |
| `answer_agent.py` / `critic_agent.py` | The two agents |
| `pipeline.py` | Orchestration and JSON logging per step |
| `routes.py` | Flask blueprint |
| `ui.py` | Gradio UI and its mounting |

## Setup

1. Install dependencies (`gradio` and `a2wsgi` were added to the root `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```
2. Set the environment variables in `.env` (see the root `.env.example`).
3. Start the hub as usual (`python app.py` or `python dev.py`), then open
   `http://127.0.0.1:8000/profile-assistant/`, or pick "Ask Virender Career Consultant"
   from the hub's All Tools menu or home page.

## Environment variables

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | Required for answers. Shared with the other modules. Never logged. |
| `PA_ANSWER_MODEL` | `gpt-4o-mini` | |
| `PA_CRITIC_MODEL` | `gpt-4o` | Must differ from `PA_ANSWER_MODEL`, otherwise startup fails. |
| `PA_PROFILE_PATH` | `profile_assistant/data/profile.json` | Primary source. Relative paths resolve from the repo root. `.json`, `.md`, `.txt` or `.pdf`. |
| `PA_EXTRA_PROFILE_PATHS` | `Profile/Profile.pdf` | Comma-separated supplementary sources. Set it empty to use only the primary. A missing file fails startup. |
| `PA_REDACT_CONTACT_DETAILS` | `true` | Removes phone numbers and email addresses from all sources before the model sees them |
| `PA_MAX_REVISIONS` | `2` | Number of revise rounds after the first answer |
| `PA_REQUEST_TIMEOUT` | `30` | Seconds per OpenAI call |
| `PA_MAX_RETRIES` | `3` | Retries for transient errors only |
| `PA_URL_PREFIX` | `/profile-assistant` | |
| `PA_ADMIN_TOKEN` | *(unset)* | Secret for `reload-profile`. If unset, that route returns 403. |
| `PA_UI_ENABLED` | `true` | Set to `false` to serve only the JSON API |

## HTTP API

```bash
# Ask a question
curl -X POST localhost:8000/profile-assistant/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Where did he study?", "history": []}'
# -> {"answer", "status", "revisions", "evidence", "trace_id", "latency_ms"}

# Health check
curl localhost:8000/profile-assistant/api/health
# -> {"status": "ok", "answer_model": "...", "critic_model": "..."}

# Reload the profile from disk without restarting
curl -X POST localhost:8000/profile-assistant/api/reload-profile -H "X-Admin-Token: $PA_ADMIN_TOKEN"
```

`status` is one of `passed`, `revised`, `fallback`, `blocked` or `error`.
`error` means OpenAI was unreachable after retries; the answer is then the
friendly "unable to answer" message. `history` accepts the OpenAI/Gradio
messages format (`[{"role": "user", "content": "..."}]`) or `[user, assistant]` pairs.

## Changing the profile

- Edit `data/profile.json`, or point `PA_PROFILE_PATH` at another file.
  JSON fields: `name` (required), `headline`, `summary`, `skills` (a list, or
  a dict of category → list), `experience`, `education`, `projects`,
  `certifications`, `languages`, `interests` and `contact_preferences`. Extra
  top-level fields (for example `location`) are kept and shown to the model too.
- A `.md`/`.txt` profile is used as-is; its first non-empty line is taken as the name.
- A `.pdf` (for example LinkedIn's "Save to PDF") is converted to text and
  cleaned of page footers. To refresh the LinkedIn data, replace
  `Profile/Profile.pdf` and reload.
- `profile.json` is the source of truth. Where a supplementary source gives a
  different title, date, number or duration, the `profile.json` value is used.
  Extra details from a supplementary source are still used.
- For the most reliable answers, copy any useful supplementary-only detail
  into `profile.json`. Two sources that disagree can make the critic loop
  unstable and occasionally end in `I don't know.`. The non-conflicting
  details of `Profile/Profile.pdf` are already in `profile.json`.
- Apply changes by restarting, or by calling `POST /api/reload-profile`. An
  invalid file is rejected and the previously loaded profile stays active.
- Everything in the profile can end up in answers. Phone numbers and email
  addresses are redacted by default (`PA_REDACT_CONTACT_DETAILS`).

## Page and theme

`{prefix}/` is a normal hub page: `templates/profile_assistant/career_consultant.html`
extends the hub's `base.html`, so it gets the same header, nav, background and
fonts. It embeds the Gradio chat (`{prefix}/ui/`) in a glass panel. The hub's
nav item and home-page card ("Ask Virender Career Consultant") link to this page.

The Gradio chat uses a matching theme (`ui.build_theme()` and `HUB_CSS`): the
palette from `static/css/style.css` (dark glass, neon blue/purple/cyan) and the
Rajdhani/Orbitron fonts, identical in light and dark mode. If the palette in
`style.css` changes, update `_HUB` in `ui.py` to match.

## Why Gradio is mounted with a2wsgi (Option A)

Gradio is an ASGI app and Flask is WSGI, so Gradio can't be registered as a
blueprint. `ui.py` builds the Gradio app on FastAPI, wraps it with
`a2wsgi.ASGIMiddleware`, and mounts it at `{prefix}/ui` with werkzeug's
`DispatcherMiddleware`. Only paths under `{prefix}/ui` go to Gradio; every
other request goes to the original Flask `wsgi_app` unchanged. The result is
one process and one port, with no extra service to deploy (unlike Option B,
a separate Gradio server shown in an iframe).

One detail matters: a2wsgi doesn't send ASGI *lifespan* events, and Gradio
starts its request queue in its lifespan handler. Without that, chat requests
wait in the queue forever. `ui._start_lifespan()` therefore runs the lifespan
on a dedicated background event loop, keeps it open for the life of the
process, and passes the same loop to `ASGIMiddleware`.

If Gradio or a2wsgi isn't installed, the UI is skipped with a warning and the
JSON API keeps working.

Gradio 5 deprecated `ChatInterface(type="messages")` and Gradio 6 removed the
parameter (the messages format is the only one), so it isn't passed.

## Tests

All OpenAI calls are mocked with a scripted fake client, so no network is used:

```bash
python -m pytest profile_assistant/tests -q
```

The root `pytest.ini` points `testpaths` at `tests/`, so pass the path
explicitly. A bare `pytest` run won't collect these tests.

## Logging

Each step logs one JSON line with the request's `trace_id`: input-guard
result, each answer and critic call (model, attempts, revision, verdict,
scores, latency, token usage), the fallback, the output-guard result, and a
final summary with total tokens and `prompt_version`. The API key, the
profile text and the question text are never logged; only the question's
length is.
