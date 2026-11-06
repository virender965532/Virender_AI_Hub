# Stock Finder

An AI equity research assistant inside Virender AI Hub. It reads current market news, identifies the key
themes and the overall sentiment, finds companies that may benefit, analyses each one in depth, and reviews
your existing holdings (BUY MORE / HOLD / TRIM / SELL).

> **Disclaimer:** This is AI-generated research for informational purposes only and is not financial advice.
> Please do your own research or consult a SEBI-registered investment adviser before investing.
> (With `SF_MARKET=US`, the disclaimer names an SEC-registered investment adviser or FINRA-registered broker-dealer.)
> The module is research and decision support only. It has **no broker integration and never places trades.**

## Setup

1. Install the dependencies: `pip install -r requirements.txt` (Stock Finder adds `yfinance`, `feedparser` and
   `plotly`; `openai`, `pydantic`, `gradio`, `a2wsgi`, `faiss-cpu`, `pymongo` and `pytest` were already there).
2. Set `OPENAI_API_KEY`. For the portfolio and the report audit trail, also set `SF_MONGODB_URI` (or reuse the
   hub's `MONGODB_URI`). Holdings and reports go to their own database, `stock_research`.
3. Run the hub (`python app.py`) and open **Stock Finder** from the home page, or go to `/stock-finder/`.

Registration is a single call in `app.py`: `register_stock_finder(app)`. Nothing heavy loads at startup.
The pipeline is built on the first API call, and the Gradio UI is built on the first visit to the page.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | (none) | Required for analysis. Never logged. |
| `SF_ROUTER_MODEL` | `gpt-4o-mini` | Classifies chat requests (temperature 0). |
| `SF_ANALYST_MODEL` | `gpt-4o` | News reduce step, screener, portfolio analyst, summaries. |
| `SF_WORKER_MODEL` | `gpt-4o-mini` | News map step and per-company analysis workers. |
| `SF_CRITIC_MODEL` | `gpt-4.1` | Must differ from `SF_ANALYST_MODEL`, or startup fails. The spec's `o3-mini` is deprecated by OpenAI and returns 404. |
| `SF_FALLBACK_MODEL` | `gpt-4o-mini` | Used when a primary model fails; the result is marked *degraded*. The critic's fallback is never the model that wrote the draft. |
| `SF_EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings for the news store. |
| `SF_MARKET` | `IN` | `IN` (NSE/BSE, `.NS` suffix) or `US`. Sets the universe, feeds, currency and disclaimer. |
| `SF_MARKET_DATA_PROVIDER` | `yfinance` | Comma-separated list, tried in order (fallback chain). |
| `SF_NEWS_PROVIDERS` | `rss,newsapi` | `newsapi` is used only when `NEWSAPI_KEY` is set. |
| `NEWSAPI_KEY` | (none) | Optional. Sent as a header, never in a URL. |
| `SF_MAX_CANDIDATES` | `8` | Maximum number of companies analysed per opportunities run. |
| `SF_MAX_PARALLEL_WORKERS` | `4` | Concurrency limit for the company and holding workers. |
| `SF_NEWS_LOOKBACK_DAYS` | `7` | News window. |
| `SF_MAX_LLM_CALLS` | `60` | Per-request budget. The run stops cleanly with partial results. |
| `SF_RISK_PROFILE_DEFAULT` | `moderate` | `conservative`, `moderate` or `aggressive`. |
| `SF_URL_PREFIX` | `/stock-finder` | Page at `{prefix}/`, API at `{prefix}/api`, UI at `{prefix}/ui`. |
| `SF_MONGODB_URI` / `SF_MONGODB_DB_NAME` | `MONGODB_URI` / `stock_research` | Separate connection from jobEngine's. |
| `SF_MAX_STOCK_PCT` / `SF_MAX_SECTOR_PCT` | `25` / `40` | Concentration warning limits (% of portfolio). |
| `SF_SCORE_WEIGHTS` | built-in | JSON override of the score weights per risk profile. Each row is re-normalized to sum to 1. |
| `SF_UI_ENABLED` | `true` | Set to `false` to serve only the API. |

Data files (edit freely):
- `config/stock_finder_feeds.yaml`: the RSS feeds per market and the company-news search URL.
- `config/stock_finder_universe.yaml`: sector-to-symbols lists used for screening and peer comparison, plus the indices.

A sample holdings file is in `stock_finder/sample_portfolio.csv`.

## Architecture

```
find opportunities:
  news analyst ──► screener ──► company analysts (parallel) ──► critic ──► synthesizer
  (RAG + map-reduce) (ReAct + tools)  (one worker per stock)      (per stock)  (ranked report)

portfolio review:
  compute (P&L, allocation, concentration) ──► portfolio analyst (parallel) ──► critic ──► synthesizer

chat:
  router ──► market overview | analyze stock | (point to opportunities / portfolio) | grounded answer
```

**Core principle: the LLM reasons, code calculates.** Prices, returns, P/E and P/B (also compared with the
sector median), ROE, ROCE, debt/equity, RSI, MACD, moving averages, volatility, drawdown, relative strength,
P&L and allocation are all computed in Python (`data/`, `analysis/`, `portfolio/service.py`). The model gets
these values as JSON and only interprets them.

- **Router** (`agents/router.py`): classifies the intent. If its confidence is below 0.5, it asks one clarifying question.
- **News analyst** (`agents/news_analyst.py`): fetches news, adds it to the FAISS news store (`rag/news_store.py`)
  and retrieves articles with market-wide queries. *Map*: the worker model finds themes in each batch.
  *Reduce*: the analyst model merges them. Evidence is given as article ids that code resolves into citations;
  a theme with no valid citation is dropped. A summary containing a figure that isn't in the data is replaced.
- **Screener** (`agents/screener.py`): runs a ReAct loop with `search_news`, `get_sector_peers`,
  `get_index_snapshot`, `screen_by_metrics` and `submit_candidates`. The number of steps is bounded. Code
  validates the submitted symbols: they are normalized, restricted to the requested sectors, deduplicated
  and checked against a penny-stock price floor. If the loop fails, a deterministic screen is used.
- **Company analysts** (`agents/company_analyst.py`): one worker per candidate, run with `asyncio` under a
  semaphore. Each worker receives only its own company's data. A failed worker is marked
  "analysis unavailable" and the rest of the run continues.
- **Critic** (`agents/critic.py`): runs a different model.
  - It first applies deterministic checks: every number in the output must appear in the input data,
    citation ids must exist, news claims must be cited, overconfident phrases are rejected, the verdict
    must match the score and data quality, at least one risk must be listed, and BUY MORE is not allowed
    on a concentrated position.
  - Then the critic model reviews tone, balance and logic.
  - A FAIL leads to one revision with the feedback. If it still fails, the verdict is downgraded to WATCHLIST
    (holdings to HOLD) with low confidence, and the critic's concerns are shown.
- **Scoring** (`analysis/scoring.py`): a transparent weighted composite of five sub-scores: fundamentals,
  valuation, momentum, news sentiment (a lexicon score) and risk. The weights depend on the risk profile.
  Each sub-score and the rule that produced it is returned so the reasoning can refer to it.
- **Synthesizer** (`agents/synthesizer.py`): assembles the report in this order: market summary, top
  opportunities ranked by score, watchlist, avoid list, portfolio actions, key risks, disclaimer.

### Reliability

- Only transient errors are retried, with exponential backoff and jitter. Each external dependency
  (OpenAI and every data and news provider) has its own circuit breaker.
- A structured-output parse error is retried once with the validation error included.
- If a model fails, the fallback model is used and the result is marked degraded.
- If a data provider fails, the next one is tried. If all fail, the run continues with partial data
  and a `data_quality` flag.
- Each request has a budget of `SF_MAX_LLM_CALLS`. When it runs out, the run stops cleanly and returns
  the work that finished.
- Every loop is bounded. An unexpected error becomes a friendly message containing the `trace_id`;
  the details go to the logs.

### Safety

- Article text is untrusted data. It is HTML-stripped, length-capped and shown to models only inside
  `<news>` tags, with an instruction to ignore any directives it contains. The UI escapes model text.
- Leverage, F&O and penny stocks are never recommended unless the user asks. Penny stocks get an
  extra risk flag.
- A disclaimer is included in every API response (JSON and SSE) and on every UI tab.

### Observability and audit

Every request has a `trace_id` and produces one JSON log line per event (`stock_finder.trace` logger).
Events cover the route decision, data fetches (provider, latency, cache hit), LLM calls (model, tokens,
estimated cost), worker results, critic verdicts and the final status. Secrets are redacted.

Every report is saved to MongoDB (`sf_reports`) with its prompt version (`prompts.PROMPT_VERSION`), model
ids, metrics and a **data snapshot**: the exact numbers and article ids each agent saw. This allows
recommendations to be audited and back-tested. `GET /api/reports/<id>?snapshot=true` returns the snapshot.

## API

Prefix: `/stock-finder/api`

| Method | Path | Body / query | Notes |
|---|---|---|---|
| GET | `/health` | | Models, circuit-breaker states, Mongo status. |
| GET | `/market-overview` | `?refresh=true` | Themes with citations, sentiment, indices. Cached for 15 minutes. |
| POST | `/find-opportunities` | `{risk_profile, sectors?, max_candidates?, stream?}` | Background job. SSE progress by default; `"stream": false` returns `202 {job_id}`. |
| POST | `/analyze` | `{symbol, risk_profile?}` | One CompanyAnalysis. Accepts `TCS`, `TCS.NS`, `Infosys` or `AAPL`. |
| POST | `/chat` | `{message, risk_profile?}` | Routed free-form question. |
| GET/POST/DELETE | `/portfolio` | `{symbol, quantity, avg_cost, buy_date?, thesis?}`, `{holdings:[...], replace}`, or a CSV with `Content-Type: text/csv` | Holdings CRUD. DELETE takes `?symbol=` or `?all=true`. |
| POST | `/portfolio/review` | `{risk_profile, stream?}` | Background job with SSE progress. |
| GET | `/jobs/<id>`, `/jobs/<id>/events` | | Job status / re-attach to the event stream. |
| GET | `/reports/<id>` | `?snapshot=true` | A saved report. |

SSE events: `session`, `job`, `progress`, `worker_result`, `portfolio_snapshot`, then `final` (the full report)
or `error`.

## UI and how it is mounted

The UI has five tabs: Market Pulse, Opportunities (live progress), Analyze a Stock (price chart with
50/200-day averages and an indicator table), My Portfolio (editable table, CSV upload, review with an
allocation chart and concentration warnings) and Chat. Each tab shows the disclaimer, and each result
shows "data as of".

It uses **option A**. Gradio is an ASGI app and Flask is WSGI, so the Gradio app is wrapped with `a2wsgi`
and attached at `{prefix}/ui` with werkzeug's `DispatcherMiddleware`, the same way as `code_assistant`
and `profile_assistant`. This was chosen over a separate port with an iframe because:
- there is one process, one port and one deploy;
- there are no CORS or mixed-origin issues;
- the UI calls the pipeline in-process.

The hub page (`templates/stock_finder/stock_finder.html`) wraps the UI in an iframe styled like the rest of the hub.

## Security

The hub has no authentication layer, so, like the other hub modules, these routes are open to anyone who can
reach the server. Keep `FLASK_HOST=127.0.0.1` or put the hub behind an authenticating proxy before exposing it.
The portfolio is a single shared `default` portfolio (`?portfolio_id=` separates portfolios, but is not access control).

## Data sources and limitations

- **Prices and fundamentals** come from Yahoo Finance via `yfinance`. The data is unofficial, delayed and
  sometimes incomplete for NSE/BSE. ROCE needs the financial statements and is often missing.
  "Promoter holding" uses Yahoo's insider-holding field as a proxy. **Pledging is not available.**
  For production use, add a licensed provider as a new adapter in `data/market_data.py`.
- **Staleness** is measured in business days; exchange holidays are not modelled, so the day after a
  holiday can be flagged as stale.
- **News** comes from public RSS feeds (and NewsAPI when configured). Feed URLs change over time;
  a failing feed is skipped and logged. News sentiment is a simple keyword lexicon.
- **Screening** only covers the universe file (liquid large and mid caps), which is deliberate: it keeps
  fetches bounded and avoids illiquid stocks.
- LLM output is checked but can still be wrong. Scores are heuristics, not predictions, and past behaviour
  does not predict returns.

## Tests

```
python -m pytest stock_finder/tests -q
```

The LLM and the data providers are mocked, and MongoDB is replaced by mongomock, so the tests make no
network calls. They cover:
- indicator, fundamentals and scoring maths;
- the critic rules (ungrounded numbers, uncited news, overconfident language, verdict and score mismatch);
- the revise and downgrade loop;
- a failed worker not failing the run;
- stale data and provider fallback;
- P&L, allocation and concentration;
- the disclaimer on every response;
- partial results when the budget is exhausted;
- the startup error when the critic and analyst models are the same;
- the Flask endpoints, via the test client.
