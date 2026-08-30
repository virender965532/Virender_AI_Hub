# Virender AI Hub

Futuristic Flask web app with an **AutoJob Nexus** module that logs into [Naukri.com](https://www.naukri.com) via Selenium (credentials from `.env`) and displays extracted job rows.

## Prerequisites

- **Python 3.10+** (3.11 recommended) — for local runs
- **Google Chrome** installed (Selenium uses Chrome via `webdriver-manager`) — for local runs
- **Docker Desktop** (or Docker Engine + Compose v2) — for container runs (amd64 / x86_64)

## Setup

1. Open a terminal in the project folder:

   `c:\Project\AI\Jobs`

2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Configure credentials (never commit real secrets):

   - Copy `.env.example` to `.env` if you prefer a template.
   - Edit `.env` and set:

     ```
     NAUKRI_EMAIL=your_real_email@example.com
     NAUKRI_PASSWORD=your_real_password
     ```

   Replace the placeholder values; the app rejects `your_email` / `your_password` as-is.

## Run (local)

```powershell
python app.py
```

Open a browser to **http://127.0.0.1:5000/**

- Home: module cards; **AutoJob Nexus** goes to `/job-search`.
- Job search: a loading indicator runs while `POST /api/jobs/naukri` drives Selenium (login → **`/jobs-in-india`** SRP → parse cards).
- By default the API uses **headless** Chrome (`headless: true`). Send `{"headless": false}` for a visible browser (the job-search page JS does this).
- Debug output is under **`scraper_debug/`** (gitignored): `login_success.png`; if **zero** jobs are found, `debug_jobs.html` + `jobs_debug.png`.

## Run with Docker

The `Dockerfile` installs Python 3.11, Google Chrome (Selenium), and Playwright Chromium. You do **not** need a local Python venv or Chrome for this path.

### 1. Environment file

Copy `.env.example` to `.env` in the project root and set real values (never commit `.env`):

```
NAUKRI_EMAIL=your_real_email@example.com
NAUKRI_PASSWORD=your_real_password
OPENAI_API_KEY=sk-...
```

Add other keys your modules need (`HF_TOKEN`, `PAGE_INDEX_API_KEY`, etc.).

### 2. Start (recommended)

From the project root (`Viru_AI_Hub`):

```powershell
docker compose up --build
```

First build can take several minutes (PyTorch, sentence-transformers, browsers). Leave the terminal open.

Open **http://127.0.0.1:5000/**

Stop with `Ctrl+C`, then:

```powershell
docker compose down
```

Run in the background:

```powershell
docker compose up --build -d
```

Logs: `docker compose logs -f web`

### 3. Without Compose

```powershell
docker build -t viru-ai-hub .
docker run --rm -p 5000:5000 --shm-size=2g --env-file .env viru-ai-hub
```

`--shm-size=2g` is required so Chrome does not crash inside the container. Compose already sets `shm_size: 2gb`.

### Docker notes

- Job search in the container is **headless** (`NAUKRI_HEADLESS=1`). A visible Chrome window is not available.
- Compose mounts `Profile`, `output`, `scraper_debug`, and RAG upload/data folders so files persist on the host.
- Image is **amd64** (Google Chrome `.deb`). ARM machines (Apple Silicon without emulation) are not supported as-is.
- On Windows, start **Docker Desktop** before `docker compose up`.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Flask factory and dev server |
| `Dockerfile` / `docker-compose.yml` | Container image and Compose service |
| `routes/main_routes.py` | Pages + JSON API |
| `services/naukri_service.py` | `login_to_naukri()`, `fetch_jobs()`, `run_login_and_fetch_jobs()` |
| `templates/` | Jinja2 HTML |
| `static/` | CSS / JS |

## Optional environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `NAUKRI_USE_UNDETECTED` | `1` | Use **undetected-chromedriver** when `1`; set `0` to force stock Selenium + webdriver-manager. |
| `NAUKRI_STEP_PAUSE` | `1.2` | Extra delay (seconds) between major steps when the browser is **visible**. |
| `NAUKRI_STEP_PAUSE_HEADLESS` | `0.25` | Same when `headless: true`. |
| `NAUKRI_HEADLESS` | (Compose: `1`) | Playwright job search: `1` = headless (required in Docker). |
| `FLASK_HOST` / `FLASK_PORT` | `127.0.0.1` / `5000` | Dev server bind. Docker sets `0.0.0.0:5000`. |

## Notes

- Naukri may show **CAPTCHA**, **OTP**, or change HTML; update selectors in `services/naukri_service.py` if automation breaks.
- First run may take longer while ChromeDriver is downloaded.
- Visible runs are slower by design so you can follow each step; increase timeouts if your network is slow.

## Logging

The app logs to **stdout** (INFO). Automation messages are prefixed with logger names (e.g. `services.naukri_service`).
