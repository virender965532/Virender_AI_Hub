# Virender AI Hub

Futuristic Flask web app with an **AutoJob Nexus** module that logs into [Naukri.com](https://www.naukri.com) via Selenium (credentials from `.env`) and displays extracted job rows.

## Prerequisites

| | macOS (Intel / Apple Silicon) | Windows 10 / 11 |
|---|---|---|
| Python | **3.10+** (3.11 recommended) — `brew install python@3.11` or [python.org](https://www.python.org/downloads/) | **3.10+** (3.11 recommended) — [python.org](https://www.python.org/downloads/) installer, tick **"Add python.exe to PATH"** |
| Browser | **Google Chrome** (Selenium uses it via `webdriver-manager`) | **Google Chrome** |
| Git | `xcode-select --install` or `brew install git` | [Git for Windows](https://git-scm.com/download/win) |
| Docker (optional) | Docker Desktop for Mac | Docker Desktop for Windows (WSL 2 backend) |
| Ollama (optional, `LLM_PROVIDER=ollama`) | `brew install ollama` or [ollama.com](https://ollama.com) | [ollama.com](https://ollama.com) installer |

## Setup (local)

### macOS

Open **Terminal** in the project folder:

```bash
cd ~/Project/Virender_AI_Hub

# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install the Playwright browser used by jobEngine
playwright install chromium

# 4. Create your env file
cp .env.example .env
```

Notes for macOS:

- Use `python3` / `pip3` if `python` is not on your PATH (it usually isn't outside the venv).
- On Apple Silicon (M1–M4), PyTorch installs from PyPI as a native arm64 wheel — keep the `torch==...+cpu` lines in `requirements.txt` **commented out** (the `+cpu` wheel does not exist for macOS).
- If `pip install` fails building a package, run `xcode-select --install` and retry.
- The first time Chrome is driven by Selenium, macOS may ask to allow `chromedriver`; if it is blocked, open **System Settings → Privacy & Security** and click **Allow Anyway**.

### Windows

Open **PowerShell** in the project folder:

```powershell
cd C:\Project\Virender_AI_Hub

# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Install the Playwright browser used by jobEngine
playwright install chromium

# 4. Create your env file
Copy-Item .env.example .env
```

Notes for Windows:

- If activation fails with *"running scripts is disabled on this system"*, run once:
  `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then activate again.
- Using **cmd.exe** instead of PowerShell? Activate with `.venv\Scripts\activate.bat` and copy with `copy .env.example .env`.
- If `python` opens the Microsoft Store, disable the alias under **Settings → Apps → Advanced app settings → App execution aliases**, or use `py -3.11` instead of `python`.
- Packages that fail to build may need the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
- Paths in `.env` (e.g. `CA_WORKSPACE_ROOTS`) can use forward slashes: `D:/Projects`.

### Configure `.env` (both platforms)

Edit `.env` and set real values (never commit it):

```
NAUKRI_EMAIL=your_real_email@example.com
NAUKRI_PASSWORD=your_real_password
OPENAI_API_KEY=sk-...
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/vai_hub
```

Replace the placeholder values; the app rejects `your_email` / `your_password` as-is. Set
`CA_WORKSPACE_ROOTS` to a folder that exists on your machine, e.g. `/Users/<you>/Project`
on macOS or `D:/Projects` on Windows. To run without OpenAI, set `LLM_PROVIDER=ollama` and see
`docs/LLM_PROVIDER.md`.

## Run (local)

Activate the venv first (`source .venv/bin/activate` on macOS, `.\.venv\Scripts\Activate.ps1` on Windows), then:

| | macOS | Windows |
|---|---|---|
| Run | `python app.py` | `python app.py` |
| Run with auto-reload | `python dev.py` | `python dev.py` |
| Tests | `pytest` | `pytest` |

Open a browser to **http://127.0.0.1:8000/**

- Home: module cards; **AutoJob Nexus** goes to `/job-search`.
- Job search: a loading indicator runs while `POST /api/jobs/naukri` drives Selenium (login → **`/jobs-in-india`** SRP → parse cards).
- By default the API uses **headless** Chrome (`headless: true`). Send `{"headless": false}` for a visible browser (the job-search page JS does this).
- Debug output is under **`scraper_debug/`** (gitignored): `login_success.png`; if **zero** jobs are found, `debug_jobs.html` + `jobs_debug.png`.
- **macOS:** port 8000 is normally free; if something else uses it, set `FLASK_PORT=8001` in `.env`. (Port 5000 is taken by AirPlay Receiver, so avoid it.)
- **Windows:** allow Python through Windows Defender Firewall if prompted (only needed for access from other devices).

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

From the project root (same commands in macOS Terminal and Windows PowerShell):

```bash
docker compose up --build
```

First build can take several minutes (PyTorch, sentence-transformers, browsers). Leave the terminal open.

Open **http://127.0.0.1:8000/**

Stop with `Ctrl+C`, then:

```bash
docker compose down
```

Run in the background:

```bash
docker compose up --build -d
```

Logs: `docker compose logs -f web`

### 3. Without Compose

```bash
docker build -t viru-ai-hub .
docker run --rm -p 8000:8000 --shm-size=2g --env-file .env viru-ai-hub
```

`--shm-size=2g` is required so Chrome does not crash inside the container. Compose already sets `shm_size: 2gb`.

### Docker notes

- Job search in the container is **headless** (`NAUKRI_HEADLESS=1`). A visible Chrome window is not available.
- Compose mounts `Profile`, `output`, `scraper_debug`, and RAG upload/data folders so files persist on the host.
- Image is **amd64** (Google Chrome `.deb`). On **Apple Silicon Macs**, build and run it under emulation:
  enable **Settings → General → "Use Rosetta for x86_64/amd64 emulation"** in Docker Desktop, then
  `DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose up --build`. Expect slower builds and runs than on Intel.
- On Windows, start **Docker Desktop** (WSL 2 backend) before `docker compose up`. Keep the project on a
  local drive; if volume mounts fail, enable file sharing for that drive in Docker Desktop settings.
- Both platforms: start Docker Desktop before running any `docker` command.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Flask factory and dev server |
| `Dockerfile` / `docker-compose.yml` | Container image and Compose service |
| `routes/main_routes.py` | Pages + JSON API |
| `services/naukri_service.py` | `login_to_naukri()`, `fetch_jobs()`, `run_login_and_fetch_jobs()` |
| `jobEngine/` | AI Job Discovery & Matching Engine — LangGraph-orchestrated, config-driven multi-source job scraping + resume matching. See `docs/job_engine/ARCHITECTURE.md`. |
| `config/job_sources.yaml` | Job sources for `jobEngine` (LinkedIn, Instinctools, ...) |
| `templates/` | Jinja2 HTML |
| `static/` | CSS / JS |

## AI Job Discovery & Matching Engine

`jobEngine/` scrapes multiple configured job websites in parallel via
Playwright, normalizes and deduplicates the results, and scores each job
against `Profile/Summary.txt` using a rule + embedding + LLM pipeline with
an explainable, configurable score. Endpoints live under `/api/jobs/*`
(`POST /api/jobs/search`, `GET /api/jobs/search/<run_id>`, `GET /api/jobs/sources`,
`GET /api/jobs/health`, ...). Full docs: `docs/job_engine/`.

## Optional environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `NAUKRI_USE_UNDETECTED` | `1` | Use **undetected-chromedriver** when `1`; set `0` to force stock Selenium + webdriver-manager. |
| `NAUKRI_STEP_PAUSE` | `1.2` | Extra delay (seconds) between major steps when the browser is **visible**. |
| `NAUKRI_STEP_PAUSE_HEADLESS` | `0.25` | Same when `headless: true`. |
| `NAUKRI_HEADLESS` | (Compose: `1`) | Playwright job search: `1` = headless (required in Docker). |
| `FLASK_HOST` / `FLASK_PORT` | `127.0.0.1` / `8000` | Dev server bind. Docker sets `0.0.0.0:8000`. |

## Notes

- Naukri may show **CAPTCHA**, **OTP**, or change HTML; update selectors in `services/naukri_service.py` if automation breaks.
- First run may take longer while ChromeDriver is downloaded.
- Visible runs are slower by design so you can follow each step; increase timeouts if your network is slow.

## Logging

The app logs to **stdout** (INFO). Automation messages are prefixed with logger names (e.g. `services.naukri_service`).
