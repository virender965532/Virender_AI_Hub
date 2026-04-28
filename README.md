# Virender AI Hub

Futuristic Flask web app with an **AutoJob Nexus** module that logs into [Naukri.com](https://www.naukri.com) via Selenium (credentials from `.env`) and displays extracted job rows.

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **Google Chrome** installed (Selenium uses Chrome via `webdriver-manager`)

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

## Run

```powershell
python app.py
```

Open a browser to **http://127.0.0.1:5000/**

- Home: module cards; **AutoJob Nexus** goes to `/job-search`.
- Job search: a loading indicator runs while `POST /api/jobs/naukri` drives Selenium (login → **`/jobs-in-india`** SRP → parse cards).
- By default the API uses **headless** Chrome (`headless: true`). Send `{"headless": false}` for a visible browser (the job-search page JS does this).
- Debug output is under **`scraper_debug/`** (gitignored): `login_success.png`; if **zero** jobs are found, `debug_jobs.html` + `jobs_debug.png`.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Flask factory and dev server |
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

## Notes

- Naukri may show **CAPTCHA**, **OTP**, or change HTML; update selectors in `services/naukri_service.py` if automation breaks.
- First run may take longer while ChromeDriver is downloaded.
- Visible runs are slower by design so you can follow each step; increase timeouts if your network is slow.

## Logging

The app logs to **stdout** (INFO). Automation messages are prefixed with logger names (e.g. `services.naukri_service`).
