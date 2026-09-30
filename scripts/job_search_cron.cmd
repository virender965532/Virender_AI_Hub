@echo off
rem Hourly scheduled job search + Pushover notification (see jobEngine/cron.py).
rem Registered with Windows Task Scheduler; output is appended to output\logs\job_search_cron.log.
cd /d "%~dp0.."
if not exist "output\logs" mkdir "output\logs"
rem UTF-8 mode: job text (emoji, non-Latin scripts) must not crash logging to the file.
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m jobEngine.cron >> "output\logs\job_search_cron.log" 2>&1
