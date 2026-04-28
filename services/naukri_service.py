"""
Naukri.com automation: login and job listing extraction via Selenium.

After login, scraping uses **jobs-in-india** (Naukri TopTier / Next.js). Job cards are
`div.flex…rounded-3xl.bg-n800` (not legacy `srp-jobtuple-wrapper`). See _find_toptier_job_cards().
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv()

logger = logging.getLogger(__name__)

NAUKRI_HOME = "https://www.naukri.com"
# NAUKRI_SORT_BY = os.getenv("NAUKRI_SORT_BY", "f")

# # Logged-in SRP: reliable job cards vs dynamic dashboard
# NAUKRI_JOBS_INDIA = "https://www.naukri.com/jobs-in-india?sort=" + NAUKRI_SORT_BY

# Optional: still keep sort configurable
NAUKRI_SORT_BY = os.getenv("NAUKRI_SORT_BY", "r")
NAUKRI_JOB_AGE = os.getenv("NAUKRI_JOB_AGE", "3")

# Base URL for JavaScript jobs with CTC filters
NAUKRI_BASE_URL = "https://www.naukri.com/javascript-jobs"

# Final URL
NAUKRI_JOBS_INDIA = (
    f"{NAUKRI_BASE_URL}"
    f"?sort={NAUKRI_SORT_BY}"
    f"&jobAge={NAUKRI_JOB_AGE}" 
    f"&ctcFilter=25to50&ctcFilter=50to75&ctcFilter=75to100"
)


# Debug artifacts (project root / scraper_debug)
SCRAPER_DEBUG_DIR = Path(__file__).resolve().parent.parent / "scraper_debug"

# Visible mode: pause between major steps (seconds)
STEP_PAUSE_VISIBLE = float(os.getenv("NAUKRI_STEP_PAUSE", "1.2"))
STEP_PAUSE_HEADLESS = float(os.getenv("NAUKRI_STEP_PAUSE_HEADLESS", "0.25"))

PAGE_LOAD_TIMEOUT = int(os.getenv("NAUKRI_PAGE_LOAD_TIMEOUT", "10"))
_no_jobs_env = (os.getenv("NAUKRI_NO_OF_JOBS") or "").strip()
NO_OF_JOBS = int(_no_jobs_env) if _no_jobs_env else 25

# After SRP scrape: visit each JD in the same session (sequential + delays; set 0 to skip).
NAUKRI_ENRICH_JD = os.getenv("NAUKRI_ENRICH_JD", "1").strip().lower() in ("1", "true", "yes")
NAUKRI_JD_PAGE_DELAY = float(os.getenv("NAUKRI_JD_PAGE_DELAY", "0.85"))
NAUKRI_JD_RETRIES = max(1, int(os.getenv("NAUKRI_JD_RETRIES", "3")))
_jd_max_raw = (os.getenv("NAUKRI_MAX_JD_ENRICH") or "").strip()
# 0 = enrich every row returned from SRP; otherwise cap detail-page visits (faster smoke tests).
NAUKRI_MAX_JD_ENRICH = int(_jd_max_raw) if _jd_max_raw else 0


def normalize(skill: str) -> str:
    """
    Normalize skill string:
    - lowercase
    - remove dots
    - remove extra spaces
    """
    skill = skill.lower()
    skill = skill.replace('.', '')
    skill = re.sub(r'\s+', ' ', skill).strip()
    return skill


def map_to_base_skill(skill: str) -> str:
    """
    Map variations to base skills
    """
    if "react" in skill:
        return "react"
    if "next" in skill:
        return "next"
    if "node" in skill:
        return "node"
    if "aws" in skill:
        return "aws"
    if "javascript" in skill:
        return "javascript"
    return skill


def _clean_text_for_skill_analysis(text: str) -> str:
    """Collapse whitespace; cap length for analyze_skills single-string path."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    return t[:50_000]


def _analyze_skill_text_blob(norm: str) -> bool:
    """
    Run target / excluded rules on one normalized blob (e.g. full JD).
    Counts every target that appears anywhere in the text (not only the first map_to_base_skill hit).
    """
    target_skills = {"react", "next", "node", "javascript", "typescript"}
    found_excluded: set[str] = set()
    if re.search(r"\bjava\b", norm):
        found_excluded.add("java")
    if "c#" in norm:
        found_excluded.add("c#")
    if re.search(r"\bnet\b", norm):
        found_excluded.add("net")
    if found_excluded:
        return False

    found_targets: set[str] = set()
    for target in target_skills:
        if target == "javascript":
            if "javascript" in norm or re.search(r"\bjs\b", norm):
                found_targets.add("javascript")
        elif target in norm:
            found_targets.add(target)

    return len(found_targets) >= 3


def analyze_skills(skill_list_or_description: str | Sequence[str]) -> bool:
    """
    Decide if a job matches the target stack. Accepts a list of skill strings, or one
    job-description string (full JD is scanned for multiple targets).
    """
    if isinstance(skill_list_or_description, str):
        raw = _clean_text_for_skill_analysis(skill_list_or_description)
        if not raw:
            return False
        norm = normalize(raw)
        return _analyze_skill_text_blob(norm)

    skill_list = [s for s in skill_list_or_description if s and str(s).strip()]

    # Target and excluded skills
    target_skills = {"react", "next", "node", "javascript", "typescript"}
    found_targets: set[str] = set()
    found_excluded: set[str] = set()

    for skill in skill_list:
        norm = normalize(str(skill))

        # Check excluded first (strict contains but avoid javascript → java issue)
        if re.search(r"\bjava\b", norm):
            found_excluded.add("java")
        if "c#" in norm:
            found_excluded.add("c#")
        if re.search(r"\bnet\b", norm):
            found_excluded.add("net")

        # Map to base skill
        base = map_to_base_skill(norm)

        # Check target match (partial allowed)
        for target in target_skills:
            if target in base:
                found_targets.add(target)

    # Final condition:
    # - No excluded skills
    # - At least 2 target skills
    if len(found_excluded) == 0 and len(found_targets) >= 2:
        return True
    return False


def relevance_from_jd(*, description: str, skills: list[str] | None) -> bool:
    """Merge structured skills and JD text, then apply the same stack rules as the full-text path."""
    parts = list(_coerce_skills_list(skills))
    desc = _clean_text_for_skill_analysis(description)
    if desc:
        parts.append(desc)
    blob = " ".join(parts).strip()
    if not blob:
        return False
    return _analyze_skill_text_blob(normalize(blob))

def _step_pause(headless: bool) -> None:
    """Slow motion in visible mode so you can follow the automation."""
    delay = STEP_PAUSE_HEADLESS if headless else STEP_PAUSE_VISIBLE
    if delay > 0:
        time.sleep(delay)


def _require_credentials() -> tuple[str, str]:
    email = (os.getenv("NAUKRI_EMAIL") or "").strip()
    password = (os.getenv("NAUKRI_PASSWORD") or "").strip()
    if not email or not password:
        raise ValueError(
            "NAUKRI_EMAIL and NAUKRI_PASSWORD must be set in your .env file "
            "(use .env.example as a template)."
        )
    if email == "your_email" or password == "your_password":
        raise ValueError(
            "Replace placeholder values in .env with your real Naukri credentials."
        )
    return email, password


def _ensure_debug_dir() -> Path:
    SCRAPER_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    return SCRAPER_DEBUG_DIR


def _save_debug_html(driver: webdriver.Chrome, filename: str = "debug_page.html") -> Path:
    path = _ensure_debug_dir() / filename
    html = driver.page_source
    path.write_text(html, encoding="utf-8", errors="replace")
    logger.info("Saved page HTML to %s (%s bytes)", path, len(html))
    return path


def create_driver(*, headless: bool = False) -> webdriver.Chrome:
    """
    Chrome with automation flags relaxed. Prefer undetected-chromedriver when installed
    (helps with bot detection); fall back to stock Selenium + webdriver-manager.
    """
    use_uc = os.getenv("NAUKRI_USE_UNDETECTED", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    if use_uc:
        try:
            import undetected_chromedriver as uc

            logger.info("Using undetected-chromedriver")
            options = uc.ChromeOptions()
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-blink-features=AutomationControlled")
            # Do not set excludeSwitches / useAutomationExtension here: Chrome 131+ and
            # matching ChromeDriver reject them as unrecognized goog:chromeOptions keys when
            # passed through uc; undetected-chromedriver already strips automation signals.

            driver = uc.Chrome(
                options=options,
                headless=headless,
                use_subprocess=True,
            )
            driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
            if not headless:
                driver.fullscreen_window()
                logger.info("Chrome window maximized (visible mode)")
            return driver
        except Exception as e:  # noqa: BLE001
            logger.warning("undetected-chromedriver unavailable or failed (%s); using stock Chrome", e)

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1440,720")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    if not headless:
        driver.maximize_window()
        logger.info("Chrome window maximized (visible mode, stock driver)")
    return driver


def _click_first_clickable(
    driver: webdriver.Chrome, locators: list[tuple[str, str]], timeout: float = 12
) -> bool:
    for by, value in locators:
        try:
            el = WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable((by, value))
            )
            el.click()
            logger.info("Clicked: %s=%s", by, value)
            return True
        except TimeoutException:
            logger.debug("Not clickable within %ss: %s=%s", timeout, by, value)
            continue
    return False


def _first_present(
    driver: webdriver.Chrome,
    selectors: list[tuple[str, str]],
    timeout_each: float = 7,
) -> Any | None:
    for by, sel in selectors:
        try:
            return WebDriverWait(driver, timeout_each).until(
                EC.presence_of_element_located((by, sel))
            )
        except TimeoutException:
            continue
    return None


def _wait_logged_in(driver: webdriver.Chrome, timeout: int = 50) -> None:
    """Wait until URL or DOM indicates a logged-in session."""

    def _is_logged_in(d: webdriver.Chrome) -> bool:
        url = (d.current_url or "").lower()
        if "/mnjuser" in url or "mynaukri" in url:
            return True
        # Guest pages often keep you on www.naukri.com without mnjuser
        try:
            if d.find_elements(
                By.CSS_SELECTOR,
                "a[href*='logout'], a[href*='mnjuser/profile'], .nI-gnb-avatar, "
                "img[alt*='profile'], [class*='gnb-user'], [data-gnb-name]",
            ):
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    WebDriverWait(driver, timeout).until(_is_logged_in)
    logger.info("Session appears logged in (URL or nav: %s)", driver.current_url[:80])


def login_to_naukri(driver: webdriver.Chrome, *, headless: bool = False) -> None:
    """
    Home → open login layer → fill credentials → submit → wait for session.
    Uses WebDriverWait for interactive elements; adds visible delays when headless=False.
    """
    email, password = _require_credentials()

    logger.info("Opening Naukri")
    driver.get(NAUKRI_HOME)
    _step_pause(headless)
    WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    logger.info("Logging in — opening login popup/modal")
    login_locators: list[tuple[str, str]] = [
        (By.CSS_SELECTOR, "a.loginLayer"),
        (By.CSS_SELECTOR, 'a[title="Login"]'),
        (By.CSS_SELECTOR, "div.headGNB a.loginLayer"),
        (By.LINK_TEXT, "Login"),
        (By.PARTIAL_LINK_TEXT, "Login"),
        (By.CSS_SELECTOR, "div.headGNB a[href*='login']"),
    ]
    if not _click_first_clickable(driver, login_locators, timeout=PAGE_LOAD_TIMEOUT):
        raise RuntimeError("Could not find or click Login. Inspect Naukri header DOM.")

    _step_pause(headless)

    user_selectors = [
        (By.CSS_SELECTOR, "input#username"),
        (By.CSS_SELECTOR, "input[name='username']"),
        (By.CSS_SELECTOR, "input[type='text'][placeholder*='Email']"),
        (By.CSS_SELECTOR, "input[placeholder*='mail']"),
    ]
    pass_selectors = [
        (By.CSS_SELECTOR, "input#password"),
        (By.CSS_SELECTOR, "input[name='password']"),
        (By.CSS_SELECTOR, "input[type='password']"),
    ]

    user_el = _first_present(driver, user_selectors)
    if user_el is None:
        raise RuntimeError("Username field not found in login modal.")

    pass_el = None
    for by, sel in pass_selectors:
        try:
            pass_el = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((by, sel))
            )
            break
        except TimeoutException:
            continue
    if pass_el is None:
        raise RuntimeError("Password field not found in login modal.")

    WebDriverWait(driver, 10).until(EC.element_to_be_clickable(user_el))
    user_el.clear()
    user_el.send_keys(email)
    _step_pause(headless)
    pass_el.clear()
    pass_el.send_keys(password)
    _step_pause(headless)
    logger.info("Credentials submitted (login form)")

    submit_locators: list[tuple[str, str]] = [
        (By.CSS_SELECTOR, "button[type='submit']"),
        (By.XPATH, "//button[contains(., 'Login')]"),
        (By.CSS_SELECTOR, ".btn-primary.loginButton"),
        (By.CSS_SELECTOR, "button.loginButton"),
    ]
    if not _click_first_clickable(driver, submit_locators, timeout=PAGE_LOAD_TIMEOUT):
        pass_el.send_keys(Keys.RETURN)
        logger.info("Submitted login via Enter on password field")

    _wait_logged_in(driver, timeout=PAGE_LOAD_TIMEOUT*2)
    _step_pause(headless)

    out = _ensure_debug_dir() / "login_success.png"
    driver.save_screenshot(str(out))
    logger.info("Screenshot after login: %s", out)


def _scroll_srp_incremental(driver: webdriver.Chrome) -> None:
    """Scroll to load more cards; brief pauses allow React to render (paired with WebDriverWait elsewhere)."""
    for _ in range(3):
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(2)
    logger.info("Completed scroll pass: 3× scrollBy(0, 1000)")


def _split_csv_trim(s: str) -> list[str]:
    if not s or not str(s).strip():
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


import re
from datetime import datetime, timedelta, timezone

def _extract_posted_text(raw: str) -> int | None:
    """Convert timeline text like '5d ago' into UTC timestamp."""
    if not raw:
        return None

    raw = raw.strip().lower()
    now = datetime.now(timezone.utc)  # ✅ always UTC

    # Match patterns like 5d ago, 2h ago, 1w ago, 3mo ago
    m = re.match(r"^(\d+)\s*(h|d|w|mo)\s*ago\b", raw)

    if m:
        value = int(m.group(1))
        unit = m.group(2)

        if unit == "h":
            dt = now - timedelta(hours=value)
        elif unit == "d":
            dt = now - timedelta(days=value)
        elif unit == "w":
            dt = now - timedelta(weeks=value)
        elif unit == "mo":
            dt = now - timedelta(days=value * 30)  # approx

        return int(dt.timestamp())

    # Special cases
    if "today" in raw or "just now" in raw:
        return int(now.timestamp())

    if "yesterday" in raw:
        return int((now - timedelta(days=1)).timestamp())

    return None


def _absolute_naukri_url(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        return "https://www.naukri.com" + href
    return href


def _is_allowed_naukri_job_url(url: str) -> bool:
    try:
        p = urlparse((url or "").strip())
        if p.scheme not in ("http", "https"):
            return False
        host = (p.netloc or "").lower()
        if host.endswith("naukri.com"):
            host_ok = True
        else:
            host_ok = False
        if not host_ok:
            return False
        return "job-listings" in (p.path or "").lower()
    except Exception:  # noqa: BLE001
        return False


def _find_meta_ul(scope: Any) -> Any | None:
    """Job meta block: ul.mb-3 with li rows identified by img[alt]."""
    for sel in ("ul.mb-3.space-y-2", "ul.mb-3"):
        try:
            uls = scope.find_elements(By.CSS_SELECTOR, sel)
        except Exception:  # noqa: BLE001
            continue
        for ul in uls:
            try:
                if ul.find_elements(By.CSS_SELECTOR, "li img[alt]"):
                    return ul
            except Exception:  # noqa: BLE001
                continue
    return None


def _parse_ul_li_by_icon_alt(ul: Any | None) -> dict[str, str]:
    """
    Walk ul > li; branch on first img[alt] (location, salary, skills, experience).
    Text from span.text-body14R (Naukri TopTier markup).
    """
    out = {"location": "", "salary": "", "skills": "", "experience": ""}
    if ul is None:
        return out
    for li in ul.find_elements(By.CSS_SELECTOR, ":scope > li"):
        try:
            imgs = li.find_elements(By.CSS_SELECTOR, "img[alt]")
            if not imgs:
                continue
            alt = (imgs[0].get_attribute("alt") or "").strip().lower()
            val = ""
            for sp in li.find_elements(By.CSS_SELECTOR, "span.text-body14R"):
                t = (sp.text or "").strip()
                if t:
                    val = t
                    break
            if alt == "location":
                out["location"] = val
            elif alt == "salary":
                out["salary"] = val
            elif alt == "skills":
                out["skills"] = val
            elif alt == "experience":
                out["experience"] = val
        except Exception:  # noqa: BLE001
            continue
    return out


def _extract_posted_from_scope(scope: Any) -> str:
    for css in (
        "p.flex.items-center.text-body12R",
        "p.flex.items-center.pt-1.text-body12R",
        "p.text-body12R.text-n400",
    ):
        try:
            for p in scope.find_elements(By.CSS_SELECTOR, css):
                t = _extract_posted_text(p.text or "")
                if t:
                    return t
        except Exception:  # noqa: BLE001
            continue
    return ""


def _job_url_from_card(card: Any, right: Any) -> str:
    for root in (right, card):
        try:
            for a in root.find_elements(By.CSS_SELECTOR, "a[href*='job-listings']"):
                href = a.get_attribute("href") or ""
                if "job-listings" in href:
                    u = _absolute_naukri_url(href).split("?")[0].rstrip("/")
                    if u:
                        return u
        except Exception:  # noqa: BLE001
            continue
    return ""


def _log_missing_list_fields(row: dict[str, Any], index: int) -> None:
    """Log mandatory / important gaps; do not raise."""
    title_hint = str(row.get("title", ""))[:50]
    if not row.get("location"):
        msg = f"Missing field: location (card {index}, {title_hint!r})"
        logger.warning(msg)
        print(msg, flush=True)
    sal_on_card = row.get("_salary_on_card")
    if sal_on_card is False:
        msg = f"Missing field: package (card {index}, {title_hint!r})"
        logger.warning(msg)
        print(msg, flush=True)
    if not row.get("skills"):
        msg = f"Missing field: skills (card {index}, {title_hint!r})"
        logger.warning(msg)
        print(msg, flush=True)
    if not (row.get("job_url") or "").strip():
        msg = f"Missing field: job_url (card {index}, {title_hint!r})"
        # logger.warning(msg)
        # print(msg, flush=True)


def _strip_internal_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not str(k).startswith("_")}


def _finalize_job_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    _log_missing_list_fields(row, index)
    row = _strip_internal_keys(row)
    ju = (row.get("job_url") or "").strip()
    row["jd_url"] = ju
    row.setdefault("description", "")
    row.setdefault("is_relevant", False)
    return row


def _find_toptier_job_cards(driver: webdriver.Chrome) -> list[Any]:
    """
    Naukri TopTier SRP (jobs-in-india): repeating row is a flex card with rounded-3xl + bg-n800.
    From saved HTML: class includes min-h-[241px], cursor-pointer, mb-[16px].
    Excludes #load-more-btn skeleton (rounded-[24px], animate-pulse).
    """
    xpath = (
        "//div[contains(@class,'rounded-3xl') and contains(@class,'bg-n800') "
        "and contains(@class,'cursor-pointer') and contains(@class,'flex') "
        "and contains(@class,'min-h-')]"
    )
    found = driver.find_elements(By.XPATH, xpath)
    cards: list[Any] = []
    for el in found:
        try:
            eid = el.get_attribute("id") or ""
            cls = el.get_attribute("class") or ""
            if eid == "load-more-btn":
                continue
            if "animate-pulse" in cls:
                continue
            cards.append(el)
        except Exception:  # noqa: BLE001
            continue
    return cards


def _toptier_left_column(card: Any) -> Any:
    try:
        return card.find_element(
            By.XPATH,
            ".//div[contains(@class,'border-r') and contains(@class,'w-[220px]')]",
        )
    except NoSuchElementException:
        return card


def _toptier_right_column(card: Any) -> Any:
    try:
        return card.find_element(
            By.XPATH,
            ".//div[contains(@class,'min-w-[480px]')]",
        )
    except NoSuchElementException:
        return card


def _row_from_toptier_card(card: Any, *, card_index: int = 0) -> dict[str, Any] | None:
    """
    Parse one TopTier job card: ul.mb-3 > li keyed by img[alt] (location, salary, skills, experience),
    posted from p.flex.items-center.text-body12R, job URL from a[href*='job-listings'].
    """
    left = _toptier_left_column(card)
    right = _toptier_right_column(card)

    title = _first_non_empty(
        right,
        [
            "div.box-border.py-5.text-headline24Sb",
            "div.text-headline24Sb.text-n100",
            "div.text-title18Sb.text-n100",
        ],
    )
    company = _first_non_empty(
        left,
        [
            "h4.text-title18Sb.text-n200",
            "h4 div.text-title16Sb.text-n200",
            "h4 div.line-clamp-1.text-title16Sb",
            "h4 div.truncate.text-title16Sb",
            "h4 div.text-title16Sb",
        ],
    )
    posted_by = _text_safe(left, "p.text-body14R.text-n400")
    if not company and posted_by:
        company = posted_by.strip()

    ul = _find_meta_ul(right)
    meta = _parse_ul_li_by_icon_alt(ul)
    salary_raw = (meta.get("salary") or "").strip()
    posted = _extract_posted_from_scope(right) or _extract_posted_from_scope(left)
    job_url = _job_url_from_card(card, right)

    if not title:
        return None

    loc_list = _split_csv_trim(meta.get("location") or "")
    skills_list = _split_csv_trim(meta.get("skills") or "")
    package = salary_raw if salary_raw else "Not Disclosed"
    card_relevant = analyze_skills(skills_list)
    row: dict[str, Any] = {
        "title": title,
        "company": company or "—",
        "location": loc_list,
        "package": package,
        "skills": skills_list,
        "experience": (meta.get("experience") or "").strip() or "—",
        "posted": posted,
        "job_url": job_url,
        "_salary_on_card": bool(salary_raw),
        "is_relevant": card_relevant,
    }
    return _finalize_job_row(row, card_index)


def _log_extracted_jobs(rows: list[dict[str, Any]]) -> None:
    print(f"Total job cards extracted: {len(rows)}", flush=True)
    logger.info("Total job cards extracted: %s", len(rows))
    for i, row in enumerate(rows, start=1):
        loc = row.get("location")
        loc_s = ", ".join(loc) if isinstance(loc, list) else repr(loc)
        skills = row.get("skills")
        sk_s = ", ".join(skills) if isinstance(skills, list) else repr(skills)
        line = (
            f"  [{i}] title={row.get('title')!r} | company={row.get('company')!r} | "
            f"location={loc_s!r} | package={row.get('package')!r} | skills={sk_s!r} | "
            f"experience={row.get('experience')!r} | posted={row.get('posted')!r} | "
            f"jd_url={row.get('jd_url')!r} | is_relevant={row.get('is_relevant')!r}"
        )
        print(line, flush=True)
        logger.info(line.strip())
        missing = [
            k
            for k in ("title", "company", "experience")
            if not row.get(k) or row.get(k) == "—"
        ]
        if missing:
            msg = f"  [{i}] missing or placeholder fields: {missing}"
            print(msg, flush=True)
            logger.warning(msg)


def _text_safe(parent: Any, css: str) -> str:
    try:
        return parent.find_element(By.CSS_SELECTOR, css).text.strip()
    except NoSuchElementException:
        return ""


def _first_non_empty(parent: Any, selectors: list[str]) -> str:
    for css in selectors:
        t = _text_safe(parent, css)
        if t:
            return t
    return ""


def _row_from_tuple_card(card: Any, *, card_index: int = 0) -> dict[str, Any] | None:
    """Parse one SRP job card; multiple selector fallbacks for Naukri DOM churn."""
    title = _first_non_empty(
        card,
        [
            "a.title",
            "h2 a",
            "h2.title a",
            ".title a",
            "h3 a",
            "a[href*='/job-listings/']",
            ".title",
            "h2",
        ],
    )
    company = _first_non_empty(
        card,
        [
            "a.comp-name",
            "span.comp-name",
            "div.comp-name",
            ".comp-name",
            "[class*='comp-name']",
            "[class*='companyName']",
            "a[class*='comp']",
        ],
    )
    location = _first_non_empty(
        card,
        [
            "span.loc-wrap",
            ".loc-wrap",
            "span.locWdth",
            "[class*='loc-wrap']",
            "span[class*='location']",
            ".location",
        ],
    )
    experience = _first_non_empty(
        card,
        [
            "span.exp-wrap",
            ".exp-wrap",
            "span.expwdth",
            "[class*='exp-wrap']",
            "span[class*='experience']",
        ],
    )
    job_url = ""
    try:
        for a in card.find_elements(By.CSS_SELECTOR, "a[href*='job-listings']"):
            job_url = _absolute_naukri_url(a.get_attribute("href") or "").split("?")[0].rstrip("/")
            if job_url:
                break
    except Exception:  # noqa: BLE001
        pass

    if not title and not company:
        return None
    print(1, job_url)
    ul = _find_meta_ul(card)
    meta = _parse_ul_li_by_icon_alt(ul)
    if meta.get("location"):
        location = meta["location"]
    if meta.get("experience"):
        experience = meta["experience"]
    salary_raw = (meta.get("salary") or "").strip()
    skills_raw = meta.get("skills") or ""
    package = salary_raw if salary_raw else "Not Disclosed"
    loc_list = _split_csv_trim(location) if location else []
    if not loc_list and location and str(location).strip() and str(location) != "—":
        loc_list = [str(location).strip()]
    sk_list = _split_csv_trim(skills_raw)
    card_relevant = analyze_skills(sk_list)

    row: dict[str, Any] = {
        "title": title or "—",
        "company": company or "—",
        "location": loc_list,
        "package": package,
        "skills": sk_list,
        "experience": experience or "—",
        "posted": _extract_posted_from_scope(card),
        "job_url": job_url,
        "_salary_on_card": bool(salary_raw),
        "is_relevant": card_relevant,
    }
    return _finalize_job_row(row, card_index)


def _extract_from_job_link_cards(driver: webdriver.Chrome, max_items: int) -> list[dict[str, Any]]:
    """
    Homepage / reco feeds often render job title as <a href*=/job-listings/>.
    Walk up a few ancestors to pick comp / loc / exp if present.
    """
    anchors = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/job-listings/"]')
    logger.info("Found %s raw job-listing links on page", len(anchors))
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    for a in anchors:
        try:
            href = (a.get_attribute("href") or "").split("?")[0].rstrip("/")
            if not href or href in seen:
                continue
            title = (a.text or "").strip()
            if len(title) < 2:
                continue
            seen.add(href)

            company, location, experience = "", "", ""
            node = a
            for _ in range(14):
                try:
                    node = node.find_element(By.XPATH, "./..")
                except NoSuchElementException:
                    break
                if not company:
                    company = (
                        _text_safe(node, "a.comp-name")
                        or _text_safe(node, ".comp-name")
                        or _text_safe(node, "[class*='comp-name']")
                        or _text_safe(node, "[class*='company']")
                    )
                if not location:
                    location = (
                        _text_safe(node, "span.loc-wrap")
                        or _text_safe(node, "[class*='loc-wrap']")
                        or _text_safe(node, "[class*='location']")
                    )
                if not experience:
                    experience = (
                        _text_safe(node, "span.exp-wrap")
                        or _text_safe(node, "[class*='exp-wrap']")
                        or _text_safe(node, "[class*='experience']")
                    )
                if company and location and experience:
                    break

            loc_list = _split_csv_trim(location) if location else []
            if not loc_list and location and str(location).strip():
                loc_list = [str(location).strip()]
            row: dict[str, Any] = {
                "title": title,
                "company": company or "—",
                "location": loc_list,
                "package": "Not Disclosed",
                "skills": [],
                "experience": experience or "—",
                "posted": "",
                "job_url": href,
                "_salary_on_card": False,
                "is_relevant": False,
            }
            rows.append(_finalize_job_row(row, len(rows) + 1))
        except Exception:  # noqa: BLE001
            logger.debug("Skipping broken job link row", exc_info=True)
            continue
        if len(rows) >= max_items:
            break

    return rows


def _extract_job_rows(driver: webdriver.Chrome, max_items: int) -> list[dict[str, Any]]:
    """
    Prefer Naukri TopTier cards (jobs-in-india Next.js layout from live HTML),
    then legacy srp-jobtuple-wrapper, then job-listing links.
    """
    tt_cards = _find_toptier_job_cards(driver)
    print(f"TopTier job card nodes found: {len(tt_cards)}", flush=True)
    logger.info("TopTier job card nodes found: %s", len(tt_cards))

    if len(tt_cards) >= 1:
        out_tt: list[dict[str, Any]] = []
        for idx, card in enumerate(tt_cards[: max_items * 2], start=1):
            try:
                row = _row_from_toptier_card(card, card_index=idx)
                if row and row.get("is_relevant"):
                    out_tt.append(row)
            except Exception:  # noqa: BLE001
                logger.debug("Skipping broken TopTier card", exc_info=True)
                continue
            if len(out_tt) >= max_items:
                break
        if out_tt:
            _log_extracted_jobs(out_tt)
            return out_tt

    card_selectors = [
        "div.srp-jobtuple-wrapper",
        "div.cust-job-tuple",
        "div.jobTuple",
        "article.jobTuple",
        "[class*='jobTuple']",
        "li.srp-jobtuple-wrapper",
        "li[class*='tuple']",
        "li[class*='job']",
    ]
    for sel in card_selectors:
        cards = driver.find_elements(By.CSS_SELECTOR, sel)
        if len(cards) >= 1:
            logger.info("Using legacy card selector %r — %s nodes", sel, len(cards))
            out: list[dict[str, Any]] = []
            for idx, card in enumerate(cards[: max_items * 2], start=1):
                try:
                    row = _row_from_tuple_card(card, card_index=idx)
                    if row and row.get("is_relevant"):
                        out.append(row)
                except Exception:  # noqa: BLE001
                    logger.debug("Skipping broken job card", exc_info=True)
                    continue
                if len(out) >= max_items:
                    break
            if out:
                _log_extracted_jobs(out)
                return out

    logger.warning("TopTier + tuple selectors found no rows; falling back to job-listings links")
    fallback = _extract_from_job_link_cards(driver, max_items)
    _log_extracted_jobs(fallback)
    return fallback


def scrape_job_detail_page(driver: webdriver.Chrome, url: str) -> dict[str, Any]:
    """
    Full job detail under #jobs-desc (TopTier JD): same ul.mb-3 icon rows, Job Description block,
    Compensation lines (p[class*='ml-1.5']).
    """
    driver.get(url)
    wait = WebDriverWait(driver, 50)
    wait.until(EC.presence_of_element_located((By.ID, "jobs-desc")))
    root = driver.find_element(By.ID, "jobs-desc")

    title = _first_non_empty(
        root,
        [
            "div.box-border.py-5.text-headline24Sb",
            "div.text-headline24Sb.text-n100",
        ],
    )
    company = _first_non_empty(
        root,
        [
            "h4.text-title18Sb.text-n200",
            "p.text-title18Sb.text-n200",
        ],
    )

    ul = _find_meta_ul(root)
    meta = _parse_ul_li_by_icon_alt(ul)
    posted = _extract_posted_from_scope(root)

    description = ""
    try:
        for hdr in root.find_elements(
            By.XPATH,
            ".//p[contains(normalize-space(.),'Job Description')]",
        ):
            try:
                sibs = hdr.find_elements(By.XPATH, "./following-sibling::div[1]")
                if sibs:
                    description = (sibs[0].text or "").strip()
                    if description:
                        break
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    if not description:
        try:
            for div in root.find_elements(By.CSS_SELECTOR, "div.text-title16R.text-n300"):
                t = (div.text or "").strip()
                if len(t) > 120:
                    description = t
                    break
        except Exception:  # noqa: BLE001
            pass

    compensation_lines: list[str] = []
    try:
        for p in root.find_elements(By.CSS_SELECTOR, "p[class*='ml-1.5']"):
            t = (p.text or "").strip()
            if t:
                compensation_lines.append(t)
    except Exception:  # noqa: BLE001
        pass

    salary_raw = (meta.get("salary") or "").strip()
    clean_url = _absolute_naukri_url(url).split("?")[0].rstrip("/")

    return {
        "title": title or "",
        "company": company or "",
        "location": _split_csv_trim(meta.get("location") or ""),
        "package": salary_raw or "Not Disclosed",
        "skills": _split_csv_trim(meta.get("skills") or ""),
        "experience": (meta.get("experience") or "").strip(),
        "posted": posted,
        "job_url": clean_url,
        "description": description,
        "compensation": compensation_lines,
    }


def _coerce_skills_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(s).strip() for s in val if s and str(s).strip()]
    return []


def enrich_jobs_with_job_details(
    driver: webdriver.Chrome,
    jobs: list[dict[str, Any]],
    *,
    headless: bool,
) -> list[dict[str, Any]]:
    """
    Reuse the logged-in driver: open each jd_url, scrape JD, merge fields, set is_relevant.
    Sequential navigation with delays and per-URL retries (rate-limit friendly).
    """
    limit = NAUKRI_MAX_JD_ENRICH
    logger.info(
        "Enriching %s jobs with JD pages (delay=%ss, retries=%s, max_detail=%s)",
        len(jobs),
        NAUKRI_JD_PAGE_DELAY,
        NAUKRI_JD_RETRIES,
        limit if limit > 0 else "all",
    )

    for idx, job in enumerate(jobs):
        url = (job.get("jd_url") or job.get("job_url") or "").strip()
        job["jd_url"] = url
        skills_before = _coerce_skills_list(job.get("skills"))

        if limit > 0 and idx >= limit:
            job["is_relevant"] = relevance_from_jd(
                description=job.get("description") or "",
                skills=skills_before,
            )
            continue

        if not url or not _is_allowed_naukri_job_url(url):
            job.setdefault("description", "")
            job["is_relevant"] = relevance_from_jd(
                description=job.get("description") or "",
                skills=skills_before,
            )
            continue

        detail: dict[str, Any] | None = None
        last_err: BaseException | None = None
        for attempt in range(NAUKRI_JD_RETRIES):
            try:
                detail = scrape_job_detail_page(driver, url)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning(
                    "JD scrape attempt %s/%s failed for %s: %s",
                    attempt + 1,
                    NAUKRI_JD_RETRIES,
                    url,
                    e,
                )
                time.sleep(NAUKRI_JD_PAGE_DELAY * (attempt + 1))
                _step_pause(headless)

        if detail is None and last_err is not None:
            logger.error("Giving up JD scrape for %s: %s", url, last_err)

        if detail:
            job["description"] = (detail.get("description") or "").strip()
            if detail.get("skills"):
                job["skills"] = detail["skills"]
            pkg = (detail.get("package") or "").strip()
            if pkg and pkg != "Not Disclosed":
                job["package"] = pkg
            exp = (detail.get("experience") or "").strip()
            if exp:
                job["experience"] = exp
            if (detail.get("title") or "").strip():
                job["title"] = detail["title"]
            if (detail.get("company") or "").strip():
                job["company"] = detail["company"]
            if detail.get("location"):
                job["location"] = detail["location"]
            ju = (detail.get("job_url") or url).strip()
            job["job_url"] = ju
            job["jd_url"] = ju
        else:
            job.setdefault("description", "")

        job["is_relevant"] = relevance_from_jd(
            description=job.get("description") or "",
            skills=_coerce_skills_list(job.get("skills")),
        )

        time.sleep(NAUKRI_JD_PAGE_DELAY)
        _step_pause(headless)

    return jobs


def run_job_detail(*, url: str, headless: bool = True) -> dict[str, Any]:
    """Login (session) then open job URL and scrape JD (#jobs-desc)."""
    url = (url or "").strip()
    if not _is_allowed_naukri_job_url(url):
        raise ValueError(
            "url must be an https://www.naukri.com URL whose path contains job-listings"
        )
    driver = create_driver(headless=headless)
    try:
        login_to_naukri(driver, headless=headless)
        return scrape_job_detail_page(driver, url)
    finally:
        try:
            driver.quit()
        except Exception:  # noqa: BLE001
            logger.exception("Error while closing WebDriver (job detail)")


def fetch_jobs(
    driver: webdriver.Chrome,
    *,
    max_items: int = NO_OF_JOBS,
    headless: bool = False,
) -> list[dict[str, Any]]:
    """
    After login: open jobs-in-india SRP, wait for tuples, scroll, extract rows.
    Retries scroll + extract once if the first pass returns zero jobs.
    """
    wait = WebDriverWait(driver, 15)

    logger.info("Navigating to jobs-in-india")
    driver.get(NAUKRI_JOBS_INDIA)
    _step_pause(headless)

    try:
        wait.until(EC.presence_of_element_located((By.ID, "jobs-list-header")))
        logger.info("jobs-list-header present (WebDriverWait)")
    except TimeoutException:
        logger.warning("Timeout waiting for #jobs-list-header")

    try:
        wait.until(
            lambda d: len(_find_toptier_job_cards(d)) >= 1
            or len(d.find_elements(By.CSS_SELECTOR, "div.srp-jobtuple-wrapper")) >= 1
        )
        logger.info("At least one job card row detected (WebDriverWait)")
    except TimeoutException:
        logger.warning("Timeout waiting for first job card — continuing with scroll")

    _scroll_srp_incremental(driver)
    n_cards = len(_find_toptier_job_cards(driver))
    n_legacy = len(driver.find_elements(By.CSS_SELECTOR, "div.srp-jobtuple-wrapper"))
    if n_cards < NO_OF_JOBS - 1 and n_legacy < NO_OF_JOBS - 1:
        logger.info(
            "Few cards visible (TopTier=%s, legacy=%s); running another scroll pass",
            n_cards,
            n_legacy,
        )
        _scroll_srp_incremental(driver)

    _step_pause(headless)

    need = max(max_items, 10)
    jobs = _extract_job_rows(driver, need)
    logger.info("Found %s jobs (pass 1)", len(jobs))

    scroll_attempts = 0
    while len(jobs) < NO_OF_JOBS - 1 :
        scroll_attempts += 1
        logger.info(
            "Fewer than %s jobs (%s); extra scroll pass %s",
            NO_OF_JOBS - 1,
            len(jobs),
            scroll_attempts,
        )
        _scroll_srp_incremental(driver)
        _step_pause(headless)
        jobs = _extract_job_rows(driver, need)
        logger.info("After extra scroll: %s jobs", len(jobs))

    if len(jobs) == 0:
        logger.info("Retry: reload jobs-in-india and scroll again")
        driver.get(NAUKRI_JOBS_INDIA)
        _step_pause(headless)
        try:
            wait.until(EC.presence_of_element_located((By.ID, "jobs-list-header")))
        except TimeoutException:
            pass
        _scroll_srp_incremental(driver)
        _scroll_srp_incremental(driver)
        jobs = _extract_job_rows(driver, need)
        logger.info("Found %s jobs (pass 2)", len(jobs))
        scroll_attempts = 0
        while len(jobs) < 10 and scroll_attempts < 6:
            scroll_attempts += 1
            _scroll_srp_incremental(driver)
            _step_pause(headless)
            jobs = _extract_job_rows(driver, need)

    if len(jobs) == 0:
        msg = "Selector mismatch"
        print(msg, flush=True)
        logger.error(msg)
        _save_debug_html(driver, "debug_jobs.html")
        debug_shot = _ensure_debug_dir() / "jobs_debug.png"
        driver.save_screenshot(str(debug_shot))
        logger.error("Saved %s and %s for inspection", SCRAPER_DEBUG_DIR / "debug_jobs.html", debug_shot)
        snippet = driver.page_source[:3000].replace("\n", " ")
        logger.error("Page source preview (3000 chars): %s", snippet)

    return jobs


def run_login_and_fetch_jobs(
    *,
    headless: bool = True,
    enrich_jd: bool | None = None,
) -> list[dict[str, Any]]:
    """
    Full pipeline. Default headless=True for servers; pass headless=False to watch the browser.
    Returns job dicts with jd_url, description (when enrich on), skills, is_relevant, plus
    title, company, location, package, experience, posted, job_url.
    """
    do_enrich = NAUKRI_ENRICH_JD if enrich_jd is None else bool(enrich_jd)
    driver = create_driver(headless=headless)
    try:
        login_to_naukri(driver, headless=headless)
        jobs = fetch_jobs(driver, max_items=NO_OF_JOBS, headless=headless)
        logger.info("Found %s jobs (SRP)", len(jobs))
        if do_enrich and jobs:
            jobs = enrich_jobs_with_job_details(driver, jobs, headless=headless)
        return jobs
    finally:
        try:
            driver.quit()
        except Exception:  # noqa: BLE001
            logger.exception("Error while closing WebDriver")
