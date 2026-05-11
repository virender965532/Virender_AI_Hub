"""Inject a styled summary panel into the active Playwright page."""

from __future__ import annotations

import logging
from typing import Any

from playwright.async_api import Page

from ..state import JobRecord, WorkflowState

logger = logging.getLogger(__name__)


_PANEL_SCRIPT = r"""
(data) => {
  const jobs = data.jobs || [];
  const errors = data.errors || [];
  const OLD_ID = "langgraph-naukri-job-panel";
  const prev = document.getElementById(OLD_ID);
  if (prev && prev.parentNode) prev.parentNode.removeChild(prev);

  const root = document.createElement("div");
  root.id = OLD_ID;
  root.style.cssText = [
    "position:fixed",
    "top:16px",
    "right:16px",
    "width:min(440px,calc(100vw - 32px))",
    "max-height:min(82vh,calc(100vh - 32px))",
    "overflow:auto",
    "z-index:2147483647",
    "background:#0f172a",
    "color:#e2e8f0",
    "font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,sans-serif",
    "font-size:13px",
    "line-height:1.35",
    "border-radius:12px",
    "box-shadow:0 18px 60px rgba(0,0,0,.45)",
    "border:1px solid rgba(148,163,184,.35)",
    "padding:14px 14px 12px",
  ].join(";");

  const header = document.createElement("div");
  header.style.cssText =
    "display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px;";
  const h = document.createElement("div");
  h.textContent = "LangGraph — Naukri jobs";
  h.style.cssText = "font-weight:650;font-size:14px;color:#f8fafc;";
  const badge = document.createElement("div");
  badge.textContent = String(jobs.length) + " roles";
  badge.style.cssText =
    "font-size:12px;color:#0f172a;background:#38bdf8;padding:3px 8px;border-radius:999px;font-weight:650;";
  header.appendChild(h);
  header.appendChild(badge);
  root.appendChild(header);

  if (errors && errors.length) {
    const errBox = document.createElement("div");
    errBox.style.cssText =
      "margin:8px 0 10px;padding:8px 10px;border-radius:8px;background:rgba(248,113,113,.12);border:1px solid rgba(248,113,113,.35);color:#fecaca;";
    errBox.textContent = errors.join(" · ");
    root.appendChild(errBox);
  }

  const list = document.createElement("div");
  list.style.cssText = "display:flex;flex-direction:column;gap:10px;";

  jobs.forEach((j, idx) => {
    const card = document.createElement("div");
    card.style.cssText =
      "padding:10px 10px;border-radius:10px;background:rgba(15,23,42,.65);border:1px solid rgba(148,163,184,.22);";

    const titleRow = document.createElement("div");
    titleRow.style.cssText = "display:flex;gap:8px;align-items:flex-start;justify-content:space-between;";
    const left = document.createElement("div");
    left.style.cssText = "min-width:0;";
    const ti = document.createElement("div");
    ti.textContent = String(idx + 1) + ". " + String(j.title || "—");
    ti.style.cssText = "font-weight:650;color:#f1f5f9;word-break:break-word;";
    left.appendChild(ti);

    const co = document.createElement("div");
    co.textContent = String(j.company || "—");
    co.style.cssText = "margin-top:4px;color:#94a3b8;font-size:12px;word-break:break-word;";
    left.appendChild(co);

    titleRow.appendChild(left);

    if (j.link) {
      const a = document.createElement("a");
      a.href = j.link;
      a.textContent = "Open";
      a.style.cssText =
        "flex:0 0 auto;color:#38bdf8;font-weight:650;text-decoration:none;padding:4px 8px;border-radius:8px;border:1px solid rgba(56,189,248,.35);background:rgba(56,189,248,.08);";
      titleRow.appendChild(a);
    }

    card.appendChild(titleRow);

    const meta = document.createElement("div");
    meta.style.cssText = "margin-top:8px;display:grid;grid-template-columns:92px 1fr;gap:6px 10px;font-size:12px;color:#cbd5e1;";
    function row(label, val) {
      const l = document.createElement("div");
      l.textContent = label;
      l.style.cssText = "color:#94a3b8;";
      const v = document.createElement("div");
      v.textContent = String(val || "—");
      v.style.cssText = "word-break:break-word;";
      meta.appendChild(l);
      meta.appendChild(v);
    }
    row("Experience", j.experience);
    row("Location", j.location);
    row("Salary", j.salary);
    card.appendChild(meta);

    list.appendChild(card);
  });

  if (!jobs.length) {
    const empty = document.createElement("div");
    empty.style.cssText =
      "padding:14px;border-radius:10px;background:rgba(30,41,59,.55);border:1px dashed rgba(148,163,184,.35);color:#cbd5e1;";
    empty.textContent = "No job cards were extracted. Check filters, DOM changes, or login state.";
    list.appendChild(empty);
  }

  root.appendChild(list);

  const foot = document.createElement("div");
  foot.style.cssText =
    "margin-top:10px;font-size:11px;color:#64748b;";
  foot.textContent = "Injected by LangGraph display_jobs_node · Close by deleting this panel in DevTools if needed.";
  root.appendChild(foot);

  document.body.appendChild(root);
}
"""


async def _inject_panel(page: Page, jobs: list[JobRecord], errors: list[str]) -> None:
    payload_jobs: list[dict[str, str]] = [
        {
            "title": j.get("title") or "",
            "company": j.get("company") or "",
            "experience": j.get("experience") or "",
            "location": j.get("location") or "",
            "salary": j.get("salary") or "",
            "link": j.get("link") or "",
        }
        for j in jobs
    ]
    await page.evaluate(_PANEL_SCRIPT, {"jobs": payload_jobs, "errors": errors})


async def display_jobs_node(state: WorkflowState) -> dict[str, Any]:
    errs = list(state.get("errors") or [])
    session = state.get("session")
    jobs = list(state.get("jobs") or [])

    if session is None:
        errs.append("display_jobs_node: no Playwright session.")
        logger.error("display_jobs_node: missing session")
        return {"display_complete": False, "errors": errs}

    page: Page = session.page
    try:
        await _inject_panel(page, jobs, errs)
        logger.info("Injected job panel into page (%s jobs).", len(jobs))
        return {"display_complete": True, "errors": errs}
    except Exception as e:  # noqa: BLE001
        logger.exception("display_jobs_node failed")
        errs.append(str(e))
        return {"display_complete": False, "errors": errs}
