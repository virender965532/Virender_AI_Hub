"""Gradio UI, mounted into the Flask app under ``{SF_URL_PREFIX}/ui``.

Tabs: Market Pulse, Opportunities (live progress), Analyze a Stock (price chart
+ indicators), My Portfolio (editable holdings, CSV upload, review with
allocation chart and concentration warnings) and Chat. The disclaimer is shown
on every tab and every result shows "data as of <timestamp>".

Mounting (option A from the spec): Gradio is ASGI and Flask is WSGI, so the
Gradio app is wrapped with ``a2wsgi`` and attached with werkzeug's
``DispatcherMiddleware``; only paths under the UI prefix are intercepted. Same
approach as ``code_assistant/ui.py``; the Gradio app is only built on the first
request to the UI, and it calls the pipeline in-process (no HTTP hop).
"""

from __future__ import annotations

import asyncio
import html
import logging
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .config import RISK_PROFILES, StockFinderSettings, get_settings
from .schemas import Citation, CompanyAnalysis, FinalReport, HoldingIn, HoldingReview, ScoreBreakdown

logger = logging.getLogger(__name__)

_running_lifespans: list[Any] = []
_DONE = object()

_HUB = {
    "bg_deep": "#05060d", "bg_panel": "#080a16", "bg_input": "rgba(8, 10, 22, 0.85)",
    "border": "rgba(120, 160, 255, 0.22)", "border_strong": "rgba(120, 160, 255, 0.35)",
    "neon_blue": "#38b6ff", "neon_purple": "#b967ff", "neon_cyan": "#5cf0e8",
    "text": "#e8ecff", "text_muted": "#98a2c9",
}
# Chart palette: categorical slots 1-3 (dark steps), validated on the chart surface #0c0e1c.
_SERIES = ("#3987e5", "#d95926", "#199e70")
_CHART_SURFACE = "#0c0e1c"
_GRID = "rgba(152, 162, 201, 0.14)"

VERDICT_BADGES = {
    "STRONG_BUY_CANDIDATE": ("Strong buy candidate", "#4ade80"),
    "WATCHLIST": ("Watchlist", "#ffc857"),
    "AVOID": ("Avoid", "#ff6b81"),
    "BUY_MORE": ("Buy more", "#4ade80"),
    "HOLD": ("Hold", "#38b6ff"),
    "TRIM": ("Trim", "#ffc857"),
    "SELL": ("Sell", "#ff6b81"),
}
HOLDING_COLUMNS = ["symbol", "quantity", "avg_cost", "buy_date", "thesis"]

CSS = """
.gradio-container { font-family: "Rajdhani", system-ui, sans-serif !important; max-width: 100% !important; }
.sf-disclaimer { border: 1px solid rgba(255, 200, 87, 0.45); border-radius: 10px; padding: 6px 12px;
    background: rgba(255, 200, 87, 0.07); font-size: 0.88rem; }
.sf-badge { display: inline-block; padding: 1px 10px; border-radius: 999px; font-weight: 600;
    border: 1px solid currentColor; font-size: 0.85rem; margin-right: 6px; }
:root { color-scheme: dark; }
footer { display: none !important; }
"""


def build_theme() -> Any:
    import gradio as gr

    values = {
        "body_background_fill": _HUB["bg_panel"], "body_text_color": _HUB["text"],
        "body_text_color_subdued": _HUB["text_muted"], "background_fill_primary": _HUB["bg_panel"],
        "background_fill_secondary": "rgba(12, 14, 28, 0.85)", "block_background_fill": "rgba(12, 14, 28, 0.55)",
        "block_border_color": _HUB["border"], "block_label_text_color": _HUB["text_muted"],
        "block_title_text_color": _HUB["text"], "border_color_primary": _HUB["border"],
        "input_background_fill": _HUB["bg_input"], "input_border_color": _HUB["border_strong"],
        "button_primary_background_fill": f"linear-gradient(120deg, {_HUB['neon_blue']}, {_HUB['neon_purple']})",
        "button_primary_text_color": _HUB["bg_deep"],
        "button_secondary_background_fill": "rgba(56, 182, 255, 0.1)", "button_secondary_text_color": _HUB["neon_cyan"],
        "color_accent": _HUB["neon_purple"], "code_background_fill": "rgba(8, 10, 22, 0.85)",
    }
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.purple, secondary_hue=gr.themes.colors.sky, neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Rajdhani"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )
    dark = {f"{key}_dark": value for key, value in values.items() if hasattr(theme, f"{key}_dark")}
    return theme.set(**values, **dark)


# --- rendering helpers (model text is escaped: it is untrusted) -----------------------------------

def esc(text: Any) -> str:
    """Escape model / news text for Markdown: no raw HTML, no injected links."""
    return html.escape(str(text or "")).replace("[", "&#91;").replace("]", "&#93;").replace("`", "&#96;")


def badge(key: str) -> str:
    label, color = VERDICT_BADGES.get(key, (key, _HUB["text_muted"]))
    return f'<span class="sf-badge" style="color:{color}">{html.escape(label)}</span>'


def fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value}{suffix}"


def citations_md(citations: list[Citation]) -> str:
    lines = []
    for c in citations:
        url = c.url if c.url.startswith(("http://", "https://")) else ""
        title = esc(c.title)
        link = f"[{title}]({url.replace(')', '%29').replace(' ', '%20')})" if url else title
        lines.append(f"- {link} — {esc(c.source)}, {c.published_date or 'date unknown'}")
    return "\n".join(lines)


def score_table(score: ScoreBreakdown) -> str:
    rows = [("Fundamentals", score.fundamentals), ("Valuation", score.valuation), ("Momentum", score.momentum),
            ("News sentiment", score.news_sentiment), ("Risk (higher = lower risk)", score.risk)]
    body = "\n".join(f"| {name} | {fmt(value)} | {score.weights.get(key, 0):.0%} |" for (name, value), key in
                     zip(rows, ("fundamentals", "valuation", "momentum", "news_sentiment", "risk")))
    return (f"| Sub-score (0-100) | Value | Weight ({score.risk_profile}) |\n|---|---|---|\n{body}\n"
            f"| **Composite** | **{fmt(score.composite)}** | |")


def critic_md(critic: Any) -> str:
    if critic.passed:
        return "✅ Critic: passed" + (" after one revision" if critic.revised else "") + f" ({esc(critic.model)})"
    concerns = "; ".join(esc(c) for c in critic.concerns) or "not reviewed"
    return f"⚠️ Critic: not passed — recommendation downgraded. Concerns: {concerns}"


def quality_md(quality: Any) -> str:
    issues = "; ".join(esc(i) for i in quality.issues[:6])
    return f"Data quality: **{quality.level}**" + (" (stale)" if quality.stale else "") + (f" — {issues}" if issues else "")


def company_md(c: CompanyAnalysis) -> str:
    title = f"### {esc(c.symbol)} — {esc(c.company_name)}"
    if c.status != "ok":
        return f"{title}\n\n_{esc(c.error)}_"
    parts = [
        title,
        f"{badge(c.verdict)} confidence **{c.confidence}** · price {fmt(c.price)} {esc(c.currency)} · "
        f"data as of {esc(c.as_of or 'unknown')}",
    ]
    if c.linked_theme:
        parts.append(f"**Theme:** {esc(c.linked_theme)}")
    parts += [f"**Thesis:** {esc(c.thesis)}", f"**Why now:** {esc(c.why_now)}", score_table(c.score_breakdown)]
    if c.strengths:
        parts.append("**Strengths**\n" + "\n".join(f"- {esc(s)}" for s in c.strengths))
    if c.risks:
        parts.append("**Risks**\n" + "\n".join(f"- {esc(r)}" for r in c.risks))
    parts += [f"**Valuation:** {esc(c.valuation_view)}", f"**Technicals:** {esc(c.technical_view)}",
              f"**News:** {esc(c.news_view)}", f"**What would change this view:** {esc(c.what_would_change_view)}"]
    if c.citations:
        parts.append("**Sources**\n" + citations_md(c.citations))
    parts += [critic_md(c.critic), quality_md(c.data_quality)]
    if c.degraded:
        parts.append("_Generated with the fallback model (degraded)._")
    return "\n\n".join(parts)


def holding_md(r: HoldingReview) -> str:
    title = f"### {esc(r.symbol)} — {esc(r.company_name)}"
    if r.status != "ok":
        return f"{title}\n\n_{esc(r.error)}_"
    parts = [
        title,
        f"{badge(r.action)} confidence **{r.confidence}** · qty {fmt(r.quantity)} · avg cost {fmt(r.avg_cost)} · "
        f"price {fmt(r.current_price)} · P&L {fmt(r.pnl_pct, '%')} · allocation {fmt(r.allocation_pct, '%')}",
        f"**Why:** {esc(r.rationale)}",
    ]
    if r.risks:
        parts.append("**Risks**\n" + "\n".join(f"- {esc(x)}" for x in r.risks))
    if r.triggers_to_revisit:
        parts.append("**Revisit if**\n" + "\n".join(f"- {esc(x)}" for x in r.triggers_to_revisit))
    if r.citations:
        parts.append("**Sources**\n" + citations_md(r.citations))
    parts += [critic_md(r.critic), quality_md(r.data_quality)]
    return "\n\n".join(parts)


def report_header(report: FinalReport) -> str:
    flags = []
    if report.partial:
        flags.append("⚠️ partial results")
    if report.degraded:
        flags.append("⚠️ degraded (fallback model or data provider)")
    if report.budget_exceeded:
        flags.append("⚠️ LLM budget reached")
    head = f"_Data as of {esc(report.as_of)} · report `{esc(report.report_id)}`_"
    if flags:
        head += "  \n" + " · ".join(flags)
    if report.error:
        head += f"\n\n**{esc(report.error)}**"
    if report.notes:
        head += "\n\n" + "\n".join(f"- {esc(n)}" for n in report.notes)
    return head


def market_md(report: FinalReport) -> str:
    m = report.market_analysis
    if m is None:
        return report_header(report)
    parts = [report_header(report), f"## Market sentiment: {badge_sentiment(m.overall_sentiment)}", esc(m.summary)]
    for theme in m.themes:
        parts.append(f"#### {esc(theme.theme)} — {badge_sentiment(theme.sentiment)} · {theme.impact_horizon} term\n"
                     f"Sectors: {esc(', '.join(theme.sectors))}\n\n{citations_md(theme.evidence)}")
    if m.key_risks:
        parts.append("**Key risks**\n" + "\n".join(f"- {esc(r)}" for r in m.key_risks))
    parts.append(quality_md(m.data_quality) + f" · {m.article_count} articles analysed")
    return "\n\n".join(parts)


def badge_sentiment(sentiment: str) -> str:
    color = {"positive": "#4ade80", "negative": "#ff6b81"}.get(sentiment, "#98a2c9")
    return f'<span class="sf-badge" style="color:{color}">{html.escape(sentiment)}</span>'


def index_frame(report: FinalReport) -> pd.DataFrame:
    rows = [{"Index": q.name, "Level": q.price, "Day %": q.change_pct, "1M %": q.return_1m_pct,
             "As of": (q.as_of or "")[:10], "Stale": "yes" if q.stale else ""}
            for q in (report.market_analysis.index_snapshot if report.market_analysis else [])]
    return pd.DataFrame(rows, columns=["Index", "Level", "Day %", "1M %", "As of", "Stale"])


def opportunities_md(report: FinalReport) -> str:
    parts = [report_header(report), f"## Summary\n{esc(report.executive_summary)}"]
    for title, items in (("Top opportunities", report.opportunities), ("Watchlist", report.watchlist),
                         ("Avoid", report.avoid), ("Analysis unavailable", report.unavailable)):
        if items:
            parts.append(f"## {title}")
            parts += [company_md(c) for c in items]
    if report.key_risks:
        parts.append("## Key risks\n" + "\n".join(f"- {esc(r)}" for r in report.key_risks))
    return "\n\n".join(parts)


def portfolio_md(report: FinalReport) -> str:
    if report.portfolio is None:
        return report_header(report) + f"\n\n{esc(report.executive_summary)}"
    snap = report.portfolio.snapshot
    parts = [
        report_header(report),
        f"## Portfolio: {fmt(snap.total_value)} {esc(snap.currency)} · cost {fmt(snap.total_cost)} · "
        f"P&L {fmt(snap.total_pnl_abs)} ({fmt(snap.total_pnl_pct, '%')})",
        esc(report.executive_summary),
    ]
    if snap.concentration_warnings:
        parts.append("**⚠️ Concentration warnings**\n" + "\n".join(f"- {esc(w)}" for w in snap.concentration_warnings))
    parts += [holding_md(r) for r in report.portfolio.reviews]
    return "\n\n".join(parts)


# --- charts ---------------------------------------------------------------------------------------

def _layout(fig: Any, title: str, y_title: str) -> Any:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15, "color": _HUB["text"]}},
        paper_bgcolor=_CHART_SURFACE, plot_bgcolor=_CHART_SURFACE, font={"color": _HUB["text_muted"]},
        margin={"l": 56, "r": 80, "t": 48, "b": 40}, height=380,
        hoverlabel={"bgcolor": "#151a33", "font": {"color": _HUB["text"]}},
    )
    fig.update_xaxes(showgrid=False, linecolor=_GRID, zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor=_GRID, zeroline=False)
    return fig


def price_chart(history: pd.DataFrame | None, symbol: str, currency: str) -> Any:
    """Close with 50/200-day averages; crosshair + unified tooltip; legend plus end-of-line labels."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if history is None or history.empty:
        fig.add_annotation(text="Price history unavailable", showarrow=False, font={"color": _HUB["text_muted"]})
        return _layout(fig, f"{symbol} price", currency)
    close = history["close"]
    series = [("Close", close), ("50-day avg", close.rolling(50).mean()), ("200-day avg", close.rolling(200).mean())]
    for (name, values), color in zip(series, _SERIES):
        fig.add_trace(go.Scatter(x=values.index, y=values.round(2), name=name, mode="lines",
                                 line={"color": color, "width": 2}, hovertemplate="%{y:,.2f}"))
        last = values.dropna()
        if not last.empty:
            fig.add_annotation(x=last.index[-1], y=float(last.iloc[-1]), text=name, showarrow=False, xanchor="left",
                               xshift=6, font={"size": 11, "color": _HUB["text_muted"]})
    fig.update_layout(hovermode="x unified", legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"})
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikecolor=_HUB["text_muted"])
    return _layout(fig, f"{symbol} — 1 year", currency)


def allocation_chart(report: FinalReport, max_stock_pct: float) -> Any:
    """Allocation per holding (one hue, sorted), with the per-stock concentration limit as a reference line."""
    import plotly.graph_objects as go

    fig = go.Figure()
    positions = sorted([p for p in report.portfolio.snapshot.positions if p.allocation_pct is not None],
                       key=lambda p: p.allocation_pct) if report.portfolio else []
    if not positions:
        fig.add_annotation(text="No allocation data", showarrow=False, font={"color": _HUB["text_muted"]})
        return _layout(fig, "Allocation", "")
    fig.add_trace(go.Bar(
        x=[p.allocation_pct for p in positions], y=[p.symbol for p in positions], orientation="h",
        marker={"color": _SERIES[0], "line": {"color": _CHART_SURFACE, "width": 2}},
        text=[f"{p.allocation_pct:.1f}%" for p in positions], textposition="outside",
        textfont={"color": _HUB["text"]}, cliponaxis=False,
        customdata=[[p.sector, p.pnl_pct] for p in positions],
        hovertemplate="%{y}: %{x:.2f}% of portfolio<br>Sector: %{customdata[0]}<br>P&L: %{customdata[1]}%<extra></extra>",
    ))
    fig.add_vline(x=max_stock_pct, line={"color": _HUB["text_muted"], "dash": "dash", "width": 1},
                  annotation_text=f"{max_stock_pct:g}% limit", annotation_font_color=_HUB["text_muted"])
    _layout(fig, "Allocation by holding", "")
    fig.update_xaxes(title="% of portfolio", gridcolor=_GRID, showgrid=True)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_layout(height=max(260, 46 * len(positions) + 90), showlegend=False, bargap=0.35)
    return fig


def indicators_frame(c: CompanyAnalysis, technicals: dict[str, Any], metrics: dict[str, Any]) -> pd.DataFrame:
    rows = [(k.replace("_", " "), v) for k, v in {**metrics, **technicals}.items()
            if isinstance(v, (int, float, bool)) and not isinstance(v, list)]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# --- event plumbing -------------------------------------------------------------------------------

def run_streaming(fn: Any) -> Iterator[tuple[str, Any]]:
    """Run ``fn(listener)`` in a thread; yield ``("event", data)`` items then ``("report", report)``."""
    events: queue.Queue = queue.Queue()
    holder: dict[str, Any] = {}

    def work() -> None:
        try:
            holder["report"] = fn(lambda event, data: events.put((event, data)))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Stock Finder UI run failed")
            holder["error"] = exc
        finally:
            events.put(_DONE)

    threading.Thread(target=work, name="sf-ui", daemon=True).start()
    while True:
        item = events.get()
        if item is _DONE:
            break
        yield item
    if "error" in holder:
        raise holder["error"]
    yield ("report", holder["report"])


def progress_line(event: str, data: dict[str, Any]) -> str | None:
    if event == "progress":
        return f"- {esc(data.get('message', ''))}"
    if event == "worker_result":
        if data.get("status") == "unavailable":
            return f"- ❌ {esc(data.get('symbol'))}: {esc(data.get('error'))}"
        verdict = data.get("verdict") or data.get("action")
        mark = "✅" if data.get("critic_passed") else "⚠️"
        return f"- {mark} {esc(data.get('symbol'))}: {esc(verdict)} ({esc(data.get('confidence'))})"
    return None


# --- UI -------------------------------------------------------------------------------------------

def build_ui(settings: StockFinderSettings | None = None) -> Any:
    """Build the Gradio Blocks app."""
    import gradio as gr

    from .pipeline import get_pipeline
    from .portfolio.models import parse_holdings_csv

    settings = settings or get_settings()
    disclaimer = f'<div class="sf-disclaimer">⚠️ {html.escape(settings.disclaimer)}</div>'

    def pipe() -> Any:
        return get_pipeline()

    def sectors() -> list[str]:
        try:
            return pipe().market.universe.sector_names()
        except Exception:  # noqa: BLE001
            return []

    # --- handlers ---

    def load_pulse(refresh: bool = False) -> tuple[str, pd.DataFrame]:
        report = pipe().market_overview(refresh=refresh)
        return market_md(report), index_frame(report)

    def run_opportunities(profile: str, chosen: list[str], max_candidates: float) -> Iterator[tuple[str, str]]:
        log: list[str] = []
        yield "_Starting… this takes a few minutes._", ""
        for kind, data in run_streaming(lambda listener: pipe().find_opportunities(
                risk_profile=profile, sectors=chosen or None, max_candidates=int(max_candidates), listener=listener)):
            if kind == "report":
                yield "\n".join(log) + "\n- **Done.**", opportunities_md(data)
                return
            line = progress_line(kind, data)
            if line:
                log.append(line)
                yield "\n".join(log[-30:]), ""

    def run_analyze(symbol: str, profile: str) -> tuple[str, Any, pd.DataFrame]:
        if not symbol.strip():
            return "_Enter a symbol, e.g. TCS, RELIANCE.NS or AAPL._", None, pd.DataFrame()
        report = pipe().analyze(symbol, risk_profile=profile)
        companies = report.opportunities + report.watchlist + report.avoid + report.unavailable
        if not companies:
            return report_header(report), None, pd.DataFrame()
        company = companies[0]
        try:
            history = pipe().market.get_history(company.symbol, "1y")
        except Exception:  # noqa: BLE001
            history = None
        doc = pipe().reports.get(report.report_id, include_snapshot=True) or {}
        snap = (doc.get("data_snapshot") or {}).get("companies", {}).get(company.symbol, {})
        table = indicators_frame(company, snap.get("technicals", {}), snap.get("metrics", {}))
        return (report_header(report) + "\n\n" + company_md(company), price_chart(history, company.symbol, settings.currency),
                table)

    def load_holdings() -> tuple[pd.DataFrame, str]:
        try:
            holdings = pipe().holdings.list()
        except Exception as exc:  # noqa: BLE001
            return pd.DataFrame(columns=HOLDING_COLUMNS), f"⚠️ Portfolio storage unavailable: {esc(type(exc).__name__)}"
        rows = [{"symbol": h.symbol, "quantity": h.quantity, "avg_cost": h.avg_cost,
                 "buy_date": h.buy_date.isoformat() if h.buy_date else "", "thesis": h.thesis or ""} for h in holdings]
        return pd.DataFrame(rows, columns=HOLDING_COLUMNS), f"{len(rows)} holdings loaded."

    def save_holdings(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        items, errors = [], []
        for i, row in enumerate((frame if frame is not None else pd.DataFrame()).to_dict("records"), start=1):
            if not str(row.get("symbol") or "").strip():
                continue
            try:
                holding = HoldingIn.model_validate({k: (row.get(k) if str(row.get(k) or "").strip() else None)
                                                    for k in HOLDING_COLUMNS})
                items.append(holding.model_copy(update={"symbol": pipe().market.normalize(holding.symbol)}))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"row {i}: {type(exc).__name__}")
        if errors:
            return frame, "⚠️ Not saved — fix: " + "; ".join(errors)
        try:
            pipe().holdings.replace_all(items)
        except Exception as exc:  # noqa: BLE001
            return frame, f"⚠️ Could not save: {esc(exc)}"
        table, _ = load_holdings()
        return table, f"Saved {len(items)} holdings."

    def upload_csv(file: Any) -> tuple[pd.DataFrame, str]:
        if file is None:
            return load_holdings()
        text = Path(file if isinstance(file, str) else file.name).read_text(encoding="utf-8-sig")
        rows, errors = parse_holdings_csv(text)
        if not rows:
            return load_holdings()[0], "⚠️ " + "; ".join(errors[:5])
        try:
            pipe().holdings.replace_all([r.model_copy(update={"symbol": pipe().market.normalize(r.symbol)}) for r in rows])
        except Exception as exc:  # noqa: BLE001
            return load_holdings()[0], f"⚠️ Could not save: {esc(exc)}"
        table, _ = load_holdings()
        note = f"Imported {len(rows)} holdings." + (f" Skipped: {'; '.join(errors[:5])}" if errors else "")
        return table, note

    def sample_csv() -> str:
        source = Path(__file__).resolve().parent / "sample_portfolio.csv"
        target = Path(tempfile.gettempdir()) / "stock_finder_sample_portfolio.csv"
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        return str(target)

    def run_review(profile: str) -> Iterator[tuple[str, str, Any]]:
        log: list[str] = []
        yield "_Reviewing holdings…_", "", None
        for kind, data in run_streaming(lambda listener: pipe().review_portfolio(risk_profile=profile, listener=listener)):
            if kind == "report":
                chart = allocation_chart(data, settings.max_stock_pct) if data.portfolio else None
                yield "\n".join(log) + "\n- **Done.**", portfolio_md(data), chart
                return
            line = progress_line(kind, data)
            if line:
                log.append(line)
                yield "\n".join(log[-30:]), "", None

    def chat(message: str, history: list[dict[str, Any]], profile: str) -> tuple[list[dict[str, Any]], str]:
        history = list(history or [])
        if not message.strip():
            return history, ""
        history.append({"role": "user", "content": message})
        report = pipe().chat(message, risk_profile=profile)
        answer = report.error or report.answer or report.executive_summary
        if report.citations:
            answer += "\n\n**Sources**\n" + citations_md(report.citations)
        answer += f"\n\n_Data as of {report.as_of[:16]} · {settings.disclaimer}_"
        history.append({"role": "assistant", "content": answer})
        return history, ""

    # --- layout ---

    with gr.Blocks(title="Stock Finder", analytics_enabled=False) as demo:
        with gr.Tab("Market Pulse"):
            gr.HTML(disclaimer)
            pulse_btn = gr.Button("Refresh market pulse", variant="primary")
            pulse_md = gr.Markdown("_Click refresh to load market news, themes and sentiment._")
            pulse_table = gr.Dataframe(label="Index snapshot", interactive=False)
            pulse_btn.click(lambda: load_pulse(True), None, [pulse_md, pulse_table])

        with gr.Tab("Opportunities"):
            gr.HTML(disclaimer)
            with gr.Row():
                opp_profile = gr.Radio(list(RISK_PROFILES), value=settings.risk_profile_default, label="Risk profile")
                opp_sectors = gr.Dropdown(sectors(), multiselect=True, label="Sectors (optional)")
                opp_max = gr.Slider(1, settings.max_candidates, value=settings.max_candidates, step=1,
                                    label="Max candidates")
            opp_btn = gr.Button("Find opportunities", variant="primary")
            opp_progress = gr.Markdown()
            opp_result = gr.Markdown()
            opp_btn.click(run_opportunities, [opp_profile, opp_sectors, opp_max], [opp_progress, opp_result])

        with gr.Tab("Analyze a Stock"):
            gr.HTML(disclaimer)
            with gr.Row():
                symbol = gr.Textbox(label="Symbol or company", placeholder="e.g. TCS, RELIANCE.NS, Infosys, AAPL")
                an_profile = gr.Radio(list(RISK_PROFILES), value=settings.risk_profile_default, label="Risk profile")
            an_btn = gr.Button("Analyze", variant="primary")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    an_md = gr.Markdown()
                with gr.Column(scale=2):
                    an_chart = gr.Plot(label="Price")
                    an_table = gr.Dataframe(label="Indicators & fundamentals (computed in code)", interactive=False)
            an_btn.click(run_analyze, [symbol, an_profile], [an_md, an_chart, an_table])
            symbol.submit(run_analyze, [symbol, an_profile], [an_md, an_chart, an_table])

        with gr.Tab("My Portfolio"):
            gr.HTML(disclaimer)
            holdings_table = gr.Dataframe(headers=HOLDING_COLUMNS, datatype=["str", "number", "number", "str", "str"],
                                          column_count=5, interactive=True,
                                          label="Holdings (editable)")
            with gr.Row():
                save_btn = gr.Button("Save table")
                reload_btn = gr.Button("Reload")
                csv_file = gr.File(label="Upload CSV (replaces holdings)", file_types=[".csv"], type="filepath")
                sample_btn = gr.DownloadButton("Sample CSV", value=sample_csv)
            holdings_note = gr.Markdown()
            with gr.Row():
                pf_profile = gr.Radio(list(RISK_PROFILES), value=settings.risk_profile_default, label="Risk profile")
                review_btn = gr.Button("Review portfolio", variant="primary")
            pf_progress = gr.Markdown()
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    pf_result = gr.Markdown()
                with gr.Column(scale=2):
                    pf_chart = gr.Plot(label="Allocation")
            save_btn.click(save_holdings, [holdings_table], [holdings_table, holdings_note])
            reload_btn.click(load_holdings, None, [holdings_table, holdings_note])
            csv_file.upload(upload_csv, [csv_file], [holdings_table, holdings_note])
            review_btn.click(run_review, [pf_profile], [pf_progress, pf_result, pf_chart])

        with gr.Tab("Chat"):
            gr.HTML(disclaimer)
            chat_profile = gr.Radio(list(RISK_PROFILES), value=settings.risk_profile_default, label="Risk profile")
            chatbot = gr.Chatbot(label="Ask Stock Finder", height=520,
                                 placeholder="Ask about the market, a stock (e.g. 'analyse TCS'), or investing concepts.")
            message = gr.Textbox(label="Message", placeholder="e.g. How is the IT sector looking?", lines=2)
            send = gr.Button("Send", variant="primary")
            send.click(chat, [message, chatbot, chat_profile], [chatbot, message])
            message.submit(chat, [message, chatbot, chat_profile], [chatbot, message])

        demo.load(load_holdings, None, [holdings_table, holdings_note])
    return demo


def _start_lifespan(asgi_app: Any, timeout: float = 30.0) -> asyncio.AbstractEventLoop:
    """Run the ASGI lifespan startup on a background loop (a2wsgi never sends lifespan events).

    Without it Gradio's request queue never starts and streamed requests hang.
    The same loop is handed to ``ASGIMiddleware`` and the lifespan stays open.
    """
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name="stock-finder-ui", daemon=True).start()
    ready = threading.Event()

    async def _hold_lifespan() -> None:
        async with asgi_app.router.lifespan_context(asgi_app):
            ready.set()
            await asyncio.Event().wait()

    future = asyncio.run_coroutine_threadsafe(_hold_lifespan(), loop)
    if not ready.wait(timeout):
        future.cancel()
        raise RuntimeError("Gradio startup did not complete")
    if future.done():
        future.result()
    _running_lifespans.append(future)  # keep a strong reference for the process lifetime
    return loop


def build_wsgi_app(settings: StockFinderSettings) -> Any | None:
    """Build the Gradio app, start its lifespan and wrap it as WSGI. Returns None (and logs) on failure.

    Called on the first request to ``{prefix}/ui`` (see ``_LazyUI`` in ``__init__.py``).
    """
    try:
        import gradio as gr
        from a2wsgi import ASGIMiddleware
        from fastapi import FastAPI
    except ImportError as exc:
        logger.warning("Stock Finder UI unavailable (missing dependency: %s); the API still works.", exc)
        return None

    mount_path = f"{settings.url_prefix}/ui"
    try:
        asgi_app = gr.mount_gradio_app(
            FastAPI(), build_ui(settings), path="/", root_path=mount_path, theme=build_theme(), css=CSS,
            head='<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&display=swap" rel="stylesheet">',
        )
        loop = _start_lifespan(asgi_app)
    except Exception:  # noqa: BLE001 - a UI failure must never take down the other hub apps
        logger.exception("Stock Finder UI failed to start; the API still works.")
        return None
    logger.info("Stock Finder UI started at %s/", mount_path)
    return ASGIMiddleware(asgi_app, loop=loop)
