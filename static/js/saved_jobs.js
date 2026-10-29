/**
 * Saved Jobs page: lists every job stored by past Remote Job Finder searches
 * (GET /api/saved-jobs) with client-side filter/sort, a per-job status, and a
 * full detail panel (LLM score, strengths, honest concerns, skills, description).
 * All scraped text goes through textContent — never innerHTML.
 *
 * The Applied Jobs page reuses this script with ``data-mode="applied"`` on
 * #sj-app: it reads /api/applied-jobs, has no status dropdown, and "remove"
 * only drops the job from the applied list. There each job can be marked
 * rejected; that (or Re-evaluate) runs the stored LLM rejection analysis,
 * which is then shown from the saved job without another call.
 */
(function () {
  "use strict";

  var STATUS_LABELS = {
    new: "New",
    interested: "Interested",
    applied: "Applied",
    not_interested: "Not interested",
  };
  var RECOMMENDATION_LABELS = {
    strong_match: "Strong match",
    good_match: "Good match",
    possible_match: "Possible match",
    poor_match: "Poor match",
    reject: "Rejected",
  };

  var APPLIED_MODE = document.getElementById("sj-app").dataset.mode === "applied";
  var MODE = APPLIED_MODE
    ? {
        api: "/api/applied-jobs",
        noun: "applied jobs",
        removeLabel: "Remove from applied",
        removeAllLabel: "Remove all applied jobs",
      }
    : {
        api: "/api/saved-jobs",
        noun: "saved jobs",
        removeLabel: "Remove from saved",
        removeAllLabel: "Remove all jobs",
      };

  var statuses = [];
  var statusesNode = document.getElementById("sj-statuses");
  try {
    statuses = statusesNode ? JSON.parse(statusesNode.textContent || "[]") : [];
  } catch (e) {
    statuses = Object.keys(STATUS_LABELS);
  }

  var els = {
    search: document.getElementById("sj-search"),
    recommendation: document.getElementById("sj-recommendation"),
    status: document.getElementById("sj-status-filter"),
    source: document.getElementById("sj-source"),
    minScore: document.getElementById("sj-min-score"),
    sort: document.getElementById("sj-sort"),
    worldwide: document.getElementById("sj-worldwide"),
    visa: document.getElementById("sj-visa"),
    remoteScope: document.getElementById("sj-remote-scope"),
    outcome: document.getElementById("sj-outcome"),
    evaluated: document.getElementById("sj-evaluated"),
    error: document.getElementById("sj-error"),
    stats: document.getElementById("sj-stats"),
    list: document.getElementById("sj-list"),
  };

  var allJobs = [];
  // Worldwide-remote classification (only present when JOB_WORLDWIDE_REMOTE_ENABLED is on).
  var WW = window.WorldwideRemote || null;

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function setVisible(node, show) {
    node.classList.toggle("hidden", !show);
  }

  function fmtDate(iso) {
    if (!iso) return null;
    var d = new Date(iso);
    if (isNaN(d.getTime())) return null;
    return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  }

  function relTime(iso) {
    if (!iso) return null;
    var d = new Date(iso);
    if (isNaN(d.getTime())) return null;
    var mins = Math.round((Date.now() - d.getTime()) / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return mins + " min ago";
    var hrs = Math.round(mins / 60);
    if (hrs < 24) return hrs + " h ago";
    var days = Math.round(hrs / 24);
    return days + (days === 1 ? " day ago" : " days ago");
  }

  function titleCase(s) {
    return String(s || "").replace(/_/g, " ").replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  function isApplied(saved) {
    return saved.status === "applied" ? 1 : 0;
  }

  function isRejected(saved) {
    return saved.outcome === "rejected";
  }

  /**
   * The score computed from the requirement checks (jobEngine/matching/fit_score.py);
   * older saves have only the LLM's own score or a blended match_score.
   */
  function score(saved) {
    var b = saved.score_breakdown, llm = saved.llm_analysis;
    var s = b ? b.score
      : llm && llm.score !== undefined && llm.score !== null ? llm.score : saved.match_score;
    s = Number(s);
    return isNaN(s) ? 0 : s;
  }

  /** Same bands as jobEngine/matching/scorer.py recommendation_for_score. */
  function recommendation(saved) {
    if (!saved.llm_analysis) return saved.recommendation || "possible_match";
    var s = score(saved);
    if (s >= 85) return "strong_match";
    if (s >= 70) return "good_match";
    if (s >= 50) return "possible_match";
    if (s >= 30) return "poor_match";
    return "reject";
  }

  function searchBlob(saved) {
    if (saved._blob) return saved._blob;
    var job = saved.job || {};
    saved._blob = [
      job.title, job.company, job.location, job.description,
      (job.skills || []).join(" "),
      (saved.matched_skills || []).join(" "),
      (saved.missing_required_skills || []).join(" "),
    ].join(" ").toLowerCase();
    return saved._blob;
  }

  // --- filters -------------------------------------------------------------

  function fillSelect(select, values, labelFn) {
    values.forEach(function (v) {
      var opt = el("option", null, labelFn(v));
      opt.value = v;
      select.appendChild(opt);
    });
  }

  function currentFilters() {
    return {
      q: els.search.value.trim().toLowerCase(),
      recommendation: els.recommendation.value,
      status: els.status ? els.status.value : "",
      source: els.source.value,
      minScore: Number(els.minScore.value) || 0,
      sort: els.sort.value,
      worldwide: els.worldwide ? els.worldwide.value : "",
      visa: els.visa ? els.visa.value : "",
      remoteScope: els.remoteScope ? els.remoteScope.value : "",
      outcome: els.outcome ? els.outcome.value : "",
      evaluated: els.evaluated ? els.evaluated.value : "",
    };
  }

  function applyFilters() {
    var f = currentFilters();
    var terms = f.q ? f.q.split(/\s+/) : [];
    var jobs = allJobs.filter(function (saved) {
      var job = saved.job || {};
      if (f.recommendation && recommendation(saved) !== f.recommendation) return false;
      if (f.status && (saved.status || "new") !== f.status) return false;
      if (f.source && job.source !== f.source) return false;
      if (score(saved) < f.minScore) return false;
      if (WW && f.worldwide && !WW.matches(job, f.worldwide)) return false;
      if (WW && f.visa && !WW.matchesVisa(job, f.visa)) return false;
      if (WW && f.remoteScope && !WW.matchesRemote(job, f.remoteScope)) return false;
      if (f.outcome && (f.outcome === "rejected") !== isRejected(saved)) return false;
      if (f.evaluated && (f.evaluated === "evaluated") !== !!saved.rejection_analysis) return false;
      if (terms.length) {
        var blob = searchBlob(saved);
        for (var i = 0; i < terms.length; i++) if (blob.indexOf(terms[i]) === -1) return false;
      }
      return true;
    });

    var sortKey = {
      last_seen: function (s) { return s.last_seen_at || ""; },
      first_seen: function (s) { return s.first_seen_at || ""; },
      posted: function (s) { return (s.job || {}).posted_date || ""; },
      applied: function (s) { return s.applied_at || ""; },
      rejected: function (s) { return s.rejected_at || ""; },
    }[f.sort];
    jobs.sort(function (a, b) {
      // Applied jobs sink to the bottom so the list leads with jobs still to look at.
      var applied = APPLIED_MODE ? 0 : isApplied(a) - isApplied(b);
      if (applied) return applied;
      if (WW && f.sort === "eligibility") {
        return WW.rank(a.job) - WW.rank(b.job) || score(b) - score(a);
      }
      if (!sortKey) return score(b) - score(a);
      var ka = sortKey(a), kb = sortKey(b);
      return ka < kb ? 1 : ka > kb ? -1 : score(b) - score(a);
    });
    return jobs;
  }

  // --- rendering -----------------------------------------------------------

  function renderStats(shown) {
    els.stats.textContent = "";
    var counts = { total: allJobs.length };
    allJobs.forEach(function (s) {
      var st = s.status || "new";
      counts[st] = (counts[st] || 0) + 1;
    });
    function stat(label, value) {
      var span = el("span", null, label + " ");
      span.appendChild(el("strong", null, value));
      els.stats.appendChild(span);
    }
    stat("Showing", shown + " of " + counts.total);
    statuses.forEach(function (st) {
      if (counts[st]) stat(STATUS_LABELS[st] || titleCase(st), counts[st]);
    });
    if (APPLIED_MODE) {
      stat("Rejected", allJobs.filter(isRejected).length);
      stat("Evaluated", allJobs.filter(function (s) { return !!s.rejection_analysis; }).length);
    }
    if (allJobs.length) {
      var clearBtn = el("button", "btn btn--ghost sj-clear", MODE.removeAllLabel);
      clearBtn.type = "button";
      clearBtn.addEventListener("click", function () { removeAll(clearBtn); });
      els.stats.appendChild(clearBtn);
    }
    setVisible(els.stats, true);
  }

  function section(title, content) {
    var wrap = el("div", "sj-section");
    wrap.appendChild(el("h3", null, title));
    wrap.appendChild(content);
    return wrap;
  }

  function pointList(items) {
    var ul = el("ul", "sj-points");
    items.forEach(function (t) { ul.appendChild(el("li", null, t)); });
    return ul;
  }

  function tagList(items, modifier) {
    var row = el("div", "sj-tags");
    items.forEach(function (t) { row.appendChild(el("span", "sj-tag" + (modifier ? " sj-tag--" + modifier : ""), t)); });
    return row;
  }

  function scoreBar(value, labelText) {
    value = Math.max(0, Math.min(100, Number(value) || 0));
    var bar = el("div", "sj-bar");
    var label = el("div", "sj-bar__label");
    label.appendChild(el("span", null, labelText));
    label.appendChild(el("strong", null, Math.round(value)));
    var track = el("div", "sj-bar__track");
    var fill = el("div", "sj-bar__fill");
    fill.style.width = value + "%";
    track.appendChild(fill);
    bar.appendChild(label);
    bar.appendChild(track);
    return bar;
  }

  // --- score breakdown (jobEngine/matching/fit_score.py) -------------------------

  var VERDICT_LABELS = { met: "Met", partial: "Partial", missing: "Missing" };
  var TIER_LABELS = { core: "Core", required: "Required", preferred: "Preferred" };

  function scoreBlock(saved) {
    var b = saved.score_breakdown;
    var wrap = el("div");
    wrap.appendChild(scoreBar(score(saved), "Match score"));
    if (b) {
      var years = b.required_years ? "job asks " + b.required_years + " yrs, you have " + b.candidate_years
        : "job states no years range";
      var note = "0.90 × skills " + Math.round(b.skills) + " + 0.05 × experience " + Math.round(b.experience) +
        " (" + years + ") + 0.05 × keywords " + Math.round(b.keywords) + ".";
      if (b.core_missing) note += " Capped at 60: a core requirement is missing.";
      wrap.appendChild(el("p", "sj-review-note", note));
    }
    return wrap;
  }

  function requirementsBlock(breakdown) {
    var ul = el("ul", "sj-points sj-reqs");
    (breakdown.requirements || []).forEach(function (r) {
      var li = el("li");
      li.appendChild(el("span", "sj-verdict sj-verdict--" + r.verdict, VERDICT_LABELS[r.verdict] || r.verdict));
      li.appendChild(document.createTextNode(" " + r.requirement + " "));
      li.appendChild(el("span", "sj-review-tag", TIER_LABELS[r.tier] || r.tier));
      // Why code overrode the LLM, e.g. "Python is listed in the profile".
      if (r.adjusted) li.appendChild(el("span", "sj-review-tag", r.adjusted));
      ul.appendChild(li);
    });
    return ul;
  }

  /** Collapsed-by-default job description with a copy-to-clipboard button. */
  function descriptionBlock(text) {
    var details = el("details", "sj-desc");
    var summary = el("summary", "sj-desc__summary");
    summary.appendChild(el("span", "sj-desc__label", "Description"));
    var copy = el("button", "btn btn--ghost sj-desc__copy", "Copy description");
    copy.type = "button";
    copy.addEventListener("click", function (e) {
      e.preventDefault(); // don't toggle the <details>
      e.stopPropagation();
      copyText(text).then(
        function () { flash(copy, "Copied ✓"); },
        function () { flash(copy, "Copy failed"); }
      );
    });
    summary.appendChild(copy);
    details.appendChild(summary);
    details.appendChild(el("pre", "sj-description", text));
    return details;
  }

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    // Fallback for plain-http hosts, where the Clipboard API is unavailable.
    return new Promise(function (resolve, reject) {
      var ta = el("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
      ta.remove();
      if (ok) resolve(); else reject(new Error("copy failed"));
    });
  }

  function flash(button, message) {
    var original = "Copy description";
    button.textContent = message;
    clearTimeout(button._flash);
    button._flash = setTimeout(function () { button.textContent = original; }, 1600);
  }

  function detailsList(saved) {
    var job = saved.job || {};
    var dl = el("dl", "sj-details");
    [
      ["Company", job.company],
      ["Location", job.location],
      ["Remote type", job.remote_type && titleCase(job.remote_type)],
      ["Employment type", job.employment_type],
      ["Arrangement", WW && WW.arrangementLabel(job)],
      ["Detected", fmtDate(job.discovered_at)],
      ["Experience required", job.experience_required],
      ["Salary", job.salary],
      ["Posted", fmtDate(job.posted_date)],
      ["Source", job.source && titleCase(job.source)],
      ["Source job ID", job.source_job_id],
      ["Applied", fmtDate(saved.applied_at)],
      ["First saved", fmtDate(saved.first_seen_at)],
      ["Last seen", fmtDate(saved.last_seen_at)],
      ["Seen in runs", (saved.run_ids || []).length || null],
    ].forEach(function (row) {
      if (row[1] === null || row[1] === undefined || row[1] === "") return;
      dl.appendChild(el("dt", null, row[0]));
      dl.appendChild(el("dd", null, row[1]));
    });
    return dl;
  }

  // --- rejection analysis (jobEngine/matching/rejection_analyzer.py) -------------

  var FINAL_VERDICT_LABELS = {
    strong_fit_rejected_anyway: "Strong fit — rejected anyway",
    partial_fit: "Partial fit",
    weak_fit: "Weak fit",
    not_eligible: "Not eligible",
  };
  var DIM_VERDICT_LABELS = { match: "Match", partial: "Partial", mismatch: "Mismatch", unknown: "Unknown" };
  var REAPPLY_LABELS = { yes: "Yes", maybe: "Maybe", no: "No" };

  function chip(className, text) {
    return el("span", "rj-chip " + className, text);
  }

  function dimensionsTable(dims) {
    var wrap = el("div", "rj-table-wrap");
    var table = el("table", "rj-table");
    var head = el("tr");
    ["Dimension", "Verdict", "Impact", "The job asks", "You have", "Why"].forEach(function (h) {
      head.appendChild(el("th", null, h));
    });
    var thead = el("thead");
    thead.appendChild(head);
    table.appendChild(thead);
    var tbody = el("tbody");
    (dims || []).forEach(function (d) {
      var tr = el("tr", "rj-row rj-row--" + d.verdict);
      tr.appendChild(el("th", null, titleCase(d.dimension)));
      var v = el("td");
      v.appendChild(chip("rj-verdict--" + d.verdict, DIM_VERDICT_LABELS[d.verdict] || d.verdict));
      tr.appendChild(v);
      var imp = el("td");
      imp.appendChild(chip("rj-impact--" + d.impact, titleCase(d.impact)));
      tr.appendChild(imp);
      tr.appendChild(el("td", null, d.job_requirement));
      tr.appendChild(el("td", null, d.candidate_fact));
      tr.appendChild(el("td", null, d.explanation));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function rejectionBlock(saved) {
    var a = saved.rejection_analysis;
    var wrap = el("div", "rj-analysis");
    var banner = el("div", "rj-banner rj-banner--" + a.final_verdict);
    var title = el("div", "rj-banner__title");
    title.appendChild(el("strong", null, FINAL_VERDICT_LABELS[a.final_verdict] || titleCase(a.final_verdict)));
    title.appendChild(chip("rj-chip--neutral", "Fit " + a.fit_score + "/100"));
    title.appendChild(chip("rj-chip--neutral", titleCase(a.confidence) + " confidence"));
    banner.appendChild(title);
    banner.appendChild(el("p", "rj-banner__summary", a.verdict_summary));
    wrap.appendChild(banner);

    if (a.honest_opinion) wrap.appendChild(section("Honest opinion", el("p", "sj-explanation", a.honest_opinion)));

    if ((a.primary_reasons || []).length) {
      var ol = el("ol", "sj-points rj-reasons");
      a.primary_reasons.forEach(function (r) {
        var li = el("li");
        li.appendChild(chip("rj-impact--" + r.impact, titleCase(r.impact)));
        li.appendChild(el("span", "sj-review-tag", titleCase(r.dimension)));
        li.appendChild(document.createTextNode(" " + r.reason));
        ol.appendChild(li);
      });
      wrap.appendChild(section("Most likely reasons", ol));
    }

    wrap.appendChild(section("Dimension by dimension", dimensionsTable(a.dimensions)));

    var skills = el("div", "sj-cols");
    if ((a.matched_skills || []).length) skills.appendChild(section("Skills you matched", tagList(a.matched_skills)));
    if ((a.missing_skills || []).length) skills.appendChild(section("Skills you were missing", tagList(a.missing_skills, "missing")));
    if (skills.childNodes.length) wrap.appendChild(skills);

    var cols = el("div", "sj-cols");
    if ((a.red_flags || []).length) cols.appendChild(section("Red flags in the posting", pointList(a.red_flags)));
    if ((a.improvement_actions || []).length) cols.appendChild(section("What to do next time", pointList(a.improvement_actions)));
    if (cols.childNodes.length) wrap.appendChild(cols);

    var reapply = el("div");
    reapply.appendChild(chip("rj-reapply--" + a.reapply_similar_roles, REAPPLY_LABELS[a.reapply_similar_roles] || a.reapply_similar_roles));
    reapply.appendChild(document.createTextNode(" " + (a.reapply_advice || "")));
    wrap.appendChild(section("Apply to similar roles?", reapply));

    var meta = [
      "Evaluated " + (fmtDate(a.evaluated_at) || "—"),
      a.model,
      a.tokens ? a.tokens + " tokens" : null,
    ].filter(Boolean).join(" · ");
    wrap.appendChild(el("p", "sj-review-note", meta));
    return wrap;
  }

  /** The rejection section of an applied job: analysis (when stored) + its buttons. */
  function rejectionSection(saved, card) {
    var wrap = el("div", "sj-section rj-section");
    wrap.appendChild(el("h3", null, "Why was I rejected?"));
    if (saved.rejection_analysis) {
      wrap.appendChild(rejectionBlock(saved));
    } else if (isRejected(saved)) {
      wrap.appendChild(el("p", "sj-review-note", "No analysis stored yet. Click Evaluate to run it."));
    } else {
      wrap.appendChild(el("p", "sj-review-note", "Mark this job as rejected to get an analysis of why."));
    }
    var actions = el("div", "sj-actions");
    if (isRejected(saved) || saved.rejection_analysis) {
      var evalBtn = el("button", "btn", saved.rejection_analysis ? "Re-evaluate" : "Evaluate");
      evalBtn.type = "button";
      evalBtn.addEventListener("click", function () { evaluate(saved, card, evalBtn); });
      actions.appendChild(evalBtn);
    }
    actions.appendChild(rejectButton(saved, card, "btn btn--ghost"));
    wrap.appendChild(actions);
    return wrap;
  }

  function rejectButton(saved, card, className) {
    var btn = el("button", className, isRejected(saved) ? "Unmark rejected" : "Mark as rejected");
    btn.type = "button";
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      setRejected(saved, card, !isRejected(saved), btn);
    });
    btn.addEventListener("keydown", function (e) { e.stopPropagation(); });
    return btn;
  }

  function buildBody(saved, card) {
    var job = saved.job || {};
    var llm = saved.llm_analysis || null;
    var body = el("div", "sj-job__body");
    if (APPLIED_MODE) body.appendChild(rejectionSection(saved, card));
    var wwPanel = WW && WW.panel(job);
    if (wwPanel) body.appendChild(section("Eligibility from India", wwPanel));
    var vrPanel = WW && WW.visaRemotePanel(job);
    if (vrPanel) body.appendChild(section("Visa sponsorship & remote scope", vrPanel));

    var explanation = saved.explanation || (llm && llm.explanation);
    if (explanation) body.appendChild(section("Why it matches", el("p", "sj-explanation", explanation)));
    var breakdown = saved.score_breakdown || null;
    if (llm) body.appendChild(section("Score", scoreBlock(saved)));

    var cols = el("div", "sj-cols");
    var strengths = saved.strengths || (llm && llm.strengths) || [];
    var concerns = saved.concerns || (llm && llm.concerns) || [];
    if (strengths.length) cols.appendChild(section("Strengths", pointList(strengths)));
    if (concerns.length) cols.appendChild(section("Honest concerns", pointList(concerns)));
    if (cols.childNodes.length) body.appendChild(cols);
    var hasChecks = breakdown && (breakdown.requirements || []).length;
    if (hasChecks) body.appendChild(section("Requirements check", requirementsBlock(breakdown)));

    var skills = el("div", "sj-cols");
    if ((saved.matched_skills || []).length) skills.appendChild(section("Matched skills", tagList(saved.matched_skills)));
    if ((saved.missing_required_skills || []).length) {
      skills.appendChild(section("Missing skills", tagList(saved.missing_required_skills, "missing")));
    }
    if ((job.skills || []).length) skills.appendChild(section("Listed skills", tagList(job.skills, "neutral")));
    if (skills.childNodes.length) body.appendChild(skills);

    var reqs = el("div", "sj-cols");
    // The requirements check supersedes the matcher's list.
    if (!hasChecks && llm && (llm.matched_requirements || []).length) {
      reqs.appendChild(section("Requirements you meet", pointList(llm.matched_requirements)));
    }
    if ((job.requirements || []).length) reqs.appendChild(section("Requirements", pointList(job.requirements)));
    if ((job.responsibilities || []).length) reqs.appendChild(section("Responsibilities", pointList(job.responsibilities)));
    if (reqs.childNodes.length) body.appendChild(reqs);

    body.appendChild(section("Job details", detailsList(saved)));
    if (job.description) body.appendChild(descriptionBlock(job.description));

    var actions = el("div", "sj-actions");
    if (job.application_url) {
      var apply = el("a", "btn sj-apply", "Open job posting");
      apply.href = job.application_url;
      apply.target = "_blank";
      apply.rel = "noopener noreferrer";
      // Opening the posting (left or middle click) marks the job applied so it is
      // flagged — and sorted last — the next time the list is scanned.
      function markApplied(e) {
        if (e.type === "auxclick" && e.button !== 1) return;
        if (!APPLIED_MODE && saved.status !== "applied") setStatus(saved, card, "applied");
      }
      apply.addEventListener("click", markApplied);
      apply.addEventListener("auxclick", markApplied);
      actions.appendChild(apply);
    }
    var del = el("button", "btn btn--ghost", MODE.removeLabel);
    del.type = "button";
    del.addEventListener("click", function () { removeJob(saved, card); });
    actions.appendChild(del);
    body.appendChild(actions);
    return body;
  }

  function statusSelect(saved) {
    var select = el("select", "job-filter-input sj-status");
    select.setAttribute("aria-label", "Status for " + ((saved.job || {}).title || "job"));
    statuses.forEach(function (st) {
      var opt = el("option", null, STATUS_LABELS[st] || titleCase(st));
      opt.value = st;
      select.appendChild(opt);
    });
    select.value = saved.status || "new";
    select.addEventListener("click", function (e) { e.stopPropagation(); });
    select.addEventListener("keydown", function (e) { e.stopPropagation(); });
    select.addEventListener("change", function () {
      setStatus(saved, select.closest(".sj-job"), select.value);
    });
    return select;
  }

  function appliedLabel(saved) {
    var when = relTime(saved.applied_at);
    return "✓ Applied" + (when ? " " + when : "");
  }

  /** Sync a card's applied badge, dimming and status dropdown with ``saved.status``. */
  function paintStatus(saved, card) {
    if (!card) return;
    var applied = saved.status === "applied";
    // Every job on the Applied Jobs page is applied: no need to dim them.
    card.classList.toggle("sj-job--applied", applied && !APPLIED_MODE);
    var select = card.querySelector(".sj-status");
    if (select) select.value = saved.status || "new";
    var badge = card.querySelector(".sj-applied");
    if (applied) {
      if (!badge) {
        badge = el("span", "sj-applied");
        card.querySelector(".sj-job__title-row").appendChild(badge);
      }
      badge.textContent = appliedLabel(saved);
      badge.title = saved.applied_at ? "Applied " + fmtDate(saved.applied_at) : "Applied";
    } else if (badge) {
      badge.remove();
    }
  }

  function renderCard(saved) {
    var job = saved.job || {};
    var rec = recommendation(saved);
    var card = el("article", "sj-job glass" + (WW ? " " + WW.cardClass(job) : ""));
    card.dataset.open = "false";

    var head = el("div", "sj-job__head");
    head.tabIndex = 0;
    head.setAttribute("role", "button");
    head.setAttribute("aria-expanded", "false");

    var badge = el("div", "sj-score sj-score--" + rec);
    badge.appendChild(document.createTextNode(Math.round(score(saved))));
    badge.appendChild(el("small", null, "score"));
    badge.title = RECOMMENDATION_LABELS[rec] || titleCase(rec);
    head.appendChild(badge);

    var main = el("div");
    var titleRow = el("div", "sj-job__title-row");
    titleRow.appendChild(el("h2", "sj-job__title", job.title || "Untitled role"));
    var stampIso = job.posted_date || saved.first_seen_at;
    if (fmtDate(stampIso)) {
      var stamp = el("time", "sj-job__time", (job.posted_date ? "Posted " : "Saved ") + fmtDate(stampIso));
      stamp.dateTime = stampIso;
      stamp.title = relTime(stampIso);
      titleRow.appendChild(stamp);
    }
    if (APPLIED_MODE && isRejected(saved)) {
      var rej = el("span", "rj-badge", "✗ Rejected");
      if (saved.rejected_at) rej.title = "Marked rejected " + fmtDate(saved.rejected_at);
      titleRow.appendChild(rej);
    }
    if (APPLIED_MODE && saved.rejection_analysis) {
      var a = saved.rejection_analysis;
      titleRow.appendChild(el("span", "rj-badge rj-badge--eval rj-banner--" + a.final_verdict,
        "Evaluated: " + (FINAL_VERDICT_LABELS[a.final_verdict] || titleCase(a.final_verdict))));
    }
    var wwBadge = WW && WW.badge(job);
    if (wwBadge) titleRow.appendChild(wwBadge);
    (WW ? WW.visaRemoteBadges(job) : []).forEach(function (b) { titleRow.appendChild(b); });
    (WW ? WW.displayTags(job) : job.tags || []).forEach(function (t) { titleRow.appendChild(el("span", "sj-pill sj-pill--tag", t)); });
    main.appendChild(titleRow);
    var meta = el("p", "sj-job__meta");
    [job.company, job.location, relTime(stampIso)]
      .filter(Boolean)
      .forEach(function (t) { meta.appendChild(el("span", null, t)); });
    main.appendChild(meta);
    head.appendChild(main);

    var side = el("div", "sj-job__side");
    side.appendChild(el("span", "sj-pill", RECOMMENDATION_LABELS[rec] || titleCase(rec)));
    if (!APPLIED_MODE) side.appendChild(statusSelect(saved));
    else side.appendChild(rejectButton(saved, card, "btn btn--ghost rj-head-btn"));
    side.appendChild(el("span", "sj-job__chev", "›"));
    head.appendChild(side);
    card.appendChild(head);

    function toggle() {
      var open = card.dataset.open !== "true";
      if (open && !card.querySelector(".sj-job__body")) card.appendChild(buildBody(saved, card));
      card.dataset.open = String(open);
      head.setAttribute("aria-expanded", String(open));
    }
    head.addEventListener("click", toggle);
    head.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggle();
      }
    });
    paintStatus(saved, card);
    return card;
  }

  function render() {
    var jobs = applyFilters();
    els.list.textContent = "";
    renderStats(jobs.length);
    if (!allJobs.length) {
      var empty = el("div", "sj-empty glass");
      if (APPLIED_MODE) {
        empty.appendChild(document.createTextNode("No applied jobs yet. Open a job posting on "));
        var savedLink = el("a", null, "Saved Jobs");
        savedLink.href = "/saved-jobs";
        empty.appendChild(savedLink);
        empty.appendChild(document.createTextNode(" or set its status to Applied, and it will be kept here."));
      } else {
        empty.appendChild(document.createTextNode("No saved jobs yet. Run a search on the "));
        var link = el("a", null, "Global Remote Job Finder");
        link.href = "/remote-job-finder";
        empty.appendChild(link);
        empty.appendChild(document.createTextNode(" and its matches will appear here."));
      }
      els.list.appendChild(empty);
      return;
    }
    if (!jobs.length) {
      els.list.appendChild(el("div", "sj-empty glass", "No " + MODE.noun + " match these filters."));
      return;
    }
    var frag = document.createDocumentFragment();
    jobs.forEach(function (saved) { frag.appendChild(renderCard(saved)); });
    els.list.appendChild(frag);
  }

  // --- API -----------------------------------------------------------------

  function showError(message) {
    els.error.textContent = message;
    setVisible(els.error, !!message);
  }

  function apiUrl(saved, suffix) {
    return MODE.api + "/" + encodeURIComponent(saved.saved_key) + (suffix || "");
  }

  function setStatus(saved, card, next) {
    var select = card && card.querySelector(".sj-status");
    if (select) select.disabled = true;
    return fetch(apiUrl(saved, "/status"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: next }),
    })
      .then(function (r) {
        return r.json().then(function (d) {
          if (!r.ok || !d.ok) throw new Error(d.error || r.statusText);
          return d;
        });
      })
      .then(function (d) {
        saved.status = d.status;
        saved.applied_at = d.applied_at || null;
        showError("");
        renderStats(applyFilters().length);
      })
      .catch(function (err) {
        showError("Could not update status: " + err.message);
      })
      .finally(function () {
        if (select) select.disabled = false;
        paintStatus(saved, card); // on failure this reverts the dropdown
      });
  }

  function postJson(url, payload) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    }).then(function (r) {
      return r.json().then(function (d) { return { ok: r.ok && d.ok, data: d, status: r.statusText }; });
    });
  }

  /** Swap in the server's copy of the job and redraw its card, opened on the analysis. */
  function replaceJob(saved, card, fresh) {
    Object.keys(saved).forEach(function (k) { delete saved[k]; });
    Object.assign(saved, fresh);
    var next = renderCard(saved);
    card.replaceWith(next);
    next.querySelector(".sj-job__head").click();
    renderStats(applyFilters().length);
    return next;
  }

  function busy(button, text) {
    button.disabled = true;
    button.dataset.label = button.textContent;
    button.textContent = text;
  }

  function idle(button) {
    button.disabled = false;
    if (button.dataset.label) button.textContent = button.dataset.label;
  }

  function setRejected(saved, card, rejected, button) {
    // Marking rejected runs the analysis when none is stored: this can take a while.
    busy(button, rejected && !saved.rejection_analysis ? "Analysing rejection…" : "Saving…");
    postJson(apiUrl(saved, "/rejected"), { rejected: rejected })
      .then(function (res) {
        if (!res.ok) throw new Error(res.data.error || res.status);
        replaceJob(saved, card, res.data.job);
        showError(res.data.analysis_error ? "Marked rejected, but the analysis failed: " + res.data.analysis_error : "");
      })
      .catch(function (err) {
        idle(button);
        showError("Could not update the job: " + err.message);
      });
  }

  function evaluate(saved, card, button) {
    busy(button, "Evaluating…");
    postJson(apiUrl(saved, "/evaluate"))
      .then(function (res) {
        if (!res.ok) throw new Error(res.data.error || res.status);
        replaceJob(saved, card, res.data.job);
        showError("");
      })
      .catch(function (err) {
        idle(button);
        showError("Could not evaluate the job: " + err.message);
      });
  }

  function removeJob(saved, card) {
    var title = (saved.job || {}).title || "this job";
    if (!window.confirm("Remove “" + title + "” from " + MODE.noun + "?")) return;
    fetch(apiUrl(saved), { method: "DELETE" })
      .then(function (r) { return r.json().then(function (d) { if (!r.ok || !d.ok) throw new Error(d.error || r.statusText); }); })
      .then(function () {
        allJobs = allJobs.filter(function (s) { return s !== saved; });
        card.remove();
        showError("");
        renderStats(applyFilters().length);
      })
      .catch(function (err) { showError("Could not remove job: " + err.message); });
  }

  function removeAll(button) {
    var n = allJobs.length;
    if (!window.confirm("Remove all " + n + " " + MODE.noun + "? This cannot be undone.")) return;
    button.disabled = true;
    fetch(MODE.api, { method: "DELETE" })
      .then(function (r) { return r.json().then(function (d) { if (!r.ok || !d.ok) throw new Error(d.error || r.statusText); }); })
      .then(function () {
        allJobs = [];
        showError("");
        render();
      })
      .catch(function (err) {
        button.disabled = false;
        showError("Could not remove jobs: " + err.message);
      });
  }

  function load() {
    els.list.appendChild(el("div", "sj-empty glass", "Loading " + MODE.noun + "…"));
    fetch(MODE.api)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data.ok) throw new Error(data.error || "request failed");
        allJobs = data.jobs || [];
        var sources = {};
        allJobs.forEach(function (s) { if ((s.job || {}).source) sources[s.job.source] = true; });
        fillSelect(els.source, Object.keys(sources).sort(), titleCase);
        render();
      })
      .catch(function (err) {
        els.list.textContent = "";
        showError("Could not load " + MODE.noun + ": " + err.message);
      });
  }

  if (els.status) fillSelect(els.status, statuses, function (st) { return STATUS_LABELS[st] || titleCase(st); });
  var debounce;
  els.search.addEventListener("input", function () {
    clearTimeout(debounce);
    debounce = setTimeout(render, 150);
  });
  if (WW && els.worldwide) WW.fillFilterSelect(els.worldwide);
  if (WW && els.visa) WW.fillVisaSelect(els.visa);
  if (WW && els.remoteScope) WW.fillRemoteSelect(els.remoteScope);
  [els.recommendation, els.status, els.source, els.minScore, els.sort, els.worldwide, els.visa, els.remoteScope, els.outcome, els.evaluated].filter(Boolean).forEach(function (node) {
    node.addEventListener("change", render);
  });
  load();
})();
