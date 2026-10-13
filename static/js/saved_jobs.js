/**
 * Saved Jobs page: lists every job stored by past Remote Job Finder searches
 * (GET /api/saved-jobs) with client-side filter/sort, a per-job status, and a
 * full detail panel (LLM score, strengths, honest concerns, skills, description).
 * All scraped text goes through textContent — never innerHTML.
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

  var statuses = [];
  try {
    statuses = JSON.parse(document.getElementById("sj-statuses").textContent || "[]");
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

  /**
   * The reviewed score when the concern reviewer checked the job, else the LLM score
   * (older saves also kept a blended match_score).
   */
  function score(saved) {
    var llm = saved.llm_analysis;
    var s = saved.review ? saved.review.score
      : Number(llm && llm.score !== undefined && llm.score !== null ? llm.score : saved.match_score);
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
      status: els.status.value,
      source: els.source.value,
      minScore: Number(els.minScore.value) || 0,
      sort: els.sort.value,
      worldwide: els.worldwide ? els.worldwide.value : "",
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
    }[f.sort];
    jobs.sort(function (a, b) {
      // Applied jobs sink to the bottom so the list leads with jobs still to look at.
      var applied = isApplied(a) - isApplied(b);
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
    if (allJobs.length) {
      var clearBtn = el("button", "btn btn--ghost sj-clear", "Remove all jobs");
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

  // --- concern review (jobEngine/matching/concern_reviewer.py) -----------------

  var VERDICT_LABELS = { met: "Met", implied: "Implied", partial: "Partial", missing: "Missing" };
  var TIER_LABELS = { core: "Core", required: "Required", preferred: "Preferred", nice_to_have: "Nice to have" };
  var DECISION_LABELS = {
    drop_implied: "Implied by",
    drop_covered: "Shown in profile",
  };

  function keptItems(review) {
    return (review.items || []).filter(function (i) { return i.decision === "keep" || i.decision === "downgrade"; });
  }

  function scoreBlock(saved) {
    var review = saved.review;
    var wrap = el("div");
    wrap.appendChild(scoreBar(score(saved), review ? "Reviewed score" : "LLM score"));
    if (review) {
      var dropped = (review.items || []).length - keptItems(review).length;
      var basis = review.score_basis === "requirements"
        ? "computed from " + (review.requirements || []).length + " requirements"
        : "kept the matcher's score (no requirements found)";
      wrap.appendChild(el("p", "sj-review-note",
        "Matcher scored " + Math.round(review.score_before) + "; the reviewer dismissed " + dropped +
        " concern" + (dropped === 1 ? "" : "s") + " and the score was " + basis + "."));
    }
    return wrap;
  }

  /** Kept concerns with their severity; dismissed ones in a collapsed list with the reason. */
  function concernsBlock(review) {
    var wrap = el("div");
    var kept = keptItems(review).filter(function (i) { return i.kind !== "missing_skill"; });
    if (kept.length) {
      var ul = el("ul", "sj-points");
      kept.forEach(function (i) {
        var li = el("li");
        li.appendChild(el("span", "sj-sev sj-sev--" + i.severity, i.severity));
        li.appendChild(document.createTextNode(" " + i.item));
        if (i.kind === "added_concern") li.appendChild(el("span", "sj-review-tag", "added by reviewer"));
        // Why code downgraded it, e.g. "GraphQL is listed in the profile".
        if (i.decision === "downgrade" && i.adjusted) li.appendChild(el("div", "sj-review-note", "Downgraded: " + i.adjusted));
        ul.appendChild(li);
      });
      wrap.appendChild(ul);
    } else {
      wrap.appendChild(el("p", "sj-review-note", "No concerns left after review."));
    }
    var dismissed = (review.items || []).filter(function (i) { return i.decision !== "keep" && i.decision !== "downgrade"; });
    if (dismissed.length) {
      var details = el("details", "sj-dismissed");
      details.appendChild(el("summary", null, "Dismissed by reviewer (" + dismissed.length + ")"));
      var list = el("ul", "sj-points");
      dismissed.forEach(function (i) {
        var li = el("li", null, i.item);
        var why = i.decision === "drop_implied" && i.implied_by
          ? DECISION_LABELS.drop_implied + " " + i.implied_by
          : DECISION_LABELS[i.decision] || titleCase(i.decision);
        if (i.reason) why += " — " + i.reason;
        li.appendChild(el("div", "sj-review-note", why));
        list.appendChild(li);
      });
      details.appendChild(list);
      wrap.appendChild(details);
    }
    return wrap;
  }

  function requirementsBlock(review) {
    var ul = el("ul", "sj-points sj-reqs");
    (review.requirements || []).forEach(function (r) {
      var li = el("li");
      li.appendChild(el("span", "sj-verdict sj-verdict--" + r.verdict, VERDICT_LABELS[r.verdict] || r.verdict));
      li.appendChild(document.createTextNode(" " + r.requirement + " "));
      li.appendChild(el("span", "sj-review-tag", TIER_LABELS[r.tier] || r.tier));
      if (r.verdict === "implied" && r.implied_by) li.appendChild(el("span", "sj-review-tag", "via " + r.implied_by));
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

  function buildBody(saved, card) {
    var job = saved.job || {};
    var llm = saved.llm_analysis || null;
    var body = el("div", "sj-job__body");
    var wwPanel = WW && WW.panel(job);
    if (wwPanel) body.appendChild(section("Eligibility from India", wwPanel));

    var explanation = saved.explanation || (llm && llm.explanation);
    if (explanation) body.appendChild(section("Why it matches", el("p", "sj-explanation", explanation)));
    var review = saved.review || null;
    if (llm) body.appendChild(section("Score", scoreBlock(saved)));

    var cols = el("div", "sj-cols");
    var strengths = saved.strengths || (llm && llm.strengths) || [];
    var concerns = saved.concerns || (llm && llm.concerns) || [];
    if (strengths.length) cols.appendChild(section("Strengths", pointList(strengths)));
    if (review) cols.appendChild(section("Honest concerns", concernsBlock(review)));
    else if (concerns.length) cols.appendChild(section("Honest concerns", pointList(concerns)));
    if (cols.childNodes.length) body.appendChild(cols);
    if (review && (review.requirements || []).length) {
      body.appendChild(section("Requirements check", requirementsBlock(review)));
    }

    var skills = el("div", "sj-cols");
    if ((saved.matched_skills || []).length) skills.appendChild(section("Matched skills", tagList(saved.matched_skills)));
    if ((saved.missing_required_skills || []).length) {
      skills.appendChild(section("Missing skills", tagList(saved.missing_required_skills, "missing")));
    }
    if ((job.skills || []).length) skills.appendChild(section("Listed skills", tagList(job.skills, "neutral")));
    if (skills.childNodes.length) body.appendChild(skills);

    var reqs = el("div", "sj-cols");
    // The reviewer's requirements check supersedes the matcher's list.
    if (!(review && (review.requirements || []).length) && llm && (llm.matched_requirements || []).length) {
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
        if (saved.status !== "applied") setStatus(saved, card, "applied");
      }
      apply.addEventListener("click", markApplied);
      apply.addEventListener("auxclick", markApplied);
      actions.appendChild(apply);
    }
    var del = el("button", "btn btn--ghost", "Remove from saved");
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
    card.classList.toggle("sj-job--applied", applied);
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
    badge.appendChild(el("small", null, saved.review ? "reviewed" : "score"));
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
    var wwBadge = WW && WW.badge(job);
    if (wwBadge) titleRow.appendChild(wwBadge);
    (job.tags || []).forEach(function (t) { titleRow.appendChild(el("span", "sj-pill sj-pill--tag", t)); });
    main.appendChild(titleRow);
    var meta = el("p", "sj-job__meta");
    [job.company, job.location, relTime(stampIso)]
      .filter(Boolean)
      .forEach(function (t) { meta.appendChild(el("span", null, t)); });
    main.appendChild(meta);
    head.appendChild(main);

    var side = el("div", "sj-job__side");
    side.appendChild(el("span", "sj-pill", RECOMMENDATION_LABELS[rec] || titleCase(rec)));
    side.appendChild(statusSelect(saved));
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
      empty.appendChild(document.createTextNode("No saved jobs yet. Run a search on the "));
      var link = el("a", null, "Global Remote Job Finder");
      link.href = "/remote-job-finder";
      empty.appendChild(link);
      empty.appendChild(document.createTextNode(" and its matches will appear here."));
      els.list.appendChild(empty);
      return;
    }
    if (!jobs.length) {
      els.list.appendChild(el("div", "sj-empty glass", "No saved jobs match these filters."));
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
    return "/api/saved-jobs/" + encodeURIComponent(saved.saved_key) + (suffix || "");
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

  function removeJob(saved, card) {
    var title = (saved.job || {}).title || "this job";
    if (!window.confirm("Remove “" + title + "” from saved jobs?")) return;
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
    if (!window.confirm("Remove all " + n + " saved jobs? This cannot be undone.")) return;
    button.disabled = true;
    fetch("/api/saved-jobs", { method: "DELETE" })
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
    els.list.appendChild(el("div", "sj-empty glass", "Loading saved jobs…"));
    fetch("/api/saved-jobs")
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
        showError("Could not load saved jobs: " + err.message);
      });
  }

  fillSelect(els.status, statuses, function (st) { return STATUS_LABELS[st] || titleCase(st); });
  var debounce;
  els.search.addEventListener("input", function () {
    clearTimeout(debounce);
    debounce = setTimeout(render, 150);
  });
  if (WW && els.worldwide) WW.fillFilterSelect(els.worldwide);
  [els.recommendation, els.status, els.source, els.minScore, els.sort, els.worldwide].filter(Boolean).forEach(function (node) {
    node.addEventListener("change", render);
  });
  load();
})();
