/**
 * Worldwide-remote classification UI helpers, shared by the Remote Job Finder and
 * Saved Jobs pages (loaded only when JOB_WORLDWIDE_REMOTE_ENABLED is on).
 * Reads ``job.worldwide`` (jobEngine/matching/worldwide_remote.py). All scraped
 * text goes through textContent — never innerHTML.
 */
(function () {
  "use strict";

  var CATEGORIES = {
    worldwide_confirmed: { label: "Worldwide Remote — Confirmed", short: "Worldwide ✓", rank: 0 },
    india_eligible: { label: "International Remote — India Eligible", short: "India eligible ✓", rank: 1 },
    needs_verification: { label: "Potentially Worldwide — Needs Verification", short: "Verify eligibility", rank: 2 },
    insufficient_info: { label: "Insufficient Information", short: "Insufficient info", rank: 3 },
    india_local: { label: "India-based — Not Remote", short: "India on-site", rank: 4 },
    country_restricted: { label: "Country Restricted — Not Eligible from India", short: "Not eligible", rank: 5 },
  };
  var ARRANGEMENTS = {
    full_time_employee: "Full-time employee",
    full_time_contractor: "Full-time contractor",
    part_time_contractor: "Part-time contractor",
    contractor: "Contractor (hours not stated)",
    freelance: "Freelance / project-based",
    temporary_contract: "Temporary contract",
    part_time: "Part-time (not stated as contract)",
    other: "Other / unspecified",
  };
  var CONTRACTOR = ["full_time_contractor", "part_time_contractor", "contractor", "freelance", "temporary_contract"];
  var FOUND_IN = {
    title: "Job title",
    location: "LinkedIn location",
    workplace_type: "Workplace type",
    employment_type: "Employment type",
    description: "Job description",
  };
  var FILTERS = [
    ["", "All jobs"],
    ["worldwide", "Confirmed worldwide only"],
    ["india", "Confirmed eligible for India"],
    ["contractor", "Contractor jobs only"],
    ["part_time_contractor", "Part-time contractor"],
    ["full_time_contractor", "Full-time contractor"],
    ["verify", "Needs verification"],
  ];

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function info(job) {
    return (job && job.worldwide) || null;
  }

  function category(job) {
    var w = info(job);
    return w && CATEGORIES[w.category] ? w.category : null;
  }

  function badge(job) {
    var cat = category(job);
    if (!cat) return null;
    var w = info(job);
    var b = el("span", "ww-badge ww-badge--" + cat, CATEGORIES[cat].short);
    b.title = CATEGORIES[cat].label + (w.method === "llm" ? " (LLM-reviewed)" : "");
    return b;
  }

  function cardClass(job) {
    var cat = category(job);
    return cat ? "ww-card ww-card--" + cat : "";
  }

  function arrangementLabel(job) {
    var w = info(job);
    return w ? ARRANGEMENTS[w.arrangement] || ARRANGEMENTS.other : null;
  }

  /** Classification details: India eligibility, arrangement, reason, quoted evidence, search URLs. */
  function panel(job) {
    var cat = category(job);
    if (!cat) return null;
    var w = info(job);
    var wrap = el("div", "ww-panel ww-panel--" + cat);
    wrap.appendChild(el("div", "ww-panel__title", CATEGORIES[cat].label));

    var facts = el("dl", "ww-facts");
    [
      ["India eligibility", w.india_eligible ? "Confirmed" : "Not confirmed"],
      ["Arrangement", arrangementLabel(job)],
      ["Decided by", w.method === "llm" ? "Rules + LLM review (quotes verified against the posting)" : "Rules"],
    ].forEach(function (row) {
      facts.appendChild(el("dt", null, row[0]));
      facts.appendChild(el("dd", null, row[1]));
    });
    wrap.appendChild(facts);
    if (w.reason) wrap.appendChild(el("p", "ww-reason", w.reason));

    var evidence = w.evidence || [];
    if (evidence.length) {
      var list = el("ul", "ww-evidence");
      evidence.forEach(function (e) {
        var li = el("li", "ww-evidence__item ww-evidence__item--" + (e.signal || "neutral"));
        li.appendChild(el("span", "ww-evidence__where", (FOUND_IN[e.found_in] || e.found_in) + ":"));
        li.appendChild(el("q", null, e.text));
        list.appendChild(li);
      });
      wrap.appendChild(list);
    } else {
      wrap.appendChild(el("p", "ww-reason", "No supporting text found."));
    }

    var urls = (job && job.search_urls) || [];
    if (urls.length) {
      var found = el("details", "ww-urls");
      found.appendChild(el("summary", null, "Found by " + urls.length + " search URL" + (urls.length === 1 ? "" : "s")));
      urls.forEach(function (u) {
        var a = el("a", null, u);
        a.href = u;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        found.appendChild(a);
      });
      wrap.appendChild(found);
    }
    return wrap;
  }

  function matches(job, filter) {
    if (!filter) return true;
    var w = info(job);
    if (!w) return false;
    if (filter === "worldwide") return w.category === "worldwide_confirmed";
    if (filter === "india") return w.category === "worldwide_confirmed" || w.category === "india_eligible";
    if (filter === "contractor") return CONTRACTOR.indexOf(w.arrangement) !== -1;
    if (filter === "verify") return w.category === "needs_verification";
    return w.arrangement === filter;
  }

  /** Lower = shown first: confirmed worldwide, then India-eligible, ..., restricted last. */
  function rank(job) {
    var cat = category(job);
    return cat ? CATEGORIES[cat].rank : 9;
  }

  function fillFilterSelect(select) {
    FILTERS.forEach(function (f) {
      var opt = el("option", null, f[1]);
      opt.value = f[0];
      select.appendChild(opt);
    });
  }

  window.WorldwideRemote = {
    badge: badge,
    cardClass: cardClass,
    panel: panel,
    matches: matches,
    rank: rank,
    arrangementLabel: arrangementLabel,
    fillFilterSelect: fillFilterSelect,
  };
})();
