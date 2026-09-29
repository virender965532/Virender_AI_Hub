/**
 * Global Remote Job Finder UI: submit a search against the existing job-engine
 * API (/api/jobs/search, unchanged) and render full job detail cards (salary,
 * employment type, skills, strengths/concerns, source/apply links) instead of
 * the compact cards used by the Job Discovery Engine page.
 */
(function () {
  var loadingEl = document.getElementById("rjf-loading");
  var errorEl = document.getElementById("rjf-error");
  var resultsEl = document.getElementById("rjf-results");
  var summaryEl = document.getElementById("rjf-summary");
  var cardsEl = document.getElementById("rjf-cards");
  var sourcesEl = document.getElementById("rjf-sources");
  var searchBtn = document.getElementById("rjf-btn-search");
  var loadingTextEl = document.getElementById("rjf-loading-text");

  var keywordsInput = document.getElementById("rjf-keywords");
  var remoteOnlyInput = document.getElementById("rjf-remote-only");
  var minScoreInput = document.getElementById("rjf-min-score");
  var maxResultsInput = document.getElementById("rjf-max-results");

  var pollTimer = null;

  function loadJsonScript(id) {
    var el = document.getElementById(id);
    if (!el) return [];
    try {
      var data = JSON.parse(el.textContent || "[]");
      return Array.isArray(data) ? data : [];
    } catch (e) {
      console.warn("Could not parse " + id, e);
      return [];
    }
  }

  function renderSourceCheckboxes() {
    var sources = loadJsonScript("rjf-sources-data");
    sourcesEl.innerHTML = "";
    if (!sources.length) {
      sourcesEl.innerHTML = '<span class="job-toolbar-note">No sources configured.</span>';
      return;
    }
    sources.forEach(function (s) {
      var label = document.createElement("label");
      label.className = "job-toolbar-check" + (s.enabled ? "" : " job-toolbar-check--disabled");
      var input = document.createElement("input");
      input.type = "checkbox";
      input.value = s.id;
      input.checked = !!s.enabled;
      input.disabled = !s.enabled;
      input.className = "rjf-source-checkbox";
      var span = document.createElement("span");
      span.textContent = s.name + (s.requires_auth ? " (auth required)" : "") + (s.enabled ? "" : " — disabled");
      label.appendChild(input);
      label.appendChild(span);
      sourcesEl.appendChild(label);
    });
  }

  function setVisible(el, show) {
    el.classList.toggle("hidden", !show);
  }

  function selectedSources() {
    var checked = Array.prototype.slice
      .call(sourcesEl.querySelectorAll(".rjf-source-checkbox"))
      .filter(function (el) { return el.checked; })
      .map(function (el) { return el.value; });
    return checked.length ? checked : null;
  }

  function parseKeywords(raw) {
    return String(raw || "")
      .split(",")
      .map(function (s) { return s.trim(); })
      .filter(Boolean);
  }

  // ---------------------------------------------------------------------------
  // Boolean keyword queries, e.g.
  //   ("Senior Full Stack Engineer" OR "Senior Full-Stack Developer") AND (React OR Next.js) NOT Java
  // The job-engine API only supports an any-of keyword list, so the query's
  // positive terms are sent as that (coarse) server-side filter and the full
  // expression is evaluated here against each returned job.
  // Precedence: NOT > AND > OR. Adjacent terms without an operator are ANDed.
  // Operators must be uppercase; they may be glued to the previous word as in
  // pasted text like "AWSNOT Python" or ")AND (".
  // ---------------------------------------------------------------------------

  var SERVER_MAX_RESULTS = 500; // schema upper bound (jobEngine/api/schemas.py)
  var SERVER_MAX_KEYWORDS = 20;

  function isBooleanQuery(raw) {
    return /[()"]|\b(AND|OR|NOT)\b|[^\s(]NOT(?=[\s("]|$)/.test(String(raw || ""));
  }

  function tokenizeQuery(raw) {
    var tokens = [];
    var re = /\s*(?:(\()|(\))|"([^"]*)"|([^\s()"]+))/g;
    var m;
    while ((m = re.exec(raw)) !== null) {
      if (m[0].trim() === "") break;
      if (m[1]) tokens.push({ type: "(" });
      else if (m[2]) tokens.push({ type: ")" });
      else if (m[3] !== undefined) {
        if (m[3].trim()) tokens.push({ type: "term", value: m[3].trim() });
      } else {
        var word = m[4];
        if (word === "AND" || word === "OR" || word === "NOT") {
          tokens.push({ type: word });
          continue;
        }
        // "AWSNOT" / "PythonNOT" from pasted text with the spaces lost.
        var glued = /^(.+?)(AND|NOT)$/.exec(word);
        if (glued) {
          tokens.push({ type: "term", value: glued[1] });
          tokens.push({ type: glued[2] });
        } else {
          tokens.push({ type: "term", value: word });
        }
      }
    }
    return tokens;
  }

  function parseQuery(raw) {
    var tokens = tokenizeQuery(String(raw || ""));
    var pos = 0;

    function peek() { return tokens[pos]; }
    function startsOperand(tok) {
      return tok && (tok.type === "term" || tok.type === "(" || tok.type === "NOT");
    }

    function parseOr() {
      var node = parseAnd();
      while (peek() && peek().type === "OR") {
        pos++;
        node = { op: "or", left: node, right: parseAnd() };
      }
      return node;
    }

    function parseAnd() {
      var node = parseUnary();
      while (peek() && (peek().type === "AND" || startsOperand(peek()))) {
        if (peek().type === "AND") pos++;
        node = { op: "and", left: node, right: parseUnary() };
      }
      return node;
    }

    function parseUnary() {
      var tok = peek();
      if (!tok) throw new Error("Query ends unexpectedly.");
      if (tok.type === "NOT") {
        pos++;
        return { op: "not", child: parseUnary() };
      }
      if (tok.type === "(") {
        pos++;
        var inner = parseOr();
        if (!peek() || peek().type !== ")") throw new Error("Missing closing parenthesis.");
        pos++;
        return inner;
      }
      if (tok.type === "term") {
        pos++;
        return { op: "term", value: tok.value };
      }
      throw new Error("Unexpected \"" + tok.type + "\" in query.");
    }

    if (!tokens.length) return null;
    var ast = parseOr();
    if (pos < tokens.length) throw new Error("Unexpected \"" + tokens[pos].type + "\" in query.");
    return ast;
  }

  function positiveTerms(node, negated, out) {
    if (!node) return out;
    if (node.op === "term") {
      if (!negated && out.indexOf(node.value) === -1) out.push(node.value);
    } else if (node.op === "not") {
      positiveTerms(node.child, !negated, out);
    } else {
      positiveTerms(node.left, negated, out);
      positiveTerms(node.right, negated, out);
    }
    return out;
  }

  function escapeRegExp(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  // Whole-word match so "Java" does not hit "JavaScript"; the boundary is only
  // enforced on alphanumeric edges so ".NET" still matches "ASP.NET".
  var termRegexCache = {};
  function termRegex(term) {
    var key = term.toLowerCase();
    if (!termRegexCache[key]) {
      var start = /^[a-z0-9]/.test(key) ? "(?:^|[^a-z0-9])" : "";
      var end = /[a-z0-9]$/.test(key) ? "(?=$|[^a-z0-9])" : "";
      var body = escapeRegExp(key).replace(/[\s-]+/g, "[\\s-]+");
      termRegexCache[key] = new RegExp(start + body + end);
    }
    return termRegexCache[key];
  }

  function evaluateQuery(node, text) {
    if (!node) return true;
    switch (node.op) {
      case "term": return termRegex(node.value).test(text);
      case "not": return !evaluateQuery(node.child, text);
      case "and": return evaluateQuery(node.left, text) && evaluateQuery(node.right, text);
      case "or": return evaluateQuery(node.left, text) || evaluateQuery(node.right, text);
    }
    return true;
  }

  function jobSearchText(job) {
    return [
      job.title || "",
      job.description || "",
      (job.skills || []).join(" "),
      (job.requirements || []).join(" "),
    ].join(" ").toLowerCase();
  }

  // Filled by buildRequestPayload for the in-flight search; used by renderResult.
  var activeSearch = { query: null, maxResults: 10 };

  function buildRequestPayload() {
    var maxResults = Number(maxResultsInput.value) || 10;
    var raw = keywordsInput.value;
    var query = null;
    var keywords;
    if (isBooleanQuery(raw)) {
      query = parseQuery(raw);
      keywords = positiveTerms(query, false, []).slice(0, SERVER_MAX_KEYWORDS);
    } else {
      keywords = parseKeywords(raw);
    }
    activeSearch = { query: query, maxResults: maxResults };

    // Client-side filtering (boolean query / remote only) runs after the
    // server truncates to max_results, so fetch the full ranked list then.
    var filtersLocally = !!query || remoteOnlyInput.checked;
    var payload = { max_results: filtersLocally ? SERVER_MAX_RESULTS : maxResults };
    if (keywords.length) payload.keywords = keywords;
    if (remoteOnlyInput.checked) payload.remote = true;
    var minScore = Number(minScoreInput.value);
    if (!isNaN(minScore)) payload.minimum_match_score = minScore;
    var sources = selectedSources();
    if (sources) payload.sources = sources;
    return payload;
  }

  function scoreClass(recommendation) {
    if (recommendation === "strong_match") return "rjf-score--strong";
    if (recommendation === "good_match") return "rjf-score--good";
    if (recommendation === "possible_match") return "rjf-score--possible";
    return "rjf-score--poor";
  }

  function recommendationLabel(recommendation) {
    return String(recommendation || "").replace(/_/g, " ");
  }

  function isRemoteFriendly(job) {
    return job.remote_type === "remote" || job.remote_type === "hybrid";
  }

  function detailRow(label, value) {
    if (!value) return "";
    return "<dt>" + label + "</dt><dd>" + value + "</dd>";
  }

  function renderCard(matched) {
    var job = matched.job || {};
    var card = document.createElement("article");
    card.className = "rjf-card glass";

    var head = document.createElement("div");
    head.className = "rjf-card__head";

    var titleWrap = document.createElement("div");
    var title = document.createElement("h3");
    title.className = "rjf-card__title";
    title.textContent = job.title || "Untitled role";
    var company = document.createElement("p");
    company.className = "rjf-card__company";
    company.textContent = [job.company, job.location].filter(Boolean).join(" · ");
    titleWrap.appendChild(title);
    titleWrap.appendChild(company);

    var score = document.createElement("div");
    score.className = "rjf-score " + scoreClass(matched.recommendation);
    score.innerHTML = Math.round(matched.match_score || 0) + "<small>MATCH</small>";

    head.appendChild(titleWrap);
    head.appendChild(score);
    card.appendChild(head);

    var recBadge = document.createElement("span");
    recBadge.className = "rjf-tag";
    recBadge.textContent = recommendationLabel(matched.recommendation);
    card.appendChild(recBadge);

    if (matched.explanation) {
      var explanation = document.createElement("p");
      explanation.className = "rjf-card__explanation";
      explanation.textContent = matched.explanation;
      card.appendChild(explanation);
    }

    var details = document.createElement("dl");
    details.className = "rjf-card__details";
    details.innerHTML =
      detailRow("Employment type", job.employment_type) +
      detailRow("Experience", job.experience_required) +
      detailRow("Salary", job.salary) +
      detailRow("Remote type", job.remote_type) +
      detailRow("Source", job.source) +
      detailRow("Posted", job.posted_date ? new Date(job.posted_date).toLocaleDateString() : "");
    if (details.innerHTML) card.appendChild(details);

    if ((matched.matched_skills || []).length || (matched.missing_required_skills || []).length) {
      var tagRow = document.createElement("div");
      tagRow.className = "rjf-tag-row";
      (matched.matched_skills || []).forEach(function (skill) {
        var tag = document.createElement("span");
        tag.className = "rjf-tag";
        tag.textContent = skill;
        tagRow.appendChild(tag);
      });
      (matched.missing_required_skills || []).forEach(function (skill) {
        var tag = document.createElement("span");
        tag.className = "rjf-tag rjf-tag--missing";
        tag.textContent = "missing: " + skill;
        tagRow.appendChild(tag);
      });
      card.appendChild(tagRow);
    }

    if ((matched.strengths || []).length) {
      var strengths = document.createElement("p");
      strengths.className = "rjf-card__section";
      strengths.innerHTML = "<strong>Strengths:</strong> " + matched.strengths.join("; ");
      card.appendChild(strengths);
    }
    if ((matched.concerns || []).length) {
      var concerns = document.createElement("p");
      concerns.className = "rjf-card__section";
      concerns.innerHTML = "<strong>Concerns:</strong> " + matched.concerns.join("; ");
      card.appendChild(concerns);
    }
    if (job.description) {
      var desc = document.createElement("p");
      desc.className = "rjf-card__section";
      var text = job.description.length > 320 ? job.description.slice(0, 320) + "…" : job.description;
      desc.innerHTML = "<strong>Description:</strong> " + text;
      card.appendChild(desc);
    }

    var footer = document.createElement("div");
    footer.className = "rjf-card__footer";
    if (job.application_url) {
      var applyLink = document.createElement("a");
      applyLink.href = job.application_url;
      applyLink.target = "_blank";
      applyLink.rel = "noopener noreferrer";
      applyLink.className = "btn btn--ghost";
      applyLink.textContent = "View / Apply";
      footer.appendChild(applyLink);
    }
    if (job.source_url && job.source_url !== job.application_url) {
      var sourceLink = document.createElement("a");
      sourceLink.href = job.source_url;
      sourceLink.target = "_blank";
      sourceLink.rel = "noopener noreferrer";
      sourceLink.className = "btn btn--ghost";
      sourceLink.textContent = "Source listing";
      footer.appendChild(sourceLink);
    }
    card.appendChild(footer);

    return card;
  }

  function renderResult(result) {
    var summary = result.summary || {};
    var parts = [
      "<span><strong>" + (summary.sources_processed || 0) + "</strong> sources processed</span>",
      "<span><strong>" + (summary.jobs_discovered || 0) + "</strong> discovered</span>",
      "<span><strong>" + (summary.jobs_recommended || 0) + "</strong> recommended</span>",
    ];
    if (summary.degraded) {
      parts.push('<span style="color:#ffb86b;">⚠ ' + (summary.degraded_reason || "degraded mode") + "</span>");
    }
    if ((result.source_errors || []).length) {
      parts.push("<span style=\"color:#ff9d9d;\">" + result.source_errors.length + " source error(s)</span>");
    }
    summaryEl.innerHTML = parts.join("");

    cardsEl.innerHTML = "";
    var jobs = result.jobs || [];
    if (remoteOnlyInput.checked) {
      jobs = jobs.filter(function (matched) { return isRemoteFriendly(matched.job || {}); });
    }
    if (activeSearch.query) {
      jobs = jobs.filter(function (matched) {
        return evaluateQuery(activeSearch.query, jobSearchText(matched.job || {}));
      });
    }
    jobs = jobs.slice(0, activeSearch.maxResults);
    if (activeSearch.query || remoteOnlyInput.checked) {
      summaryEl.innerHTML += "<span><strong>" + jobs.length + "</strong> shown after filters</span>";
    }

    if (!jobs.length) {
      var empty = document.createElement("div");
      empty.className = "rjf-empty glass";
      empty.textContent = "No jobs matched the selected filters for this search.";
      cardsEl.appendChild(empty);
    } else {
      jobs.forEach(function (matched) { cardsEl.appendChild(renderCard(matched)); });
    }

    setVisible(resultsEl, true);
  }

  function showError(message) {
    errorEl.textContent = message;
    setVisible(errorEl, true);
    setVisible(loadingEl, false);
    searchBtn.disabled = false;
  }

  function poll(runId) {
    fetch("/api/jobs/search/" + encodeURIComponent(runId))
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (!data.ok) {
          showError(data.error || "Search run not found.");
          return;
        }
        if (data.status === "running") {
          pollTimer = window.setTimeout(function () { poll(runId); }, 2000);
          return;
        }
        setVisible(loadingEl, false);
        searchBtn.disabled = false;
        if (data.status === "failed" && !(data.jobs || []).length) {
          showError("The search run failed. Check server logs for details.");
          return;
        }
        renderResult(data);
      })
      .catch(function (err) {
        showError("Could not reach the server: " + err);
      });
  }

  function startSearch() {
    if (pollTimer) window.clearTimeout(pollTimer);
    var payload;
    try {
      payload = buildRequestPayload();
    } catch (err) {
      showError("Invalid keyword query: " + err.message);
      return;
    }
    setVisible(errorEl, false);
    setVisible(resultsEl, false);
    setVisible(loadingEl, true);
    loadingTextEl.textContent = "Starting discovery workflow…";
    searchBtn.disabled = true;

    fetch("/api/jobs/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (res) { return res.json().then(function (data) { return { status: res.status, data: data }; }); })
      .then(function (result) {
        if (result.status >= 400 || !result.data.ok) {
          var detail = (result.data.details || []).map(function (d) { return d.msg; }).join("; ");
          showError(result.data.error + (detail ? ": " + detail : ""));
          return;
        }
        loadingTextEl.textContent = "Discovery running (run " + result.data.run_id + ")…";
        poll(result.data.run_id);
      })
      .catch(function (err) {
        showError("Could not reach the server: " + err);
      });
  }

  renderSourceCheckboxes();
  searchBtn.addEventListener("click", startSearch);
})();
