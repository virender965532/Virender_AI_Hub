/**
 * Global Remote Job Finder UI: submit a search against the existing job-engine
 * API (/api/jobs/search, unchanged) and render full job detail cards (salary,
 * employment type, skills, strengths/concerns, source/apply links).
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
      input.checked = s.checked === undefined ? !!s.enabled : !!s.checked;
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

  var SERVER_MAX_RESULTS = 500; // schema upper bound (jobEngine/api/schemas.py)

  // Filled by buildRequestPayload for the in-flight search; used by renderResult.
  var activeSearch = { minScore: 0, maxJobs: 10 };

  function buildRequestPayload() {
    // Max results = how many jobs to scrape from each search URL; -1 = all.
    var maxJobs = parseInt(maxResultsInput.value, 10);
    if (!(maxJobs === -1 || maxJobs > 0)) throw new Error("Max results must be -1 (all) or a positive number.");
    // Keywords only replace the keywords= parameter of the sources' search URLs.
    var searchKeywords = keywordsInput.value.trim();

    // Every filter runs server-side, so what is shown is exactly what is saved.
    var payload = { max_jobs: maxJobs, max_results: SERVER_MAX_RESULTS };
    if (searchKeywords) payload.search_keywords = searchKeywords;
    if (remoteOnlyInput.checked) payload.remote_only = true;
    // Always send the field's value so the engine default (JOB_MIN_MATCH_SCORE, 70) never applies.
    var minScore = minScoreInput.value.trim() === "" ? NaN : Number(minScoreInput.value);
    if (isNaN(minScore) || minScore < 0 || minScore > 100) throw new Error("Min match score must be between 0 and 100.");
    payload.minimum_match_score = minScore;
    activeSearch = { minScore: minScore, maxJobs: maxJobs };
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
    var metrics = result.metrics || {};
    var processed = metrics.jobs_processed != null ? metrics.jobs_processed : (summary.jobs_discovered || 0);
    var parts = [
      "<span><strong>" + (summary.sources_processed || 0) + "</strong> sources processed</span>",
      "<span><strong>" + (summary.jobs_discovered || 0) + "</strong> discovered</span>",
      "<span><strong>" + processed + "</strong> processed (" + (activeSearch.maxJobs === -1 ? "all" : "max " + activeSearch.maxJobs + " per search URL") + ")</span>",
      "<span><strong>" + (summary.jobs_validated || 0) + "</strong> matched filters</span>",
      "<span><strong>" + (summary.jobs_recommended || 0) + "</strong> scored ≥ " + activeSearch.minScore + " (shown &amp; saved)</span>",
    ];
    var filteredOut = metrics.jobs_filtered_out || {};
    var reasons = Object.keys(filteredOut).map(function (k) { return filteredOut[k] + " " + k.replace(/_/g, " "); });
    if (reasons.length) {
      parts.push('<span style="color:#ffb86b;">Filtered out: ' + reasons.join(", ") + "</span>");
    }
    if (summary.degraded) {
      parts.push('<span style="color:#ffb86b;">⚠ ' + (summary.degraded_reason || "degraded mode") + "</span>");
    }
    if ((result.source_errors || []).length) {
      parts.push("<span style=\"color:#ff9d9d;\">" + result.source_errors.length + " source error(s)</span>");
    }
    var linkedinAuthError = (result.source_errors || []).some(function (e) {
      return e.source_id === "linkedin" && (e.status === "auth_required" || e.status === "anti_bot_blocked");
    });
    if (linkedinAuthError && window.LinkedInConnect) {
      parts.push('<span style="color:#ffb86b;">LinkedIn needs you to sign in. ' +
        '<button type="button" class="btn btn--ghost" id="rjf-li-signin">Connect LinkedIn</button></span>');
    }
    summaryEl.innerHTML = parts.join("");
    var signinBtn = document.getElementById("rjf-li-signin");
    if (signinBtn) {
      signinBtn.addEventListener("click", function () {
        window.LinkedInConnect.open("LinkedIn asked for a sign-in during the last search. Sign in below, then run the search again.");
      });
    }

    cardsEl.innerHTML = "";
    var jobs = result.jobs || [];

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
      showError("Invalid search settings: " + err.message);
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
