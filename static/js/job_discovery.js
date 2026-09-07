/**
 * AI Job Discovery Engine UI: submit a search, poll run status, render matched job cards.
 */
(function () {
  var loadingEl = document.getElementById("jd-loading");
  var errorEl = document.getElementById("jd-error");
  var resultsEl = document.getElementById("jd-results");
  var summaryEl = document.getElementById("jd-summary");
  var cardsEl = document.getElementById("jd-cards");
  var sourcesEl = document.getElementById("jd-sources");
  var searchBtn = document.getElementById("jd-btn-search");
  var loadingTextEl = document.getElementById("jd-loading-text");

  var keywordsInput = document.getElementById("jd-keywords");
  var locationInput = document.getElementById("jd-location");
  var remoteInput = document.getElementById("jd-remote");
  var minScoreInput = document.getElementById("jd-min-score");
  var maxResultsInput = document.getElementById("jd-max-results");

  var pollTimer = null;

  function loadSources() {
    var el = document.getElementById("jd-sources-data");
    if (!el) return [];
    try {
      var data = JSON.parse(el.textContent || "[]");
      return Array.isArray(data) ? data : [];
    } catch (e) {
      console.warn("Could not parse job sources", e);
      return [];
    }
  }

  function renderSourceCheckboxes() {
    var sources = loadSources();
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
      input.className = "jd-source-checkbox";
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
      .call(sourcesEl.querySelectorAll(".jd-source-checkbox"))
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

  function buildRequestPayload() {
    var payload = { max_results: Number(maxResultsInput.value) || 50 };
    var keywords = parseKeywords(keywordsInput.value);
    if (keywords.length) payload.keywords = keywords;
    var location = locationInput.value.trim();
    if (location) payload.location = location;
    if (remoteInput.checked) payload.remote = true;
    var minScore = Number(minScoreInput.value);
    if (!isNaN(minScore)) payload.minimum_match_score = minScore;
    var sources = selectedSources();
    if (sources) payload.sources = sources;
    return payload;
  }

  function scoreClass(recommendation) {
    if (recommendation === "strong_match") return "jd-score--strong";
    if (recommendation === "good_match") return "jd-score--good";
    if (recommendation === "possible_match") return "jd-score--possible";
    return "jd-score--poor";
  }

  function recommendationLabel(recommendation) {
    return String(recommendation || "").replace(/_/g, " ");
  }

  function renderCard(matched) {
    var job = matched.job || {};
    var card = document.createElement("article");
    card.className = "jd-card glass";

    var head = document.createElement("div");
    head.className = "jd-card__head";

    var titleWrap = document.createElement("div");
    var title = document.createElement("h3");
    title.className = "jd-card__title";
    title.textContent = job.title || "Untitled role";
    var company = document.createElement("p");
    company.className = "jd-card__company";
    company.textContent = [job.company, job.location].filter(Boolean).join(" · ");
    titleWrap.appendChild(title);
    titleWrap.appendChild(company);

    var score = document.createElement("div");
    score.className = "jd-score " + scoreClass(matched.recommendation);
    score.innerHTML = Math.round(matched.match_score || 0) + "<small>MATCH</small>";

    head.appendChild(titleWrap);
    head.appendChild(score);
    card.appendChild(head);

    var recBadge = document.createElement("span");
    recBadge.className = "jd-tag";
    recBadge.textContent = recommendationLabel(matched.recommendation);
    card.appendChild(recBadge);

    if (matched.explanation) {
      var explanation = document.createElement("p");
      explanation.className = "jd-card__explanation";
      explanation.textContent = matched.explanation;
      card.appendChild(explanation);
    }

    if ((matched.matched_skills || []).length || (matched.missing_required_skills || []).length) {
      var tagRow = document.createElement("div");
      tagRow.className = "jd-tag-row";
      (matched.matched_skills || []).forEach(function (skill) {
        var tag = document.createElement("span");
        tag.className = "jd-tag";
        tag.textContent = skill;
        tagRow.appendChild(tag);
      });
      (matched.missing_required_skills || []).forEach(function (skill) {
        var tag = document.createElement("span");
        tag.className = "jd-tag jd-tag--missing";
        tag.textContent = "missing: " + skill;
        tagRow.appendChild(tag);
      });
      card.appendChild(tagRow);
    }

    var meta = document.createElement("div");
    meta.className = "jd-card__meta";
    meta.innerHTML =
      "<span>Source: " + (job.source || "—") + "</span>" +
      "<span>Remote: " + (job.remote_type || "unknown") + "</span>";
    card.appendChild(meta);

    var footer = document.createElement("div");
    footer.className = "jd-card__footer";
    if (job.application_url) {
      var applyLink = document.createElement("a");
      applyLink.href = job.application_url;
      applyLink.target = "_blank";
      applyLink.rel = "noopener noreferrer";
      applyLink.className = "btn btn--ghost";
      applyLink.textContent = "View / Apply";
      footer.appendChild(applyLink);
    }
    card.appendChild(footer);

    return card;
  }

  function renderResult(result) {
    var summary = result.summary || {};
    var parts = [
      "<span><strong>" + (summary.sources_processed || 0) + "</strong> sources processed</span>",
      "<span><strong>" + (summary.jobs_discovered || 0) + "</strong> discovered</span>",
      "<span><strong>" + (summary.duplicates_removed || 0) + "</strong> duplicates removed</span>",
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
    if (!jobs.length) {
      var empty = document.createElement("div");
      empty.className = "jd-empty glass";
      empty.textContent = "No jobs met the minimum match score for this search.";
      cardsEl.appendChild(empty);
    } else {
      jobs.forEach(function (matched) {
        cardsEl.appendChild(renderCard(matched));
      });
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
    setVisible(errorEl, false);
    setVisible(resultsEl, false);
    setVisible(loadingEl, true);
    loadingTextEl.textContent = "Starting discovery workflow…";
    searchBtn.disabled = true;

    fetch("/api/jobs/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildRequestPayload()),
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
