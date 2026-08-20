/**
 * Fetch Naukri jobs, show loading state, sort/filter in the browser.
 * sortJobs / renderJobs / openJob are exposed globally for HTML hooks.
 */
(function () {
  var loadingEl = document.getElementById("loading-state");
  var errorEl = document.getElementById("error-state");
  var resultsEl = document.getElementById("results-state");
  var tbody = document.getElementById("job-tbody");
  var countEl = document.getElementById("job-count");
  var thresholdNoteEl = document.getElementById("job-threshold-note");
  var refreshBtn = document.getElementById("btn-refresh");
  var fetchBtn = document.getElementById("btn-fetch");
  var sortSelect = document.getElementById("sort-select");
  var keywordInput = document.getElementById("job-keyword");
  var searchKeywordInput = document.getElementById("search-keyword");
  var jobAgeInput = document.getElementById("job-age");
  var noOfJobsInput = document.getElementById("no-of-jobs");
  var maxPagesInput = document.getElementById("max-pages");
  var relevanceInput = document.getElementById("relevance-min");
  var ctcGridEl = document.getElementById("ctc-filter-grid");
  var urlPreviewEl = document.getElementById("naukri-url-preview");
  var enrichChk = document.getElementById("chk-enrich-jd");

  function loadDefaults() {
    var el = document.getElementById("naukri-defaults");
    if (!el) return {};
    try {
      return JSON.parse(el.textContent || "{}") || {};
    } catch (e) {
      console.warn("Could not parse naukri defaults", e);
      return {};
    }
  }

  var defaults = loadDefaults();
  var DEFAULT_SEARCH_KEYWORD = defaults.keyword || "javascript";
  var DEFAULT_JOB_AGE = String(defaults.job_age != null ? defaults.job_age : "3");
  var DEFAULT_NO_OF_JOBS = Number(defaults.no_of_jobs) || 25;
  var DEFAULT_MAX_PAGES = Number(defaults.max_pages) || 100;
  var DEFAULT_RELEVANCE = Number(defaults.relevance_min_pct);
  if (isNaN(DEFAULT_RELEVANCE)) DEFAULT_RELEVANCE = 80;
  var DEFAULT_CTC = Array.isArray(defaults.ctc_filters) ? defaults.ctc_filters.slice() : [];
  var CTC_OPTIONS = Array.isArray(defaults.ctc_options) ? defaults.ctc_options : [];

  /** Full list from API */
  var allJobs = [];
  /** Current sort key */
  var currentSort = "";

  function setVisible(el, show) {
    el.classList.toggle("hidden", !show);
  }

  /**
   * Map free-text keyword to Naukri path slug.
   * javascript -> javascript, node.js -> node-dot-js, node js -> node-js
   */
  function keywordToNaukriSlug(keyword) {
    var raw = String(keyword == null ? "" : keyword)
      .trim()
      .toLowerCase();
    if (!raw) raw = DEFAULT_SEARCH_KEYWORD;
    var slug = raw
      .replace(/\./g, "-dot-")
      .replace(/\s+/g, "-")
      .replace(/-{2,}/g, "-")
      .replace(/^-+|-+$/g, "");
    return slug || DEFAULT_SEARCH_KEYWORD;
  }

  function buildNaukriJobsUrl(keyword) {
    return "https://www.naukri.com/" + keywordToNaukriSlug(keyword) + "-jobs";
  }

  function getSearchKeyword() {
    var value =
      searchKeywordInput && searchKeywordInput.value
        ? searchKeywordInput.value.trim()
        : "";
    return value || DEFAULT_SEARCH_KEYWORD;
  }

  function getSelectedCtcFilters() {
    if (!ctcGridEl) return DEFAULT_CTC.slice();
    var checked = ctcGridEl.querySelectorAll('input[type="checkbox"][data-ctc]:checked');
    var out = [];
    checked.forEach(function (chk) {
      out.push(chk.getAttribute("data-ctc"));
    });
    return out;
  }

  function getJobAge() {
    var v = jobAgeInput && jobAgeInput.value ? String(jobAgeInput.value).trim() : "";
    return v || DEFAULT_JOB_AGE;
  }

  function getNoOfJobs() {
    var n = noOfJobsInput ? Number(noOfJobsInput.value) : DEFAULT_NO_OF_JOBS;
    if (isNaN(n) || n < 1) return DEFAULT_NO_OF_JOBS;
    return Math.floor(n);
  }

  function getMaxPages() {
    var n = maxPagesInput ? Number(maxPagesInput.value) : DEFAULT_MAX_PAGES;
    if (isNaN(n) || n < 1) return DEFAULT_MAX_PAGES;
    return Math.floor(n);
  }

  function getRelevanceMin() {
    var n = relevanceInput ? Number(relevanceInput.value) : DEFAULT_RELEVANCE;
    if (isNaN(n)) return DEFAULT_RELEVANCE;
    return n;
  }

  function buildPreviewQuery() {
    var params = [];
    params.push("k=" + encodeURIComponent(getSearchKeyword()));
    params.push("jobAge=" + encodeURIComponent(getJobAge()));
    getSelectedCtcFilters().forEach(function (band) {
      params.push("ctcFilter=" + encodeURIComponent(band));
    });
    return "?" + params.join("&");
  }

  function updateUrlPreview() {
    if (!urlPreviewEl) return;
    urlPreviewEl.textContent =
      "URL: " + buildNaukriJobsUrl(getSearchKeyword()) + buildPreviewQuery();
  }

  function renderCtcOptions() {
    if (!ctcGridEl) return;
    var selected = {};
    DEFAULT_CTC.forEach(function (v) {
      selected[v] = true;
    });
    ctcGridEl.innerHTML = "";
    CTC_OPTIONS.forEach(function (opt) {
      var value = opt.value;
      var label = opt.label || value;
      var id = "ctc-" + value;
      var wrap = document.createElement("label");
      wrap.className = "job-toolbar-check";
      wrap.htmlFor = id;
      var input = document.createElement("input");
      input.type = "checkbox";
      input.id = id;
      input.setAttribute("data-ctc", value);
      input.checked = !!selected[value];
      input.addEventListener("change", updateUrlPreview);
      var span = document.createElement("span");
      span.textContent = label;
      wrap.appendChild(input);
      wrap.appendChild(span);
      ctcGridEl.appendChild(wrap);
    });
  }

  function applyDefaultsToForm() {
    if (searchKeywordInput) searchKeywordInput.value = DEFAULT_SEARCH_KEYWORD;
    if (jobAgeInput) jobAgeInput.value = DEFAULT_JOB_AGE;
    if (noOfJobsInput) noOfJobsInput.value = String(DEFAULT_NO_OF_JOBS);
    if (maxPagesInput) maxPagesInput.value = String(DEFAULT_MAX_PAGES);
    if (relevanceInput) relevanceInput.value = String(DEFAULT_RELEVANCE);
    renderCtcOptions();
    MIN_RELEVANCE_PCT = DEFAULT_RELEVANCE;
    updateThresholdNote();
    updateUrlPreview();
  }

  function collectSearchConfig() {
    return {
      keyword: getSearchKeyword(),
      job_age: getJobAge(),
      ctc_filters: getSelectedCtcFilters(),
      no_of_jobs: getNoOfJobs(),
      max_pages: getMaxPages(),
      relevance_min_pct: getRelevanceMin(),
    };
  }

  function escapeHtml(s) {
    var div = document.createElement("div");
    div.textContent = s == null ? "" : String(s);
    return div.innerHTML;
  }

  function jobLocationText(job) {
    if (Array.isArray(job.location)) return job.location.join(", ");
    return job.location == null ? "" : String(job.location);
  }

  function jobSkillsText(job) {
    if (Array.isArray(job.skills)) return job.skills.join(", ");
    return job.skills == null ? "" : String(job.skills);
  }

  function jobDetailUrl(job) {
    var u = job.jd_url != null ? String(job.jd_url).trim() : "";
    if (!u && job.job_url != null) u = String(job.job_url).trim();
    if (!u && job.link != null) u = String(job.link).trim();
    return u;
  }

  function jobPackageText(job) {
    if (job.package != null && String(job.package).trim() !== "") {
      return String(job.package);
    }
    return job.salary == null ? "" : String(job.salary);
  }

  function formatPosted(job) {
    if (job.uploaded_at != null && String(job.uploaded_at).trim() !== "") {
      return String(job.uploaded_at).trim();
    }
    if (job.posted != null && job.posted !== "") {
      var ts = Number(job.posted);
      if (!isNaN(ts)) {
        try {
          return new Date(ts * 1000).toLocaleString();
        } catch (e) {
          return String(job.posted);
        }
      }
    }
    return "";
  }

  /** Minimum relevance score (inclusive); synced from form / API response. */
  var MIN_RELEVANCE_PCT = DEFAULT_RELEVANCE;

  function updateThresholdNote() {
    if (!thresholdNoteEl) return;
    thresholdNoteEl.textContent =
      "Showing jobs with relevance \u2265 " + MIN_RELEVANCE_PCT + "% (server threshold).";
  }

  function jobRelevanceScore(job) {
    var pct = Number(job.relevant_percentage);
    return isNaN(pct) ? 0 : pct;
  }

  function jobMeetsThreshold(job) {
    return jobRelevanceScore(job) >= MIN_RELEVANCE_PCT;
  }

  function jobHaystack(job) {
    return [
      job.title,
      job.company,
      jobLocationText(job),
      job.package,
      jobSkillsText(job),
      job.experience,
      job.posted,
      job.job_url,
      job.jd_url,
      job.is_remote,
      job.relevant_percentage,
      job.uploaded_at,
    ]
      .map(function (x) {
        return String(x == null ? "" : x).toLowerCase();
      })
      .join(" ");
  }

  function getFilteredJobs() {
    var q = (keywordInput && keywordInput.value ? keywordInput.value : "").trim().toLowerCase();
    var list = allJobs.filter(jobMeetsThreshold);
    if (q) {
      list = list.filter(function (j) {
        return jobHaystack(j).indexOf(q) !== -1;
      });
    }
    return list;
  }

  function relevanceBadgeHtml(job) {
    var pct = jobRelevanceScore(job);
    var pctStr = pct + "%";
    var titleOk =
      "Relevance score is " + MIN_RELEVANCE_PCT + "% or higher based on matched stack skills.";
    var titleLow = "Below " + MIN_RELEVANCE_PCT + "% relevance threshold.";
    if (jobMeetsThreshold(job)) {
      return (
        '<span class="job-badge job-badge--relevant" title="' +
        titleOk +
        '">✅ Relevant · ' +
        escapeHtml(pctStr) +
        "</span>"
      );
    }
    return (
      '<span class="job-badge job-badge--not" title="' +
      titleLow +
      '">❌ Not relevant · ' +
      escapeHtml(pctStr) +
      "</span>"
    );
  }

  function renderJobs(jobs) {
    tbody.innerHTML = "";
    if (!jobs.length && allJobs.length) {
      var trEmpty = document.createElement("tr");
      trEmpty.innerHTML =
        '<td colspan="10" class="job-skills-cell">' +
        escapeHtml(
          "No jobs meet the " +
            MIN_RELEVANCE_PCT +
            "% relevance threshold. " +
            allJobs.length +
            " job(s) were scraped — lower NAUKRI_JOB_RELEVANCE_MIN_PCT in .env or check skills on the listing."
        ) +
        "</td>";
      tbody.appendChild(trEmpty);
    }
    jobs.forEach(function (job) {
      var loc = jobLocationText(job);
      var remote = job.is_remote === true ? "Yes" : "No";
      var skillsFull = jobSkillsText(job);
      var pkg = jobPackageText(job);
      var posted = formatPosted(job);
      var url = jobDetailUrl(job);
      var relHtml = relevanceBadgeHtml(job);

      var actions = "—";
      if (url) {
        actions =
          '<a class="btn btn--ghost" href="' +
          String(url)
            .replace(/&/g, "&amp;")
            .replace(/"/g, "&quot;") +
          '" target="_blank" rel="noopener noreferrer">View Job</a>';
      }

      var tr = document.createElement("tr");
      if (jobMeetsThreshold(job)) {
        tr.className = "job-row--relevant";
      }
      tr.innerHTML =
        "<td>" +
        escapeHtml(job.title) +
        "</td><td>" +
        escapeHtml(job.company) +
        "</td><td>" +
        escapeHtml(loc) +
        "</td><td>" +
        escapeHtml(remote) +
        "</td><td>" +
        escapeHtml(pkg) +
        '</td><td class="job-skills-cell">' +
        escapeHtml(skillsFull) +
        "</td><td>" +
        relHtml +
        "</td><td>" +
        escapeHtml(job.experience == null ? "" : String(job.experience)) +
        "</td><td>" +
        escapeHtml(posted) +
        '</td><td class="job-actions-cell">' +
        actions +
        "</td>";
      tbody.appendChild(tr);
    });
    var q = keywordInput && keywordInput.value ? keywordInput.value.trim() : "";
    var suffix = q ? " (filtered)" : "";
    var aboveThreshold = allJobs.filter(jobMeetsThreshold).length;
    if (allJobs.length && !jobs.length && !q) {
      countEl.textContent =
        "0 of " +
        allJobs.length +
        " scraped jobs meet the " +
        MIN_RELEVANCE_PCT +
        "% relevance threshold" +
        suffix;
      return;
    }
    countEl.textContent = jobs.length + " positions" + suffix;
    if (allJobs.length && aboveThreshold !== allJobs.length && !q) {
      countEl.textContent +=
        " (" + aboveThreshold + " of " + allJobs.length + " scraped meet threshold)";
    }
  }

  /**
   * @param {string} criteria - from <select>
   */
  function sortJobs(criteria) {
    currentSort = criteria || "";
    var jobs = getFilteredJobs();
    if (criteria === "title") {
      jobs.sort(function (a, b) {
        return String(a.title).localeCompare(String(b.title));
      });
    } else if (criteria === "company") {
      jobs.sort(function (a, b) {
        return String(a.company).localeCompare(String(b.company));
      });
    } else if (criteria === "location") {
      jobs.sort(function (a, b) {
        return jobLocationText(a).localeCompare(jobLocationText(b));
      });
    } else if (criteria === "package") {
      jobs.sort(function (a, b) {
        return String(a.package || "").localeCompare(String(b.package || ""));
      });
    } else if (criteria === "experience") {
      jobs.sort(function (a, b) {
        return String(a.experience).localeCompare(String(b.experience));
      });
    } else if (criteria === "posted") {
      jobs.sort(function (a, b) {
        var ap = Number(a.posted || 0);
        var bp = Number(b.posted || 0);
        if (!isNaN(ap) && !isNaN(bp) && (ap !== 0 || bp !== 0)) {
          if (bp !== ap) return bp - ap;
        }
        return formatPosted(b).localeCompare(formatPosted(a));
      });
    } else if (criteria === "relevant") {
      jobs.sort(function (a, b) {
        var ap = jobRelevanceScore(a);
        var bp = jobRelevanceScore(b);
        if (bp !== ap) return bp - ap;
        return String(a.title).localeCompare(String(b.title));
      });
    }
    renderJobs(jobs);
  }

  window.sortJobs = sortJobs;
  window.renderJobs = renderJobs;

  window.openJob = function (url) {
    if (url) window.location.assign(url);
  };

  function setFetchBusy(busy) {
    if (fetchBtn) fetchBtn.disabled = busy;
    if (refreshBtn) refreshBtn.disabled = busy;
  }

  var progressPollTimer = null;
  var loadingTextEl = document.getElementById("loading-text");
  var loadingHintEl = document.getElementById("loading-hint");
  var progressCountEl = document.getElementById("scrape-progress-count");
  var progressPctEl = document.getElementById("scrape-progress-pct");
  var progressBarEl = document.getElementById("scrape-progress-bar");
  var progressDetailEl = document.getElementById("scrape-progress-detail");

  function makeProgressId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID().replace(/-/g, "");
    }
    return (
      "p" +
      Date.now().toString(36) +
      Math.random().toString(36).slice(2, 10)
    );
  }

  function stopProgressPolling() {
    if (progressPollTimer) {
      clearInterval(progressPollTimer);
      progressPollTimer = null;
    }
  }

  function renderProgress(data, fallbackTarget) {
    var found = Number(data && data.found);
    var target = Number(data && data.target);
    if (isNaN(found)) found = 0;
    if (isNaN(target) || target < 1) target = fallbackTarget || 1;
    var pct = Number(data && data.percent);
    if (isNaN(pct)) pct = Math.min(100, (found / target) * 100);
    pct = Math.max(0, Math.min(100, pct));

    if (progressCountEl) {
      progressCountEl.textContent = found + " / " + target + " matching jobs";
    }
    if (progressPctEl) {
      progressPctEl.textContent = Math.round(pct) + "%";
    }
    if (progressBarEl) {
      progressBarEl.style.width = pct + "%";
    }
    if (loadingTextEl && data && data.message) {
      loadingTextEl.textContent = data.message;
    }
    if (progressDetailEl) {
      var bits = [];
      if (data && data.page) bits.push("page " + data.page);
      if (data && data.scanned != null) bits.push(data.scanned + " scanned");
      if (data && data.phase) bits.push(data.phase);
      progressDetailEl.textContent = bits.length
        ? bits.join(" · ")
        : "Collecting jobs that meet your relevance threshold.";
    }
  }

  function startProgressPolling(progressId, target) {
    stopProgressPolling();
    renderProgress(
      {
        found: 0,
        target: target,
        percent: 0,
        message: "Starting job search…",
        phase: "starting",
        scanned: 0,
        page: 0,
      },
      target
    );
    if (loadingHintEl) {
      loadingHintEl.textContent =
        "Progress updates live while Chrome logs in and scrapes listings.";
    }

    function applyFinishedProgress(data) {
      stopProgressPolling();
      setVisible(loadingEl, false);
      setFetchBusy(false);

      if (data.error || data.status === "error") {
        errorEl.textContent =
          data.error || data.message || "Scrape failed. Check server logs.";
        setVisible(errorEl, true);
        return;
      }

      allJobs = Array.isArray(data.jobs) ? data.jobs : [];
      if (data.relevance_min_pct != null) {
        var threshold = Number(data.relevance_min_pct);
        if (!isNaN(threshold)) {
          MIN_RELEVANCE_PCT = threshold;
          updateThresholdNote();
        }
      }
      if (data.errors && data.errors.length) {
        console.warn("Job search workflow notes:", data.errors);
      }
      if (data.display_complete === false) {
        console.warn("Playwright job panel injection did not complete.");
      }
      if (sortSelect) {
        sortSelect.value = "";
      }
      currentSort = "";
      if (keywordInput) {
        keywordInput.value = "";
      }
      renderJobs(getFilteredJobs());
      setVisible(resultsEl, true);
    }

    function tick() {
      fetch("/api/jobs/naukri/progress/" + encodeURIComponent(progressId), {
        cache: "no-store",
      })
        .then(function (res) {
          return res.json().then(function (data) {
            return { ok: res.ok, data: data };
          });
        })
        .then(function (payload) {
          if (!payload.ok || !payload.data.ok) return;
          renderProgress(payload.data, target);
          if (payload.data.done) {
            applyFinishedProgress(payload.data);
          }
        })
        .catch(function () {
          /* ignore transient poll errors while scrape runs */
        });
    }

    tick();
    progressPollTimer = setInterval(tick, 800);
  }

  function fetchJobs() {
    setVisible(loadingEl, true);
    setVisible(errorEl, false);
    setVisible(resultsEl, false);
    setFetchBusy(true);

    var enrich = enrichChk ? enrichChk.checked : false;
    var config = collectSearchConfig();
    if (searchKeywordInput) {
      searchKeywordInput.value = config.keyword;
    }
    if (jobAgeInput) jobAgeInput.value = config.job_age;
    if (noOfJobsInput) noOfJobsInput.value = String(config.no_of_jobs);
    if (maxPagesInput) maxPagesInput.value = String(config.max_pages);
    if (relevanceInput) relevanceInput.value = String(config.relevance_min_pct);
    MIN_RELEVANCE_PCT = config.relevance_min_pct;
    updateThresholdNote();
    updateUrlPreview();

    var progressId = makeProgressId();
    startProgressPolling(progressId, config.no_of_jobs);

    fetch("/api/jobs/naukri", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      body: JSON.stringify({
        headless: false,
        enrich_jd: enrich,
        keyword: config.keyword,
        job_age: config.job_age,
        ctc_filters: config.ctc_filters,
        no_of_jobs: config.no_of_jobs,
        max_pages: config.max_pages,
        relevance_min_pct: config.relevance_min_pct,
        progress_id: progressId,
      }),
    })
      .then(function (res) {
        return res.json().then(function (data) {
          console.log(data);
          return { ok: res.ok, data: data };
        });
      })
      .then(function (payload) {
        if (!payload.ok || !payload.data.ok) {
          stopProgressPolling();
          setVisible(loadingEl, false);
          setFetchBusy(false);
          var msg =
            (payload.data && payload.data.error) ||
            "Request failed. Check server logs and Naukri selectors.";
          errorEl.textContent = msg;
          setVisible(errorEl, true);
          return;
        }
        // Scrape continues in background; polling applies results when done.
        if (payload.data.progress_id) {
          // keep polling the same id (server may echo it)
        }
      })
      .catch(function () {
        stopProgressPolling();
        setVisible(loadingEl, false);
        setFetchBusy(false);
        errorEl.textContent = "Network error — is the Flask server running?";
        setVisible(errorEl, true);
      });
  }

  if (fetchBtn) {
    fetchBtn.addEventListener("click", fetchJobs);
  }
  if (refreshBtn) {
    refreshBtn.addEventListener("click", fetchJobs);
  }
  if (searchKeywordInput) {
    searchKeywordInput.addEventListener("input", updateUrlPreview);
    searchKeywordInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        fetchJobs();
      }
    });
  }
  [jobAgeInput, noOfJobsInput, maxPagesInput, relevanceInput].forEach(function (el) {
    if (!el) return;
    el.addEventListener("input", updateUrlPreview);
    el.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        fetchJobs();
      }
    });
  });
  if (keywordInput) {
    keywordInput.addEventListener("input", function () {
      sortJobs(currentSort);
    });
  }

  applyDefaultsToForm();
})();
