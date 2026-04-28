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
  var refreshBtn = document.getElementById("btn-refresh");
  var sortSelect = document.getElementById("sort-select");
  var keywordInput = document.getElementById("job-keyword");
  var enrichChk = document.getElementById("chk-enrich-jd");
  var relevantOnlyChk = document.getElementById("chk-relevant-only");

  /** Full list from API */
  var allJobs = [];
  /** Current sort key */
  var currentSort = "";

  function setVisible(el, show) {
    el.classList.toggle("hidden", !show);
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
    return u;
  }

  function jobIsRelevant(job) {
    return job.is_relevant === true;
  }

  function truncate(s, max) {
    s = String(s || "");
    if (s.length <= max) return s;
    return s.slice(0, max - 1) + "…";
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
      job.description,
    ]
      .map(function (x) {
        return String(x == null ? "" : x).toLowerCase();
      })
      .join(" ");
  }

  function getFilteredJobs() {
    var q = (keywordInput && keywordInput.value ? keywordInput.value : "").trim().toLowerCase();
    var list = allJobs.slice();
    if (relevantOnlyChk && relevantOnlyChk.checked) {
      list = list.filter(jobIsRelevant);
    }
    if (q) {
      list = list.filter(function (j) {
        return jobHaystack(j).indexOf(q) !== -1;
      });
    }
    return list;
  }

  function relevanceBadgeHtml(job) {
    if (jobIsRelevant(job)) {
      return '<span class="job-badge job-badge--relevant" title="Matches target stack (React/Next/Node/TS/JS)">✅ Relevant</span>';
    }
    return '<span class="job-badge job-badge--not" title="Excluded stack or not enough target skills">❌ Not relevant</span>';
  }

  function renderJobs(jobs) {
    tbody.innerHTML = "";
    jobs.forEach(function (job) {
      var date;
      if (job.posted != null) {
        date = new Date(Number(job.posted) * 1000);
      }
      var loc = jobLocationText(job);
      var skillsFull = jobSkillsText(job);
      var skills = truncate(skillsFull, 72);
      var pkg = job.package != null ? String(job.package) : "";
      var posted = date != null ? date.toString() : "";
      var url = jobDetailUrl(job);
      var relHtml = relevanceBadgeHtml(job);

      var actions = "—";
      if (url) {
        actions =
          '<button type="button" class="btn btn--ghost" onclick="openJob(' +
          JSON.stringify(url) +
          ')">View Job</button>' +
          ' <a class="btn btn--ghost" href="/job-detail?url=' +
          encodeURIComponent(url) +
          '" target="_blank" rel="noopener noreferrer">JD</a>';
      }

      var tr = document.createElement("tr");
      if (jobIsRelevant(job)) {
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
        escapeHtml(pkg) +
        '</td><td title="' +
        escapeHtml(skillsFull) +
        '">' +
        escapeHtml(skills) +
        "</td><td>" +
        relHtml +
        "</td><td>" +
        escapeHtml(job.experience) +
        "</td><td>" +
        escapeHtml(posted) +
        '</td><td class="job-actions-cell">' +
        actions +
        "</td>";
      tbody.appendChild(tr);
    });
    var q = keywordInput && keywordInput.value ? keywordInput.value.trim() : "";
    var suffix = q ? " (filtered)" : "";
    countEl.textContent = jobs.length + " positions" + suffix;
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
        return String(b.posted || "").localeCompare(String(a.posted || ""));
      });
    } else if (criteria === "relevant") {
      jobs.sort(function (a, b) {
        var ar = jobIsRelevant(a) ? 1 : 0;
        var br = jobIsRelevant(b) ? 1 : 0;
        if (br !== ar) return br - ar;
        return String(a.title).localeCompare(String(b.title));
      });
    }
    renderJobs(jobs);
  }

  window.sortJobs = sortJobs;
  window.renderJobs = renderJobs;

  window.openJob = function (url) {
    if (url) window.open(url, "_blank", "noopener,noreferrer");
  };

  function fetchJobs() {
    setVisible(loadingEl, true);
    setVisible(errorEl, false);
    setVisible(resultsEl, false);

    var enrich = enrichChk ? enrichChk.checked : true;

    fetch("/api/jobs/naukri", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ headless: false, enrich_jd: enrich }),
    })
      .then(function (res) {
        return res.json().then(function (data) {
          return { ok: res.ok, data: data };
        });
      })
      .then(function (payload) {
        setVisible(loadingEl, false);
        if (!payload.ok || !payload.data.ok) {
          var msg =
            (payload.data && payload.data.error) ||
            "Request failed. Check server logs and Naukri selectors.";
          errorEl.textContent = msg;
          setVisible(errorEl, true);
          return;
        }
        allJobs = payload.data.jobs || [];
        if (sortSelect) {
          sortSelect.value = "";
        }
        currentSort = "";
        if (keywordInput) {
          keywordInput.value = "";
        }
        renderJobs(getFilteredJobs());
        setVisible(resultsEl, true);
      })
      .catch(function () {
        setVisible(loadingEl, false);
        errorEl.textContent = "Network error — is the Flask server running?";
        setVisible(errorEl, true);
      });
  }

  if (refreshBtn) {
    refreshBtn.addEventListener("click", fetchJobs);
  }
  if (keywordInput) {
    keywordInput.addEventListener("input", function () {
      sortJobs(currentSort);
    });
  }
  if (relevantOnlyChk) {
    relevantOnlyChk.addEventListener("change", function () {
      sortJobs(currentSort);
    });
  }

  fetchJobs();
})();
