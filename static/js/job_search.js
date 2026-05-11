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
    var pct = Number(job.relevant_percentage);
    if (isNaN(pct)) pct = 0;
    var pctStr = pct + "%";
    var titleOk =
      "All of React, Next, Node, JavaScript, TypeScript appear in skills; no Java / C# / .NET.";
    var titleNo = "Needs all five stack skills and no Java/C#/NET in the skill list.";
    if (jobIsRelevant(job)) {
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
      titleNo +
      '">❌ Not relevant · ' +
      escapeHtml(pctStr) +
      "</span>"
    );
  }

  function renderJobs(jobs) {
    tbody.innerHTML = "";
    jobs.forEach(function (job) {
      var loc = jobLocationText(job);
      var remote = job.is_remote === true ? "Yes" : "No";
      var skillsFull = jobSkillsText(job);
      var skills = truncate(skillsFull, 72);
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
          '">View Job</a>';
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
        escapeHtml(remote) +
        "</td><td>" +
        escapeHtml(pkg) +
        '</td><td title="' +
        escapeHtml(skillsFull) +
        '">' +
        escapeHtml(skills) +
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
        var ap = Number(a.posted || 0);
        var bp = Number(b.posted || 0);
        if (!isNaN(ap) && !isNaN(bp) && (ap !== 0 || bp !== 0)) {
          if (bp !== ap) return bp - ap;
        }
        return formatPosted(b).localeCompare(formatPosted(a));
      });
    } else if (criteria === "relevant") {
      jobs.sort(function (a, b) {
        var ar = jobIsRelevant(a) ? 1 : 0;
        var br = jobIsRelevant(b) ? 1 : 0;
        if (br !== ar) return br - ar;
        var ap = Number(a.relevant_percentage || 0);
        var bp = Number(b.relevant_percentage || 0);
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

  function fetchJobs() {
    setVisible(loadingEl, true);
    setVisible(errorEl, false);
    setVisible(resultsEl, false);

    var enrich = enrichChk ? enrichChk.checked : false;

    fetch("/api/jobs/naukri", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ headless: false, enrich_jd: enrich }),
    })
      .then(function (res) {
        return res.json().then(function (data) {
          console.log(data);
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
