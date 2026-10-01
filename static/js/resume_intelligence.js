/**
 * Resume Intelligence — submit resume + JD, render grounded gap analysis.
 */
(function () {
  var form = document.getElementById("resume-intel-form");
  var loadingEl = document.getElementById("loading-state");
  var errorEl = document.getElementById("error-state");
  var resultEl = document.getElementById("result-state");
  var submitBtn = document.getElementById("btn-analyze");

  var sourceEl = document.getElementById("result-source");
  var summaryEl = document.getElementById("result-summary");
  var listElements = {
    matched_skills: document.getElementById("result-matched"),
    missing_skills: document.getElementById("result-missing"),
    strong_evidence: document.getElementById("result-strong"),
    weak_evidence: document.getElementById("result-weak"),
    relevant_projects: document.getElementById("result-projects"),
    resume_suggestions: document.getElementById("result-suggestions"),
    concerns: document.getElementById("result-concerns"),
  };

  function setVisible(el, show) {
    el.classList.toggle("hidden", !show);
  }

  function showError(msg) {
    errorEl.textContent = msg;
    setVisible(errorEl, true);
    setVisible(resultEl, false);
  }

  function clearError() {
    errorEl.textContent = "";
    setVisible(errorEl, false);
  }

  function renderList(el, items) {
    el.innerHTML = "";
    if (!items || !items.length) {
      var li = document.createElement("li");
      li.textContent = "None found.";
      el.appendChild(li);
      return;
    }
    items.forEach(function (item) {
      var li = document.createElement("li");
      li.textContent = item;
      el.appendChild(li);
    });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    clearError();
    setVisible(resultEl, false);

    var jdText = (document.getElementById("jd-text").value || "").trim();
    if (!jdText) {
      showError("Job description is required.");
      return;
    }

    var formData = new FormData(form);

    setVisible(loadingEl, true);
    submitBtn.disabled = true;

    fetch("/api/resume-intel/analyze", {
      method: "POST",
      body: formData,
    })
      .then(function (res) {
        return res.json().then(function (data) {
          return { ok: res.ok, data: data };
        });
      })
      .then(function (_ref) {
        var ok = _ref.ok;
        var data = _ref.data;
        if (!ok || !data.ok) {
          throw new Error((data && data.error) || "Request failed.");
        }
        var analysis = data.analysis || {};
        sourceEl.textContent = "Resume analyzed: " + (data.resume_source || "");
        summaryEl.textContent = analysis.overall_fit_summary || "";
        Object.keys(listElements).forEach(function (key) {
          renderList(listElements[key], analysis[key]);
        });
        setVisible(resultEl, true);
      })
      .catch(function (err) {
        showError(err.message || "Something went wrong.");
      })
      .finally(function () {
        setVisible(loadingEl, false);
        submitBtn.disabled = false;
      });
  });
})();
