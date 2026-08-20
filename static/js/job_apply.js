/**
 * Job Apply Pro — submit JD + hiring manager email, run agent pipeline.
 */
(function () {
  var form = document.getElementById("job-apply-form");
  var loadingEl = document.getElementById("loading-state");
  var errorEl = document.getElementById("error-state");
  var resultEl = document.getElementById("result-state");
  var subjectEl = document.getElementById("result-subject");
  var bodyEl = document.getElementById("result-body");
  var statusEl = document.getElementById("result-status");
  var submitBtn = document.getElementById("btn-submit");

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

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    clearError();
    setVisible(resultEl, false);

    var jobDescription = (document.getElementById("job-description").value || "").trim();
    var hiringEmail = (document.getElementById("hiring-email").value || "").trim();

    if (!jobDescription) {
      showError("Job description is required.");
      return;
    }
    if (!hiringEmail) {
      showError("Hiring manager email is required.");
      return;
    }

    setVisible(loadingEl, true);
    submitBtn.disabled = true;

    fetch("/api/job-apply/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        job_description: jobDescription,
        hiring_manager_email: hiringEmail,
      }),
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
        subjectEl.textContent = "Subject: " + (data.subject || "");
        bodyEl.textContent = data.body || "";
        var emailStatus = data.email_status || {};
        statusEl.textContent =
          emailStatus.status === "success"
            ? "Gmail compose opened in your browser."
            : "Email draft ready. " + (emailStatus.error || emailStatus.message || "");
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
