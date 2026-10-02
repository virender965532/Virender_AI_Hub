/**
 * "Connect LinkedIn" popup: drives a browser running on the server
 * (/api/jobs/linkedin/login/*). The popup shows screenshots of that browser and
 * forwards clicks, keys, pasted text and scrolling to it, so the candidate can
 * sign in by hand (2FA / CAPTCHA included). The server saves the session to
 * MongoDB once LinkedIn signs them in.
 */
(function () {
  var API = "/api/jobs/linkedin";
  var NAMED_KEYS = ["Enter", "Tab", "Backspace", "Delete", "Escape", "ArrowLeft", "ArrowRight",
    "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"];

  var modal = document.getElementById("li-modal");
  if (!modal) return;
  var connectBtn = document.getElementById("li-btn-connect");
  var closeBtn = document.getElementById("li-btn-close");
  var statusEl = document.getElementById("li-status");
  var messageEl = document.getElementById("li-message");
  var screen = document.getElementById("li-screen");
  var frameImg = document.getElementById("li-frame");
  var hintEl = document.getElementById("li-screen-hint");
  var textInput = document.getElementById("li-text");
  var sendBtn = document.getElementById("li-btn-send");

  var login = null; // {id, viewport}
  var frameTimer = null;
  var statusTimer = null;
  var frameUrl = null;
  var queue = [];
  var sending = false;
  var wheelDy = 0;
  var wheelTimer = null;

  function setMessage(text, kind) {
    messageEl.textContent = text;
    messageEl.className = "li-modal__message" + (kind ? " li-modal__message--" + kind : "");
  }

  function refreshStatus() {
    if (!statusEl) return;
    fetch(API + "/session")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (data.connected) {
          var when = data.updated_at ? new Date(data.updated_at).toLocaleString() : "";
          var until = data.expires_at ? new Date(data.expires_at).toLocaleDateString() : "";
          statusEl.textContent = "LinkedIn connected" + (when ? " (saved " + when + ")" : "")
            + (until ? ", valid until " + until : "");
          statusEl.className = "li-status li-status--ok";
        } else {
          statusEl.textContent = "LinkedIn not connected";
          statusEl.className = "li-status li-status--missing";
        }
      })
      .catch(function () { statusEl.textContent = ""; });
  }

  // --- input queue: sent one at a time so keys arrive in order ---------------

  function enqueue(event) {
    if (!login) return;
    var last = queue[queue.length - 1];
    if (event.type === "type" && last && last.type === "type") {
      last.text += event.text; // coalesce fast typing into one request
    } else {
      queue.push(event);
    }
    pump();
  }

  function pump() {
    if (sending || !queue.length || !login) return;
    sending = true;
    var event = queue.shift();
    fetch(API + "/login/" + encodeURIComponent(login.id) + "/input", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(event),
    })
      .catch(function () { /* status polling reports a dead session */ })
      .then(function () {
        sending = false;
        scheduleFrame(120); // show the effect quickly
        pump();
      });
  }

  // --- screen ----------------------------------------------------------------

  function scheduleFrame(delay) {
    if (!login) return;
    window.clearTimeout(frameTimer);
    frameTimer = window.setTimeout(loadFrame, delay);
  }

  function loadFrame() {
    if (!login) return;
    var id = login.id;
    fetch(API + "/login/" + encodeURIComponent(id) + "/frame?t=" + Date.now(), { cache: "no-store" })
      .then(function (res) { return res.ok ? res.blob() : null; })
      .then(function (blob) {
        if (!blob || !login || login.id !== id) return;
        if (frameUrl) URL.revokeObjectURL(frameUrl);
        frameUrl = URL.createObjectURL(blob);
        frameImg.src = frameUrl;
        hintEl.classList.add("hidden");
      })
      .catch(function () {})
      .then(function () { scheduleFrame(600); });
  }

  function pollStatus() {
    if (!login) return;
    var id = login.id;
    fetch(API + "/login/" + encodeURIComponent(id))
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (!login || login.id !== id) return;
        if (!data.ok) {
          finish(data.error || "The sign-in session ended.", "error");
          return;
        }
        if (data.state === "waiting") {
          statusTimer = window.setTimeout(pollStatus, 1500);
          return;
        }
        finish(data.message, data.state === "saved" ? "ok" : "error");
        if (data.state === "saved") window.setTimeout(close, 1800);
      })
      .catch(function () { statusTimer = window.setTimeout(pollStatus, 3000); });
  }

  function stopTimers() {
    window.clearTimeout(frameTimer);
    window.clearTimeout(statusTimer);
    window.clearTimeout(wheelTimer);
    queue = [];
  }

  function finish(message, kind) {
    stopTimers();
    login = null;
    setMessage(message, kind);
    refreshStatus();
  }

  function open(reason) {
    modal.classList.remove("hidden");
    frameImg.removeAttribute("src");
    hintEl.textContent = "Loading LinkedIn…";
    hintEl.classList.remove("hidden");
    setMessage(reason || "Starting a browser on the server…");
    connectBtn && (connectBtn.disabled = true);

    fetch(API + "/login", { method: "POST" })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        connectBtn && (connectBtn.disabled = false);
        if (!data.ok) {
          setMessage(data.message || data.error || "Could not start the sign-in.", "error");
          hintEl.textContent = "";
          return;
        }
        login = { id: data.login_id, viewport: data.viewport };
        setMessage(data.message + " Time limit: " + Math.round(data.seconds_left / 60) + " min.");
        loadFrame();
        statusTimer = window.setTimeout(pollStatus, 1500);
        screen.focus();
      })
      .catch(function (err) {
        connectBtn && (connectBtn.disabled = false);
        setMessage("Could not reach the server: " + err, "error");
      });
  }

  function close() {
    if (login) {
      fetch(API + "/login/" + encodeURIComponent(login.id), { method: "DELETE" }).catch(function () {});
    }
    stopTimers();
    login = null;
    modal.classList.add("hidden");
    refreshStatus();
  }

  // --- event wiring ----------------------------------------------------------

  frameImg.addEventListener("click", function (e) {
    if (!login) return;
    var rect = frameImg.getBoundingClientRect();
    enqueue({
      type: "click",
      x: Math.round(((e.clientX - rect.left) * login.viewport.width) / rect.width),
      y: Math.round(((e.clientY - rect.top) * login.viewport.height) / rect.height),
    });
    screen.focus();
  });

  screen.addEventListener("keydown", function (e) {
    if (!login) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return; // let paste (Ctrl+V) reach the paste handler
    if (NAMED_KEYS.indexOf(e.key) !== -1) {
      e.preventDefault();
      enqueue({ type: "press", key: e.key });
    } else if (e.key.length === 1) {
      e.preventDefault();
      enqueue({ type: "type", text: e.key });
    }
  });

  screen.addEventListener("paste", function (e) {
    var text = (e.clipboardData || window.clipboardData).getData("text");
    if (text) {
      e.preventDefault();
      enqueue({ type: "type", text: text });
    }
  });

  screen.addEventListener("wheel", function (e) {
    if (!login) return;
    e.preventDefault();
    wheelDy += e.deltaY;
    if (wheelTimer) return;
    wheelTimer = window.setTimeout(function () {
      wheelTimer = null;
      if (wheelDy) enqueue({ type: "scroll", dy: Math.round(wheelDy) });
      wheelDy = 0;
    }, 150);
  }, { passive: false });

  function sendText() {
    if (textInput.value) enqueue({ type: "type", text: textInput.value });
    textInput.value = "";
  }
  sendBtn.addEventListener("click", sendText);
  textInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      e.preventDefault();
      sendText();
    }
  });
  Array.prototype.forEach.call(modal.querySelectorAll("[data-li-key]"), function (btn) {
    btn.addEventListener("click", function () { enqueue({ type: "press", key: btn.getAttribute("data-li-key") }); });
  });

  closeBtn.addEventListener("click", close);
  if (connectBtn) connectBtn.addEventListener("click", function () { open(); });

  refreshStatus();
  window.LinkedInConnect = { open: open, refreshStatus: refreshStatus };
})();
