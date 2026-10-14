// Tabs, educational assistant, and the clinician feedback workspace.
//
// Deliberately kept out of app.js: the analysis workflow is the part that has
// been validated against the paper, and nothing here reaches into its state.
// The only thing read across the boundary is the rendered verdict text, and
// only when the user ticks the box asking about the on-screen result.

document.addEventListener("DOMContentLoaded", () => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------- tabs
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".tab-panel");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const name = tab.dataset.tab;
      tabs.forEach((t) => {
        const on = t === tab;
        t.classList.toggle("is-active", on);
        t.setAttribute("aria-selected", String(on));
      });
      panels.forEach((p) => p.classList.toggle("is-active", p.id === `tab-${name}`));
      window.scrollTo({ top: 0, behavior: "smooth" });
      if (name === "feedback") refreshQueue();
    });
  });

  const escapeHtml = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // Minimal markdown: **bold** only. Everything else is escaped first, so a
  // model response cannot inject markup.
  const renderText = (s) =>
    escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br>");

  // ------------------------------------------------------ educational chat
  const chatLog = $("chat-log");
  const chatForm = $("chat-form");
  const chatInput = $("chat-input");
  const chatSend = $("chat-send");
  const chatIncludeResult = $("chat-include-result");
  const assistantStatus = $("assistant-status");

  fetch("/api/v1/assistant/status")
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      if (!d) return;
      if (!d.language_model_live) {
        assistantStatus.hidden = false;
        assistantStatus.textContent =
          "Language service not configured — answers come straight from the built-in " +
          "reference notes. Safety checks and sources are unaffected.";
      }
    })
    .catch(() => {
      assistantStatus.hidden = false;
      assistantStatus.textContent = "Could not reach the assistant service.";
    });

  const BOT_ICON =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2a3 3 0 0 0-3 3v2a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>' +
    '<path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3"/></svg>';
  const ALERT_ICON =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>' +
    '<path d="M12 9v4M12 17h.01"/></svg>';
  const LINK_ICON =
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/>' +
    '<path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/></svg>';

  // A bubble is wrapped in a row so the assistant can carry an avatar; the
  // returned node is the bubble itself, so callers update it as before.
  function appendBubble(role, html, extraClass = "") {
    const empty = chatLog.querySelector(".chat-empty");
    if (empty) empty.remove();

    const row = document.createElement("div");
    row.className = `bubble-row ${role === "user" ? "is-user" : "is-bot"}`;

    if (role !== "user") {
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.innerHTML = BOT_ICON;
      row.appendChild(avatar);
    }

    const el = document.createElement("div");
    el.className = `bubble bubble-${role} ${extraClass}`.trim();
    el.innerHTML = html;
    row.appendChild(el);

    chatLog.appendChild(row);
    chatLog.scrollTop = chatLog.scrollHeight;
    return el;
  }

  // Only the rendered verdict and probability, never the image, and only on
  // explicit opt-in.
  function displayedResultSummary() {
    if (!chatIncludeResult || !chatIncludeResult.checked) return null;
    const title = document.getElementById("verdict-title");
    const conf = document.getElementById("verdict-confidence");
    if (!title || !title.textContent.trim()) return null;
    const parts = [`Displayed prediction: ${title.textContent.trim()}`];
    if (conf && conf.textContent.trim()) parts.push(`Reported confidence: ${conf.textContent.trim()}`);
    return parts.join(". ");
  }

  async function ask(question) {
    if (!question.trim()) return;
    appendBubble("user", renderText(question));
    chatInput.value = "";
    chatSend.disabled = true;
    chatSend.classList.add("is-loading");
    const thinking = appendBubble("bot", '<span class="typing"><i></i><i></i><i></i></span>');

    try {
      const res = await fetch("/api/v1/assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          displayed_result: displayedResultSummary(),
        }),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        thinking.className = "bubble bubble-bot bubble-error";
        thinking.textContent = detail.detail
          ? `Could not answer: ${typeof detail.detail === "string" ? detail.detail : "invalid request"}`
          : `Assistant unavailable (HTTP ${res.status}).`;
        return;
      }

      const data = await res.json();
      thinking.className = `bubble bubble-bot${data.is_emergency ? " bubble-emergency" : ""}`;
      if (data.is_emergency) {
        const av = thinking.parentElement.querySelector(".avatar");
        if (av) { av.classList.add("is-emergency"); av.innerHTML = ALERT_ICON; }
      }
      let html = renderText(data.answer);

      if (data.sources && data.sources.length) {
        const seen = new Set();
        const chips = data.sources
          .filter((s) => !seen.has(s.name) && seen.add(s.name))
          .map((s) => `<a class="source-chip" href="${escapeHtml(s.url)}" target="_blank" ` +
                      `rel="noopener noreferrer">${LINK_ICON}${escapeHtml(s.name)}</a>`)
          .join("");
        html += `<div class="bubble-sources">${chips}</div>`;
      }
      if (data.mode === "knowledge_base") {
        html += `<div class="bubble-mode">Answered from built-in reference notes.</div>`;
      }
      thinking.innerHTML = html;
    } catch (err) {
      thinking.className = "bubble bubble-bot bubble-error";
      thinking.textContent = "Network error — the assistant could not be reached.";
    } finally {
      chatSend.disabled = false;
      chatSend.classList.remove("is-loading");
      chatLog.scrollTop = chatLog.scrollHeight;
    }
  }

  if (chatForm) {
    chatForm.addEventListener("submit", (e) => {
      e.preventDefault();
      ask(chatInput.value);
    });
  }
  document.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip[data-ask]");
    if (chip) ask(chip.dataset.ask);
  });

  // -------------------------------------------------- clinician feedback
  const fbFile = $("fb-file");
  const fbRun = $("fb-run");
  const fbResult = $("fb-result");
  const fbReview = $("fb-review");
  const fbCorrectionRow = $("fb-correction-row");
  const fbCorrected = $("fb-corrected");
  const fbNote = $("fb-note");
  const fbSubmit = $("fb-submit");
  const fbReviewer = $("fb-reviewer");
  const fbModel = $("fb-model");
  const fbInstitution = $("fb-institution");
  const fbQueue = $("fb-queue");
  const fbCounts = $("fb-counts");
  const fbExports = $("fb-exports");

  let selectedVerdict = null;

  function fbNotify(msg, kind = "info") {
    fbResult.hidden = false;
    fbResult.className = `fb-result fb-${kind}`;
    fbResult.innerHTML = msg;
  }

  if (fbRun) {
    fbRun.addEventListener("click", async () => {
      const file = fbFile.files && fbFile.files[0];
      if (!file) return fbNotify("Choose a radiograph first.", "error");
      if (file.size > 20 * 1024 * 1024) return fbNotify("Image exceeds the 20 MB limit.", "error");
      if (!["image/jpeg", "image/jpg", "image/png"].includes(file.type)) {
        return fbNotify("Unsupported image type. Use JPEG or PNG.", "error");
      }

      fbRun.disabled = true;
      fbRun.classList.add("is-loading");
      fbNotify("Running prediction…");
      try {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("model_name", fbModel.value);
        const res = await fetch("/api/v1/predict", { method: "POST", body: fd });
        if (!res.ok) {
          const d = await res.json().catch(() => ({}));
          fbRun.disabled = false;
          return fbNotify(`Prediction failed: ${escapeHtml(d.detail || res.status)}`, "error");
        }
        const d = await res.json();
        const p = (d.probabilities.PNEUMONIA * 100).toFixed(1);
        fbNotify(
          `<strong>${escapeHtml(d.prediction_class)}</strong> · P(pneumonia) = ${p}% ` +
          `· model <code>${escapeHtml(d.model_name)}</code>`
        );
        fbReview.hidden = false;
      } catch {
        fbNotify("Network error while predicting.", "error");
      } finally {
        fbRun.disabled = false;
        fbRun.classList.remove("is-loading");
      }
    });
  }

  document.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip[data-verdict]");
    if (!chip) return;
    selectedVerdict = chip.dataset.verdict;
    document.querySelectorAll(".chip[data-verdict]").forEach((c) =>
      c.classList.toggle("is-active", c === chip));
    fbCorrectionRow.hidden = selectedVerdict !== "incorrect";
  });

  if (fbSubmit) {
    fbSubmit.addEventListener("click", async () => {
      const file = fbFile.files && fbFile.files[0];
      if (!file) return fbNotify("Choose a radiograph first.", "error");
      if (!selectedVerdict) return fbNotify("Mark the prediction correct or incorrect.", "error");
      if (selectedVerdict === "incorrect" && !fbCorrected.value) {
        return fbNotify("A corrected label is required when marking a prediction incorrect.", "error");
      }
      if (!fbReviewer.value.trim()) return fbNotify("Enter a reviewer name.", "error");

      fbSubmit.disabled = true;
      try {
        const fd = new FormData();
        fd.append("file", file);
        fd.append("reviewer", fbReviewer.value.trim());
        fd.append("verdict", selectedVerdict);
        fd.append("model_name", fbModel.value);
        if (fbInstitution && fbInstitution.value.trim()) {
          fd.append("institution", fbInstitution.value.trim());
        }
        if (fbCorrected.value) fd.append("corrected_label", fbCorrected.value);
        if (fbNote.value.trim()) fd.append("note", fbNote.value.trim());

        const res = await fetch("/api/v1/feedback/submit", { method: "POST", body: fd });
        const d = await res.json().catch(() => ({}));
        if (!res.ok) {
          fbSubmit.disabled = false;
          return fbNotify(`Submission failed: ${escapeHtml(d.detail || res.status)}`, "error");
        }
        fbNotify(`Review queued (<code>${escapeHtml(d.id.slice(0, 8))}</code>). Nothing was retrained.`, "ok");
        fbReview.hidden = true;
        selectedVerdict = null;
        fbCorrected.value = "";
        fbNote.value = "";
        document.querySelectorAll(".chip[data-verdict]").forEach((c) => c.classList.remove("is-active"));
        refreshQueue();
      } catch {
        fbNotify("Network error while submitting.", "error");
      } finally {
        fbSubmit.disabled = false;
      }
    });
  }

  async function refreshQueue() {
    if (!fbQueue) return;
    try {
      const res = await fetch("/api/v1/feedback?limit=50");
      if (!res.ok) throw new Error();
      const d = await res.json();

      fbCounts.innerHTML =
        Object.entries(d.counts)
          .map(([k, v]) => `<span class="count-pill">${escapeHtml(k)}: <strong>${v}</strong></span>`)
          .join("") +
        Object.entries(d.institutions || {})
          .map(([k, v]) => `<span class="count-pill is-site">${escapeHtml(k)}: <strong>${v}</strong></span>`)
          .join("");

      if (!d.items.length) {
        fbQueue.innerHTML =
          "<div class='empty-state'><svg viewBox='0 0 24 24' aria-hidden='true'>" +
          "<path d='M9 11l3 3L22 4'/><path d='M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11'/>" +
          "</svg><p>No reviews yet. Run a case to start the queue.</p></div>";
      } else {
        fbQueue.innerHTML = d.items.map((it) => {
          const p = (it.predicted_probability * 100).toFixed(1);
          const corrected = it.corrected_label
            ? ` → <strong>${escapeHtml(it.corrected_label)}</strong>` : "";
          const frozen = it.status === "exported";
          return `<div class="queue-item ${frozen ? "is-frozen" : ""}">
            <div class="queue-head">
              <span class="badge ${it.verdict === "correct" ? "badge-ok" : "badge-warn"}">${escapeHtml(it.verdict)}</span>
              <span class="badge ${frozen ? "badge-neutral" : "badge-accent"}">${escapeHtml(it.status)}</span>
            </div>
            <div class="queue-body">
              <code>${escapeHtml(it.model_version)}</code> predicted
              <strong>${escapeHtml(it.predicted_label)}</strong> at ${p}%${corrected}
            </div>
            <div class="queue-meta">
              ${escapeHtml(it.reviewer)} · ${escapeHtml(it.created_at.slice(0, 16).replace("T", " "))}
              · sha256 <code>${escapeHtml(it.image_sha256.slice(0, 10))}</code>
              ${it.exported_in ? `· export <code>${escapeHtml(it.exported_in)}</code>` : ""}
            </div>
            ${it.note ? `<div class="queue-note">${escapeHtml(it.note)}</div>` : ""}
            ${frozen ? "" : `<button class="btn-link" data-edit-note="${escapeHtml(it.id)}">Edit note</button>`}
          </div>`;
        }).join("");
      }

      const ex = await fetch("/api/v1/feedback/exports").then((r) => r.json()).catch(() => null);
      if (ex && ex.exports.length) {
        fbExports.innerHTML = "<p class='field-label'>Exports</p>" + ex.exports
          .map((e) => `<div class="export-row"><code>${escapeHtml(e.tag)}</code> — ${e.records} records</div>`)
          .join("");
      } else if (fbExports) {
        fbExports.innerHTML = "";
      }
    } catch {
      fbQueue.innerHTML =
        "<div class='empty-state'><svg viewBox='0 0 24 24' aria-hidden='true'>" +
        "<circle cx='12' cy='12' r='9'/><path d='M12 8v5M12 16h.01'/></svg>" +
        "<p>Could not load the review queue.</p></div>";
    }
  }

  document.addEventListener("click", async (e) => {
    const btn = e.target.closest("[data-edit-note]");
    if (!btn) return;
    const note = window.prompt("Review note:");
    if (note === null) return;
    const res = await fetch(`/api/v1/feedback/${btn.dataset.editNote}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ note }),
    });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      window.alert(`Could not update: ${d.detail || res.status}`);
    }
    refreshQueue();
  });

  const fbRefresh = $("fb-refresh");
  if (fbRefresh) fbRefresh.addEventListener("click", refreshQueue);

  const fbExport = $("fb-export");
  if (fbExport) {
    fbExport.addEventListener("click", async () => {
      if (!window.confirm(
        "Write a versioned export of all pending reviews?\n\n" +
        "This freezes those records. It does not start training."
      )) return;
      fbExport.disabled = true;
      try {
        const res = await fetch("/api/v1/feedback/export", { method: "POST" });
        const d = await res.json().catch(() => ({}));
        if (!res.ok) {
          fbNotify(`Export failed: ${escapeHtml(d.detail || res.status)}`, "error");
        } else {
          fbNotify(
            `Export <code>${escapeHtml(d.export_tag)}</code> written with ${d.record_count} records. ` +
            `No training was started.`, "ok");
        }
        refreshQueue();
      } finally {
        fbExport.disabled = false;
      }
    });
  }
});

/* ---------------------------------------------------------------- report
 * app.js flattens the generated report before writing it: the markdown is
 * stripped and the newlines go with it, so sections run together as
 * "...infiltrates.Pleural Spaces No evidence of..." and the result is a wall
 * of text. The structure still exists in the API response, so fetch is
 * wrapped to keep the original sections and the three nodes are re-rendered
 * from those. app.js is untouched.
 */
(() => {
  "use strict";

  const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const inline = (t) =>
    esc(t)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(\d+(?:\.\d+)?\s?%)/g, '<span class="num">$1</span>');

  function toHtml(raw) {
    const lines = String(raw).split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    let html = "", open = false;
    for (const line of lines) {
      const bullet = line.replace(/^[-*\u2022]\s*/, "");
      if (bullet !== line) {
        if (!open) { html += '<ul class="rep-list">'; open = true; }
        const m = bullet.match(/^\*\*\s*(.+?)\s*:?\s*\*\*:?\s*(.*)$/);
        html += m
          ? `<li><span class="rep-key">${esc(m[1])}</span>${m[2] ? inline(m[2]) : ""}</li>`
          : `<li>${inline(bullet)}</li>`;
      } else {
        if (open) { html += "</ul>"; open = false; }
        html += `<p>${inline(line)}</p>`;
      }
    }
    if (open) html += "</ul>";
    return html;
  }

  function render(sections) {
    const map = {
      "report-findings": sections.findings,
      "report-impression": sections.impression,
      "report-recommendations": sections.recommendations,
    };
    for (const [id, raw] of Object.entries(map)) {
      const node = document.getElementById(id);
      if (!node || !raw) continue;
      const html = toHtml(raw);
      if (!html) continue;
      node.innerHTML = html;
      node.classList.add("is-formatted");
    }
  }

  const originalFetch = window.fetch;
  window.fetch = async function (...args) {
    const response = await originalFetch.apply(this, args);
    try {
      const url = typeof args[0] === "string" ? args[0] : (args[0] && args[0].url) || "";
      if (url.includes("/api/v1/analyze") && !url.includes("/pdf") && response.ok) {
        // Read a clone so app.js still gets an unconsumed body.
        response.clone().json().then((data) => {
          const sections = data && data.report && data.report.sections;
          // app.js writes after its own await resolves; run last.
          if (sections) setTimeout(() => render(sections), 0);
        }).catch(() => {});
      }
    } catch {
      /* never let instrumentation break the request */
    }
    return response;
  };
})();

/* ------------------------------------------------------- fine-tuning run
 * Starts scripts/finetune_from_feedback.py on the server and streams its
 * log back. The script owns every safety guarantee; this only drives it.
 */
document.addEventListener("DOMContentLoaded", () => {
  "use strict";
  const btn = document.getElementById("fb-train");
  const log = document.getElementById("fb-train-log");
  const title = document.querySelector(".train-title");
  if (!btn || !log) return;

  let poll = null;

  function show(text) {
    log.hidden = false;
    log.textContent = text;
    log.scrollTop = log.scrollHeight;
  }

  async function tick() {
    try {
      const d = await fetch("/api/v1/feedback/train/status").then((r) => r.json());
      if (d.log) show(d.log);
      if (!d.running) {
        clearInterval(poll);
        poll = null;
        btn.disabled = false;
        btn.classList.remove("is-loading");
        if (title) title.classList.remove("is-running");
        if (d.produced_checkpoint) {
          show(`${d.log}\n--- finished ---\nNew version: ${d.produced_checkpoint}\n` +
               `Nothing was promoted. Review the comparison above before using it.`);
        }
      }
    } catch {
      clearInterval(poll);
      poll = null;
      btn.disabled = false;
      btn.classList.remove("is-loading");
      if (title) title.classList.remove("is-running");
    }
  }

  btn.addEventListener("click", async () => {
    if (!window.confirm(
      "Start a fine-tuning run on the latest export?\n\n" +
      "This creates a NEW model version. The current model stays in place and " +
      "nothing is promoted automatically. It can take several minutes."
    )) return;

    btn.disabled = true;
    btn.classList.add("is-loading");
    if (title) title.classList.add("is-running");
    show("Starting…");

    try {
      const res = await fetch("/api/v1/feedback/train", { method: "POST", body: new FormData() });
      const d = await res.json().catch(() => ({}));
      if (!res.ok) {
        show(`Could not start: ${d.detail || res.status}`);
        btn.disabled = false;
        btn.classList.remove("is-loading");
        if (title) title.classList.remove("is-running");
        return;
      }
      show(`Started on ${d.manifest} (${d.cases} cases, ${d.epochs} epochs).\nWaiting for output…`);
      poll = setInterval(tick, 2500);
    } catch {
      show("Network error while starting the run.");
      btn.disabled = false;
      btn.classList.remove("is-loading");
      if (title) title.classList.remove("is-running");
    }
  });
});
