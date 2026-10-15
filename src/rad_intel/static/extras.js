// Tabs, the educational assistant, and report readability.
//
// Deliberately kept out of app.js: the analysis workflow is the part that has
// been validated against the project's reported results, and nothing here
// reaches into its state. The only thing read across the boundary is the
// rendered verdict text, and only when the reader ticks the box asking about
// the on-screen result.
//
// The review workspace lives in workspace.js, which is a closed-access area
// with its own state and does not belong in the same file as the public tabs.

document.addEventListener("DOMContentLoaded", () => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------- tabs
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".tab-panel");

  const TAB_NAMES = [...tabs].map((t) => t.dataset.tab);

  function activate(name, { scroll = true, pushHash = true } = {}) {
    if (!TAB_NAMES.includes(name)) name = TAB_NAMES[0];
    tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("is-active", on);
      t.setAttribute("aria-selected", String(on));
    });
    panels.forEach((p) => p.classList.toggle("is-active", p.id === `tab-${name}`));
    if (pushHash && window.location.hash.slice(1) !== name) {
      history.replaceState(null, "", `#${name}`);
    }
    if (scroll) window.scrollTo({ top: 0, behavior: "smooth" });
    document.dispatchEvent(new CustomEvent("radintel:tab", { detail: { name } }));
  }

  tabs.forEach((tab) => tab.addEventListener("click", () => activate(tab.dataset.tab)));
  window.addEventListener("hashchange", () =>
    activate(window.location.hash.slice(1), { scroll: false, pushHash: false }));

  // A reload should land where the reader was, not back at the first tab.
  const initial = window.location.hash.slice(1);
  if (initial) activate(initial, { scroll: false });

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

/* Click a reviewed thumbnail to see it full size. */
document.addEventListener("click", (e) => {
  const thumb = e.target.closest(".queue-thumb");
  if (!thumb) return;
  const box = document.createElement("div");
  box.className = "lightbox";
  box.innerHTML = `<img src="${thumb.getAttribute("src")}" alt="Reviewed radiograph">`;
  const close = () => box.remove();
  box.addEventListener("click", close);
  document.addEventListener("keydown", function esc(ev) {
    if (ev.key === "Escape") { close(); document.removeEventListener("keydown", esc); }
  });
  document.body.appendChild(box);
});


/* ------------------------------------------------------- model chooser
 * The list on the Analyse tab is filled in from the server rather than
 * written into the page, so a version an administrator releases from
 * reviewed cases appears without the page having to be edited -- and the
 * scores beside each name come from one place instead of being copied into
 * the markup where they can drift.
 */
document.addEventListener("DOMContentLoaded", async () => {
  "use strict";
  const select = document.getElementById("select-model");
  if (!select) return;

  const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const NAMES = {
    hybrid: "Hybrid (DenseNet + Swin + CBAM)",
    hybrid_no_cbam: "Fusion without CBAM",
    densenet121: "DenseNet121",
    swin_t: "Swin-T",
    resnet50: "ResNet50",
  };

  try {
    const data = await fetch("/api/v1/models").then((r) => r.json());
    if (!data.available_models || !data.available_models.length) return;

    const chosen = select.value || "hybrid";
    select.innerHTML = data.available_models.map((m) => {
      const name = NAMES[m.name] || m.name;
      // Two kinds reach this list and they are not the same claim: one was
      // retrained from nothing on everything, the other was nudged from the
      // project's own checkpoint by a batch of reviews.
      const released = m.is_released
        ? (m.name.startsWith("community")
            ? " · retrained on the public dataset plus reviewed cases"
            : " · fine-tuned on reviewed cases")
        : "";
      const label = m.label ? ` — ${m.label}` : "";
      return `<option value="${esc(m.name)}">${esc(name + label + released)}</option>`;
    }).join("");
    select.value = data.available_models.some((m) => m.name === chosen) ? chosen : "hybrid";

    const released = data.available_models.filter((m) => m.is_released);
    const hint = document.getElementById("model-hint");
    if (hint && released.length) {
      hint.textContent = `${released.length} version` +
        `${released.length === 1 ? "" : "s"} built with clinician-reviewed cases ` +
        `${released.length === 1 ? "is" : "are"} available here, shown with the scores ` +
        `measured on the same held-out set as the rest. The default is unchanged — ` +
        `these are only ever something you pick.`;
    }
  } catch {
    /* Leave the list as the page shipped it: the tool still works. */
  }
});
