/* ===========================================================================
 * The clinician review workspace.
 *
 * Kept in its own file because it is a closed-access area with its own state,
 * and does not belong in the same place as the public tabs. Nothing here is a
 * security control: every rule it draws on screen -- who may edit a case, who
 * may cancel a run, who may see the console -- is enforced again by the
 * server, which is where it counts. Hiding a button only keeps the interface
 * honest about what will work.
 * ======================================================================== */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const esc = (value) =>
    String(value ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const gate = $("fb-gate");
  const workspace = $("fb-workspace");
  if (!gate || !workspace) return;

  /** The signed-in account, or null. Set only from /auth/me and /auth/login. */
  let me = null;
  let queueScope = "all";
  let trainPoll = null;
  let onFeedbackTab = (window.location.hash.slice(1) || "analyse") === "feedback";

  // --- plumbing ------------------------------------------------------------

  async function api(path, options = {}) {
    const response = await fetch(path, { credentials: "same-origin", ...options });
    let body = null;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    if (!response.ok) {
      const error = new Error((body && body.detail) || `HTTP ${response.status}`);
      error.status = response.status;
      error.detail = (body && body.detail) || null;
      throw error;
    }
    return body;
  }

  const asJson = (payload, method = "POST") => ({
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  function show(node, message, kind = "info") {
    if (!node) return;
    node.hidden = false;
    node.className = `fb-result fb-${kind}`;
    node.innerHTML = message;
  }

  function hide(node) {
    if (node) node.hidden = true;
  }

  const stamp = (iso) =>
    iso ? esc(String(iso).slice(0, 16).replace("T", " ")) + " UTC" : "—";

  /** Any error, turned into something a clinician can act on. */
  function readable(error) {
    if (error.status === 401) {
      return "Your session has ended. Log in again.";
    }
    if (error.status === 429) {
      return error.detail || "Too many requests just now. Give it a minute.";
    }
    if (error.detail) return error.detail;
    return "Could not reach the server. Check your connection and try again.";
  }

  function busy(button, isBusy) {
    if (!button) return;
    button.disabled = isBusy;
    button.classList.toggle("is-loading", isBusy);
  }

  // =========================================================== logging in

  const loginForm = $("login-form");
  const loginError = $("login-error");
  const loginStep2 = $("login-step2");
  let loginMode = "clinician";

  function setLoginMode(mode) {
    loginMode = mode;
    document.querySelectorAll("[data-login-mode]").forEach((button) => {
      const on = button.dataset.loginMode === mode;
      button.classList.toggle("is-active", on);
      button.setAttribute("aria-selected", String(on));
    });
    const clinician = mode === "clinician";
    $("row-radid").hidden = !clinician;
    $("row-password").hidden = clinician;
    $("login-radid").required = clinician;
    $("login-password").required = !clinician;
    $("login-heading").textContent = clinician ? "Log in" : "Administrator log in";
    $("login-blurb").textContent = clinician
      ? "Your email address and the Rad-Intel ID the administrator gave you. A one-time code is then sent to your registered phone."
      : "Your email address and password. A one-time code is then sent to your registered phone, same as everybody else.";
    resetToStepOne();
  }

  function resetToStepOne() {
    loginStep2.hidden = true;
    $("btn-send-code").hidden = false;
    $("login-code").value = "";
    hide(loginError);
  }

  document.querySelectorAll("[data-login-mode]").forEach((button) =>
    button.addEventListener("click", () => setLoginMode(button.dataset.loginMode)));

  async function sendCode() {
    const email = $("login-email").value.trim();
    if (!email) return show(loginError, "Enter your email address.", "error");

    let path, payload;
    if (loginMode === "clinician") {
      const radid = $("login-radid").value.trim();
      if (!radid) {
        return show(loginError,
          "Enter your Rad-Intel ID — the code starting RAD that the administrator gave you.",
          "error");
      }
      path = "/api/v1/auth/code";
      payload = { email, radintel_id: radid };
    } else {
      const password = $("login-password").value;
      if (!password) return show(loginError, "Enter your password.", "error");
      path = "/api/v1/auth/admin/code";
      payload = { email, password };
    }

    const button = $("btn-send-code");
    busy(button, true);
    hide(loginError);
    try {
      const result = await api(path, asJson(payload));
      $("code-sent").innerHTML =
        `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6L9 17l-5-5"/></svg>` +
        `<span>${esc(result.detail)}</span>`;
      loginStep2.hidden = false;
      button.hidden = true;
      $("login-code").focus();
    } catch (error) {
      show(loginError, esc(readable(error)), "error");
    } finally {
      busy(button, false);
    }
  }

  $("btn-send-code").addEventListener("click", sendCode);
  $("btn-resend").addEventListener("click", sendCode);

  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (loginStep2.hidden) return sendCode();

    const button = $("login-submit");
    busy(button, true);
    hide(loginError);
    try {
      const account = await api("/api/v1/auth/login", asJson({
        email: $("login-email").value.trim(),
        code: $("login-code").value.trim(),
      }));
      // Clear both factors from the form before the workspace appears.
      $("login-password").value = "";
      $("login-radid").value = "";
      $("login-code").value = "";
      applySession(account);
    } catch (error) {
      show(loginError, esc(readable(error)), "error");
      $("login-code").select();
    } finally {
      busy(button, false);
    }
  });

  // --- asking for access ---------------------------------------------------

  const requestPanel = $("request-panel");
  $("btn-open-request").addEventListener("click", () => {
    const opening = requestPanel.hidden;
    requestPanel.hidden = !opening;
    $("btn-open-request").setAttribute("aria-expanded", String(opening));
    $("btn-open-request").classList.toggle("is-open", opening);
    if (opening) $("req-name").focus();
  });

  $("request-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("request-submit");
    const result = $("request-result");
    busy(button, true);
    try {
      const response = await api("/api/v1/auth/request-access", asJson({
        full_name: $("req-name").value.trim(),
        email: $("req-email").value.trim(),
        institution: $("req-institution").value.trim(),
        phone: $("req-phone").value.trim(),
        registration: $("req-registration").value.trim() || null,
        message: $("req-message").value.trim() || null,
      }));
      show(result,
        `${esc(response.detail)}<br><br>Your reference is ` +
        `<code>${esc(response.reference)}</code> — quote it if you follow up.`, "ok");
      $("request-form").reset();
    } catch (error) {
      show(result, esc(readable(error)), "error");
    } finally {
      busy(button, false);
    }
  });

  // --- session -------------------------------------------------------------

  function applySession(account) {
    me = account && account.authenticated ? account : null;
    gate.hidden = !!me;
    workspace.hidden = !me;
    if (!me) {
      stopTrainPolling();
      resetToStepOne();
      return;
    }

    $("session-name").textContent = me.full_name || me.email;
    $("session-site").textContent = me.institution || "";
    $("session-role").hidden = me.role !== "admin";
    $("fb-site-label").textContent = me.institution || "your institution";
    $("admin-console").hidden = me.role !== "admin";
    $("must-change").hidden = !me.must_change_password;

    renderMyDetails();
    refreshQueue();
    loadParticipants();
    loadModelVersions();
    loadStorage();
    if (me.role === "admin") loadAdminConsole();
    startTrainPolling();
  }

  $("btn-signout").addEventListener("click", async () => {
    await fetch("/api/v1/auth/logout", { method: "POST", credentials: "same-origin" })
      .catch(() => {});
    applySession(null);
  });

  function renderMyDetails() {
    const rows = [
      ["Name", me.full_name],
      ["Institution", me.institution],
      ["Email", me.email],
      ["Phone", me.phone || "not on file"],
      ["Rad-Intel ID", me.radintel_id || "—"],
      ["Role", me.role === "admin" ? "administrator" : "clinician"],
    ];
    $("my-details").innerHTML = rows.map(([label, value]) =>
      `<div class="detail-row"><dt>${esc(label)}</dt>` +
      `<dd${label === "Rad-Intel ID" ? ' class="mono"' : ""}>${esc(value)}</dd></div>`
    ).join("");
  }

  // =========================================================== submitting

  let verdict = null;

  document.addEventListener("click", (event) => {
    const chip = event.target.closest(".chip[data-verdict]");
    if (!chip) return;
    verdict = chip.dataset.verdict;
    document.querySelectorAll(".chip[data-verdict]").forEach((other) =>
      other.classList.toggle("is-active", other === chip));
    $("fb-correction-row").hidden = verdict !== "incorrect";
  });

  function chosenFile() {
    const input = $("fb-file");
    return input.files && input.files[0];
  }

  $("fb-run").addEventListener("click", async () => {
    const file = chosenFile();
    const result = $("fb-result");
    if (!file) return show(result, "Choose a radiograph first.", "error");
    if (file.size > 20 * 1024 * 1024) {
      return show(result, "That image is over the 20 MB limit.", "error");
    }
    if (!["image/jpeg", "image/jpg", "image/png"].includes(file.type)) {
      return show(result, "Use a JPEG or a PNG. Convert a DICOM first.", "error");
    }

    busy($("fb-run"), true);
    show(result, "Running the shared model…");
    try {
      const form = new FormData();
      form.append("file", file);
      const prediction = await api("/api/v1/predict", { method: "POST", body: form });
      const percent = (prediction.probabilities.PNEUMONIA * 100).toFixed(1);
      show(result,
        `<strong>${esc(prediction.prediction_class)}</strong> · ` +
        `P(pneumonia) = ${percent}% · model <code>${esc(prediction.model_name)}</code>`);
      $("fb-review").hidden = false;
    } catch (error) {
      show(result, esc(readable(error)), "error");
    } finally {
      busy($("fb-run"), false);
    }
  });

  $("fb-submit").addEventListener("click", async () => {
    const file = chosenFile();
    const result = $("fb-result");
    if (!file) return show(result, "Choose a radiograph first.", "error");
    if (!verdict) return show(result, "Say whether the prediction was right.", "error");
    if (verdict === "incorrect" && !$("fb-corrected").value) {
      return show(result,
        "Choose what the label should have been — a correction is what the model learns from.",
        "error");
    }

    busy($("fb-submit"), true);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("verdict", verdict);
      if ($("fb-corrected").value) form.append("corrected_label", $("fb-corrected").value);
      if ($("fb-note").value.trim()) form.append("note", $("fb-note").value.trim());

      const saved = await api("/api/v1/feedback/submit", { method: "POST", body: form });
      show(result,
        `Review queued as <code>${esc(saved.id.slice(0, 8))}</code>. Nothing was ` +
        `retrained — it joins the next export.`, "ok");
      $("fb-review").hidden = true;
      $("fb-file").value = "";
      $("fb-corrected").value = "";
      $("fb-note").value = "";
      verdict = null;
      document.querySelectorAll(".chip[data-verdict]").forEach((chip) =>
        chip.classList.remove("is-active"));
      refreshQueue();
      loadStorage();
    } catch (error) {
      show(result, esc(readable(error)), "error");
    } finally {
      busy($("fb-submit"), false);
    }
  });

  // =============================================================== the queue

  document.querySelectorAll("[data-scope]").forEach((button) =>
    button.addEventListener("click", () => {
      queueScope = button.dataset.scope;
      document.querySelectorAll("[data-scope]").forEach((other) =>
        other.classList.toggle("is-active", other === button));
      refreshQueue();
    }));

  $("fb-sort").addEventListener("change", refreshQueue);
  $("fb-refresh").addEventListener("click", () => {
    refreshQueue();
    loadParticipants();
    loadStorage();
  });

  function queueQuery() {
    const params = new URLSearchParams({ limit_rows: "100", sort: $("fb-sort").value });
    if (queueScope === "mine") params.set("scope", "mine");
    if (queueScope === "pending") params.set("status", "pending");
    return params.toString();
  }

  function emptyQueue(message) {
    return `<div class="empty-state">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 11l3 3L22 4"/>
      <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
      <p>${esc(message)}</p></div>`;
  }

  function queueItem(item) {
    const percent = (item.predicted_probability * 100).toFixed(1);
    const corrected = item.corrected_label
      ? ` → should be <strong>${esc(item.corrected_label)}</strong>` : "";
    const frozen = item.status === "exported";
    const mine = item.owner_id && me && item.owner_id === me.id;

    return `<div class="queue-item ${frozen ? "is-frozen" : ""} ${mine ? "is-mine" : ""}"
                 data-record="${esc(item.id)}">
      <div class="queue-head">
        <span class="badge ${item.verdict === "correct" ? "badge-ok" : "badge-warn"}">${esc(item.verdict)}</span>
        <span class="badge ${frozen ? "badge-neutral" : "badge-accent"}">${esc(frozen ? "exported" : "awaiting export")}</span>
        ${mine ? '<span class="badge badge-mine">yours</span>' : ""}
      </div>
      <div class="queue-thumb-row">
        <img class="queue-thumb" loading="lazy" alt="Reviewed radiograph"
             src="/api/v1/feedback/image/${esc(item.id)}">
        <div class="queue-thumb-meta">
          <div class="queue-body">
            predicted <strong>${esc(item.predicted_label)}</strong> at ${percent}%${corrected}
          </div>
          <div class="queue-meta">
            <strong>${esc(item.reviewer)}</strong>
            ${item.institution ? ` · ${esc(item.institution)}` : ""}
            <br>${stamp(item.created_at)}
            · sha256 <code>${esc(item.image_sha256.slice(0, 10))}</code>
            ${item.exported_in ? `<br>export <code>${esc(item.exported_in)}</code>` : ""}
          </div>
        </div>
      </div>
      ${item.note ? `<div class="queue-note">${esc(item.note)}</div>` : ""}
      ${item.editable ? `<div class="queue-actions">
        <button class="btn-link" data-edit="${esc(item.id)}">Edit</button>
        <button class="btn-link btn-link-danger" data-delete="${esc(item.id)}">Delete</button>
      </div>` : ""}
      <div class="queue-edit" hidden></div>
    </div>`;
  }

  async function refreshQueue() {
    if (!me) return;
    const queue = $("fb-queue");
    try {
      const data = await api(`/api/v1/feedback?${queueQuery()}`);

      $("fb-counts").innerHTML =
        Object.entries(data.counts).map(([key, value]) =>
          `<span class="count-pill">${esc(key === "total" ? "all cases" : key)}: <strong>${value}</strong></span>`
        ).join("") +
        Object.entries(data.institutions || {}).map(([key, value]) =>
          `<span class="count-pill is-site">${esc(key)}: <strong>${value}</strong></span>`
        ).join("");

      if (!data.items.length) {
        queue.innerHTML = emptyQueue(
          queueScope === "mine"
            ? "You have not submitted a case yet. Run one on the left to start."
            : queueScope === "pending"
              ? "Nothing waiting. Every case has been exported."
              : "No reviews yet. Run a case to start the queue."
        );
      } else {
        queue.innerHTML = data.items.map(queueItem).join("");
      }

      const exports = await api("/api/v1/feedback/exports").catch(() => null);
      $("fb-exports").innerHTML = exports && exports.exports.length
        ? "<p class='field-label'>Exports written</p>" + exports.exports.map((row) =>
            `<div class="export-row"><code>${esc(row.tag)}</code> — ${row.records} cases</div>`
          ).join("")
        : "";
    } catch (error) {
      if (error.status === 401) return applySession(null);
      queue.innerHTML = emptyQueue(readable(error));
    }
  }

  // --- editing and deleting your own case ----------------------------------

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-edit]");
    if (!button) return;
    const item = button.closest(".queue-item");
    const panel = item.querySelector(".queue-edit");
    if (!panel.hidden) {
      panel.hidden = true;
      panel.innerHTML = "";
      return;
    }
    panel.hidden = false;
    panel.innerHTML = `
      <p class="muted small">
        The verdict, the corrected label and the note can change. The image cannot:
        swapping the pixels under a verdict other people have already read would make
        the record untrue. Delete the case and upload it again instead.
      </p>
      <div class="form-row">
        <span class="field-label">Was the prediction correct?</span>
        <div class="chip-row">
          <button class="chip" type="button" data-ev="correct">Correct</button>
          <button class="chip" type="button" data-ev="incorrect">Incorrect</button>
        </div>
      </div>
      <div class="form-row">
        <label class="field-label">What it should have been</label>
        <select class="input edit-label">
          <option value="">Unchanged</option>
          <option value="NORMAL">NORMAL</option>
          <option value="PNEUMONIA">PNEUMONIA</option>
        </select>
      </div>
      <div class="form-row">
        <label class="field-label">Review note</label>
        <textarea class="input edit-note" rows="2" maxlength="2000"></textarea>
      </div>
      <div class="fb-actions">
        <button class="btn-primary" type="button" data-save="${esc(button.dataset.edit)}">Save changes</button>
        <button class="btn-link" type="button" data-cancel-edit>Cancel</button>
      </div>
      <div class="fb-result edit-result" hidden></div>`;
    const note = item.querySelector(".queue-note");
    if (note) panel.querySelector(".edit-note").value = note.textContent.trim();
  });

  document.addEventListener("click", (event) => {
    const chip = event.target.closest(".queue-edit .chip[data-ev]");
    if (!chip) return;
    chip.closest(".chip-row").querySelectorAll(".chip").forEach((other) =>
      other.classList.toggle("is-active", other === chip));
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest("[data-cancel-edit]")) return;
    const panel = event.target.closest(".queue-edit");
    panel.hidden = true;
    panel.innerHTML = "";
  });

  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-save]");
    if (!button) return;
    const panel = button.closest(".queue-edit");
    const result = panel.querySelector(".edit-result");
    const chosen = panel.querySelector(".chip.is-active");
    const payload = { note: panel.querySelector(".edit-note").value.trim() || null };
    if (chosen) payload.verdict = chosen.dataset.ev;
    const label = panel.querySelector(".edit-label").value;
    if (label) payload.corrected_label = label;

    busy(button, true);
    try {
      await api(`/api/v1/feedback/${button.dataset.save}`, asJson(payload, "PATCH"));
      refreshQueue();
    } catch (error) {
      show(result, esc(readable(error)), "error");
    } finally {
      busy(button, false);
    }
  });

  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-delete]");
    if (!button) return;
    const confirmed = window.confirm(
      "Delete this case?\n\n" +
      "It will not appear in any future export, and its image is removed unless " +
      "another case uses the same file.\n\n" +
      "An export already written, and any model already trained from it, stay as " +
      "they are — those record what was actually trained on."
    );
    if (!confirmed) return;
    busy(button, true);
    try {
      const outcome = await api(`/api/v1/feedback/${button.dataset.delete}`,
                                { method: "DELETE" });
      show($("fb-result"), esc(outcome.detail), "ok");
      refreshQueue();
      loadStorage();
    } catch (error) {
      show($("fb-result"), esc(readable(error)), "error");
    } finally {
      busy(button, false);
    }
  });

  // --- export --------------------------------------------------------------

  $("fb-export").addEventListener("click", async () => {
    const confirmed = window.confirm(
      "Write a training export of every case not yet exported?\n\n" +
      "This freezes those records so they keep matching what gets trained on. " +
      "It does not start training."
    );
    if (!confirmed) return;
    busy($("fb-export"), true);
    try {
      const result = await api("/api/v1/feedback/export", { method: "POST" });
      show($("fb-result"),
        `Export <code>${esc(result.export_tag)}</code> written with ` +
        `${result.record_count} cases. No training has started.`, "ok");
    } catch (error) {
      show($("fb-result"), esc(readable(error)), "error");
    } finally {
      busy($("fb-export"), false);
      refreshQueue();
    }
  });

  // ============================================================== training

  const trainLog = $("fb-train-log");

  function writeLog(text) {
    trainLog.hidden = false;
    trainLog.textContent = text;
    trainLog.scrollTop = trainLog.scrollHeight;
  }

  /**
   * Reflect a run's state in the interface for everybody, not just its holder.
   *
   * While a run is going, export and train are held and say who is holding
   * them, so a second clinician sees a reason rather than a dead button.
   */
  function applyTrainState(state) {
    const banner = $("train-banner");
    const start = $("fb-train");
    const cancel = $("fb-train-cancel");

    if (state.running) {
      const who = state.is_holder
        ? "You started a fine-tuning run"
        : `${state.holder_name} is running a fine-tuning job`;
      const site = state.holder_institution && !state.is_holder
        ? ` (${state.holder_institution})` : "";
      banner.hidden = false;
      $("train-banner-text").innerHTML =
        `<strong>${esc(who)}${esc(site)}</strong> — started ${stamp(state.started_at)}, ` +
        `${state.cases} cases, ${state.epochs} epochs. Exporting and training are held ` +
        `until it finishes. You can carry on uploading and correcting cases.`;
      start.disabled = true;
      start.classList.add("is-loading");
      start.textContent = state.is_holder ? "Your run is in progress…"
                                          : `Held — ${state.holder_name} is training`;
      $("fb-export").disabled = true;
      cancel.hidden = !state.can_cancel;
    } else {
      banner.hidden = true;
      start.disabled = false;
      start.classList.remove("is-loading");
      start.textContent = "Start fine-tuning run";
      $("fb-export").disabled = false;
      cancel.hidden = true;
    }

    if (state.log) writeLog(state.log);

    if (!state.running && state.cancel_reason) {
      writeLog(`${state.log || ""}\n--- cancelled (${state.cancel_reason}) ---\n` +
               `No new version was written and the current model is unchanged. ` +
               `The export is still there, so the run can be retried.`);
    } else if (!state.running && state.produced_checkpoint) {
      writeLog(`${state.log || ""}\n--- finished ---\nNew version: ` +
               `${state.produced_checkpoint}\nIt is not being served yet. Read the ` +
               `before-and-after comparison above; an administrator activates a ` +
               `version from "The shared model" panel.`);
      loadModelVersions();
    }
  }

  async function pollTrainState() {
    if (!me) return;
    try {
      applyTrainState(await api("/api/v1/feedback/train/status"));
    } catch (error) {
      if (error.status === 401) applySession(null);
    }
  }

  function startTrainPolling() {
    stopTrainPolling();
    pollTrainState();
    // Slower when the reader is on another tab: the poll is also the holder's
    // heartbeat, so it has to keep going, but it need not be brisk.
    trainPoll = setInterval(pollTrainState, onFeedbackTab ? 4000 : 20000);
  }

  function stopTrainPolling() {
    if (trainPoll) clearInterval(trainPoll);
    trainPoll = null;
  }

  document.addEventListener("radintel:tab", (event) => {
    onFeedbackTab = event.detail.name === "feedback";
    if (me) {
      startTrainPolling();
      if (onFeedbackTab) refreshQueue();
    }
  });

  $("fb-train").addEventListener("click", async () => {
    const confirmed = window.confirm(
      "Start a fine-tuning run on the latest export?\n\n" +
      "It creates a NEW version of the shared model. The current one keeps being " +
      "served until an administrator activates the new one.\n\n" +
      "It takes several minutes. Keep this tab open: if the browser that started a " +
      "run stops reporting in for three minutes, the run cancels itself."
    );
    if (!confirmed) return;

    busy($("fb-train"), true);
    writeLog("Starting…");
    try {
      const started = await api("/api/v1/feedback/train",
                                { method: "POST", body: new FormData() });
      writeLog(`Started on ${started.manifest} — ${started.cases} cases, ` +
               `${started.epochs} epochs.\n${started.note}\n\nWaiting for output…`);
      startTrainPolling();
    } catch (error) {
      writeLog(readable(error));
      busy($("fb-train"), false);
      pollTrainState();
    }
  });

  $("fb-train-cancel").addEventListener("click", async () => {
    if (!window.confirm("Cancel the run? No new version will be written.")) return;
    busy($("fb-train-cancel"), true);
    try {
      const outcome = await api("/api/v1/feedback/train/cancel", { method: "POST" });
      writeLog(outcome.detail);
    } catch (error) {
      writeLog(readable(error));
    } finally {
      busy($("fb-train-cancel"), false);
      pollTrainState();
    }
  });

  // ======================================================= the shared model

  async function loadModelVersions() {
    if (!me) return;
    try {
      const data = await api("/api/v1/feedback/models");
      $("models-note").textContent = `${data.note} ${data.where_to_test}`;
      $("model-versions").innerHTML = data.versions.map((version) => {
        const metric = (value, digits = 3) =>
          typeof value === "number" ? value.toFixed(digits) : "—";
        return `<div class="version-row ${version.is_live ? "is-live" : ""}">
          <div class="version-head">
            <code>${esc(version.file)}</code>
            ${version.is_live ? '<span class="badge badge-ok">being served</span>' : ""}
            ${version.is_base ? '<span class="badge badge-neutral">base</span>' : ""}
          </div>
          <div class="version-label">${esc(version.label)}</div>
          ${version.accuracy != null ? `<div class="version-metrics">
            <span>acc ${metric(version.accuracy)}</span>
            <span>AUROC ${metric(version.auroc)}</span>
            <span>sens ${metric(version.sensitivity)}</span>
            <span>spec ${metric(version.specificity)}</span>
          </div>` : ""}
          ${version.reviewed_cases ? `<div class="version-meta">from
            ${version.reviewed_cases} reviewed cases · ${stamp(version.created_at)}</div>` : ""}
          ${me.role === "admin" && !version.is_live
            ? `<button class="btn-secondary btn-sm" data-activate="${esc(version.file)}">
                 Serve this version</button>`
            : ""}
        </div>`;
      }).join("");
    } catch (error) {
      if (error.status === 401) applySession(null);
    }
  }

  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-activate]");
    if (!button) return;
    const file = button.dataset.activate;
    const isBase = file === "best_hybrid_model.pt";
    const confirmed = window.confirm(
      isBase
        ? "Go back to serving the base checkpoint?\n\nThis is the version the project's " +
          "reported results describe."
        : `Serve ${file} to everybody?\n\nEvery prediction, here and on the Analyse ` +
          `tab, will come from it. The project's reported figures describe the base ` +
          `checkpoint, which stays on disk — you can switch back at any time.`
    );
    if (!confirmed) return;
    busy(button, true);
    try {
      const body = new FormData();
      body.append("file", file);
      const outcome = await api("/api/v1/feedback/models/activate", { method: "POST", body });
      show($("fb-result"), esc(outcome.detail), "ok");
      loadModelVersions();
    } catch (error) {
      show($("fb-result"), esc(readable(error)), "error");
    } finally {
      busy(button, false);
    }
  });

  async function loadStorage() {
    if (!me) return;
    try {
      const data = await api("/api/v1/feedback/storage");
      $("storage-summary").innerHTML =
        `<div class="storage-head">
           <span class="count-pill">${data.image_count} images</span>
           <span class="count-pill">${esc(data.human)}</span>
         </div>
         <div class="storage-path"><code>${esc(data.directory)}</code></div>
         <p class="muted small">${esc(data.note)}</p>`;
    } catch {
      $("storage-summary").innerHTML = "";
    }
  }

  // ============================================================ participants

  async function loadParticipants() {
    if (!me) return;
    try {
      const data = await api("/api/v1/feedback/participants");
      $("participant-count").textContent = data.count;
      $("participants").innerHTML = data.participants.map((person) => {
        const initials = person.full_name.split(/\s+/).filter(Boolean)
          .slice(0, 2).map((word) => word[0]).join("").toUpperCase();
        return `<div class="roster-row">
          <span class="avatar" aria-hidden="true">${esc(initials)}</span>
          <span class="roster-text">
            <strong>${esc(person.full_name)}</strong>
            <small>${esc(person.institution)}</small>
          </span>
          ${person.role === "admin"
            ? '<span class="badge badge-neutral">admin</span>' : ""}
        </div>`;
      }).join("");
    } catch (error) {
      if (error.status === 401) applySession(null);
    }
  }

  // ======================================================= admin console

  async function loadAdminConsole() {
    await Promise.all([
      loadAdminOverview(),
      loadAdminUsers(),
      loadAdminRequests(),
      loadAdminSessions(),
      loadAdminActivity(),
    ]);
    $("ap-name").value = me.full_name || "";
    $("ap-institution").value = me.institution || "";
    $("ap-email").value = me.email || "";
    $("ap-phone").value = me.phone || "";
  }

  async function loadAdminOverview() {
    try {
      const data = await api("/api/v1/admin/overview");
      $("admin-contact-note").textContent = data.contact_note;
      $("admin-stats").innerHTML = [
        ["Clinicians", data.users.active, `${data.users.disabled} disabled`],
        ["Requests waiting", data.access_requests_waiting, "verify before issuing"],
        ["Open sessions", data.open_sessions, "logged in right now"],
        ["Log entries", data.activity_entries, "kept for a year"],
      ].map(([label, value, hint]) =>
        `<div class="stat"><span class="stat-value">${value}</span>
         <span class="stat-label">${esc(label)}</span>
         <span class="stat-hint">${esc(hint)}</span></div>`
      ).join("");

      const warning = $("delivery-warning");
      const missing = [];
      if (!data.delivery.sms_configured) {
        missing.push(data.delivery.echo_to_server_log
          ? "No message provider is configured, so login codes are written to the " +
            "<strong>server log</strong> instead of being sent to anyone's phone. That " +
            "is a local test mode — configure a provider before anyone else uses this."
          : "No message provider is configured, so <strong>nobody can log in</strong>: " +
            "codes cannot be delivered. Run <code>python scripts/manage_clinicians.py " +
            "delivery</code> to see what to set.");
      }
      if (!data.delivery.email_configured) {
        missing.push(`No SMTP server is configured, so access requests are not emailed ` +
                     `to ${esc(data.delivery.admin_email)}. They are still recorded and ` +
                     `appear above.`);
      }
      warning.hidden = !missing.length;
      warning.innerHTML = missing.length
        ? `<svg class="notice-ico" viewBox="0 0 24 24" aria-hidden="true">
             <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>
             <path d="M12 9v4M12 17h.01"/></svg>
           <span>${missing.join("<br><br>")}</span>`
        : "";
    } catch (error) {
      if (error.status === 401) applySession(null);
    }
  }

  // --- creating accounts ---------------------------------------------------

  $("nu-generate").addEventListener("click", async () => {
    busy($("nu-generate"), true);
    try {
      const { radintel_id } = await api("/api/v1/admin/new-id");
      $("nu-radid").value = radintel_id;
    } catch (error) {
      show($("nu-result"), esc(readable(error)), "error");
    } finally {
      busy($("nu-generate"), false);
    }
  });

  $("admin-add-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const result = $("nu-result");
    busy($("nu-submit"), true);
    try {
      const created = await api("/api/v1/admin/users", asJson({
        full_name: $("nu-name").value.trim(),
        email: $("nu-email").value.trim(),
        institution: $("nu-institution").value.trim(),
        phone: $("nu-phone").value.trim(),
        admin_note: $("nu-note").value.trim() || null,
        radintel_id: $("nu-radid").value.trim() || null,
      }));
      show(result,
        `<strong>${esc(created.user.full_name)}</strong> can now log in.<br><br>` +
        `Rad-Intel ID: <code class="id-reveal">${esc(created.radintel_id)}</code><br><br>` +
        `${esc(created.detail)}`, "ok");
      $("admin-add-form").reset();
      $("nu-radid").value = "";
      loadAdminUsers();
      loadAdminOverview();
      loadParticipants();
    } catch (error) {
      show(result, esc(readable(error)), "error");
    } finally {
      busy($("nu-submit"), false);
    }
  });

  // --- the clinician table -------------------------------------------------

  let searchTimer = null;
  $("admin-search").addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(loadAdminUsers, 250);
  });
  $("admin-search-clear").addEventListener("click", () => {
    $("admin-search").value = "";
    loadAdminUsers();
  });

  async function loadAdminUsers() {
    const body = $("admin-users").querySelector("tbody");
    const query = $("admin-search").value.trim();
    try {
      const data = await api(`/api/v1/admin/users?q=${encodeURIComponent(query)}`);
      if (!data.users.length) {
        body.innerHTML = `<tr><td colspan="7" class="table-empty">
          No account matches that. A Rad-Intel ID has to be given in full — IDs are
          stored encrypted, so there is nothing to match a fragment against.</td></tr>`;
        return;
      }
      body.innerHTML = data.users.map((user) => `
        <tr data-user="${esc(user.id)}" class="${user.status === "active" ? "" : "is-disabled"}">
          <td>
            <strong>${esc(user.full_name)}</strong>
            ${user.role === "admin" ? '<span class="badge badge-accent">admin</span>' : ""}
            ${user.admin_note ? `<br><small class="muted">${esc(user.admin_note)}</small>` : ""}
          </td>
          <td>${esc(user.institution)}</td>
          <td><small>${esc(user.email)}<br>${esc(user.phone || "no phone")}</small></td>
          <td class="mono">${esc(user.radintel_id || "—")}</td>
          <td><small>${user.last_login_at ? stamp(user.last_login_at) : "never"}</small></td>
          <td><span class="badge ${user.status === "active" ? "badge-ok" : "badge-neutral"}">${esc(user.status)}</span></td>
          <td class="row-actions">
            <button class="btn-link" data-user-edit="${esc(user.id)}">Edit</button>
            <button class="btn-link" data-user-rotate="${esc(user.id)}">New ID</button>
            <button class="btn-link" data-user-toggle="${esc(user.id)}"
                    data-next="${user.status === "active" ? "disabled" : "active"}">
              ${user.status === "active" ? "Disable" : "Enable"}</button>
            <button class="btn-link btn-link-danger" data-user-delete="${esc(user.id)}">Delete</button>
          </td>
        </tr>
        <tr class="edit-row" hidden><td colspan="7"></td></tr>`).join("");
    } catch (error) {
      if (error.status === 401) return applySession(null);
      body.innerHTML = `<tr><td colspan="7" class="table-empty">${esc(readable(error))}</td></tr>`;
    }
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-user-edit]");
    if (!button) return;
    const row = button.closest("tr");
    const editRow = row.nextElementSibling;
    const cell = editRow.querySelector("td");
    if (!editRow.hidden) {
      editRow.hidden = true;
      cell.innerHTML = "";
      return;
    }
    const cells = row.querySelectorAll("td");
    const name = cells[0].querySelector("strong").textContent;
    const institution = cells[1].textContent.trim();
    const contact = cells[2].querySelector("small").innerHTML.split("<br>");
    editRow.hidden = false;
    cell.innerHTML = `
      <div class="inline-form">
        <label><span class="field-label">Name</span>
          <input class="input ef-name" value="${esc(name)}" maxlength="120"></label>
        <label><span class="field-label">Institution</span>
          <input class="input ef-institution" value="${esc(institution)}" maxlength="160"></label>
        <label><span class="field-label">Email</span>
          <input class="input ef-email" value="${esc(contact[0] || "")}" maxlength="200"></label>
        <label><span class="field-label">Phone</span>
          <input class="input ef-phone" value="${esc((contact[1] || "").replace("no phone", ""))}" maxlength="24"></label>
        <label><span class="field-label">Private note</span>
          <input class="input ef-note" maxlength="500"></label>
        <div class="fb-actions">
          <button class="btn-primary btn-sm" data-user-save="${esc(button.dataset.userEdit)}">Save</button>
          <button class="btn-link" data-user-cancel>Cancel</button>
        </div>
        <div class="fb-result ef-result" hidden></div>
      </div>`;
    const note = cells[0].querySelector("small");
    if (note) cell.querySelector(".ef-note").value = note.textContent.trim();
  });

  document.addEventListener("click", (event) => {
    if (!event.target.closest("[data-user-cancel]")) return;
    const editRow = event.target.closest("tr");
    editRow.hidden = true;
    editRow.querySelector("td").innerHTML = "";
  });

  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-user-save]");
    if (!button) return;
    const form = button.closest(".inline-form");
    const result = form.querySelector(".ef-result");
    busy(button, true);
    try {
      await api(`/api/v1/admin/users/${button.dataset.userSave}`, asJson({
        full_name: form.querySelector(".ef-name").value.trim(),
        institution: form.querySelector(".ef-institution").value.trim(),
        email: form.querySelector(".ef-email").value.trim(),
        phone: form.querySelector(".ef-phone").value.trim(),
        admin_note: form.querySelector(".ef-note").value.trim(),
      }, "PATCH"));
      loadAdminUsers();
      loadParticipants();
    } catch (error) {
      show(result, esc(readable(error)), "error");
    } finally {
      busy(button, false);
    }
  });

  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-user-toggle]");
    if (!button) return;
    const next = button.dataset.next;
    if (next === "disabled" && !window.confirm(
      "Disable this account?\n\nThey are logged out immediately and cannot log in " +
      "again. Their cases stay in the queue."
    )) return;
    busy(button, true);
    try {
      await api(`/api/v1/admin/users/${button.dataset.userToggle}`,
                asJson({ status: next }, "PATCH"));
      loadAdminUsers();
      loadAdminOverview();
      loadAdminSessions();
      loadParticipants();
    } catch (error) {
      window.alert(readable(error));
    } finally {
      busy(button, false);
    }
  });

  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-user-rotate]");
    if (!button) return;
    if (!window.confirm(
      "Issue a replacement Rad-Intel ID?\n\nThe current one stops working at once and " +
      "their open sessions are closed. You will need to pass the new ID to them."
    )) return;
    busy(button, true);
    try {
      const outcome = await api(`/api/v1/admin/users/${button.dataset.userRotate}/rotate-id`,
                                { method: "POST" });
      window.alert(`${outcome.detail}\n\nNew Rad-Intel ID: ${outcome.radintel_id}`);
      loadAdminUsers();
      loadAdminSessions();
    } catch (error) {
      window.alert(readable(error));
    } finally {
      busy(button, false);
    }
  });

  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-user-delete]");
    if (!button) return;
    if (!window.confirm(
      "Delete this account?\n\nThey can no longer log in. Their reviews stay in the " +
      "queue under their name, because other participants have already seen and " +
      "trained on them — delete those from the queue separately if that is what you want."
    )) return;
    busy(button, true);
    try {
      const outcome = await api(`/api/v1/admin/users/${button.dataset.userDelete}`,
                                { method: "DELETE" });
      window.alert(outcome.detail);
      loadAdminUsers();
      loadAdminOverview();
      loadParticipants();
    } catch (error) {
      window.alert(readable(error));
    } finally {
      busy(button, false);
    }
  });

  // --- access requests -----------------------------------------------------

  async function loadAdminRequests() {
    try {
      const data = await api("/api/v1/admin/requests");
      $("req-count").textContent = data.count;
      if (!data.count) {
        $("admin-requests").innerHTML =
          "<p class='muted small'>Nobody is waiting. Requests arrive here and, when " +
          "SMTP is configured, by email too.</p>";
        return;
      }
      $("admin-requests").innerHTML = data.requests.map((request) => `
        <div class="request-row">
          <div class="request-row-head">
            <strong>${esc(request.full_name)}</strong>
            <span class="badge ${request.status === "new" ? "badge-accent" : "badge-neutral"}">${esc(request.status)}</span>
          </div>
          <div class="request-row-body">
            ${esc(request.institution)}<br>
            <a href="mailto:${esc(request.email)}">${esc(request.email)}</a><br>
            ${esc(request.phone || "no phone given")}
            ${request.registration ? `<br>registration ${esc(request.registration)}` : ""}
            ${request.message ? `<br><em>${esc(request.message)}</em>` : ""}
          </div>
          <div class="request-row-meta">
            ref <code>${esc(request.reference)}</code> · ${stamp(request.created_at)}
            · emailed to you: ${request.emailed_to_admin ? "yes" : "no"}
          </div>
          <div class="fb-actions">
            <button class="btn-secondary btn-sm" data-prefill='${esc(JSON.stringify({
              name: request.full_name, email: request.email,
              institution: request.institution, phone: request.phone || "",
            }))}'>Use these details</button>
            <button class="btn-link" data-request-status="${esc(request.id)}" data-next="contacted">Mark contacted</button>
            <button class="btn-link" data-request-status="${esc(request.id)}" data-next="declined">Decline</button>
            <button class="btn-link btn-link-danger" data-request-delete="${esc(request.id)}">Remove</button>
          </div>
        </div>`).join("");
    } catch (error) {
      if (error.status === 401) applySession(null);
    }
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-prefill]");
    if (!button) return;
    const details = JSON.parse(button.dataset.prefill);
    $("nu-name").value = details.name;
    $("nu-email").value = details.email;
    $("nu-institution").value = details.institution;
    $("nu-phone").value = details.phone;
    $("nu-name").scrollIntoView({ behavior: "smooth", block: "center" });
    $("nu-generate").focus();
  });

  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-request-status]");
    if (!button) return;
    try {
      await api(`/api/v1/admin/requests/${button.dataset.requestStatus}` +
                `?status=${button.dataset.next}`, { method: "PATCH" });
      loadAdminRequests();
      loadAdminOverview();
    } catch (error) {
      window.alert(readable(error));
    }
  });

  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-request-delete]");
    if (!button) return;
    if (!window.confirm("Remove this request from the list?")) return;
    try {
      await api(`/api/v1/admin/requests/${button.dataset.requestDelete}`,
                { method: "DELETE" });
      loadAdminRequests();
      loadAdminOverview();
    } catch (error) {
      window.alert(readable(error));
    }
  });

  // --- sessions and the log ------------------------------------------------

  async function loadAdminSessions() {
    try {
      const data = await api("/api/v1/admin/sessions");
      $("session-count").textContent = data.count;
      $("admin-sessions").innerHTML = data.count
        ? data.sessions.map((row) => `
          <div class="roster-row">
            <span class="roster-text">
              <strong>${esc(row.full_name)}${row.is_you ? " (you)" : ""}</strong>
              <small>${esc(row.institution || "")} · last seen ${stamp(row.last_seen_at)}
              · ${esc(row.ip || "unknown address")}</small>
            </span>
            ${row.is_you ? "" :
              `<button class="btn-link btn-link-danger" data-session-close="${esc(row.id)}">Close</button>`}
          </div>`).join("")
        : "<p class='muted small'>Nobody has the workspace open.</p>";
    } catch (error) {
      if (error.status === 401) applySession(null);
    }
  }

  document.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-session-close]");
    if (!button) return;
    try {
      await api(`/api/v1/admin/sessions/${button.dataset.sessionClose}`,
                { method: "DELETE" });
      loadAdminSessions();
      loadAdminOverview();
    } catch (error) {
      window.alert(readable(error));
    }
  });

  async function loadAdminActivity() {
    const body = $("admin-activity").querySelector("tbody");
    try {
      const data = await api("/api/v1/admin/activity?limit_rows=120");
      $("activity-count").textContent = data.total;
      body.innerHTML = data.entries.length
        ? data.entries.map((entry) => `
          <tr class="${entry.outcome === "ok" ? "" : "is-refused"}">
            <td><small>${stamp(entry.at)}</small></td>
            <td><small>${esc(entry.actor)}</small></td>
            <td><code>${esc(entry.action)}</code>
              ${entry.outcome !== "ok" ? `<span class="badge badge-warn">${esc(entry.outcome)}</span>` : ""}</td>
            <td><small>${esc(entry.detail || "")}</small></td>
            <td><small>${esc(entry.ip || "")}</small></td>
          </tr>`).join("")
        : `<tr><td colspan="5" class="table-empty">Nothing logged yet.</td></tr>`;
    } catch (error) {
      if (error.status === 401) applySession(null);
    }
  }

  // --- the administrator's own account -------------------------------------

  $("admin-profile-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const result = $("ap-result");
    try {
      await api("/api/v1/admin/profile", asJson({
        full_name: $("ap-name").value.trim(),
        institution: $("ap-institution").value.trim(),
        email: $("ap-email").value.trim(),
        phone: $("ap-phone").value.trim(),
      }, "PATCH"));
      const refreshed = await api("/api/v1/auth/me");
      me = refreshed;
      $("session-name").textContent = me.full_name;
      $("session-site").textContent = me.institution;
      renderMyDetails();
      show(result, "Saved.", "ok");
    } catch (error) {
      show(result, esc(readable(error)), "error");
    }
  });

  $("admin-password-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const result = $("pw-result");
    try {
      const outcome = await api("/api/v1/auth/password", asJson({
        current_password: $("pw-current").value,
        new_password: $("pw-new").value,
      }));
      show(result, esc(outcome.detail), "ok");
      $("admin-password-form").reset();
      $("must-change").hidden = true;
    } catch (error) {
      show(result, esc(readable(error)), "error");
    }
  });

  // --- start up ------------------------------------------------------------

  setLoginMode("clinician");
  api("/api/v1/auth/me").then(applySession).catch(() => applySession(null));
})();
