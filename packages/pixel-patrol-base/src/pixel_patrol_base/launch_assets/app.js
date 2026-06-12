"use strict";

const POLL_INTERVAL_MS = 500;

const form = document.getElementById("process-form");
const startBtn = document.getElementById("start-btn");

const loaderSelect = document.getElementById("loader");
const fileExtInput = document.getElementById("file-extensions");
const fileExtHelp = document.getElementById("file-extensions-help");
const processorsIncludeSelect = document.getElementById("processors-include");
const processorsExcludeSelect = document.getElementById("processors-exclude");

const versionInfoEl = document.getElementById("version-info");
const openExistingBtn = document.getElementById("open-existing-btn");
const openExistingOverlay = document.getElementById("open-existing-overlay");
const browserPathInput = document.getElementById("browser-path");
const browserUpBtn = document.getElementById("browser-up-btn");
const browserListEl = document.getElementById("browser-list");
const existingReportError = document.getElementById("existing-report-error");
const existingReportCancelBtn = document.getElementById("existing-report-cancel");
const existingReportOpenBtn = document.getElementById("existing-report-open");

const statusEl = document.getElementById("progress-status");
const progressContainer = document.getElementById("progress-bar-container");
const progressBar = document.getElementById("progress-bar");
const progressLabel = document.getElementById("progress-bar-label");
const detailsEl = document.getElementById("progress-details");
const errorEl = document.getElementById("error-message");
const warningsEl = document.getElementById("warnings-display");
const actionButtonsEl = document.getElementById("action-buttons");

let availableLoaders = [];
let pollTimer = null;
const dismissedWarnings = new Set();

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function warningKey(w) {
  return `${w.timestamp}|${w.level}|${w.message}`;
}

// ---------------------------------------------------------------------
// Initial setup: loaders / processors
// ---------------------------------------------------------------------

async function loadLoaders() {
  const res = await fetch("/api/loaders");
  availableLoaders = await res.json();
  loaderSelect.innerHTML = "";
  for (const loader of availableLoaders) {
    const opt = document.createElement("option");
    opt.value = loader.value;
    opt.textContent = loader.label;
    loaderSelect.appendChild(opt);
  }
  // Default to the first real loader (index 1), matching previous behaviour.
  if (availableLoaders.length > 1) {
    loaderSelect.value = availableLoaders[1].value;
  }
  updateFileExtensionsHelp();
}

async function loadProcessors() {
  const res = await fetch("/api/processors");
  const processors = await res.json();
  for (const select of [processorsIncludeSelect, processorsExcludeSelect]) {
    select.innerHTML = "";
    for (const proc of processors) {
      const opt = document.createElement("option");
      opt.value = proc.id;
      opt.textContent = proc.name;
      select.appendChild(opt);
    }
  }
}

function updateFileExtensionsHelp() {
  const loader = availableLoaders.find((l) => l.value === loaderSelect.value);
  if (loader && loader.extensions && loader.extensions.length) {
    const extStr = loader.extensions.join(", ");
    const shown = extStr.length > 50 ? extStr.slice(0, 50) + "..." : extStr;
    fileExtInput.placeholder = `e.g., ${shown}`;
    const supported = loader.extensions.slice(0, 10).join(", ");
    const more = loader.extensions.length > 10 ? "..." : "";
    fileExtHelp.innerHTML =
      "Leave empty for all supported extensions. Otherwise, comma-separated extensions:<br>" +
      `Supported: ${escapeHtml(supported)}${more}`;
  } else {
    fileExtInput.placeholder = "Leave empty for all supported extensions";
    fileExtHelp.textContent = "Comma-separated extensions (leave empty for all)";
  }
}

loaderSelect.addEventListener("change", updateFileExtensionsHelp);

// ---------------------------------------------------------------------
// Info popover
// ---------------------------------------------------------------------

document.getElementById("config-info-btn").addEventListener("click", () => {
  const panel = document.getElementById("config-info-panel");
  panel.hidden = !panel.hidden;
});

// ---------------------------------------------------------------------
// Form submission
// ---------------------------------------------------------------------

function selectedValues(select) {
  return Array.from(select.selectedOptions).map((o) => o.value);
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const payload = {
    base_directory: form.base_directory.value.trim(),
    output_path: form.output_path.value.trim(),
    project_name: form.project_name.value.trim(),
    loader: form.loader.value,
    paths: form.paths.value.trim(),
    file_extensions: form.file_extensions.value.trim(),
    flavor: form.flavor.value.trim(),
    description: form.description.value.trim(),
    processors_include: selectedValues(processorsIncludeSelect),
    processors_exclude: selectedValues(processorsExcludeSelect),
    max_workers: form.max_workers.value ? parseInt(form.max_workers.value, 10) : null,
    scheduler: form.scheduler.value.trim(),
    mb_per_task: form.mb_per_task.value ? parseFloat(form.mb_per_task.value) : null,
    max_images_per_task: form.max_images_per_task.value ? parseInt(form.max_images_per_task.value, 10) : null,
    rows_per_part: form.rows_per_part.value ? parseInt(form.rows_per_part.value, 10) : null,
    parquet_row_group_size: form.parquet_row_group_size.value ? parseInt(form.parquet_row_group_size.value, 10) : null,
    slice_size: form.slice_size.value.trim(),
    log_file: form.log_file.checked,
  };

  dismissedWarnings.clear();

  const res = await fetch("/api/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const state = await res.json();
  renderState(state);
  startPolling();
});

// ---------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(pollStatus, POLL_INTERVAL_MS);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function pollStatus() {
  const res = await fetch("/api/status");
  const state = await res.json();
  renderState(state);
  if (state.status !== "running") {
    stopPolling();
  }
}

// ---------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------

function renderState(state) {
  const { status, progress, message, error, processed_files, total_files, output_parquet, warnings } = state;

  startBtn.disabled = status === "running";

  // Status text
  statusEl.className = "status-text";
  if (status === "idle") {
    statusEl.classList.add("status-muted");
    statusEl.textContent = "Ready to start processing";
  } else if (status === "running") {
    statusEl.classList.add("status-running");
    statusEl.innerHTML = `<strong>Processing...</strong><br>${escapeHtml(message)}`;
  } else if (status === "completed") {
    statusEl.classList.add("status-success");
    statusEl.innerHTML = `<strong>Processing completed!</strong><br>${escapeHtml(message)}`;
  } else if (status === "cancelled") {
    statusEl.classList.add("status-muted");
    statusEl.innerHTML = `<strong>Processing cancelled.</strong>`;
  } else if (status === "error") {
    statusEl.classList.add("status-danger");
    statusEl.innerHTML = `<strong>Error occurred</strong><br>${escapeHtml(error || "Unknown error")}`;
  }

  // Progress bar
  if (status === "running" || status === "completed") {
    progressContainer.hidden = false;
    progressBar.style.width = `${progress}%`;
    progressBar.classList.toggle("completed", status === "completed");
    progressLabel.textContent = `${progress.toFixed(1)}%`;
  } else {
    progressContainer.hidden = true;
  }

  // Details
  if (total_files > 0) {
    detailsEl.innerHTML = `<strong>Progress: ${processed_files}/${total_files} files</strong>`;
  } else if (status === "running" && processed_files > 0) {
    detailsEl.innerHTML = `<strong>Records processed: ${processed_files}</strong>`;
  } else {
    detailsEl.innerHTML = "";
  }

  // Error message
  if (error && status === "error") {
    errorEl.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(error)}</span></div>`;
  } else {
    errorEl.innerHTML = "";
  }

  // Warnings
  renderWarnings(warnings || []);

  // Action buttons
  if (status === "running") {
    actionButtonsEl.innerHTML = `<button type="button" id="cancel-btn" class="btn btn-secondary btn-lg btn-block">Cancel Processing</button>`;
    document.getElementById("cancel-btn").addEventListener("click", cancelProcessing);
  } else if (status === "completed" && output_parquet) {
    actionButtonsEl.innerHTML = `<button type="button" id="open-viewer-btn" class="btn btn-success btn-lg btn-block">Open in Viewer</button>`;
    document.getElementById("open-viewer-btn").addEventListener("click", () => openViewer(output_parquet));
  } else {
    actionButtonsEl.innerHTML = "";
  }
}

function renderWarnings(warnings) {
  const visible = warnings.filter((w) => !dismissedWarnings.has(warningKey(w)));
  warningsEl.innerHTML = "";
  for (const w of visible.slice(-10)) {
    const div = document.createElement("div");
    const color = w.level === "ERROR" ? "alert-danger" : "alert-warning";
    div.className = `alert ${color}`;
    div.innerHTML = `<span><strong>${escapeHtml(w.level)}:</strong> ${escapeHtml(w.message)}</span>`;
    const dismiss = document.createElement("button");
    dismiss.className = "alert-dismiss";
    dismiss.textContent = "×";
    dismiss.setAttribute("aria-label", "Dismiss");
    dismiss.addEventListener("click", () => {
      dismissedWarnings.add(warningKey(w));
      div.remove();
    });
    div.appendChild(dismiss);
    warningsEl.appendChild(div);
  }
}

// ---------------------------------------------------------------------
// Cancel processing
// ---------------------------------------------------------------------

async function cancelProcessing() {
  const btn = document.getElementById("cancel-btn");
  btn.disabled = true;
  btn.textContent = "Cancelling...";
  const res = await fetch("/api/cancel", { method: "POST" });
  const state = await res.json();
  renderState(state);
}

// ---------------------------------------------------------------------
// Open in viewer
// ---------------------------------------------------------------------

async function openViewer(outputParquet) {
  const btn = document.getElementById("open-viewer-btn");
  btn.disabled = true;
  btn.textContent = "Launching viewer...";
  try {
    const res = await fetch("/api/open-viewer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        output_parquet: outputParquet,
        group_by: form.group_by.value.trim(),
        filter_col: form.filter_col.value.trim(),
        filter_op: form.filter_op.value,
        filter_value: form.filter_value.value.trim(),
        dimensions: form.dimensions.value.trim(),
        widgets_exclude: form.widgets_exclude.value.trim(),
        is_show_significance: form.is_show_significance.checked,
        palette: form.palette.value.trim(),
      }),
    });
    const data = await res.json();
    if (data.url) {
      window.open(data.url, "_blank");
    } else {
      errorEl.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(data.error || "Failed to launch viewer")}</span></div>`;
    }
  } finally {
    btn.disabled = false;
    btn.textContent = "Open in Viewer";
  }
}

// ---------------------------------------------------------------------
// Open existing report
// ---------------------------------------------------------------------

const LAST_BROWSE_DIR_KEY = "pixelPatrolLastBrowseDir";
let selectedReportPath = null;
let currentParentDir = null;

function showOpenExistingDialog() {
  existingReportError.innerHTML = "";
  selectedReportPath = null;
  existingReportOpenBtn.disabled = true;
  openExistingOverlay.hidden = false;
  const startDir = localStorage.getItem(LAST_BROWSE_DIR_KEY) || "";
  loadBrowser(startDir);
}

function hideOpenExistingDialog() {
  openExistingOverlay.hidden = true;
}

async function loadBrowser(path) {
  existingReportError.innerHTML = "";
  try {
    const url = path ? `/api/browse?path=${encodeURIComponent(path)}` : "/api/browse";
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok || data.error) {
      existingReportError.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(data.error || "Failed to browse folder")}</span></div>`;
      return;
    }
    renderBrowser(data);
    localStorage.setItem(LAST_BROWSE_DIR_KEY, data.path);
  } catch (err) {
    existingReportError.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(String(err))}</span></div>`;
  }
}

function renderBrowser(data) {
  browserPathInput.value = data.path;
  browserUpBtn.disabled = !data.parent;
  currentParentDir = data.parent;

  selectedReportPath = null;
  existingReportOpenBtn.disabled = true;

  browserListEl.innerHTML = "";
  if (!data.entries.length) {
    browserListEl.innerHTML = `<div class="browser-empty">No subfolders or .parquet files here</div>`;
    return;
  }

  for (const entry of data.entries) {
    const row = document.createElement("div");
    row.className = "browser-entry";
    const icon = entry.is_dir ? "📁" : "📄";
    row.innerHTML = `<span class="browser-entry-icon">${icon}</span><span>${escapeHtml(entry.name)}</span>`;
    const fullPath = data.path.replace(/\/$/, "") + "/" + entry.name;
    if (entry.is_dir) {
      row.addEventListener("click", () => loadBrowser(fullPath));
    } else {
      row.addEventListener("click", () => {
        for (const el of browserListEl.querySelectorAll(".browser-entry.selected")) {
          el.classList.remove("selected");
        }
        row.classList.add("selected");
        selectedReportPath = fullPath;
        existingReportOpenBtn.disabled = false;
      });
    }
    browserListEl.appendChild(row);
  }
}

async function openExistingReport() {
  if (!selectedReportPath) return;
  existingReportOpenBtn.disabled = true;
  existingReportOpenBtn.textContent = "Opening...";
  try {
    const res = await fetch("/api/open-viewer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ output_parquet: selectedReportPath }),
    });
    const data = await res.json();
    if (data.url) {
      window.open(data.url, "_blank");
      hideOpenExistingDialog();
    } else {
      existingReportError.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(data.error || "Failed to open report")}</span></div>`;
    }
  } catch (err) {
    existingReportError.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(String(err))}</span></div>`;
  } finally {
    existingReportOpenBtn.disabled = false;
    existingReportOpenBtn.textContent = "Open";
  }
}

openExistingBtn.addEventListener("click", showOpenExistingDialog);
existingReportCancelBtn.addEventListener("click", hideOpenExistingDialog);
existingReportOpenBtn.addEventListener("click", openExistingReport);
browserUpBtn.addEventListener("click", () => {
  if (currentParentDir) loadBrowser(currentParentDir);
});
browserPathInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") loadBrowser(browserPathInput.value.trim());
});

// ---------------------------------------------------------------------
// Version check
// ---------------------------------------------------------------------

async function checkVersion() {
  try {
    const res = await fetch("/api/version");
    const data = await res.json();
    if (!data.update_available) return;

    if (data.managed) {
      versionInfoEl.innerHTML = `
        <span>Update available: pixel-patrol v${escapeHtml(data.latest)}</span>
        <button type="button" id="install-update-btn" class="btn btn-secondary btn-sm">Install Update</button>
      `;
      document.getElementById("install-update-btn").addEventListener("click", installUpdate);
    } else {
      versionInfoEl.innerHTML = `
        <a href="${data.pypi_url}" target="_blank">Update available: pixel-patrol v${escapeHtml(data.latest)} (PyPI)</a>
      `;
    }
    versionInfoEl.hidden = false;
  } catch (err) {
    // Offline or PyPI unreachable - silently skip the version check.
  }
}

async function installUpdate() {
  const btn = document.getElementById("install-update-btn");
  btn.disabled = true;
  btn.textContent = "Installing...";
  try {
    const res = await fetch("/api/update", { method: "POST" });
    const data = await res.json();
    if (res.ok) {
      versionInfoEl.innerHTML = `<span>Update installed — close this tab and reopen Pixel Patrol to use the new version.</span>`;
    } else {
      btn.disabled = false;
      btn.textContent = "Install Update";
      errorEl.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(data.error || "Update failed")}</span></div>`;
    }
  } catch (err) {
    btn.disabled = false;
    btn.textContent = "Install Update";
    errorEl.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(String(err))}</span></div>`;
  }
}

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------

(async function init() {
  await Promise.all([loadLoaders(), loadProcessors()]);
  checkVersion();
  const res = await fetch("/api/status");
  const state = await res.json();
  renderState(state);
  if (state.status === "running") {
    startPolling();
  }
})();
