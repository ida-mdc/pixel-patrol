"use strict";

const POLL_INTERVAL_MS = 500;

const form = document.getElementById("process-form");
const startBtn = document.getElementById("start-btn");

const loaderSelect = document.getElementById("loader");
const fileExtInput = document.getElementById("file-extensions");
const fileExtHelp = document.getElementById("file-extensions-help");
const processorsIncludeSelect = document.getElementById("processors-include");
const processorsExcludeSelect = document.getElementById("processors-exclude");

const reportsDirEl = document.getElementById("reports-dir");
const reportsListEl = document.getElementById("reports-list");
const addReportBtn = document.getElementById("add-report-btn");
const addCancelBtn = document.getElementById("add-cancel-btn");
const addReportOverlay = document.getElementById("add-report-overlay");
const refreshBtn = document.getElementById("refresh-btn");

const versionInfoEl = document.getElementById("version-info");
const importBtn = document.getElementById("import-existing-btn");
const importOverlay = document.getElementById("import-overlay");
const browserPathInput = document.getElementById("browser-path");
const browserUpBtn = document.getElementById("browser-up-btn");
const browserListEl = document.getElementById("browser-list");
const importError = document.getElementById("import-error");
const importCancelBtn = document.getElementById("import-cancel");
const importConfirmBtn = document.getElementById("import-confirm");

const statusBanner = document.getElementById("status-banner");
const statusEl = document.getElementById("status-text");
const progressContainer = document.getElementById("progress-bar-container");
const progressBar = document.getElementById("progress-bar");
const progressLabel = document.getElementById("progress-bar-label");
const detailsEl = document.getElementById("progress-details");
const errorEl = document.getElementById("error-message");
const warningsEl = document.getElementById("warnings-display");
const actionButtonsEl = document.getElementById("action-buttons");

let availableLoaders = [];
let pollTimer = null;
let lastCompletedReport = null;
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

function reportUrl(path) {
  return `/report?path=${encodeURIComponent(path)}`;
}

// ---------------------------------------------------------------------
// Reports list
// ---------------------------------------------------------------------

async function loadReports(refresh = false) {
  try {
    const res = await fetch(refresh ? "/api/reports?refresh=1" : "/api/reports");
    const data = await res.json();
    if (data.reports_dir) {
      reportsDirEl.textContent = `Reports directory: ${data.reports_dir}`;
    }
    const reports = data.reports || [];
    renderReports(reports);
    return reports.length;
  } catch (err) {
    reportsListEl.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(String(err))}</span></div>`;
    return null;
  }
}

function renderReports(reports) {
  reportsListEl.innerHTML = "";
  if (!reports.length) {
    reportsListEl.innerHTML =
      `<div class="reports-empty">No reports yet. Click <strong>+ New Report</strong> to process a folder of images.</div>`;
    return;
  }
  for (const r of reports) {
    reportsListEl.appendChild(renderReportCard(r));
  }
}

function formatDate(iso) {
  if (!iso) return { date: "", time: "" };
  const d = new Date(iso);
  if (isNaN(d)) return { date: iso.slice(0, 10), time: iso.slice(11, 16) };
  const pad = (n) => String(n).padStart(2, "0");
  return {
    date: `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`,
    time: `${pad(d.getHours())}:${pad(d.getMinutes())}`,
  };
}

function renderReportCard(r) {
  const card = document.createElement("div");
  card.className = "report-card";
  const imported = r.source === "imported";
  const missing = r.exists === false;
  if (missing) card.classList.add("report-missing");

  const { date, time } = formatDate(r.created_at);

  const thumb = r.thumbnail_b64
    ? `<img class="report-thumb" src="data:image/jpeg;base64,${r.thumbnail_b64}" alt="">`
    : `<div class="report-thumb report-thumb-empty"></div>`;

  const stats = [];
  if (r.n_files) stats.push(`${r.n_files.toLocaleString()} files`);
  if (r.total_size_bytes && r.size_readable) stats.push(escapeHtml(r.size_readable));
  for (const [ext, n] of Object.entries(r.file_type_counts || {}).slice(0, 4)) {
    stats.push(`.${escapeHtml(ext)} ×${n}`);
  }

  const title = r.project_name || r.filename;
  const badge = imported ? `<span class="report-badge">imported</span>` : "";
  // Show where the data came from (base directory); the report file path is on hover.
  const source = r.base_dir || r.path;
  const meta = missing
    ? `<div class="report-meta report-danger" title="${escapeHtml(r.path)}">File no longer found: ${escapeHtml(r.path)}</div>`
    : `<div class="report-meta" title="Report file: ${escapeHtml(r.path)}">${escapeHtml(source)}</div>` +
      (stats.length ? `<div class="report-meta">${stats.join("  ·  ")}</div>` : "");

  const openBtn = missing
    ? ""
    : `<a class="btn btn-primary report-open" href="${reportUrl(r.path)}" target="_blank" rel="noopener">Open</a>`;

  // Imported reports are only dropped from the list; internal ones are deleted
  // from disk - hence the different colour, label, and tooltip.
  const removeBtn = imported
    ? `<button type="button" class="btn btn-secondary report-delete" title="Remove from this list (the report file stays on disk)">Remove</button>`
    : `<button type="button" class="btn btn-danger report-delete" title="Delete the report file from disk">Delete</button>`;

  card.innerHTML = `
    <div class="report-thumb-wrap">${thumb}</div>
    <div class="report-main">
      <div class="report-name">${escapeHtml(title)}${badge}</div>
      ${meta}
    </div>
    <div class="report-date">${escapeHtml(date)} ${escapeHtml(time)}</div>
    <div class="report-actions">
      ${openBtn}
      ${removeBtn}
    </div>
  `;

  card.querySelector(".report-delete").addEventListener("click", () => deleteReport(r));
  return card;
}

async function deleteReport(r) {
  const imported = r.source === "imported";
  const prompt = imported
    ? `Remove "${r.filename}" from the list?\n\nThe file stays where it is:\n${r.path}`
    : `Delete report "${r.filename}" from disk?\n\n${r.path}`;
  if (!confirm(prompt)) return;
  try {
    const res = await fetch("/api/delete-report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: r.path }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      alert(data.error || "Could not remove report.");
      return;
    }
    loadReports();
  } catch (err) {
    alert(String(err));
  }
}

// ---------------------------------------------------------------------
// Add Report modal
// ---------------------------------------------------------------------

function showAddReport() {
  addReportOverlay.hidden = false;
}

function hideAddReport() {
  addReportOverlay.hidden = true;
}

addReportBtn.addEventListener("click", showAddReport);
addCancelBtn.addEventListener("click", hideAddReport);
refreshBtn.addEventListener("click", () => loadReports(true));

// ---------------------------------------------------------------------
// Loaders / processors
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
// Form submission
// ---------------------------------------------------------------------

function selectedValues(select) {
  return Array.from(select.selectedOptions).map((o) => o.value);
}

function viewerDefaults() {
  return {
    group_by: form.group_by.value.trim(),
    filter_col: form.filter_col.value.trim(),
    filter_op: form.filter_op.value,
    filter_value: form.filter_value.value.trim(),
    dimensions: form.dimensions.value.trim(),
    widgets_exclude: form.widgets_exclude.value.trim(),
    is_show_significance: form.is_show_significance.checked,
    palette: form.palette.value.trim(),
  };
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
  lastCompletedReport = null;

  const res = await fetch("/api/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const state = await res.json();
  hideAddReport();
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
    if (state.status === "completed") loadReports();
  }
}

// ---------------------------------------------------------------------
// Rendering the processing banner
// ---------------------------------------------------------------------

function renderState(state) {
  const { status, progress, message, error, processed_files, total_files, output_parquet, warnings } = state;

  startBtn.disabled = status === "running";

  if (status === "idle") {
    statusBanner.hidden = true;
    return;
  }
  statusBanner.hidden = false;

  statusEl.className = "status-text";
  if (status === "running") {
    statusEl.classList.add("status-running");
    statusEl.innerHTML = `<strong>Processing…</strong> ${escapeHtml(message)}`;
  } else if (status === "completed") {
    statusEl.classList.add("status-success");
    statusEl.innerHTML = `<strong>Processing completed.</strong> ${escapeHtml(message)}`;
  } else if (status === "cancelled") {
    statusEl.classList.add("status-muted");
    statusEl.innerHTML = `<strong>Processing cancelled.</strong>`;
  } else if (status === "error") {
    statusEl.classList.add("status-danger");
    statusEl.innerHTML = `<strong>Error.</strong> ${escapeHtml(error || "Unknown error")}`;
  }

  // Only show the progress bar while running; once finished the status text
  // and action buttons convey the result.
  if (status === "running") {
    progressContainer.hidden = false;
    progressBar.style.width = `${progress}%`;
    progressLabel.textContent = `${progress.toFixed(0)}%`;
  } else {
    progressContainer.hidden = true;
  }

  if (total_files > 0) {
    detailsEl.innerHTML = `<strong>Progress: ${processed_files}/${total_files} files</strong>`;
  } else if (status === "running" && processed_files > 0) {
    detailsEl.innerHTML = `<strong>Records processed: ${processed_files}</strong>`;
  } else {
    detailsEl.innerHTML = "";
  }

  errorEl.innerHTML =
    error && status === "error"
      ? `<div class="alert alert-danger"><span>${escapeHtml(error)}</span></div>`
      : "";

  renderWarnings(warnings || []);

  if (output_parquet) lastCompletedReport = output_parquet;

  if (status === "running") {
    actionButtonsEl.innerHTML = `<button type="button" id="cancel-btn" class="btn btn-secondary">Cancel Processing</button>`;
    document.getElementById("cancel-btn").addEventListener("click", cancelProcessing);
  } else if (status === "completed" && lastCompletedReport) {
    actionButtonsEl.innerHTML =
      `<button type="button" id="open-report-btn" class="btn btn-success">Open Report</button>
       <button type="button" id="dismiss-banner-btn" class="btn btn-secondary">Dismiss</button>`;
    document.getElementById("open-report-btn").addEventListener("click", () => openProcessedReport(lastCompletedReport));
    document.getElementById("dismiss-banner-btn").addEventListener("click", () => { statusBanner.hidden = true; });
  } else {
    actionButtonsEl.innerHTML =
      `<button type="button" id="dismiss-banner-btn" class="btn btn-secondary">Dismiss</button>`;
    document.getElementById("dismiss-banner-btn").addEventListener("click", () => { statusBanner.hidden = true; });
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

async function cancelProcessing() {
  const btn = document.getElementById("cancel-btn");
  btn.disabled = true;
  btn.textContent = "Cancelling...";
  const res = await fetch("/api/cancel", { method: "POST" });
  const state = await res.json();
  renderState(state);
}

// Open a freshly processed report, applying the form's viewer defaults.
async function openProcessedReport(outputParquet) {
  const btn = document.getElementById("open-report-btn");
  btn.disabled = true;
  btn.textContent = "Opening…";
  try {
    const res = await fetch("/api/report-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ output_parquet: outputParquet, ...viewerDefaults() }),
    });
    const data = await res.json();
    if (data.url) {
      window.open(data.url, "_blank", "noopener");
      btn.disabled = false;
      btn.textContent = "Open Report";
    } else {
      errorEl.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(data.error || "Failed to open report")}</span></div>`;
      btn.disabled = false;
      btn.textContent = "Open Report";
    }
  } catch (err) {
    errorEl.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(String(err))}</span></div>`;
    btn.disabled = false;
    btn.textContent = "Open Report";
  }
}

// ---------------------------------------------------------------------
// Import existing report (directory browser → add to the index)
// ---------------------------------------------------------------------

const LAST_BROWSE_DIR_KEY = "pixelPatrolLastBrowseDir";
let selectedReportPath = null;
let currentParentDir = null;

function showImportDialog() {
  importError.innerHTML = "";
  selectedReportPath = null;
  importConfirmBtn.disabled = true;
  importOverlay.hidden = false;
  const startDir = localStorage.getItem(LAST_BROWSE_DIR_KEY) || "";
  loadBrowser(startDir);
}

function hideImportDialog() {
  importOverlay.hidden = true;
}

async function loadBrowser(path) {
  importError.innerHTML = "";
  try {
    const url = path ? `/api/browse?path=${encodeURIComponent(path)}` : "/api/browse";
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok || data.error) {
      importError.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(data.error || "Failed to browse folder")}</span></div>`;
      return;
    }
    renderBrowser(data);
    localStorage.setItem(LAST_BROWSE_DIR_KEY, data.path);
  } catch (err) {
    importError.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(String(err))}</span></div>`;
  }
}

function renderBrowser(data) {
  browserPathInput.value = data.path;
  browserUpBtn.disabled = !data.parent;
  currentParentDir = data.parent;

  selectedReportPath = null;
  importConfirmBtn.disabled = true;

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
        importConfirmBtn.disabled = false;
      });
    }
    browserListEl.appendChild(row);
  }
}

async function importSelectedReport() {
  if (!selectedReportPath) return;
  importConfirmBtn.disabled = true;
  importConfirmBtn.textContent = "Importing…";
  try {
    const res = await fetch("/api/import-report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: selectedReportPath }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      importError.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(data.error || "Failed to import report")}</span></div>`;
      return;
    }
    hideImportDialog();
    loadReports();
  } catch (err) {
    importError.innerHTML = `<div class="alert alert-danger"><span>${escapeHtml(String(err))}</span></div>`;
  } finally {
    importConfirmBtn.disabled = false;
    importConfirmBtn.textContent = "Import";
  }
}

importBtn.addEventListener("click", showImportDialog);
importCancelBtn.addEventListener("click", hideImportDialog);
importConfirmBtn.addEventListener("click", importSelectedReport);
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
      versionInfoEl.innerHTML = `<span>Update installed — close this tab and reopen PixelPatrol to use the new version.</span>`;
    } else {
      btn.disabled = false;
      btn.textContent = "Install Update";
      alert(data.error || "Update failed");
    }
  } catch (err) {
    btn.disabled = false;
    btn.textContent = "Install Update";
    alert(String(err));
  }
}

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------

(async function init() {
  await Promise.all([loadLoaders(), loadProcessors()]);
  const reportCount = await loadReports();
  checkVersion();
  const res = await fetch("/api/status");
  const state = await res.json();
  renderState(state);
  if (state.status === "running") {
    startPolling();
  } else if (reportCount === 0) {
    // Nothing to look at yet - guide first-time users straight to processing.
    showAddReport();
  }
})();
