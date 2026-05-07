import './style.css';
import { initDuckDB, loadFromUrl, loadFromFile, finishLoad } from './loader.js';
import { SERVER_MODE, makeServerConn } from './query.js';
import { initControls } from './controls.js';
import { renderAll } from './renderer.js';
import { registry } from './plugin-registry.js';
import { state, on } from './state.js';
import { buildWhere } from './sql.js';
import { getPaletteNames } from './colors.js';
import { readUrlParams, writeUrlParams } from './url-params.js';
import { exportBakedHtml } from './export-snapshot.js';
import { ID_WELCOME_SCREEN, ID_MAIN_APP, ID_LOADING_OVERLAY, ID_SIDEBAR_BACKDROP } from './constants.js';

// ── Module-level handles ──────────────────────────────────────────────────────
let db          = null;
let conn        = null;
let schema      = null;
let totalRows   = 0;
let projectName  = null;
let description  = null;

/** Serialize render passes so concurrent triggers cannot interleave (duplicate widgets / stale cards). */
let renderQueue = Promise.resolve();

// ── Bootstrap ─────────────────────────────────────────────────────────────────

/**
 * Consume window.__PP_PLUGINS (inline objects) and window.__PP_PLUGIN_URLS
 * (ES module URLs) declared by the host page before the viewer boots.
 *
 * Call once during boot, before the first render, so every declared plugin
 * is already in the registry when afterLoad() runs.
 */
async function loadExternalPlugins() {
  const params = new URLSearchParams(window.location.search);
  let bundledExtUrls = [];
  try {
    const res = await fetch('./pp_extension_urls.json', { cache: 'no-store' });
    if (res.ok) {
      const json = await res.json();
      if (Array.isArray(json)) bundledExtUrls = json.filter(Boolean);
    }
  } catch {
    // Optional for older deployments that do not ship this file.
  }

  // ?extension=<url> or window.__PP_EXTENSION_URLS (injected by local server)
  // Manifest format: { "plugins": ["./a.js", "./b.js"] }
  // Relative URLs are resolved against the manifest's own URL.
  const pageExtUrls   = Array.isArray(window.__PP_EXTENSION_URLS) ? window.__PP_EXTENSION_URLS : [];
  const extensionUrls = [...bundledExtUrls, ...pageExtUrls, ...params.getAll('extension').filter(Boolean)];
  const manifests = (
    await Promise.all(extensionUrls.map(async extUrl => {
      try {
        const res      = await fetch(extUrl);
        const manifest = await res.json();
        const base     = new URL(extUrl, window.location.href);
        const packageKey = String(manifest.package_name ?? manifest.name ?? extUrl).toLowerCase();
        const pluginUrls = (manifest.plugins ?? []).map(p => new URL(p, base).href);
        return { packageKey, pluginUrls };
      } catch (err) {
        console.warn(`[viewer] failed to load extension manifest "${extUrl}":`, err);
        return { packageKey: extUrl.toLowerCase(), pluginUrls: [] };
      }
    }))
  ).sort((a, b) => a.packageKey.localeCompare(b.packageKey));

  // Deterministic ordering:
  // 1) package alphabetical (manifest package_name/name), then
  // 2) plugin order from each package's extension manifest.
  for (const { pluginUrls } of manifests) {
    for (const url of pluginUrls) {
      try {
        await registry.loadFromUrl(url);
      } catch (err) {
        console.warn(`[viewer] failed to load plugin from "${url}":`, err);
      }
    }
  }
}

async function boot() {
  // Wire state → render once, at startup. afterLoad() must not re-register these.
  on('query',  doRender);
  on('render', doRender);
  registry.onAdd(doRender);

  // ── Server mode: native DuckDB via local Python server ──────────────────────
  if (SERVER_MODE) {
    await loadExternalPlugins();
    setLoading('Loading report data…');
    conn = makeServerConn();
    try {
      ({ schema, totalRows, projectName, description } = await finishLoad(conn));
      projectName  ??= window.__PP_PROJECT_NAME  ?? null;
      description  ??= window.__PP_DESCRIPTION   ?? null;

      hideFileOpenControls();
      document.getElementById('current-filename').textContent =
        window.__PP_FILENAME ?? 'data.parquet';
      hideLoading();
      afterLoad();
    } catch (err) {
      showFatalError('Failed to connect to local server', err);
    }
    return;
  }

  // ── Unpacked offline ZIP (viewer_dist layout + snapshot.parquet + __PP_SNAPSHOT_BUNDLE)
  if (window.__PP_SNAPSHOT_BUNDLE) {
    await loadExternalPlugins();
    await bootWasmOfflineSnapshot();
    attachWasmFileUi();
    return;
  }

  await loadExternalPlugins();

  // ── WASM mode: DuckDB WASM in the browser ────────────────────────────────────
  setLoading('Initialising DuckDB…');

  try {
    ({ db, conn } = await initDuckDB());
  } catch (err) {
    showFatalError('Failed to initialise DuckDB WASM', err);
    return;
  }

  hideLoading();

  // Check for ?data=<url> query parameter.
  const dataUrl = new URLSearchParams(window.location.search).get('data');
  if (dataUrl) {
    await openUrl(dataUrl);
  } else {
    showWelcome();
  }

  attachWasmFileUi();
}

function attachWasmFileUi() {
  document.getElementById('topbar-brand').addEventListener('click', () => {
    if (state.sidebarLocked) return;
    showWelcome();
  });

  document.getElementById('file-input-welcome').addEventListener('change', e => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = '';
    if (files.length) openFiles(files);
  });
  document.getElementById('file-input-top').addEventListener('change', e => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = '';
    if (files.length) openFiles(files);
  });
}

async function bootWasmOfflineSnapshot() {
  setLoading('Initialising DuckDB…');
  try {
    ({ db, conn } = await initDuckDB());
  } catch (err) {
    showFatalError('Failed to initialise DuckDB WASM', err);
    return;
  }

  const bundle = window.__PP_SNAPSHOT_BUNDLE;
  const snapUrl = new URL(bundle.parquetFile, window.location.href).href;

  setLoading('Loading snapshot…');
  try {
    ({ schema, totalRows, projectName, description } = await loadFromUrl(db, conn, snapUrl, null));
  } catch (err) {
    showFatalError('Failed to load snapshot parquet', err, loadErrorHint(err));
    return;
  }

  document.getElementById('current-filename').textContent = bundle.parquetFile;
  hideFileOpenControls();
  hideLoading();
  afterLoad({ snapshotBundle: bundle });
}

// ── Open data ─────────────────────────────────────────────────────────────────

async function openUrl(url) {
  const filename = url.split('/').pop();
  setLoading(`Downloading ${filename}…`);
  try {
    ({ schema, totalRows, projectName, description } = await loadFromUrl(db, conn, url, (loaded, total) => {
      if (total > 0) {
        const pct = Math.round(loaded / total * 100);
        setLoadingProgress(pct, `Downloading ${filename}… ${pct}%`);
      }
    }));
    document.getElementById('current-filename').textContent = filename;
    hideLoading();
    afterLoad();
  } catch (err) {
    showFatalError(`Failed to load ${filename}`, err, loadErrorHint(err));
  }
}

async function openFiles(files) {
  const file = files[0];
  setLoading(`Loading ${file.name}…`);
  try {
    ({ schema, totalRows, projectName, description } = await loadFromFile(db, conn, file));
    document.getElementById('current-filename').textContent = file.name;
    hideLoading();
    afterLoad();
  } catch (err) {
    showFatalError(`Failed to load ${file.name}`, err, loadErrorHint(err));
  }
}

/**
 * Return a user-friendly hint when a parquet load error looks like a format
 * incompatibility (old report generated with an earlier Pixel Patrol version).
 */
function loadErrorHint(err) {
  const msg = String(err?.message ?? err).toLowerCase();
  const isFormatError = /tprotocolexception|invalid data|parquetexception|not a parquet|magic number|invalid parquet|thrift|footer|corrupt/i.test(msg);
  if (isFormatError) {
    return 'This file may have been created with an older version of Pixel Patrol. '
      + 'Please re-run <code>pixel-patrol process</code> to regenerate the report.';
  }
  return null;
}

// ── After data is loaded ───────────────────────────────────────────────────────

function applySnapshotBundle(bundle) {
  const rs = bundle.renderState ?? {};
  state.palette          = rs.palette ?? state.palette;
  state.groupCol         = rs.groupCol !== undefined ? rs.groupCol : state.groupCol;
  state.filter           = { col: '', op: '', val: '' };
  state.dimensions       = {};
  state.showSignificance = !!rs.showSignificance;
  state.hiddenWidgets    = new Set(Array.isArray(rs.hiddenWidgets) ? rs.hiddenWidgets : []);
  state.sidebarLocked    = true;
}

function afterLoad(options = {}) {
  const snapshotBundle = options.snapshotBundle;

  // 1. Set schema defaults into state.
  state.dimensions      = {};
  state.groupCol        = schema.defaultGroupCol ?? null;
  state.hiddenWidgets   = new Set();
  state.sidebarLocked   = false;
  state.filter          = { col: '', op: '', val: '' };

  if (snapshotBundle) {
    applySnapshotBundle(snapshotBundle);
  }

  // 2. Overlay URL params where valid (skipped for offline snapshots).
  const urlParams = snapshotBundle ? {} : readUrlParams();
  if (urlParams.palette && getPaletteNames().includes(urlParams.palette)) {
    state.palette = urlParams.palette;
  }
  if ('groupCol' in urlParams) {
    const gc = urlParams.groupCol;
    if (!gc || schema.groupCols.includes(gc)) state.groupCol = gc;
  }
  if (urlParams.filter)           state.filter           = urlParams.filter;
  if (urlParams.dimensions)       state.dimensions       = urlParams.dimensions;
  if ('showSignificance' in urlParams) state.showSignificance = urlParams.showSignificance;
  if (urlParams.hiddenWidgets)    state.hiddenWidgets    = urlParams.hiddenWidgets;

  // 3. Init controls (syncs DOM from state).
  initControls(schema, totalRows, registry.plugins, handleExportCsv,
    SERVER_MODE ? handleExportParquet : null, {
    sidebarLocked: state.sidebarLocked,
    frozenSidebar: snapshotBundle?.frozenSidebar,
    onExportBakedHtml: state.sidebarLocked ? undefined : handleExportBakedHtml,
  });

  const nameEl = document.getElementById('project-name');
  if (nameEl) {
    nameEl.textContent = '';
    if (projectName) {
      const title = document.createElement('span');
      title.className   = 'project-title';
      title.textContent = projectName;
      nameEl.appendChild(title);
    }
    if (description) {
      const desc = document.createElement('span');
      desc.className   = 'project-description';
      desc.textContent = description;
      nameEl.appendChild(desc);
    }
  }

  showApp();
  doRender();
}

function doRender() {
  renderQueue = renderQueue.catch(() => {}).then(async () => {
    if (!conn || !schema) return;
    try {
      await renderAll(registry.plugins, conn, schema, state, totalRows);
      if (!state.sidebarLocked) writeUrlParams(state);
    } catch (err) {
      console.error('[viewer] renderAll error:', err);
    }
  });
}

// ── Export CSV ────────────────────────────────────────────────────────────────

async function handleExportBakedHtml() {
  if (state.sidebarLocked) return;
  try {
    setLoading('Building HTML snapshot…');
    const html = await exportBakedHtml(state, schema, registry.plugins);
    const blob = new Blob([html], { type: 'text/html' });
    const raw = document.getElementById('current-filename')?.textContent ?? 'report';
    const base = raw.replace(/\.[^/.]+$/, '').replace(/[^\w.-]+/g, '_').trim().slice(0, 80) || 'report';
    triggerDownload(blob, `${base}-snapshot.html`);
  } catch (err) {
    console.error('[viewer] baked HTML export error:', err);
    alert(`HTML snapshot failed: ${err?.message ?? err}`);
  } finally {
    hideLoading();
  }
}

async function handleExportCsv() {
  try {
    // COPY … TO is not available in WASM; use JSON serialisation instead.
    const rows  = await conn.query(`SELECT * FROM pp_data ${buildWhere(state.filter)} LIMIT 100000`);
    const json  = rows.toArray().map(r => r.toJSON());
    const cols  = Object.keys(json[0] ?? {});
    const lines = [
      cols.join(','),
      ...json.map(r => cols.map(c => csvCell(r[c])).join(',')),
    ];
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    triggerDownload(blob, 'pixel_patrol_export.csv');
  } catch (err) {
    console.error('[viewer] CSV export error:', err);
  }
}

async function handleExportParquet() {
  try {
    const where = buildWhere(state.filter);
    const qs    = where ? `?where=${encodeURIComponent(where)}` : '';
    const resp  = await fetch(`/api/export-parquet${qs}`);
    if (!resp.ok) throw new Error(await resp.text());
    const blob  = await resp.blob();
    triggerDownload(blob, document.getElementById('current-filename').textContent.replace('.parquet', '_filtered.parquet'));
  } catch (err) {
    console.error('[viewer] Parquet export error:', err);
  }
}

function csvCell(v) {
  if (v == null) return '';
  const s = String(v);
  return s.includes(',') || s.includes('"') || s.includes('\n')
    ? `"${s.replace(/"/g, '""')}"`
    : s;
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a   = Object.assign(document.createElement('a'), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ── UI state helpers ───────────────────────────────────────────────────────────

function setLoading(msg) {
  document.getElementById('loading-text').textContent = msg;
  document.getElementById(ID_LOADING_OVERLAY).style.display = 'flex';
}

function setLoadingProgress(pct, msg) {
  const bar = document.getElementById('loading-progress');
  if (bar) { bar.style.display = 'block'; bar.value = pct; }
  if (msg) document.getElementById('loading-text').textContent = msg;
}

function hideLoading() {
  const bar = document.getElementById('loading-progress');
  if (bar) { bar.style.display = 'none'; bar.value = 0; }
  document.getElementById(ID_LOADING_OVERLAY).style.display = 'none';
}

function showWelcome() {
  document.getElementById(ID_WELCOME_SCREEN).style.display = 'flex';
  document.getElementById(ID_MAIN_APP).style.display       = 'none';
}

function showApp() {
  document.getElementById(ID_WELCOME_SCREEN).style.display = 'none';
  document.getElementById(ID_MAIN_APP).style.display       = 'flex';
}

function hideFileOpenControls() {
  document.querySelector('label[for="file-input-welcome"]')?.style.setProperty('display', 'none');
  document.querySelector('label[for="file-input-top"]')?.style.setProperty('display', 'none');
  document.getElementById('open-file-top-btn')?.style.setProperty('display', 'none');
}

function showFatalError(message, err, hint = null) {
  hideLoading();
  showWelcome();

  // Show a dismissible error banner above the welcome content.
  const existing = document.getElementById('error-banner');
  if (existing) existing.remove();

  const banner = document.createElement('div');
  banner.id = 'error-banner';
  banner.className = 'alert alert-danger alert-dismissible mx-auto mt-3';
  banner.style.cssText = 'max-width:680px;position:absolute;top:12px;left:0;right:0;z-index:200';
  banner.innerHTML = `
    <button type="button" class="btn-close" onclick="this.closest('#error-banner').remove()"></button>
    <strong>${message}</strong>
    ${hint ? `<p class="mb-1 mt-2">${hint}</p>` : ''}
    <details class="mt-2">
      <summary class="small text-muted" style="cursor:pointer">Technical details</summary>
      <pre class="mt-1 mb-0 small" style="white-space:pre-wrap">${err?.message ?? err}</pre>
    </details>
  `;

  // Insert into the welcome screen's flex container without destroying it.
  const ws = document.getElementById(ID_WELCOME_SCREEN);
  ws.style.position = 'relative';
  ws.appendChild(banner);
}

// ── Sidebar toggle (mobile) ───────────────────────────────────────────────────

(function initSidebarToggle() {
  const toggle   = document.getElementById('sidebar-toggle');
  const sidebar  = document.getElementById('sidebar');
  const backdrop = document.getElementById(ID_SIDEBAR_BACKDROP);

  function setSidebarOpen(open) {
    sidebar.classList.toggle('sidebar-open', open);
    backdrop.classList.toggle('sidebar-open', open);
  }

  toggle?.addEventListener('click',   () => setSidebarOpen(!sidebar.classList.contains('sidebar-open')));
  backdrop?.addEventListener('click', () => setSidebarOpen(false));

  // Close sidebar when a filter/apply button is tapped on mobile.
  document.getElementById('apply-btn')?.addEventListener('click', () => {
    if (window.innerWidth < 768) setSidebarOpen(false);
  });
})();

// ── Go ─────────────────────────────────────────────────────────────────────────
boot();
