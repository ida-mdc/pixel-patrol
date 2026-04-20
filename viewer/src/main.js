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

// ── Module-level handles ──────────────────────────────────────────────────────
let db          = null;
let conn        = null;
let schema      = null;
let totalRows   = 0;
let projectName = null;
let authors     = null;

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

  // ?extension=<url> or window.__PP_EXTENSION_URLS (injected by local server)
  // Manifest format: { "plugins": ["./a.js", "./b.js"] }
  // Relative URLs are resolved against the manifest's own URL.
  const pageExtUrls   = Array.isArray(window.__PP_EXTENSION_URLS) ? window.__PP_EXTENSION_URLS : [];
  const extensionUrls = [...pageExtUrls, ...params.getAll('extension').filter(Boolean)];
  const urls = (
    await Promise.all(extensionUrls.map(async extUrl => {
      try {
        const res      = await fetch(extUrl);
        const manifest = await res.json();
        const base     = new URL(extUrl, window.location.href);
        return (manifest.plugins ?? []).map(p => new URL(p, base).href);
      } catch (err) {
        console.warn(`[viewer] failed to load extension manifest "${extUrl}":`, err);
        return [];
      }
    }))
  ).flat();

  await Promise.all(
    urls.map(url =>
      registry.loadFromUrl(url).catch(err =>
        console.warn(`[viewer] failed to load plugin from "${url}":`, err),
      ),
    ),
  );
}

async function boot() {
  // Wire state → render once, at startup. afterLoad() must not re-register these.
  on('query',  doRender);
  on('render', doRender);
  registry.onAdd(doRender);

  await loadExternalPlugins();

  // ── Server mode: native DuckDB via local Python server ──────────────────────
  if (SERVER_MODE) {
    setLoading('Connecting to local server…');
    conn = makeServerConn();
    try {
      ({ schema, totalRows, projectName, authors } = await finishLoad(conn));
      projectName ??= window.__PP_PROJECT_NAME ?? null;
      authors     ??= window.__PP_AUTHORS      ?? null;

      document.getElementById('current-filename').textContent =
        window.__PP_FILENAME ?? 'data.parquet';
      hideLoading();
      afterLoad();
    } catch (err) {
      showFatalError('Failed to connect to local server', err);
    }
    return;
  }

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

  // Clicking the brand name returns to the welcome screen.
  document.getElementById('topbar-brand').addEventListener('click', () => {
    showWelcome();
  });

  // File input handlers (top bar + welcome screen).
  document.getElementById('file-input-welcome').addEventListener('change', e => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = '';          // reset so the same file can be re-opened
    if (files.length) openFiles(files);
  });
  document.getElementById('file-input-top').addEventListener('change', e => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = '';          // reset so the same file can be re-opened
    if (files.length) openFiles(files);
  });
}

// ── Open data ─────────────────────────────────────────────────────────────────

async function openUrl(url) {
  const filename = url.split('/').pop();
  setLoading(`Downloading ${filename}…`);
  try {
    ({ schema, totalRows, projectName, authors } = await loadFromUrl(db, conn, url, (loaded, total) => {
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
    ({ schema, totalRows, projectName, authors } = await loadFromFile(db, conn, file));
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

function afterLoad() {
  // 1. Set schema defaults into state.
  state.dimensions   = {};
  state.groupCol     = schema.defaultGroupCol ?? null;
  state.hiddenWidgets = new Set();

  // 2. Overlay URL params where valid.
  const urlParams = readUrlParams();
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
  initControls(schema, totalRows, registry.plugins, handleExportCsv);

  const nameEl = document.getElementById('project-name');
  if (nameEl) {
    nameEl.textContent = '';
    if (projectName) {
      const title = document.createElement('span');
      title.className   = 'project-title';
      title.textContent = projectName;
      nameEl.appendChild(title);
    }
    if (authors) {
      const auth = document.createElement('span');
      auth.className   = 'project-authors';
      auth.textContent = authors;
      nameEl.appendChild(auth);
    }
  }

  showApp();
  doRender();
}

async function doRender() {
  try {
    await renderAll(registry.plugins, conn, schema, state, totalRows);
    writeUrlParams(state);
  } catch (err) {
    console.error('[viewer] renderAll error:', err);
  }
}

// ── Export CSV ────────────────────────────────────────────────────────────────

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
  document.getElementById('loading-overlay').style.display = 'flex';
}

function setLoadingProgress(pct, msg) {
  const bar = document.getElementById('loading-progress');
  if (bar) { bar.style.display = 'block'; bar.value = pct; }
  if (msg) document.getElementById('loading-text').textContent = msg;
}

function hideLoading() {
  const bar = document.getElementById('loading-progress');
  if (bar) { bar.style.display = 'none'; bar.value = 0; }
  document.getElementById('loading-overlay').style.display = 'none';
}

function showWelcome() {
  document.getElementById('welcome-screen').style.display = 'flex';
  document.getElementById('main-app').style.display       = 'none';
}

function showApp() {
  document.getElementById('welcome-screen').style.display = 'none';
  document.getElementById('main-app').style.display       = 'flex';
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
  const ws = document.getElementById('welcome-screen');
  ws.style.position = 'relative';
  ws.appendChild(banner);
}

// ── Sidebar toggle (mobile) ───────────────────────────────────────────────────

(function initSidebarToggle() {
  const toggle   = document.getElementById('sidebar-toggle');
  const sidebar  = document.getElementById('sidebar');
  const backdrop = document.getElementById('sidebar-backdrop');

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
