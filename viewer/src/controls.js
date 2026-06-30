import { state, setState, resetState, emit } from './state.js';
import { BLOB_COLS } from './schema.js';
import { getPaletteNames } from './colors.js';
import { pluginGroup, orderedGroupNames } from './plugin-groups.js';
import { formatFrozenSidebarHtml } from './export-snapshot.js';

/**
 * Wire up all sidebar controls for a loaded schema.
 * State must already have been set (from URL params or defaults) before calling.
 * This function only syncs DOM to state - it does not overwrite state.
 *
 * @param {object}   schema
 * @param {number}   totalRows
 * @param {object[]} plugins  - all registered plugins (for widget toggles)
 * @param {Function} onExport  - called with (format, scope) where format ∈ {'csv','parquet'} and scope ∈ {'summary','full'}
 * @param {object}   [opts]
 * @param {boolean}  [opts.sidebarLocked]
 * @param {object}   [opts.frozenSidebar]  - payload from buildFrozenSidebarPayload
 * @param {Function} [opts.onExportBakedHtml] - baked static HTML snapshot
 */
let syncViewToggle = () => {};

/** Wire up controls that are independent of schema and only need to run once. */
export function initStaticUi() {
  const viewBtnOverview = el('view-btn-overview');
  const viewBtnFull     = el('view-btn-full');
  syncViewToggle = () => {
    viewBtnOverview?.setAttribute('aria-pressed', String(state.condensedMode));
    viewBtnFull?.setAttribute('aria-pressed',     String(!state.condensedMode));
  };
  if (viewBtnOverview && viewBtnFull) {
    syncViewToggle();
    viewBtnOverview.onclick = () => { if (state.condensedMode) return; state.condensedMode = true;  syncViewToggle(); emit('render'); };
    viewBtnFull.onclick     = () => { if (!state.condensedMode) return; state.condensedMode = false; syncViewToggle(); emit('render'); };
  }
  initCollapseToggle('appearance-section-header', 'appearance-section');
  initHeaderPopover('export-menu-btn', 'export-menu-panel');
  initHeaderPopover('feedback-menu-btn', 'feedback-menu-panel');
}

export function initControls(schema, totalRows, plugins, onExport, canParquet, opts = {}) {
  // ── Palette ──────────────────────────────────────────────────────────
  const paletteEl = el('palette-selector');
  paletteEl.innerHTML = getPaletteNames().map(p => opt(p, p)).join('');
  paletteEl.value = state.palette;
  paletteEl.onchange = () => {
    state.palette = paletteEl.value;
  };

  // ── Group By ─────────────────────────────────────────────────────────
  const groupEl = el('groupby-selector');
  groupEl.innerHTML =
    `<option value="">None</option>` +
    schema.groupCols.map(c => opt(c, c)).join('');
  groupEl.value = state.groupCol ?? '';

  // ── Dimension selectors ───────────────────────────────────────────────
  buildDimensionControls(schema.dimensionInfo, state.dimensions);

  // ── Filter column dropdown ────────────────────────────────────────────
  el('filter-column').innerHTML =
    `<option value="">Column…</option>` +
    schema.allCols.filter(c => !BLOB_COLS.has(c)).map(c => opt(c, c)).join('');

  // Sync filter DOM from state
  el('filter-column').value = state.filter.col;
  el('filter-op').value     = state.filter.op;
  el('filter-value').value  = state.filter.val;

  // Sync significance checkbox
  const sigCb = el('show-significance-cb');
  if (sigCb) sigCb.checked = state.showSignificance;

  // ── Widget toggles (live inside the collapsible Appearance section; Apply
  //    required to take effect) ─
  buildWidgetToggles(plugins, schema);

  // ── Apply button ──────────────────────────────────────────────────────
  el('apply-btn').onclick = () => {
    state.groupCol = groupEl.value || null;
    state.filter = {
      col: el('filter-column').value,
      op:  el('filter-op').value,
      val: el('filter-value').value.trim(),
    };
    state.dimensions       = readDimensions(schema.dimensionInfo);
    state.showSignificance = el('show-significance-cb')?.checked ?? false;
    applyWidgetToggles(plugins, schema);
    emit('query');
  };

  // ── Reset button ──────────────────────────────────────────────────────
  el('reset-btn').onclick = () => {
    el('filter-column').value = '';
    el('filter-op').value     = '';
    el('filter-value').value  = '';
    if (sigCb) sigCb.checked  = false;
    syncViewToggle();
    resetDimensions(schema.dimensionInfo);
    resetState(schema.defaultGroupCol);
    // Sync DOM after reset
    groupEl.value      = state.groupCol ?? '';
    paletteEl.value    = state.palette;
    buildWidgetToggles(plugins, schema);
  };

  buildExportControls(schema, onExport, !!canParquet);

  const bakedBtn = el('export-baked-btn');
  if (bakedBtn) {
    bakedBtn.onclick = opts.onExportBakedHtml ?? (() => {});
    bakedBtn.disabled = !opts.onExportBakedHtml;
  }

  // ── Row count display ─────────────────────────────────────────────────
  el('row-count-badge').textContent = `${totalRows.toLocaleString()} records`;

  if (opts.sidebarLocked && opts.frozenSidebar) {
    const banner = el('sidebar-frozen-banner');
    const bodyEl = el('sidebar-frozen-body');
    if (banner && bodyEl) {
      banner.classList.remove('d-none');
      bodyEl.innerHTML = formatFrozenSidebarHtml(opts.frozenSidebar);
    }

    paletteEl.disabled = true;
    groupEl.disabled   = true;
    el('filter-column').disabled = true;
    el('filter-op').disabled     = true;
    el('filter-value').disabled  = true;
    if (sigCb) sigCb.disabled = true;

    el('dimension-controls')?.querySelectorAll('select').forEach(sel => { sel.disabled = true; });

    el('apply-btn').style.display = 'none';
    el('reset-btn').style.display = 'none';

    el('widget-toggles')?.querySelectorAll('input[type="checkbox"]').forEach(cb => { cb.disabled = true; });

    if (bakedBtn) bakedBtn.style.display = 'none';
  }
}

/** Update the filtered row count shown in the header badge. */
export function updateFilteredInfo(filteredRows, totalRows) {
  const isFiltered = filteredRows !== totalRows;
  const summary = isFiltered
    ? `${filteredRows.toLocaleString()} / ${totalRows.toLocaleString()} records`
    : `${totalRows.toLocaleString()} records`;
  el('row-count-badge').textContent = summary;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

function el(id) { return document.getElementById(id); }
function opt(val, label) { return `<option value="${val}">${label}</option>`; }

function buildDimensionControls(dimensionInfo, activeDimensions = {}) {
  const container = el('dimension-controls');
  const entries   = Object.entries(dimensionInfo);

  if (!entries.length) {
    container.innerHTML = '<small class="text-muted">No dimensions detected.</small>';
    return;
  }

  container.innerHTML = entries.map(([dim, indices]) => `
    <div style="flex:1 1 60px;min-width:60px">
      <label class="form-label small mb-1 d-block text-center">${dim.toUpperCase()}</label>
      <select id="dim-sel-${dim}" class="form-select form-select-sm">
        <option value="">All</option>
        ${indices.map(i => `<option value="${i}">${i}</option>`).join('')}
      </select>
    </div>
  `).join('');

  // Sync select values from activeDimensions (URL params or state).
  for (const [dim] of entries) {
    const selEl = document.getElementById(`dim-sel-${dim}`);
    if (selEl) selEl.value = activeDimensions[dim] ?? '';
  }
}

function buildWidgetToggles(plugins, schema) {
  const container = el('widget-toggles');
  if (!container) return;

  const applicable = plugins.filter(p => {
    try { return p.requires(schema); } catch { return false; }
  });

  if (!applicable.length) {
    container.innerHTML = '<small class="text-muted">No widgets available.</small>';
    return;
  }

  const grouped = new Map();
  for (const p of applicable) {
    const grp = pluginGroup(p);
    if (!grouped.has(grp)) grouped.set(grp, []);
    grouped.get(grp).push(p);
  }
  const orderedGroups = orderedGroupNames(applicable);

  container.innerHTML = orderedGroups.map(g => {
    const rows = grouped.get(g).map(p => `
      <div class="form-check">
        <input class="form-check-input" type="checkbox" id="wt-${p.id}"
               ${state.hiddenWidgets.has(p.id) ? '' : 'checked'}>
        <label class="form-check-label small" for="wt-${p.id}">${p.label}</label>
      </div>
    `).join('');
    return `
      <div class="mt-2 mb-1 small text-uppercase text-muted fw-bold">${g}</div>
      ${rows}
    `;
  }).join('');

  // No onchange handler - changes are applied only when the Apply button is clicked.
}

/** Read current checkbox values into state.hiddenWidgets. */
function applyWidgetToggles(plugins, schema) {
  const applicable = plugins.filter(p => {
    try { return p.requires(schema); } catch { return false; }
  });
  for (const p of applicable) {
    const cb = document.getElementById(`wt-${p.id}`);
    if (!cb) continue;
    if (cb.checked) state.hiddenWidgets.delete(p.id);
    else            state.hiddenWidgets.add(p.id);
  }
}

/**
 * Wire a collapsible sidebar section. `header` must be a <button> with
 * aria-expanded; `content` is shown/hidden via the `hidden` attribute. The
 * chevron rotation is driven from aria-expanded in CSS. Keyboard activation
 * (Enter/Space) works for free because the header is a real button.
 */
function initCollapseToggle(headerId, contentId) {
  const header  = el(headerId);
  const content = el(contentId);
  if (!header || !content) return;
  header.addEventListener('click', () => {
    const open = header.getAttribute('aria-expanded') === 'true';
    header.setAttribute('aria-expanded', String(!open));
    content.hidden = open;
  });
}

// Registry of topbar popovers so opening one closes the others.
const headerPopovers = [];

/**
 * Wire a topbar popover (Export, Feedback): toggle on the trigger button,
 * close on outside-click or Escape, close sibling popovers when this one
 * opens, and keep aria-expanded in sync for screen readers.
 */
function initHeaderPopover(btnId, panelId) {
  const btn   = el(btnId);
  const panel = el(panelId);
  if (!btn || !panel) return;

  const setOpen = (open) => {
    panel.hidden = !open;
    btn.setAttribute('aria-expanded', String(open));
  };
  headerPopovers.push(setOpen);

  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const willOpen = panel.hidden;
    headerPopovers.forEach(close => close(false));
    setOpen(willOpen);
  });

  // Clicks inside the panel must not bubble up to the document closer.
  panel.addEventListener('click', (e) => e.stopPropagation());

  document.addEventListener('click', () => { if (!panel.hidden) setOpen(false); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !panel.hidden) { setOpen(false); btn.focus(); }
  });
}

function readDimensions(dimensionInfo) {
  const dims = {};
  for (const dim of Object.keys(dimensionInfo)) {
    const selEl = document.getElementById(`dim-sel-${dim}`);
    if (!selEl) continue;
    // Only include explicit selections; empty means "All".
    if (selEl.value !== '') dims[dim] = selEl.value;
  }
  return dims;
}

function resetDimensions(dimensionInfo) {
  for (const dim of Object.keys(dimensionInfo)) {
    const selEl = document.getElementById(`dim-sel-${dim}`);
    if (selEl) selEl.value = '';
  }
}

const EXPORT_HINTS = {
  'csv:summary':     'Only stats for full image - One row per image. Excludes thumbnail column.',
  'csv:full':        'Full table - including rows of dim slice stats. Excludes thumbnail column.',
  'parquet:summary': 'Only stats for full image - One row per image.',
  'parquet:full':    'Full table - including rows of dim slice stats.',
};

// One labelled button per export option (no dropdown), each with its own
// description line, so every choice is visible at a glance.
function buildExportControls(schema, onExport, canParquet) {
  const listEl = el('export-buttons');
  if (!listEl) return;

  const hasSlicing = (schema.dimCols ?? []).length > 0;

  const options = [];
  if (hasSlicing) {
    options.push({ value: 'csv:summary',     label: 'CSV – summary' });
    options.push({ value: 'csv:full',        label: 'CSV – full' });
    if (canParquet) {
      options.push({ value: 'parquet:summary', label: 'Parquet – summary' });
      options.push({ value: 'parquet:full',    label: 'Parquet – full' });
    }
  } else {
    options.push({ value: 'csv:summary',     label: 'CSV' });
    if (canParquet) {
      options.push({ value: 'parquet:summary', label: 'Parquet' });
    }
  }

  listEl.innerHTML = '';
  for (const o of options) {
    const wrap = document.createElement('div');

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'export-opt-btn';
    btn.innerHTML = `<i class="bi bi-download me-1"></i>${o.label}`;
    btn.onclick = () => {
      const [format, scope] = o.value.split(':');
      onExport(format, scope);
    };

    const desc = document.createElement('div');
    desc.className = 'export-opt-desc';
    desc.textContent = EXPORT_HINTS[o.value] ?? '';

    wrap.appendChild(btn);
    if (desc.textContent) wrap.appendChild(desc);
    listEl.appendChild(wrap);
  }
}
