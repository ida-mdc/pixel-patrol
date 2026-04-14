import { state, setState, resetState, emit } from './state.js';
import { BLOB_COLS } from './schema.js';
import { getPaletteNames } from './colors.js';

/**
 * Wire up all sidebar controls for a loaded schema.
 * State must already have been set (from URL params or defaults) before calling.
 * This function only syncs DOM to state — it does not overwrite state.
 *
 * @param {object}   schema
 * @param {number}   totalRows
 * @param {object[]} plugins  — all registered plugins (for widget toggles)
 * @param {Function} onExportCsv
 */
export function initControls(schema, totalRows, plugins, onExportCsv) {
  // ── Palette ──────────────────────────────────────────────────────────
  const paletteEl = el('palette-selector');
  paletteEl.innerHTML = getPaletteNames().map(p => opt(p, p)).join('');
  paletteEl.value = state.palette;
  paletteEl.onchange = () => {
    setState({ palette: paletteEl.value }, 'render');
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

  // ── Widget toggles ────────────────────────────────────────────────────
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
    emit('query');
  };

  // ── Reset button ──────────────────────────────────────────────────────
  el('reset-btn').onclick = () => {
    el('filter-column').value = '';
    el('filter-op').value     = '';
    el('filter-value').value  = '';
    if (sigCb) sigCb.checked  = false;
    resetDimensions(schema.dimensionInfo);
    resetState(schema.defaultGroupCol);
    // Sync DOM after reset
    groupEl.value      = state.groupCol ?? '';
    paletteEl.value    = state.palette;
    buildWidgetToggles(plugins, schema);
  };

  // ── Export CSV ────────────────────────────────────────────────────────
  el('export-csv-btn').onclick = onExportCsv;

  // ── Row count display ─────────────────────────────────────────────────
  el('row-count-badge').textContent = `${totalRows.toLocaleString()} records`;
}

/** Update the filtered row count shown in the header badge and sidebar. */
export function updateFilteredInfo(filteredRows, totalRows) {
  const isFiltered = filteredRows !== totalRows;
  const summary = isFiltered
    ? `${filteredRows.toLocaleString()} / ${totalRows.toLocaleString()} records`
    : `${totalRows.toLocaleString()} records`;
  el('filtered-info').textContent  = summary;
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
    <div class="mb-2">
      <label class="form-label small mb-1">${dim.toUpperCase()}</label>
      <select id="dim-sel-${dim}" class="form-select form-select-sm">
        <option value="">All</option>
        ${indices.map(i => `<option value="${i}">${dim}${i}</option>`).join('')}
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

  container.innerHTML = applicable.map(p => `
    <div class="form-check">
      <input class="form-check-input" type="checkbox" id="wt-${p.id}"
             ${state.hiddenWidgets.has(p.id) ? '' : 'checked'}>
      <label class="form-check-label small" for="wt-${p.id}">${p.label}</label>
    </div>
  `).join('');

  for (const p of applicable) {
    const cb = document.getElementById(`wt-${p.id}`);
    if (!cb) continue;
    cb.onchange = () => {
      if (cb.checked) state.hiddenWidgets.delete(p.id);
      else            state.hiddenWidgets.add(p.id);
      emit('render');
    };
  }
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
