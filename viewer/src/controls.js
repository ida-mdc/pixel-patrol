import { state, setState, resetState, emit } from './state.js';
import { BLOB_COLS } from './schema.js';
import { getPaletteNames } from './colors.js';
import { pluginGroup, orderedGroupNames } from './plugin-groups.js';
import { formatFrozenSidebarHtml } from './export-snapshot.js';

/**
 * Wire up all sidebar controls for a loaded schema.
 * State must already have been set (from URL params or defaults) before calling.
 * This function only syncs DOM to state — it does not overwrite state.
 *
 * @param {object}   schema
 * @param {number}   totalRows
 * @param {object[]} plugins  — all registered plugins (for widget toggles)
 * @param {Function} onExport  — called with (format, scope) where format ∈ {'csv','parquet'} and scope ∈ {'summary','full'}
 * @param {object}   [opts]
 * @param {boolean}  [opts.sidebarLocked]
 * @param {object}   [opts.frozenSidebar]  — payload from buildFrozenSidebarPayload
 * @param {Function} [opts.onExportBakedHtml] — baked static HTML snapshot
 */
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

  // ── Export dropdown ───────────────────────────────────────────────────
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

const EXPORT_HINTS = {
  'csv:summary':     'Only stats for full image - One row per image. Excludes thumbnail column.',
  'csv:full':        'Full table - including rows of dim slice stats. Excludes thumbnail column.',
  'parquet:summary': 'Only stats for full image - One row per image.',
  'parquet:full':    'Full table - including rows of dim slice stats.',
};

function buildExportControls(schema, onExport, canParquet) {
  const selectEl = el('export-select');
  const hintEl   = el('export-hint');
  const btnEl    = el('export-btn');
  if (!selectEl || !btnEl) return;

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

  selectEl.innerHTML = options.map(o => `<option value="${o.value}">${o.label}</option>`).join('');

  function updateHint() {
    hintEl.textContent = EXPORT_HINTS[selectEl.value] ?? '';
  }
  selectEl.onchange = updateHint;
  updateHint();

  btnEl.onclick = () => {
    const [format, scope] = selectEl.value.split(':');
    onExport(format, scope);
  };
}
