/**
 * Shared Plotly utilities for viewer plugins.
 *
 * Mirrors the role of `factory.py` in the Dash app:
 *   LAYOUT     ↔  _STANDARD_LAYOUT_KWARGS
 *   LEGEND     ↔  _STANDARD_LEGEND_KWARGS
 *   appendPlot ↔  dcc.Graph + html.Div wrapper (handles DOM-first requirement)
 *   bargap     ↔  _apply_standard_styling bargap logic
 */

import Plotly from 'plotly.js-dist-min';

// Expose globally so dynamically-loaded extension plugins can use it.
window.Plotly = Plotly;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Standard layout defaults - spread into your layout override object.
 *   { ...LAYOUT, title: 'My Chart', height: 400 }
 */
export const LAYOUT = {
  template:  'plotly_white',
  margin:    { l: 50, r: 50, t: 50, b: 50 },
  hovermode: 'closest',
};

/**
 * Standard legend positioning - mirrors Dash _STANDARD_LEGEND_KWARGS.
 * Use as:  { ...LAYOUT, showlegend: true, legend: LEGEND }
 */
export const LEGEND = {
  orientation: 'v',
  yanchor:     'top',
  y:           1,
  xanchor:     'left',
  x:           1.02,
  title:       null,
};

/** Build a grouping label from the selected group column. */
export function groupingLabel(state, fallback = '') {
  return state?.groupCol ? `${state.groupCol}` : fallback;
}

/** Return a legend config with grouping title applied. */
export function legendWithGrouping(baseLegend, state, fallback = '') {
  return {
    ...baseLegend,
    title: { text: groupingLabel(state, fallback) },
  };
}

// Expand-corners icon (500×500 SVG coordinate space).
const _EXPAND_ICON = {
  width: 500, height: 500,
  path: 'M80,80 L80,200 L120,200 L120,120 L200,120 L200,80 Z ' +
        'M420,80 L300,80 L300,120 L380,120 L380,200 L420,200 Z ' +
        'M80,420 L200,420 L200,380 L120,380 L120,300 L80,300 Z ' +
        'M420,420 L420,300 L380,300 L380,380 L300,380 L300,420 Z',
};

const CHART_CONFIG = {
  responsive: true,
  displayModeBar: 'hover',
  modeBarButtonsToRemove: ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
  modeBarButtonsToAdd: [{ name: 'maximize', title: 'Maximize', icon: _EXPAND_ICON, click: (gd) => {
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.75);z-index:9999;display:flex;align-items:center;justify-content:center;';
    const box = document.createElement('div');
    box.style.cssText = 'background:#fff;border-radius:6px;width:95vw;height:93vh;display:flex;flex-direction:column;';
    const bar = document.createElement('div');
    bar.style.cssText = 'text-align:right;padding:4px 8px;border-bottom:1px solid #ddd;flex-shrink:0;';
    bar.innerHTML = '<button style="border:none;background:none;cursor:pointer;font-size:1rem;" title="Close (Esc)">&#x2715;</button>';
    const wrap = document.createElement('div');
    wrap.style.cssText = 'flex:1;min-height:0;';
    box.append(bar, wrap);
    overlay.appendChild(box);
    document.body.appendChild(overlay);
    Plotly.newPlot(wrap, gd.data, { ...gd.layout, autosize: true, height: undefined }, { ...CHART_CONFIG, displayModeBar: true, modeBarButtonsToAdd: [] });
    Plotly.Plots.resize(wrap);
    const close = () => { Plotly.purge(wrap); overlay.remove(); document.removeEventListener('keydown', onKey); };
    bar.querySelector('button').onclick = close;
    overlay.addEventListener('click', e => { if (e.target === overlay) close(); });
    const onKey = e => { if (e.key === 'Escape') close(); };
    document.addEventListener('keydown', onKey);
  } }],
  displaylogo: false,
  toImageButtonOptions: { format: 'png', scale: 3 },
};

// Config for the small "hero" plot shown inside a condensed-mode tile: no toolbar,
// no interactivity hints - it's a glanceable preview, the full plot lives in the
// expanded detail view.
const MINI_CONFIG = {
  responsive: true,
  displayModeBar: false,
  staticPlot: false,
};

/**
 * Compact layout defaults for condensed-mode tile previews. Tiny margins, no
 * legend, small fonts - just enough to read the shape of the data at a glance.
 */
export const MINI_LAYOUT = {
  margin:     { l: 38, r: 12, t: 10, b: 30 },
  showlegend: false,
  font:       { size: 10 },
};

/**
 * Merge a plugin's layout overrides on top of LAYOUT, with `automargin: true`
 * defaulted on both axes so Plotly expands the plot area to fit tick labels
 * and axis titles instead of letting them overlap.
 */
function mergeLayout(layout) {
  return {
    ...LAYOUT,
    ...layout,
    xaxis: { automargin: true, ...layout.xaxis },
    yaxis: { automargin: true, ...layout.yaxis },
  };
}

// ---------------------------------------------------------------------------
// Core plot helper
// ---------------------------------------------------------------------------

/**
 * Create a new <div>, append it to `container` (must already be in the DOM),
 * then call Plotly.newPlot.
 *
 * WHY the order matters: Plotly reads `div.clientWidth` at render time to
 * determine the chart width.  If the element is detached, clientWidth = 0
 * and Plotly falls back to its 700 px default, causing overflow.
 *
 * @param {HTMLElement} container  Parent element (already attached to the DOM).
 * @param {object[]}    traces     Plotly trace array.
 * @param {object}      layout     Layout overrides, merged on top of LAYOUT.
 * @param {string}      [divStyle] CSS string for the wrapper div.
 * @returns {HTMLElement} The created div (Plotly has rendered into it).
 */
export function appendPlot(container, traces, layout, divStyle = '') {
  const div = document.createElement('div');
  if (divStyle) div.style.cssText = divStyle;
  container.appendChild(div);              // ← must precede newPlot
  Plotly.newPlot(div, traces, mergeLayout(layout), CHART_CONFIG);
  return div;
}

/**
 * Render a single compact "hero" plot for a condensed-mode tile. Fills the
 * given container (which should already be sized by CSS), merges MINI_LAYOUT
 * defaults, and hides the mode bar. Same DOM-first ordering rule as appendPlot.
 *
 * @param {HTMLElement} container  Tile plot area (already attached + sized).
 * @param {object[]}    traces
 * @param {object}      [layout]   Layout overrides on top of MINI_LAYOUT.
 * @returns {HTMLElement} the plot div.
 */
export function appendMiniPlot(container, traces, layout = {}) {
  const div = document.createElement('div');
  div.style.cssText = 'width:100%;height:100%';
  container.appendChild(div);
  const merged = {
    ...LAYOUT, ...MINI_LAYOUT, ...layout,
    margin: { ...MINI_LAYOUT.margin, ...layout.margin },
    xaxis:  { automargin: true, ...layout.xaxis },
    yaxis:  { automargin: true, ...layout.yaxis },
  };
  Plotly.newPlot(div, traces, merged, MINI_CONFIG);
  return div;
}

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

/**
 * Convert a snake_case column name to Title Case for display labels.
 * e.g. 'mean_intensity' → 'Mean Intensity'
 */
export function niceName(col) {
  return col.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/** Escape a string for safe insertion into HTML content or attributes. */
export function escapeHtml(s) {
  return String(s)
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}

/**
 * Bargap for bar charts - mirrors Dash _apply_standard_styling.
 *   1 category → 0.7,  2 → 0.4,  3+ → 0.1
 */
export function bargap(nCategories) {
  if (nCategories === 1) return 0.7;
  if (nCategories === 2) return 0.4;
  return 0.1;
}

/**
 * Render multiple plots side-by-side in a flex container, all DOM-inserted
 * before any Plotly.newPlot call so that flex layout is fully computed when
 * Plotly reads clientWidth for each chart.
 *
 * @param {HTMLElement} container  Parent element (already in the DOM).
 * @param {Array<{traces, layout, divStyle}>} plotDefs  Plot definitions.
 * @param {string} [wrapStyle]  CSS for the flex wrapper.
 * @returns {HTMLElement} The flex wrapper div.
 */
export function appendPlots(container, plotDefs, wrapStyle = 'display:flex;flex-wrap:wrap;gap:16px') {
  const wrap = document.createElement('div');
  wrap.style.cssText = wrapStyle;
  container.appendChild(wrap);

  // Insert all divs first so the browser computes the final flex sizes.
  const divs = plotDefs.map(({ divStyle }) => {
    const div = document.createElement('div');
    if (divStyle) div.style.cssText = divStyle;
    wrap.appendChild(div);
    return div;
  });

  // Then render - every div is already in the live DOM at correct size.
  plotDefs.forEach(({ traces, layout }, i) => {
    Plotly.newPlot(divs[i], traces, mergeLayout(layout), CHART_CONFIG);
  });

  return wrap;
}

/**
 * Create a flex-wrap grid container and append it to `container`.
 *
 * Returns { wrap, flexBasisPct } where:
 *   wrap          - the grid div (already in the DOM; append your plot items here)
 *   flexBasisPct  - the CSS flex-basis percentage for each item
 *
 * Usage:
 *   const { wrap, flexBasisPct } = createFlexGrid(container, plotsPerRow);
 *   for (...) {
 *     appendPlot(wrap, traces, layout,
 *       `flex:0 0 ${flexBasisPct}%;min-width:300px;margin-bottom:20px;box-sizing:border-box`);
 *   }
 */
export function createFlexGrid(container, plotsPerRow, gap = '15px') {
  const flexBasisPct = Math.max(30, Math.floor(100 / plotsPerRow) - 2);
  const wrap = document.createElement('div');
  wrap.style.cssText = `display:flex;flex-wrap:wrap;gap:${gap};width:100%`;
  container.appendChild(wrap);
  return { wrap, flexBasisPct };
}

/**
 * The "Slice by" control row shared by the violin and custom-plot widgets: a
 * labelled row of dim toggles that add/remove dimension letters from `splitDims`
 * (mutated in place) and re-render via `onChange`. The caller owns where its
 * scope badge lives, so badge-syncing just happens inside `onChange`.
 *
 * @param {string[]} dims         Dimension letters that can be split on.
 * @param {Set<string>} splitDims Mutated in place as toggles flip.
 * @param {() => void} onChange   Run after each toggle (sync badge + redraw).
 * @returns {HTMLElement} The control row (not yet attached).
 */
export function sliceToggles(dims, splitDims, onChange) {
  const row = document.createElement('div');
  row.className = 'violin-controls';
  row.innerHTML = '<span class="violin-controls-label">Slice by:</span>';
  for (const letter of dims) {
    const sw = document.createElement('label');
    sw.className = 'dim-switch';
    sw.innerHTML = `<input type="checkbox"><span class="dim-switch-track"></span><span class="dim-switch-label">${letter.toUpperCase()}</span>`;
    sw.querySelector('input').addEventListener('change', e => {
      if (e.target.checked) splitDims.add(letter); else splitDims.delete(letter);
      onChange();
    });
    row.appendChild(sw);
  }
  return row;
}

const WARNING_ICONS = { red: '🚩', yellow: '⚠️' };

/**
 * Insert a warning banner as the first child of `container` - used to flag
 * the most obvious data-quality issues right where the relevant widget
 * renders them (e.g. mixed dtypes at the top of the Metadata widget).
 *
 * Mirrors the red/yellow flag styling used in the report-walkthrough docs,
 * so the language and color-coding feel familiar to anyone who's read it.
 *
 * @param {HTMLElement} container  Widget body (already in the DOM).
 * @param {object} opts
 * @param {'red'|'yellow'} [opts.level='yellow']  Severity - red for things
 *   worth investigating, yellow for things that are merely worth a look.
 * @param {string} opts.html  Pre-built HTML for the message body. Callers are
 *   responsible for escaping any dynamic values (see `escapeHtml`).
 * @returns {HTMLElement} The created banner element.
 */
export function prependWarning(container, { level = 'yellow', html }) {
  const div = document.createElement('div');
  div.className = `widget-warning widget-warning-${level}`;
  div.innerHTML = `<span class="widget-warning-icon">${WARNING_ICONS[level] ?? WARNING_ICONS.yellow}</span><div>${html}</div>`;
  container.insertBefore(div, container.firstChild);
  return div;
}

/** Pixel value ranges for common integer dtypes - used to phrase condensed-mode summaries. */
const DTYPE_RANGES = {
  uint8:  [0, 255],
  int8:   [-128, 127],
  uint16: [0, 65535],
  int16:  [-32768, 32767],
  uint32: [0, 4294967295],
  int32:  [-2147483648, 2147483647],
};

/** Returns [min, max] for a known integer dtype name, or null (e.g. for float dtypes). */
export function dtypeRange(name) {
  return DTYPE_RANGES[String(name || '').toLowerCase()] ?? null;
}

/**
 * Warn if one or more named values aren't reported by every row - e.g. a
 * metric or dimension size that's only present for part of the dataset.
 * Shows nothing if everything is fully present, since that's the common
 * case and not worth calling out.
 *
 * @param {HTMLElement} container
 * @param {{label: string, present: number}[]} items
 * @param {number} total
 * @param {object} [opts]
 * @param {'files'|'images'|'slices'} [opts.unit='files']  what one row represents,
 *   matching the widget's scope badge (per file/image/slice).
 * @param {'red'|'yellow'} [opts.level='red']
 * @returns {HTMLElement|undefined} The created banner, if any.
 */
export function dataAvailabilityWarning(container, items, total, { unit = 'files', level = 'red' } = {}) {
  const incomplete = items.filter(({ present }) => present < total);
  if (!incomplete.length) return;

  const pct = n => (total > 0 ? ((n / total) * 100).toFixed(2) : '0.00');

  if (incomplete.length === 1) {
    const { label, present } = incomplete[0];
    return prependWarning(container, {
      level,
      html: `Only ${present.toLocaleString()} of ${total.toLocaleString()} ${unit} (${pct(present)}%) have <code>${escapeHtml(label)}</code> information.`,
    });
  }

  const rows = incomplete.map(({ label, present }) => `
    <li style="display:flex;justify-content:space-between;gap:16px;max-width:320px">
      <span><code>${escapeHtml(label)}</code></span>
      <span style="font-variant-numeric:tabular-nums;white-space:nowrap">${present.toLocaleString()} of ${total.toLocaleString()} (${pct(present)}%)</span>
    </li>`).join('');

  return prependWarning(container, {
    level,
    html: `Not all ${unit} report every value:<ul style="margin:6px 0 0;padding-left:0;list-style:none">${rows}</ul>`,
  });
}

/** Join items into prose: [] → "", [a] → "a", [a,b] → "a and b", [a,b,c] → "a, b and c". */
export function humanList(items) {
  const a = (items ?? []).filter(x => x != null && x !== '');
  if (a.length <= 1) return a[0] ?? '';
  return `${a.slice(0, -1).join(', ')} and ${a.at(-1)}`;
}

/** Human-readable byte size, e.g. 1536 → "1.5 KB", 12_000_000 → "11 MB". */
export function formatBytes(v) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let val = Number(v), u = 0;
  while (val >= 1024 && u < units.length - 1) { val /= 1024; u++; }
  return val >= 10 ? `${Math.round(val)} ${units[u]}` : `${val.toFixed(1)} ${units[u]}`;
}

/**
 * Build a `<table class="stat-table">` from header strings and an array of row
 * cell arrays. Cells are inserted as text (no HTML interpretation).
 */
export function statTable(headers, rowData) {
  const t = document.createElement('table');
  t.className = 'stat-table';
  const thead = document.createElement('thead');
  thead.appendChild(headers.reduce((tr, h) => {
    tr.appendChild(Object.assign(document.createElement('th'), { textContent: h }));
    return tr;
  }, document.createElement('tr')));
  const tbody = document.createElement('tbody');
  for (const cells of rowData) {
    tbody.appendChild(cells.reduce((tr, c) => {
      tr.appendChild(Object.assign(document.createElement('td'), { textContent: c }));
      return tr;
    }, document.createElement('tr')));
  }
  t.append(thead, tbody);
  return t;
}

/**
 * Render a cropped table "peek" into a condensed-mode tile plot area. Used as a
 * tile preview when the full widget would only show a table (e.g. a metadata
 * column that's constant across every image, so there's no distribution to plot).
 * Mirrors the Image Table tile's markup so the shared clip/fade styling applies.
 *
 * @param {HTMLElement} container  the tile plot area
 * @param {string[]} headers
 * @param {Array<Array<string|number>>} rows
 */
export function tilePreviewTable(container, headers, rows) {
  const thead = `<tr>${headers.map(h => `<th>${escapeHtml(String(h))}</th>`).join('')}</tr>`;
  const tbody = rows.map(r =>
    `<tr>${r.map(c => `<td>${escapeHtml(String(c))}</td>`).join('')}</tr>`).join('');
  container.classList.add('widget-tile-plot-table');
  container.innerHTML =
    `<table class="widget-tile-table-preview"><thead>${thead}</thead><tbody>${tbody}</tbody></table>`;
}

/**
 * Append a titled two-column "invariant" section: an optional `<hr>`, an `<h6>`
 * heading, then a stat-table. Used by widgets to list properties that are
 * constant across the dataset (single file extension, no-variance metrics, …).
 *
 * @param {HTMLElement} container
 * @param {object} o
 * @param {string} o.title            heading text
 * @param {[string,string]} o.headers column headers
 * @param {Array<[string,string]>} o.rows  cell pairs
 * @param {boolean} [o.hr=false]      prepend a horizontal rule
 */
export function appendInvariantTable(container, { title, headers, rows, hr = false }) {
  if (hr) container.appendChild(document.createElement('hr'));
  const h = document.createElement('h6');
  h.style.cssText = 'margin-top:20px;margin-bottom:12px';
  h.textContent = title;
  container.appendChild(h);
  container.appendChild(statTable(headers, rows));
}

/**
 * Append a compact DOM color legend for the current groups.
 * Supports optional grouping header using the selected group column.
 */
export function appendGroupLegend(
  container,
  groups,
  colorFn,
  { state = null, minGroups = 2, labelFn = null } = {},
) {
  if ((groups?.length ?? 0) < minGroups) return;
  const title = document.createElement('div');
  title.style.cssText = 'font-size:13px;font-weight:600;margin-top:8px;margin-bottom:6px';
  title.textContent = groupingLabel(state, '');
  container.appendChild(title);
  const div = document.createElement('div');
  div.style.cssText = 'display:flex;flex-wrap:wrap;gap:6px 16px;margin-bottom:12px';
  for (const g of groups) {
    const item   = document.createElement('span');
    item.style.cssText = 'display:flex;align-items:center;gap:5px;font-size:0.85rem';
    const swatch = document.createElement('span');
    swatch.style.cssText = `display:inline-block;width:12px;height:12px;border-radius:2px;flex-shrink:0;background:${colorFn(g)}`;
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(labelFn ? labelFn(g) : String(g)));
    div.appendChild(item);
  }
  container.appendChild(div);
}
