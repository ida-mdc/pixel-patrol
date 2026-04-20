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
 * Standard layout defaults — spread into your layout override object.
 *   { ...LAYOUT, title: 'My Chart', height: 400 }
 */
export const LAYOUT = {
  template:  'plotly_white',
  margin:    { l: 50, r: 50, t: 50, b: 50 },
  hovermode: 'closest',
};

/**
 * Standard legend positioning — mirrors Dash _STANDARD_LEGEND_KWARGS.
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

// Applied to every chart — never show the mode-bar, always be responsive.
const CHART_CONFIG = { responsive: true, displayModeBar: false };

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
  Plotly.newPlot(div, traces, { ...LAYOUT, ...layout }, CHART_CONFIG);
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
 * Bargap for bar charts — mirrors Dash _apply_standard_styling.
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

  // Then render — every div is already in the live DOM at correct size.
  plotDefs.forEach(({ traces, layout }, i) => {
    Plotly.newPlot(divs[i], traces, { ...LAYOUT, ...layout }, CHART_CONFIG);
  });

  return wrap;
}

/**
 * Create a flex-wrap grid container and append it to `container`.
 *
 * Returns { wrap, flexBasisPct } where:
 *   wrap          — the grid div (already in the DOM; append your plot items here)
 *   flexBasisPct  — the CSS flex-basis percentage for each item
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
