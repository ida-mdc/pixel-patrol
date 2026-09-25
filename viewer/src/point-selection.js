// Cross-plot point selection: a click pub/sub plus the linked highlight ring.

import { accent } from './plot-utils.js';

const linkedPlots = new Set();
let selectedFrn = null;

// ── click pub/sub ─────────────────────────────────────────────────────────────
const clickHandlers = new Set();

/** Subscribe to point clicks; returns an unsubscribe function. */
export function onPointClick(handler) {
  clickHandlers.add(handler);
  return () => clickHandlers.delete(handler);
}

function emitPointClick(payload) {
  for (const handler of [...clickHandlers]) {
    try { handler(payload); } catch (err) { console.warn('[viewer] point-click handler failed:', err); }
  }
}

// ── linked highlight ──────────────────────────────────────────────────────────
function highlightInPlot(plotDiv, frn, color) {
  if (!plotDiv || !plotDiv.isConnected || typeof Plotly === 'undefined') return;
  try {
    const stale = (plotDiv.data || []).map((t, i) => (t._highlight ? i : -1)).filter(i => i >= 0);
    if (stale.length) Plotly.deleteTraces(plotDiv, stale);
    if (frn == null) return;
    for (const t of plotDiv.data || []) {
      const cd = t.customdata; if (!cd || !cd.length) continue;
      for (let i = 0; i < cd.length; i++) {
        if (Number(cd[i]) === Number(frn)) {
          Plotly.addTraces(plotDiv, {
            type: 'scatter', mode: 'markers',
            x: [Array.isArray(t.x) ? t.x[i] : t.x], y: [Array.isArray(t.y) ? t.y[i] : t.y],
            marker: { size: 15, color: 'rgba(0,0,0,0)', line: { width: 3, color } },
            hoverinfo: 'skip', showlegend: false, cliponaxis: false, _highlight: true,
          });
          return;
        }
      }
    }
  } catch { /* linking is best-effort; never break a plot */ }
}

/** Ring `frn` across every registered plot; pass null to clear. */
export function setSelectedPoint(frn) {
  selectedFrn = frn;
  const color = accent();
  for (const p of [...linkedPlots]) {
    if (!p.isConnected) { linkedPlots.delete(p); continue; }
    highlightInPlot(p, frn, color);
  }
}

// Make a Plotly div (whose points carry file_row_number in customdata) clickable
// and part of the linked highlight. Clicks emit {fileRowNumber, ctx, metric}.
export function registerPointPlot(plotDiv, ctx, metric) {
  if (!plotDiv || typeof plotDiv.on !== 'function') return;
  plotDiv.style.cursor = 'pointer';
  linkedPlots.add(plotDiv);
  // A plot rendered while a point is already selected picks up the ring too.
  if (selectedFrn != null) highlightInPlot(plotDiv, selectedFrn, accent());
  plotDiv.on('plotly_click', (e) => {
    const frn = e?.points?.[0]?.customdata;
    if (frn != null) emitPointClick({ fileRowNumber: Number(frn), ctx, metric });
  });
}
