import Plotly from 'plotly.js-dist-min';
import { q, groupCol, groupExpr } from '../sql.js';
import { groupColor } from '../colors.js';
import { DIM_PATTERN } from '../schema.js';
import { appendPlot, createFlexGrid, niceName } from '../plot-utils.js';

const VIOLIN_MAX_POINTS = 2000;  // per group, matches Dash app
const MAX_METRICS = 12;

// Matches BasicStatsProcessor.OUTPUT_SCHEMA
const BASIC_METRIC_BASES = new Set([
  'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
]);

// Matches QualityMetricsProcessor.OUTPUT_SCHEMA
const QUALITY_METRIC_BASES = new Set([
  'laplacian_variance', 'tenengrad', 'brenner', 'noise_std', 'blocking_records', 'ringing_records',
]);

function matchesBases(col, bases) {
  for (const base of bases) {
    if (col === base || col.startsWith(base + '_')) return true;
  }
  return false;
}

const SIGNIFICANCE_HELP = [
  '**Statistical Comparisons**',
  '',
  'Pairwise group comparisons use the Mann–Whitney U test (a non-parametric test that makes no assumptions about the data distribution) with Bonferroni correction for multiple comparisons.',
  '',
  '**Significance levels:**',
  '- `ns`: not significant (p ≥ 0.05)',
  '- `*`: p < 0.05',
  '- `**`: p < 0.01',
  '- `***`: p < 0.001',
].join('\n');

const BASIC_INFO = [
  'Shows **per-image intensity statistics** across groups.',
  '',
  'You can choose which statistic to plot and filter by image dimensions.',
  '',
  'In the plot each point is one image; the box shows the distribution per group.',
  '',
  SIGNIFICANCE_HELP,
].join('\n');

const QUALITY_INFO = [
  'Visualizes **image quality metrics** as violin plots across groups.',
  '',
  'Use these plots to quickly spot outliers, compare image sets, and detect quality differences.',
  '',
  '**Metrics**',
  '- **Laplacian variance** – Edge-based sharpness estimate. Higher values indicate a sharper image.',
  '- **Tenengrad** – Focus measure based on Sobel gradients; captures overall edge strength.',
  '- **Brenner** – Measures fine structural detail using pixel intensity differences.',
  '- **Noise std** – Estimated pixel-level noise standard deviation; higher noise reduces clarity.',
  '- **Blocking records** – Strength of blocky compression artifacts (e.g. JPEG blocking).',
  '- **Ringing records** – Edge oscillation artifacts around sharp boundaries, often due to compression.',
  '',
  SIGNIFICANCE_HELP,
].join('\n');

async function renderViolins(container, ctx, filterMetric) {
  const metrics = resolveMetrics(ctx.schema, ctx.state.dimensions)
    .filter(filterMetric)
    .slice(0, MAX_METRICS);

  if (!metrics.length) {
    container.innerHTML = '<div class="no-data">No numeric metric columns.</div>';
    return;
  }

  // Per-group reservoir sample — mirrors Dash's per-group sample(n=2000, seed=42).
  const gc  = groupCol(ctx.state);
  const sql = `
    WITH ranked AS (
      SELECT ${groupExpr(ctx.state)},
             ${metrics.map(q).join(', ')},
             ROW_NUMBER() OVER (PARTITION BY ${gc} ORDER BY random()) AS __rn__,
             COUNT(*)     OVER (PARTITION BY ${gc}) AS __group_size__
      FROM pp_data ${ctx.where}
    )
    SELECT * EXCLUDE (__rn__) FROM ranked WHERE __rn__ <= ${VIOLIN_MAX_POINTS}
  `;

  const rows = await ctx.queryRows(sql);
  if (!rows.length) {
    container.innerHTML = '<div class="no-data">No rows match the current filter.</div>';
    return;
  }

  const groups = [...new Set(rows.map(r => String(r.__group__)))].sort();

  // Separate metrics with variance from those without.
  const toPlot     = [];
  const noVariance = [];
  for (const metric of metrics) {
    const vals = rows.map(r => r[metric]).filter(v => v != null).map(Number);
    if (!vals.length) continue;
    if (new Set(vals).size <= 1) noVariance.push({ metric, value: vals[0] });
    else                         toPlot.push(metric);
  }

  const numGroups  = groups.length;
  const plotsPerRow = numGroups <= 2 ? 3 : numGroups === 3 ? 2 : 1;
  const sampledGroups = new Set(
    rows.filter(r => Number(r.__group_size__) > VIOLIN_MAX_POINTS).map(r => String(r.__group__))
  );
  const showSig = ctx.state.showSignificance && groups.length >= 2;

  if (toPlot.length) {
    const { wrap, flexBasisPct } = createFlexGrid(container, plotsPerRow);

    for (const metric of toPlot) {
      const label     = niceName(metric);
      const isSampled = sampledGroups.size > 0;
      const title     = `Distribution of ${label}` +
        (isSampled ? `<br><sup>sampled to ${VIOLIN_MAX_POINTS} per group</sup>` : '');

      const groupData = {};
      for (const g of groups) {
        groupData[g] = rows
          .filter(r => String(r.__group__) === g)
          .map(r => r[metric])
          .filter(v => v != null)
          .map(Number);
      }

      const traces = groups.map(g => {
        const vals       = groupData[g];
        const showPoints = vals.length < 1000 ? 'all' : 'outliers';
        return {
          type:     'violin',
          y:        vals,
          name:     g,
          box:      { visible: true },
          meanline: { visible: true },
          points:   showPoints,
          pointpos: 0,
          opacity:  0.9,
          marker:   { color: groupColor(ctx.colorMap, g), line: { width: 1, color: 'black' } },
          hovertemplate: '<b>Group:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>',
        };
      });

      const outerDiv = appendPlot(wrap, traces, {
        title:      { text: title },
        yaxis:      { title: label },
        xaxis:      { title: '' },
        showlegend: false,
      }, `flex:0 0 ${flexBasisPct}%;min-width:300px;margin-bottom:20px;box-sizing:border-box`);

      if (showSig) {
        const pairs = computeSignificancePairs(groups, groupData);
        addSignificanceBrackets(outerDiv, pairs, groups);
      }
    }
  }

  // No-variance table — matches Dash's "Metrics with No Variance" section.
  if (noVariance.length) {
    const hr = document.createElement('hr');
    container.appendChild(hr);
    const h = document.createElement('h6');
    h.style.cssText = 'margin-top:20px;margin-bottom:12px';
    h.textContent = 'Metrics with No Variance';
    container.appendChild(h);

    const table = document.createElement('table');
    table.className = 'stat-table';
    table.innerHTML = `
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        ${noVariance.map(({ metric, value }) =>
          `<tr><td>${niceName(metric)}</td><td>${Number(value).toFixed(4)}</td></tr>`
        ).join('')}
      </tbody>
    `;
    container.appendChild(table);
  }
}

function makeViolinPlugin(id, label, info, filterMetric) {
  return {
    id,
    label,
    info,
    requires(schema) {
      return schema.metricCols.some(filterMetric)
        || (schema.dimMetricCols ?? []).some(filterMetric);
    },
    async render(container, ctx) {
      await renderViolins(container, ctx, filterMetric);
    },
  };
}

export default [
  makeViolinPlugin('violin-basic',   'Pixel Value Statistics', BASIC_INFO,   m => matchesBases(m, BASIC_METRIC_BASES)),
  makeViolinPlugin('violin-quality', 'Image Quality Metrics',  QUALITY_INFO, m => matchesBases(m, QUALITY_METRIC_BASES)),
];

// ── Dimension-aware metric resolution ─────────────────────────────────────────

function resolveMetrics(schema, dimensions) {
  const dimEntries = Object.entries(dimensions);

  const fromMetricCols = schema.metricCols.filter(col => {
    if (!DIM_PATTERN.test(col)) return true;
    for (const [letter, idx] of dimEntries) {
      const m = new RegExp(`_${letter}(\\d+)`).exec(col);
      if (m && m[1] !== String(idx)) return false;
    }
    return true;
  });

  const fromDimMetricCols = (schema.dimMetricCols ?? []).filter(col => {
    const tokens = [...col.matchAll(/_([a-z])(\d+)/g)];
    return tokens.length > 0 &&
      tokens.every(([, letter, idx]) =>
        dimensions[letter] !== undefined && String(dimensions[letter]) === idx,
      );
  });

  return [...fromMetricCols, ...fromDimMetricCols];
}

// ── Statistical significance (Mann-Whitney U, Bonferroni corrected) ───────────

const THRESHOLDS = [[0.001, '***'], [0.01, '**'], [0.05, '*']];
function sigSymbol(p) {
  for (const [t, s] of THRESHOLDS) if (p < t) return s;
  return 'ns';
}

/** Abramowitz & Stegun erf approximation (max error < 1.5e-7). */
function erf(x) {
  const a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((a[4]*t + a[3])*t + a[2])*t + a[1])*t + a[0]) * t * Math.exp(-x * x);
  return sign * y;
}
const normalCDF = z => 0.5 * (1 + erf(z / Math.SQRT2));

/** Two-sided Mann-Whitney U p-value via normal approximation. */
function mannWhitneyP(a, b) {
  const n1 = a.length, n2 = b.length;
  if (n1 < 3 || n2 < 3) return 1.0;

  const combined = [
    ...a.map(v => ({ v, g: 0 })),
    ...b.map(v => ({ v, g: 1 })),
  ].sort((x, y) => x.v - y.v);

  // Average ranks for ties.
  const ranks = new Array(combined.length);
  let i = 0;
  while (i < combined.length) {
    let j = i;
    while (j < combined.length && combined[j].v === combined[i].v) j++;
    const avgRank = (i + j + 1) / 2;
    for (let k = i; k < j; k++) ranks[k] = avgRank;
    i = j;
  }

  let R1 = 0;
  for (let k = 0; k < combined.length; k++) if (combined[k].g === 0) R1 += ranks[k];

  const U1  = R1 - n1 * (n1 + 1) / 2;
  const mu  = n1 * n2 / 2;
  const sig = Math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12);
  if (sig === 0) return 1.0;

  return 2 * (1 - normalCDF(Math.abs(U1 - mu) / sig));
}

/**
 * Pairwise Mann-Whitney U with Bonferroni correction.
 * Returns [{g1, g2, symbol}] sorted narrowest span first (matches Dash).
 */
function computeSignificancePairs(groups, groupData) {
  const pairs = [];
  for (let i = 0; i < groups.length; i++) {
    for (let j = i + 1; j < groups.length; j++) {
      pairs.push({ g1: groups[i], g2: groups[j],
                   p: mannWhitneyP(groupData[groups[i]], groupData[groups[j]]) });
    }
  }
  const n = pairs.length;
  return pairs
    .map(p => ({ ...p, symbol: sigSymbol(Math.min(p.p * n, 1.0)) }))
    .sort((a, b) => {
      const sa = Math.abs(groups.indexOf(a.g1) - groups.indexOf(a.g2));
      const sb = Math.abs(groups.indexOf(b.g1) - groups.indexOf(b.g2));
      return sa - sb;
    });
}

/**
 * Add significance brackets to an already-rendered Plotly violin div.
 */
function addSignificanceBrackets(plotDiv, pairs, groups) {
  const sigPairs = pairs.filter(p => p.symbol !== 'ns');
  if (!sigPairs.length) return;

  const renderedRange = plotDiv._fullLayout?.yaxis?.range;
  const yBottom = renderedRange?.[0] ?? 0;
  const yTop    = renderedRange?.[1] ?? 1;
  const span    = Math.abs(yTop - yBottom) || 1;

  const gap   = span * 0.06;
  const tickH = span * 0.04;
  const xPos  = Object.fromEntries(groups.map((g, i) => [g, i]));

  const shapes      = [];
  const annotations = [];
  let   currentY    = yTop + gap;

  for (const { g1, g2, symbol } of sigPairs) {
    const x1 = Math.min(xPos[g1], xPos[g2]);
    const x2 = Math.max(xPos[g1], xPos[g2]);

    shapes.push(
      { type:'line', x0:x1, x1:x1, y0:currentY-tickH, y1:currentY,
        xref:'x', yref:'y', line:{ color:'black', width:1 } },
      { type:'line', x0:x1, x1:x2, y0:currentY,       y1:currentY,
        xref:'x', yref:'y', line:{ color:'black', width:1 } },
      { type:'line', x0:x2, x1:x2, y0:currentY-tickH, y1:currentY,
        xref:'x', yref:'y', line:{ color:'black', width:1 } },
    );
    annotations.push({
      x: (x1 + x2) / 2, y: currentY + tickH * 0.5,
      text: symbol, showarrow: false,
      font: { size: 12, color: 'black' },
      xref: 'x', yref: 'y',
    });

    currentY += gap + tickH;
  }

  Plotly.relayout(plotDiv, {
    shapes,
    annotations,
    'yaxis.range': [yBottom, currentY + gap],
  });
}
