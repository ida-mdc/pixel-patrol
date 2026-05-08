// Matches BasicStatsProcessor.OUTPUT_SCHEMA
const BASIC_METRIC_BASES = new Set([
  'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
]);

// Matches the default PIXEL_PATROL_METRICS_QUALITY group (intensity-independent metrics).
// Legacy gradient metrics (tenengrad, laplacian_variance, brenner) are kept here so that
// datasets built with PIXEL_PATROL_METRICS_GRADIENT_FOCUS=1 still appear in this widget.
const QUALITY_METRIC_BASES = new Set([
  'michelson_contrast', 'mscn_variance', 'local_std_ratio',
  'laplacian_variance', 'tenengrad', 'brenner',
  'noise_std', 'blocking_records', 'ringing_records',
]);

function matchesBases(col, bases) {
  for (const base of bases) {
    if (col === base || col.startsWith(base + '_')) return true;
  }
  return false;
}

const VIOLIN_MAX_POINTS = 2000;
const MAX_METRICS       = 12;

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
  '', 'You can choose which statistic to plot and filter by image dimensions.',
  '', 'In the plot each point is one image; the box shows the distribution per group.',
  '', SIGNIFICANCE_HELP,
].join('\n');

const QUALITY_INFO = [
  'Visualizes **image quality metrics** as violin plots across groups.',
  '', 'All metrics are invariant to both additive intensity offsets (camera background) and',
  'multiplicative scaling (gain, fluorophore concentration), making them comparable across',
  'images acquired under different conditions.',
  '', 'Use these plots to spot outliers, compare image sets, and detect quality differences.',
  '', '**Default metrics**',
  '- **Michelson contrast** – Mean local intensity range (3×3 window) divided by tile std.',
  '  Higher = more local contrast; lower = blurry or uniform tiles.',
  '- **MSCN variance** – Variance of locally-normalised pixel values (BRISQUE/NIQE framework).',
  '  Captures texture richness independent of background level.',
  '- **Local std ratio** – Mean local std divided by tile std.',
  '  Approaches 1 for sharp structured tiles; lower for smooth or out-of-focus tiles.',
  '', '**Legacy metrics** (enabled via `PIXEL_PATROL_METRICS_GRADIENT_FOCUS=1`)',
  '- **Laplacian variance**, **Tenengrad**, **Brenner** – Gradient-based focus measures,',
  '  normalised by tile variance. Useful for within-image comparisons.',
  '', '**Compression artifact metrics** (enabled via `PIXEL_PATROL_METRICS_COMPRESSION=1`)',
  '- **Blocking records** – Strength of JPEG-style block boundary artifacts.',
  '- **Ringing records** – High-frequency oscillation artifacts near edges.',
  '', SIGNIFICANCE_HELP,
].join('\n');

async function renderViolins(container, ctx, filterMetric) {
  const { q, groupCol: gcFn, groupExpr: geFn } = ctx.sql;
  const { append: appendPlot, flexGrid: createFlexGrid, niceName } = ctx.plot;

  const metrics = resolveMetrics(ctx.schema, ctx.state.dimensions)
    .filter(filterMetric)
    .slice(0, MAX_METRICS);

  if (!metrics.length) {
    container.innerHTML = '<div class="no-data">No numeric metric columns.</div>';
    return;
  }

  // Per-group reservoir sample — mirrors Dash's per-group sample(n=2000, seed=42).
  const dimFilters = Object.entries(ctx.state.dimensions ?? {})
    .map(([letter, idxRaw]) => {
      const idx = Number(idxRaw);
      if (!Number.isFinite(idx)) return null;
      return `${q(`dim_${letter}`)} = ${idx}`;
    })
    .filter(Boolean);
  const activeDims = ctx.state.dimensions ?? {};
  const xSelected = activeDims.x !== undefined;
  const ySelected = activeDims.y !== undefined;
  // obs_level matches RasterImageDaskProcessor grouping depth: global → 0,
  // one dim fixed → 1, two dims fixed → 2, …
  const longObsLevel = dimFilters.length === 0 ? 0 : dimFilters.length;
  const sourceTable = 'pp_all';
  const whereParts = [];
  const baseWhere = ctx.userWhere ? ctx.userWhere.replace(/^\s*WHERE\s+/i, '') : '';
  if (baseWhere) whereParts.push(baseWhere);
  whereParts.push(`obs_level = ${longObsLevel}`);
  whereParts.push(...dimFilters);
  // For tiled datasets, there can be many rows per file at (x,y) tile level.
  // This widget is intended to show per-image (non-spatial) distributions by default,
  // so exclude spatial rows unless the user explicitly slices on x/y.
  if (ctx.schema?.dimCols?.includes('dim_x') && !xSelected) whereParts.push(`${q('dim_x')} IS NULL`);
  if (ctx.schema?.dimCols?.includes('dim_y') && !ySelected) whereParts.push(`${q('dim_y')} IS NULL`);
  const combinedWhere = whereParts.length ? `WHERE ${whereParts.join(' AND ')}` : '';

  const gc  = gcFn();
  const sql = `
    WITH ranked AS (
      SELECT ${geFn()},
             ${metrics.map(q).join(', ')},
             ROW_NUMBER() OVER (PARTITION BY ${gc} ORDER BY random()) AS __rn__,
             COUNT(*)     OVER (PARTITION BY ${gc}) AS __group_size__
      FROM ${sourceTable} ${combinedWhere}
    )
    SELECT * EXCLUDE (__rn__) FROM ranked WHERE __rn__ <= ${VIOLIN_MAX_POINTS}
  `;

  const rows = await ctx.queryRows(sql);
  if (!rows.length) {
    container.innerHTML = '<div class="no-data">No rows match the current filter.</div>';
    return;
  }

  const rowGroups = new Set(rows.map(r => String(r.__group__)));
  const groups = ctx.groups.filter(g => rowGroups.has(g));
  const toPlot = [], noVariance = [];
  for (const metric of metrics) {
    const vals = rows
      .map(r => r[metric])
      .filter(v => v != null)
      .map(Number)
      .filter(Number.isFinite);
    if (!vals.length) continue;
    if (new Set(vals).size <= 1) noVariance.push({ metric, value: vals[0] });
    else toPlot.push(metric);
  }

  const numGroups   = groups.length;
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
          .map(Number)
          .filter(Number.isFinite);
      }

      const traces = groups.map(g => {
        const vals       = groupData[g];
        const showPoints = vals.length < 1000 ? 'all' : 'outliers';
        return {
          type: 'violin', y: vals, name: ctx.groupLabel(g), box: { visible: true }, meanline: { visible: true },
          points: showPoints, pointpos: 0, opacity: 0.9,
          marker: { color: ctx.color.group(g), line: { width: 1, color: 'black' } },
          hovertemplate: '<b>Group:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>',
        };
      });

      const outerDiv = appendPlot(wrap, traces, {
        title: { text: title },
        yaxis: { title: label },
        xaxis: { title: ctx.plot.groupingLabel ? ctx.plot.groupingLabel('') : '', type: 'category' },
        showlegend: false,
      }, `flex:0 0 ${flexBasisPct}%;min-width:300px;margin-bottom:20px;box-sizing:border-box`);

      if (showSig) {
        const pairs = computeSignificancePairs(groups, groupData);
        addSignificanceBrackets(outerDiv, pairs, groups);
      }
    }
  }

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
      <tbody>${noVariance.map(({ metric, value }) => `<tr><td>${niceName(metric)}</td><td>${Number(value).toFixed(4)}</td></tr>`).join('')}</tbody>
    `;
    container.appendChild(table);
  }
}

function makeViolinPlugin(id, label, info, filterMetric) {
  return {
    id, label, info, group: 'Dataset Stats',
    requires(schema) {
      return !!schema.isLongFormat && schema.metricCols.some(filterMetric);
    },
    async render(container, ctx) {
      try {
        await renderViolins(container, ctx, filterMetric);
      } catch {
        container.innerHTML = '<div class="no-data">Failed to load data.</div>';
      }
    },
  };
}

export default [
  makeViolinPlugin('violin-basic',   'Pixel Value Statistics', BASIC_INFO,   m => matchesBases(m, BASIC_METRIC_BASES)),
  makeViolinPlugin('violin-quality', 'Image Quality Metrics',  QUALITY_INFO, m => matchesBases(m, QUALITY_METRIC_BASES)),
];

function resolveMetrics(schema, dimensions) {
  if (!schema.isLongFormat) return [];
  // Long format keeps metrics as base columns; dim filtering is row-based via dim_* + obs_level.
  return schema.metricCols;
}

// ── Statistical significance (Mann-Whitney U, Bonferroni corrected) ────────────

const THRESHOLDS = [[0.001, '***'], [0.01, '**'], [0.05, '*']];
function sigSymbol(p) {
  for (const [t, s] of THRESHOLDS) if (p < t) return s;
  return 'ns';
}

function erf(x) {
  const a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((a[4]*t + a[3])*t + a[2])*t + a[1])*t + a[0]) * t * Math.exp(-x * x);
  return sign * y;
}
const normalCDF = z => 0.5 * (1 + erf(z / Math.SQRT2));

function mannWhitneyP(a, b) {
  const n1 = a.length, n2 = b.length;
  if (n1 < 3 || n2 < 3) return 1.0;
  const combined = [...a.map(v => ({ v, g: 0 })), ...b.map(v => ({ v, g: 1 }))].sort((x, y) => x.v - y.v);
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

function computeSignificancePairs(groups, groupData) {
  const pairs = [];
  for (let i = 0; i < groups.length; i++)
    for (let j = i + 1; j < groups.length; j++)
      pairs.push({ g1: groups[i], g2: groups[j], p: mannWhitneyP(groupData[groups[i]], groupData[groups[j]]) });
  const n = pairs.length;
  return pairs
    .map(p => ({ ...p, symbol: sigSymbol(Math.min(p.p * n, 1.0)) }))
    .sort((a, b) => Math.abs(groups.indexOf(a.g1) - groups.indexOf(a.g2)) - Math.abs(groups.indexOf(b.g1) - groups.indexOf(b.g2)));
}

function addSignificanceBrackets(plotDiv, pairs, groups) {
  const sigPairs = pairs.filter(p => p.symbol !== 'ns');
  if (!sigPairs.length) return;
  const renderedRange = plotDiv._fullLayout?.yaxis?.range;
  const yBottom = renderedRange?.[0] ?? 0;
  const yTop    = renderedRange?.[1] ?? 1;
  const span    = Math.abs(yTop - yBottom) || 1;
  const gap     = span * 0.06, tickH = span * 0.04;
  const xPos    = Object.fromEntries(groups.map((g, i) => [g, i]));
  const shapes = [], annotations = [];
  let currentY = yTop + gap;
  for (const { g1, g2, symbol } of sigPairs) {
    const x1 = Math.min(xPos[g1], xPos[g2]);
    const x2 = Math.max(xPos[g1], xPos[g2]);
    shapes.push(
      { type:'line', x0:x1, x1:x1, y0:currentY-tickH, y1:currentY, xref:'x', yref:'y', line:{color:'black',width:1} },
      { type:'line', x0:x1, x1:x2, y0:currentY,       y1:currentY, xref:'x', yref:'y', line:{color:'black',width:1} },
      { type:'line', x0:x2, x1:x2, y0:currentY-tickH, y1:currentY, xref:'x', yref:'y', line:{color:'black',width:1} },
    );
    annotations.push({ x:(x1+x2)/2, y:currentY+tickH*0.5, text:symbol, showarrow:false, font:{size:12,color:'black'}, xref:'x', yref:'y' });
    currentY += gap + tickH;
  }
  Plotly.relayout(plotDiv, { shapes, annotations, 'yaxis.range': [yBottom, currentY + gap] });
}
