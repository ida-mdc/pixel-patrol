const BASIC_METRIC_BASES = new Set([
  'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
]);
const QUALITY_METRIC_BASES = new Set([
  'michelson_contrast', 'mscn_variance', 'texture_heterogeneity', 'laplacian_variance',
  'blocking_index', 'ringing_index',
]);

const COL_BG    = ['#ffffff', '#f4f6f9'];
const MIN_COL_PX = 180;  // min width per "Across 'X' slices" column in the grid

const RAGGED_INFO = 'The dashed line (right-hand axis) shows the **% of images that still have a ' +
  'slice at this position**, per condition - useful for spotting datasets where images have ' +
  'different numbers of slices.';

const BASIC_INFO = [
  'Shows how image statistics (e.g., mean, std, min, max) change **across different dimension slices**.',
  '', 'Useful for identifying drift, artifacts, or unexpected variation within (e.g.) T/C/Z/S dimensions.',
  '', 'You can select slices in the dropdowns to filter the tables.',
  '', RAGGED_INFO,
].join('\n');

const QUALITY_INFO = [
  'Shows how **image quality metrics** change across (e.g. T, C, Z, S) slices.',
  '', 'Use this view to detect:',
  '- drift in focus or noise over time (T)',
  '- channel-specific artifacts (C)',
  '- depth-dependent quality changes (Z)',
  '', RAGGED_INFO,
].join('\n');

// X and Y are spatial tile coordinates. When querying non-spatial dims we
// exclude tile rows (which carry dim_x/dim_y alongside dim_c etc.) so that
// only the clean per-slice aggregated rows are scanned - the main speedup.
const SPATIAL_LETTERS = new Set(['x', 'y']);

async function renderAcrossDims(container, ctx, filterMetric) {
  const renderStartMs = performance.now();
  if (!ctx.schema.isLongFormat) {
    container.innerHTML = '<div class="no-data">This widget requires long-format data.</div>';
    return;
  }

  const metrics = (ctx.schema.metricCols ?? []).filter(filterMetric).sort();
  const activeDims = ctx.state.dimensions ?? {};
  const dimLetters = Object.keys(ctx.schema.dimensionInfo ?? {})
    .filter(letter => activeDims[letter] === undefined)
    .sort();
  if (!metrics.length || !dimLetters.length) {
    container.innerHTML = '<div class="no-data">No dimension slices found with the current filter.</div>';
    return;
  }

  const fixed = Object.fromEntries(
    Object.entries(activeDims).map(([l, i]) => [l, Number(i)]).filter(([, i]) => Number.isFinite(i)));
  const dimAggMap = await fetchDimAggregates(ctx, metrics, dimLetters, fixed);

  const dimLettersVisible = dimLetters.filter(letter =>
    dimAggMap[letter] && metrics.some(m => (dimAggMap[letter][m] ?? []).some(r => Number.isFinite(r.x))));
  if (!dimLettersVisible.length) {
    container.innerHTML = '<div class="no-data">No dimension slices found with plottable data for the current filter.</div>';
    return;
  }

  appendRetentionLegend(container, ctx);
  const metricSelect = appendMetricSelect(container, ctx, metrics);
  const tableHost = document.createElement('div');
  container.appendChild(tableHost);

  const grid = { metrics, dimLettersVisible, dimAggMap, layout: buildStatsLayout(ctx) };
  const draw = (metric) => renderMetricGrid(tableHost, ctx, grid, metric);
  metricSelect.addEventListener('change', () => draw(metricSelect.value || metrics[0]));
  const initialPlots = draw('__all__');

  console.info(`[stats-across-dims] total_render_ms=${(performance.now() - renderStartMs).toFixed(1)} ` +
    `metrics=${metrics.length} dims_visible=${dimLettersVisible.length} initial_plots=${initialPlots}`);
}

// Mean/std/count of every metric per (group, slice index), for each varying dim.
// Returns dimAggMap[letter][metric] = [{ __group__, x, y_mean, y_std, n }].
async function fetchDimAggregates(ctx, metrics, dimLetters, fixed) {
  const { q, groupCol: gcFn } = ctx.sql;
  const gcExpr = gcFn();
  const dimAggMap = {};
  for (const varyingLetter of dimLetters) {
    const t0 = performance.now();
    // Vary one dim, fix the rest, require every other dim NULL so we don't pool
    // values from different aggregated row classes (tile rows, x/y strips, …).
    // Spatial strips (x/y) span files with mixed dim_orders, so obs_level is
    // unreliable there - omit it and rely on the NULL constraints alone.
    const whereParts = ctx.sql.dimSubsetWhere({
      fixed, split: new Set([varyingLetter]),
      obsLevel: SPATIAL_LETTERS.has(varyingLetter) ? null : undefined,
    });
    const aggExprs = metrics.flatMap((m, i) => {
      const mq = q(m);
      return [`AVG(${mq}) AS m_${i}`, `COALESCE(STDDEV_SAMP(${mq}), 0) AS s_${i}`, `COUNT(${mq}) AS n_${i}`];
    }).join(', ');

    const rows = await ctx.queryRows(`
      SELECT ${gcExpr} AS __group__, ${q(`dim_${varyingLetter}`)} AS x, ${aggExprs}
      FROM pp_all WHERE ${whereParts.join(' AND ')}
      GROUP BY 1, 2 ORDER BY 2
    `);
    dimAggMap[varyingLetter] = pivotDimRows(rows, metrics);
    console.info(`[stats-across-dims] dim=${varyingLetter} query_ms=${(performance.now() - t0).toFixed(1)} rows=${rows.length}`);
  }
  return dimAggMap;
}

// Wide m_i/s_i/n_i columns → per-metric point lists, dropping empty (n=0) cells.
function pivotDimRows(rows, metrics) {
  const byMetric = Object.fromEntries(metrics.map(m => [m, []]));
  for (const r of rows) {
    const x = Number(r.x);
    if (!Number.isFinite(x)) continue;
    const group = String(r.__group__);
    metrics.forEach((m, i) => {
      const n = Number(r[`n_${i}`] ?? 0);
      if (n > 0) byMetric[m].push({ __group__: group, x, y_mean: Number(r[`m_${i}`] ?? 0), y_std: Number(r[`s_${i}`] ?? 0), n });
    });
  }
  return byMetric;
}

// Shared legend above the grid: group color chips + the dashed retention line.
function appendRetentionLegend(container, ctx) {
  const { escapeHtml } = ctx.plot;
  const legend = document.createElement('div');
  legend.style.cssText = 'display:flex;flex-wrap:wrap;gap:12px;margin-bottom:10px;font-size:0.85rem;align-items:center';
  if (ctx.groups.length > 1) {
    for (const g of ctx.groups) {
      const item = document.createElement('span');
      item.style.cssText = 'display:flex;align-items:center;gap:5px';
      item.innerHTML = `<span style="display:inline-block;width:12px;height:3px;border-radius:2px;background:${ctx.color.group(g)}"></span>${escapeHtml(String(g))}`;
      legend.appendChild(item);
    }
  }
  const dash = document.createElement('span');
  dash.style.cssText = 'display:flex;align-items:center;gap:5px;color:#6c757d';
  dash.innerHTML = '<span style="display:inline-block;width:16px;height:0;border-top:1px dashed #6c757d"></span>'
    + '% of images with this slice (right axis)';
  legend.appendChild(dash);
  container.appendChild(legend);
}

// The "Metric:" dropdown (with an "All metrics" option); returns the <select>.
function appendMetricSelect(container, ctx, metrics) {
  const { escapeHtml, niceName } = ctx.plot;
  const controls = document.createElement('div');
  controls.style.cssText = 'display:flex;align-items:center;gap:8px;margin:6px 0 10px 0';
  controls.innerHTML = `
    <label for="stats-across-metric" style="font-weight:600;font-size:0.9rem;margin:0">Metric:</label>
    <select id="stats-across-metric" class="form-select form-select-sm" style="width:auto;min-width:220px">
      ${metrics.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(niceName(m))}</option>`).join('')}
      <option value="__all__" selected>All metrics</option>
    </select>
  `;
  container.appendChild(controls);
  return controls.querySelector('#stats-across-metric');
}

// Plot layout shared by every cell in the grid (clean framed axes, no legend).
function buildStatsLayout(ctx) {
  return {
    ...ctx.plot.LAYOUT,
    showlegend: false,
    xaxis: { showgrid: false, zeroline: false, showline: true, mirror: true, ticks: 'outside', title: null, tickmode: 'auto', nticks: 8, tickformat: 'd' },
    yaxis: { showgrid: false, zeroline: false, showline: true, mirror: true, ticks: 'outside', title: null },
  };
}

// Build the dimension × metric table of mini scatter plots; returns the plot count.
function renderMetricGrid(tableHost, ctx, { metrics, dimLettersVisible, dimAggMap, layout }, selected) {
  const { niceName } = ctx.plot;
  tableHost.innerHTML = '';
  const metricsToRender = selected === '__all__' ? metrics : [selected];
  const plotJobs = [];

  const tableWrap = document.createElement('div');
  tableWrap.className = 'across-dims-wrap';
  tableWrap.style.overflowX = 'auto';
  if (dimLettersVisible.length === 1) {
    const letter = dimLettersVisible[0];
    const xCount = Math.max(...metrics.map(m => new Set((dimAggMap[letter]?.[m] ?? []).map(r => r.x)).size));
    tableWrap.style.maxWidth = `clamp(25%, ${xCount * 3}%, 100%)`;
  }
  const table = document.createElement('table');
  table.style.cssText = `border-collapse:collapse;table-layout:fixed;width:100%;min-width:${dimLettersVisible.length * MIN_COL_PX}px`;

  const hRow = table.createTHead().insertRow();
  dimLettersVisible.forEach((letter, ci) => {
    const th = document.createElement('th');
    th.style.cssText = `padding:8px 12px;text-align:left;border-bottom:2px solid #dee2e6;font-weight:600;background:${COL_BG[ci % 2]};${ci > 0 ? 'border-left:1px solid #dee2e6;' : ''}`;
    th.textContent = `Across '${letter.toUpperCase()}' slices`;
    hRow.appendChild(th);
  });

  const tbody = table.createTBody();
  for (const metric of metricsToRender) {
    const titleCell = tbody.insertRow().insertCell();
    titleCell.colSpan = dimLettersVisible.length;
    titleCell.style.cssText = 'font-weight:500;font-size:0.95rem;color:#343a40;padding:5px 10px;border-top:1px solid #e9ecef;background:#f8f9fa';
    titleCell.textContent = niceName(metric);

    const plotRow = tbody.insertRow();
    dimLettersVisible.forEach((letter, ci) => {
      const cell = plotRow.insertCell();
      cell.style.cssText = `padding:4px;vertical-align:top;background:${COL_BG[ci % 2]};${ci > 0 ? 'border-left:1px solid #dee2e6;' : ''}`;
      const rows = (dimAggMap[letter]?.[metric] ?? []).filter(r => Number.isFinite(r.x));
      if (!rows.length) { cell.innerHTML = '<div style="text-align:center;color:#6c757d;padding:15px">No data</div>'; return; }
      plotJobs.push({ cell, agg: rows });
    });
  }

  tableWrap.appendChild(table);
  tableHost.appendChild(tableWrap);
  for (const { cell, agg } of plotJobs) renderAggScatter(cell, agg, ctx, layout, ctx.plot.append);
  return plotJobs.length;
}

async function acrossDimsCondensedSummary(ctx) {
  try {
    const letters = Object.keys(ctx.schema.dimensionInfo ?? {})
      .map(l => l.toUpperCase())
      .sort();
    if (!letters.length) return null;
    return `Per-slice trend along <strong>${letters.join(', ')}</strong>.`;
  } catch { return null; }
}

// One simple "mean across slices" line plot - the single most representative
// metric, along the first non-spatial dimension that has data. Mirrors the
// non-spatial query path in renderAcrossDims, kept minimal for the tile preview.
async function acrossDimsCondensedPlot(ctx, container, filterMetric, metricPref) {
  if (!ctx.schema.isLongFormat) return false;
  const { q, groupCol: gcFn } = ctx.sql;
  const metrics = (ctx.schema.metricCols ?? []).filter(filterMetric);
  const metric  = metricPref.find(m => metrics.includes(m)) ?? metrics[0];
  if (!metric) return false;

  const activeDims = ctx.state.dimensions ?? {};
  const fixedDims  = Object.entries(activeDims).map(([l, i]) => [l, Number(i)]).filter(([, i]) => Number.isFinite(i));
  const letters    = Object.keys(ctx.schema.dimensionInfo ?? {})
    .filter(l => l !== 'x' && l !== 'y' && activeDims[l] === undefined)
    .sort();
  if (!letters.length) return false;

  const fixed     = Object.fromEntries(fixedDims);
  const gcExpr    = gcFn();

  for (const letter of letters) {
    const parts = ctx.sql.dimSubsetWhere({ fixed, split: new Set([letter]) });

    const rows = await ctx.queryRows(`
      SELECT ${gcExpr} AS __group__, ${q(`dim_${letter}`)} AS x,
             AVG(${q(metric)}) AS y, COALESCE(STDDEV_SAMP(${q(metric)}), 0) AS s,
             COUNT(${q(metric)}) AS n
      FROM pp_all WHERE ${parts.join(' AND ')}
      GROUP BY 1, 2 ORDER BY 2
    `);
    const valid = rows.filter(r => Number.isFinite(Number(r.x)) && Number(r.n) > 0);
    if (!valid.length) continue;

    const traces = [];
    for (const g of ctx.groups) {
      const gr = valid.filter(r => String(r.__group__) === g).sort((a, b) => Number(a.x) - Number(b.x));
      if (!gr.length) continue;
      const color = ctx.color.group(g);
      const xs = gr.map(r => Number(r.x));
      const ym = gr.map(r => Number(r.y));
      const ys = gr.map(r => Number(r.s ?? 0));
      traces.push(
        { type: 'scatter', mode: 'lines', x: xs, y: ym.map((m, i) => m + ys[i]),
          line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
        { type: 'scatter', mode: 'lines', x: xs, y: ym.map((m, i) => m - ys[i]),
          line: { width: 0 }, fill: 'tonexty', fillcolor: ctx.color.hexToRgba(color, 0.2),
          showlegend: false, hoverinfo: 'skip' },
        { type: 'scatter', mode: 'lines+markers', name: ctx.groupLabel(g), x: xs, y: ym,
          line: { color, width: 1.5 }, marker: { size: 3, color }, hoverinfo: 'skip' },
      );
    }
    if (!traces.length) continue;

    ctx.plot.appendMini(container, traces, {
      xaxis:  { title: `${letter.toUpperCase()} slice`, tickformat: 'd' },
      yaxis:  { title: ctx.plot.niceName(metric) },
      margin: { l: 52 },
    });
    return true;
  }
  return false;
}

function makeAcrossDimsPlugin(id, label, info, filterMetric, condensedSummary, metricPref = [], shortLabel) {
  return {
    id, label, info, shortLabel, group: 'Dataset Stats', scope: 'slice', multiPlot: true,
    requires(schema) {
      return !!schema.isLongFormat && schema.metricCols.some(filterMetric);
    },
    ...(condensedSummary ? { condensedSummary } : {}),
    async condensedPlot(container, ctx) {
      return acrossDimsCondensedPlot(ctx, container, filterMetric, metricPref);
    },
    async render(container, ctx) {
      try {
        await renderAcrossDims(container, ctx, filterMetric);
      } catch {
        container.innerHTML = '<div class="no-data">Failed to load data.</div>';
      }
    },
  };
}

export default [
  makeAcrossDimsPlugin(
    'stats-across-dims-basic', 'Basic Statistics Across Dimensions', BASIC_INFO,
    base => BASIC_METRIC_BASES.has(base),
    ctx => acrossDimsCondensedSummary(ctx, { metricLabel: 'brightness statistics' }),
    ['mean_intensity'], 'Intensity across Slices',
  ),
  makeAcrossDimsPlugin(
    'stats-across-dims-quality', 'Quality Metrics Across Dimensions', QUALITY_INFO,
    base => QUALITY_METRIC_BASES.has(base),
    ctx => acrossDimsCondensedSummary(ctx, { metricLabel: 'image quality metrics' }),
    ['laplacian_variance', 'michelson_contrast', 'mscn_variance', 'texture_heterogeneity'], 'Quality across Slices',
  ),
];

function renderAggScatter(container, agg, ctx, STATS_DIMS_LAYOUT, appendPlot) {
  const traces = [];
  const retentionTraces = [];
  for (const g of ctx.groups) {
    const gRows  = agg.filter(r => String(r.__group__) === g).sort((a, b) => a.x - b.x);
    if (!gRows.length) continue;
    const color  = ctx.color.group(g);
    const xVals  = gRows.map(r => r.x);
    const yMean  = gRows.map(r => r.y_mean);
    const yStd   = gRows.map(r => r.y_std ?? 0);
    const ns     = gRows.map(r => r.n);
    const nMax   = Math.max(...ns);
    const retention = ns.map(n => (n / nMax) * 100);
    const yUpper = yMean.map((m, i) => m + yStd[i]);
    const yLower = yMean.map((m, i) => m - yStd[i]);
    const sizes  = ns.map(n => Math.max(4, Math.min(12, 3 + 3 * Math.log10(Math.max(n, 1)))));
    const hover  = gRows.map((r, i) => `<b>${g}</b><br>Slice: ${r.x}<br>Mean: ${yMean[i].toFixed(3)}<br>Std: ${yStd[i].toFixed(3)}<br><b>n=${ns[i]}</b> (${retention[i].toFixed(0)}% of images)`);
    const rgba   = ctx.color.hexToRgba(color, 0.2);
    traces.push(
      { type:'scatter', x:xVals, y:yUpper, mode:'lines', line:{width:0}, showlegend:false, hoverinfo:'skip' },
      { type:'scatter', x:xVals, y:yLower, mode:'lines', line:{width:0}, fill:'tonexty', fillcolor:rgba, showlegend:false, hoverinfo:'skip' },
      { type:'scatter', mode:'lines+markers', name:ctx.groupLabel(g), x:xVals, y:yMean, line:{width:2, color}, marker:{size:sizes, color, line:{width:1, color:'white'}}, hovertemplate:'%{text}<extra></extra>', text:hover },
    );
    const retentionHover = gRows.map((r, i) =>
      `<b>${ctx.groupLabel(g)}</b><br>Slice: ${r.x}<br>${ns[i]} of ${nMax} images (${retention[i].toFixed(0)}%)`);
    retentionTraces.push({ x: xVals, y: retention, color, hover: retentionHover });
  }

  // "% of images" curve - shown on every plot (even when flat at 100%) so its
  // absence doesn't read as "no data for this slice".
  for (const rt of retentionTraces) {
    traces.push({
      type: 'scatter', mode: 'lines', x: rt.x, y: rt.y, yaxis: 'y2',
      line: { width: 1, dash: 'dash', shape: 'hv', color: rt.color },
      opacity: 0.7, showlegend: false,
      hovertemplate: '%{text}<extra></extra>', text: rt.hover,
    });
  }

  const allXVals = [...new Set(agg.map(r => r.x))].sort((a, b) => a - b);
  const xaxisOverride = allXVals.length <= 8 ? { tickmode: 'array', tickvals: allXVals } : {};
  const layout = {
    ...STATS_DIMS_LAYOUT,
    xaxis: { ...STATS_DIMS_LAYOUT.xaxis, ...xaxisOverride },
    margin: { l: 36, r: 34, t: 8, b: 28 },
    height: 140,
    // Range extends a bit past 100 so a flat 100% line doesn't sit flush on the plot border.
    yaxis2: {
      overlaying: 'y', side: 'right', range: [0, 105], ticksuffix: '%',
      showgrid: false, zeroline: false, showline: false, tickfont: { size: 9 },
    },
  };
  appendPlot(container, traces, layout);
}
