const BASIC_METRIC_BASES = new Set([
  'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
]);
const QUALITY_METRIC_BASES = new Set([
  'laplacian_variance', 'tenengrad', 'brenner', 'noise_std', 'blocking_records', 'ringing_records',
]);

const COL_BG = ['#ffffff', '#f4f6f9'];

const BASIC_INFO = [
  'Shows how image statistics (e.g., mean, std, min, max) change **across different dimension slices**.',
  '', 'Useful for identifying drift, artifacts, or unexpected variation within (e.g.) T/C/Z/S dimensions.',
  '', 'You can select slices in the dropdowns to filter the tables.',
].join('\n');

const QUALITY_INFO = [
  'Shows how **image quality metrics** change across (e.g. T, C, Z, S) slices.',
  '', 'Use this view to detect:',
  '- drift in focus or noise over time (T)',
  '- channel-specific artifacts (C)',
  '- depth-dependent quality changes (Z)',
].join('\n');

async function renderAcrossDims(container, ctx, filterMetric) {
  const { q, groupCol: gcFn } = ctx.sql;
  const { append: appendPlot, niceName, LAYOUT } = ctx.plot;

  const allGroups      = detectDimMetricGroups(ctx.schema.dimMetricCols ?? []);
  const gcExpr         = gcFn();
  const activeDims     = ctx.state.dimensions ?? {};
  const filteredGroups = allGroups.filter(({ letter, items }) => {
    if (activeDims[letter] !== undefined) return false;
    for (const item of items) {
      for (const [d, idx] of Object.entries(item.allDims)) {
        if (d !== letter && activeDims[d] !== undefined && String(activeDims[d]) !== String(idx)) return false;
      }
    }
    return true;
  });

  const byMetric = {};
  for (const g of filteredGroups) {
    if (!filterMetric(g.base)) continue;
    if (!byMetric[g.base]) byMetric[g.base] = [];
    byMetric[g.base].push(g);
  }

  const metrics    = Object.keys(byMetric).sort();
  const dimLetters = [...new Set(Object.values(byMetric).flat().map(g => g.letter))].sort();

  if (!metrics.length || !dimLetters.length) {
    container.innerHTML = '<div class="no-data">No dimension slices found with the current filter.</div>';
    return;
  }

  const allCols  = [...new Set(Object.values(byMetric).flat().flatMap(g => g.items.map(i => i.col)))];
  const selParts = allCols.flatMap((c, i) => [
    `AVG(${q(c)})                       AS m_${i}`,
    `COALESCE(STDDEV_SAMP(${q(c)}), 0)  AS s_${i}`,
    `COUNT(${q(c)})                      AS n_${i}`,
  ]);

  const groupRows = await ctx.queryRows(`
    SELECT ${gcExpr} AS __group__, ${selParts.join(', ')}
    FROM pp_data ${ctx.where} GROUP BY 1
  `);
  const colIdx    = Object.fromEntries(allCols.map((c, i) => [c, i]));
  const colLookup = Object.fromEntries(allCols.map(c => [c, []]));
  for (const row of groupRows) {
    const grp = String(row.__group__);
    for (const col of allCols) {
      const i = colIdx[col];
      colLookup[col].push({ __group__: grp, y_mean: Number(row[`m_${i}`] ?? 0), y_std: Number(row[`s_${i}`] ?? 0), n: Number(row[`n_${i}`] ?? 0) });
    }
  }

  const plotJobs  = [];
  const minColPx  = 180;
  const tableWrap = document.createElement('div');
  tableWrap.style.overflowX = 'auto';
  const table = document.createElement('table');
  table.style.cssText = `border-collapse:collapse;table-layout:fixed;width:100%;min-width:${dimLetters.length * minColPx}px`;

  const thead = table.createTHead();
  const hRow  = thead.insertRow();
  for (let ci = 0; ci < dimLetters.length; ci++) {
    const th = document.createElement('th');
    th.style.cssText = `padding:8px 12px;text-align:left;border-bottom:2px solid #dee2e6;font-weight:600;background:${COL_BG[ci % 2]};${ci > 0 ? 'border-left:1px solid #dee2e6;' : ''}`;
    th.textContent = `Across '${dimLetters[ci].toUpperCase()}' slices`;
    hRow.appendChild(th);
  }

  const tbody = table.createTBody();
  for (const metric of metrics) {
    const titleRow  = tbody.insertRow();
    const titleCell = titleRow.insertCell();
    titleCell.colSpan = dimLetters.length;
    titleCell.style.cssText = 'font-weight:500;font-size:0.95rem;color:#343a40;padding:5px 10px;border-top:1px solid #e9ecef;background:#f8f9fa';
    titleCell.textContent = niceName(metric);

    const plotRow = tbody.insertRow();
    for (let ci = 0; ci < dimLetters.length; ci++) {
      const cell = plotRow.insertCell();
      cell.style.cssText = `padding:4px;vertical-align:top;background:${COL_BG[ci % 2]};${ci > 0 ? 'border-left:1px solid #dee2e6;' : ''}`;
      const match = byMetric[metric]?.find(g => g.letter === dimLetters[ci]);
      if (!match || match.items.length < 2) { cell.innerHTML = '<div style="text-align:center;color:#6c757d;padding:15px">N/A</div>'; continue; }
      const agg = match.items.flatMap(item =>
        (colLookup[item.col] ?? []).map(row => ({ __group__: row.__group__, x: item.idx, y_mean: row.y_mean, y_std: row.y_std, n: row.n }))
      );
      if (!agg.length) { cell.innerHTML = '<div style="text-align:center;color:#6c757d;padding:15px">No data</div>'; continue; }
      plotJobs.push({ cell, agg });
    }
  }

  tableWrap.appendChild(table);
  container.appendChild(tableWrap);

  const STATS_DIMS_LAYOUT = {
    ...LAYOUT,
    showlegend: false,
    xaxis: { showgrid: false, zeroline: false, showline: true, mirror: true, ticks: 'outside', title: null, tickmode: 'linear', tick0: 0, dtick: 1, tickformat: 'd' },
    yaxis: { showgrid: false, zeroline: false, showline: true, mirror: true, ticks: 'outside', title: null },
  };

  for (const { cell, agg } of plotJobs) {
    renderAggScatter(cell, agg, ctx, STATS_DIMS_LAYOUT, appendPlot);
  }
}

function makeAcrossDimsPlugin(id, label, info, filterMetric) {
  return {
    id, label, info,
    requires(schema) {
      return detectDimMetricGroups(schema.dimMetricCols ?? []).some(g => filterMetric(g.base));
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
  makeAcrossDimsPlugin('stats-across-dims-basic',   'Basic Statistics Across Dimensions', BASIC_INFO,   base => BASIC_METRIC_BASES.has(base)),
  makeAcrossDimsPlugin('stats-across-dims-quality', 'Quality Metrics Across Dimensions',  QUALITY_INFO, base => QUALITY_METRIC_BASES.has(base)),
];

function renderAggScatter(container, agg, ctx, STATS_DIMS_LAYOUT, appendPlot) {
  const traces = [];
  for (const g of ctx.groups) {
    const gRows  = agg.filter(r => String(r.__group__) === g).sort((a, b) => a.x - b.x);
    if (!gRows.length) continue;
    const color  = ctx.color.group(g);
    const xVals  = gRows.map(r => r.x);
    const yMean  = gRows.map(r => r.y_mean);
    const yStd   = gRows.map(r => r.y_std ?? 0);
    const ns     = gRows.map(r => r.n);
    const yUpper = yMean.map((m, i) => m + yStd[i]);
    const yLower = yMean.map((m, i) => m - yStd[i]);
    const sizes  = ns.map(n => Math.max(4, Math.min(12, 3 + 3 * Math.log10(Math.max(n, 1)))));
    const hover  = gRows.map((r, i) => `<b>${g}</b><br>Slice: ${r.x}<br>Mean: ${yMean[i].toFixed(3)}<br>Std: ${yStd[i].toFixed(3)}<br><b>n=${ns[i]}</b>`);
    const rgba   = ctx.color.hexToRgba(color, 0.2);
    traces.push(
      { type:'scatter', x:xVals, y:yUpper, mode:'lines', line:{width:0}, showlegend:false, hoverinfo:'skip' },
      { type:'scatter', x:xVals, y:yLower, mode:'lines', line:{width:0}, fill:'tonexty', fillcolor:rgba, showlegend:false, hoverinfo:'skip' },
      { type:'scatter', mode:'lines+markers', name:g, x:xVals, y:yMean, line:{width:2, color}, marker:{size:sizes, color, line:{width:1, color:'white'}}, hovertemplate:'%{text}<extra></extra>', text:hover },
    );
  }
  appendPlot(container, traces, { ...STATS_DIMS_LAYOUT, margin: { l:36, r:8, t:8, b:28 }, height: 140 });
}

function detectDimMetricGroups(cols) {
  const DIM_TOKEN = /_([a-z])(\d+)/g;
  const map = {};
  for (const col of cols) {
    const tokens = [...col.matchAll(DIM_TOKEN)];
    if (!tokens.length) continue;
    const base = col.replace(/_[a-z]\d+/g, '');
    if (!base) continue;
    const allDims = {};
    for (const t of tokens) allDims[t[1]] = parseInt(t[2]);
    for (const [letter, idx] of Object.entries(allDims)) {
      const key = `${base}|${letter}`;
      if (!map[key]) map[key] = [];
      map[key].push({ col, base, letter, idx, allDims });
    }
  }
  return Object.entries(map)
    .filter(([, items]) => new Set(items.map(i => i.idx)).size >= 2)
    .map(([key, items]) => {
      const [base, letter] = key.split('|');
      return { base, letter, items: [...items].sort((a, b) => a.idx - b.idx) };
    });
}
