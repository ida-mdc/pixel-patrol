const BASIC_METRIC_BASES = new Set([
  'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
]);
const QUALITY_METRIC_BASES = new Set([
  'michelson_contrast', 'mscn_variance', 'local_std_ratio',
  'laplacian_variance', 'tenengrad', 'brenner',
  'noise_std', 'blocking_records', 'ringing_records',
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

// X and Y are spatial tile coordinates. When querying non-spatial dims we
// exclude tile rows (which carry dim_x/dim_y alongside dim_c etc.) so that
// only the clean per-slice aggregated rows are scanned — the main speedup.
const SPATIAL_LETTERS = new Set(['x', 'y']);

async function renderAcrossDims(container, ctx, filterMetric) {
  const renderStartMs = performance.now();
  if (!ctx.schema.isLongFormat) {
    container.innerHTML = '<div class="no-data">This widget requires long-format data.</div>';
    return;
  }
  const { q, groupCol: gcFn } = ctx.sql;
  const { append: appendPlot, niceName, escapeHtml, LAYOUT } = ctx.plot;

  const metrics = (ctx.schema.metricCols ?? []).filter(filterMetric).sort();
  const activeDims = ctx.state.dimensions ?? {};
  // Non-spatial slice queries use obs_level = 1 (single leading dim coordinate).
  // Spatial strip queries must NOT filter obs_level: datasets often mix dim_orders
  // (YX vs CYX vs TCYX in one parquet) while dimensionInfo unions every letter that
  // appears anywhere — obs_level would be wrong per row class. Tile rows always have
  // both dim_x and dim_y; strip aggregates have exactly one spatial coord set.
  const dimLetters = Object.keys(ctx.schema.dimensionInfo ?? {})
    .filter(letter => activeDims[letter] === undefined)
    .sort();

  if (!metrics.length || !dimLetters.length) {
    container.innerHTML = '<div class="no-data">No dimension slices found with the current filter.</div>';
    return;
  }

  const baseWhere = ctx.userWhere ? ctx.userWhere.replace(/^\s*WHERE\s+/i, '') : '';
  const gcExpr   = gcFn();
  const minColPx = 180;
  const plotJobs = [];
  const fixedDims = Object.entries(activeDims)
    .map(([letter, idxRaw]) => [letter, Number(idxRaw)])
    .filter(([, idx]) => Number.isFinite(idx));

  const dimAggMap = {};
  for (const varyingLetter of dimLetters) {
    const dimStartMs = performance.now();
    const whereParts = [];
    if (baseWhere) whereParts.push(baseWhere);
    const relevantLetters = new Set([varyingLetter]);
    for (const [letter, idx] of fixedDims) {
      whereParts.push(`${q(`dim_${letter}`)} = ${idx}`);
      relevantLetters.add(letter);
    }
    whereParts.push(`${q(`dim_${varyingLetter}`)} IS NOT NULL`);

    // Restrict to pre-aggregated rows whenever possible:
    // - non-spatial dims: use the pre-aggregated rows at the expected depth
    //   (fixed dims + varying dim)
    // - across X: use x-aggregate rows (dim_x set, dim_y NULL)
    // - across Y: use y-aggregate rows (dim_y set, dim_x NULL)
    if (!SPATIAL_LETTERS.has(varyingLetter)) {
      // obs_level matches the number of fixed dims plus the varying dim.
      // Example: varying 'c' with y fixed → obs_level = 2.
      whereParts.push(`obs_level = ${fixedDims.length + 1}`);
      for (const dimCol of (ctx.schema.dimCols ?? [])) {
        const letter = dimCol.replace(/^dim_/, '');
        // Only force spatial dims to NULL if they aren't explicitly fixed.
        if (SPATIAL_LETTERS.has(letter) && !relevantLetters.has(letter)) {
          whereParts.push(`${q(dimCol)} IS NULL`);
        }
      }
    } else if (varyingLetter === 'x') {
      // Only require dim_y NULL if Y isn't explicitly fixed.
      if (!relevantLetters.has('y')) whereParts.push(`${q('dim_y')} IS NULL`);
    } else if (varyingLetter === 'y') {
      // Only require dim_x NULL if X isn't explicitly fixed.
      if (!relevantLetters.has('x')) whereParts.push(`${q('dim_x')} IS NULL`);
    }

    // For spatial strips (across x/y), avoid mixing multiple aggregated row classes.
    // Long-format parquet can contain many derived rows (tile rows, x-strip aggregates,
    // y-strip aggregates, higher-level aggregates with extra dims set, etc.). If we
    // don't constrain the "other" dimensions to NULL, we end up pooling values from
    // different slices into one x (or y) bin, producing an apparent distribution
    // even when there's only one source file per group.
    //
    // So: unless a dim is explicitly fixed in the sidebar (or is the varying dim),
    // require it to be NULL for the rows we aggregate.
    for (const dimCol of (ctx.schema.dimCols ?? [])) {
      const letter = dimCol.replace(/^dim_/, '');
      if (!relevantLetters.has(letter)) {
        whereParts.push(`${q(dimCol)} IS NULL`);
      }
    }

    const aggExprs = metrics.flatMap((metric, i) => {
      const mq = q(metric);
      return [
        `AVG(${mq}) AS m_${i}`,
        `COALESCE(STDDEV_SAMP(${mq}), 0) AS s_${i}`,
        `COUNT(${mq}) AS n_${i}`,
      ];
    }).join(',\n               ');

    const rows = await ctx.queryRows(`
      SELECT ${gcExpr} AS __group__,
             ${q(`dim_${varyingLetter}`)} AS x,
             ${aggExprs}
      FROM pp_all
      WHERE ${whereParts.join(' AND ')}
      GROUP BY 1, 2
      ORDER BY 2
    `);
    const dimQueryMs = performance.now() - dimStartMs;

    const byMetric = Object.fromEntries(metrics.map(m => [m, []]));
    for (const r of rows) {
      const x = Number(r.x);
      if (!Number.isFinite(x)) continue;
      const group = String(r.__group__);
      for (let i = 0; i < metrics.length; i++) {
        const n = Number(r[`n_${i}`] ?? 0);
        if (n <= 0) continue;
        byMetric[metrics[i]].push({
          __group__: group,
          x,
          y_mean: Number(r[`m_${i}`] ?? 0),
          y_std:  Number(r[`s_${i}`] ?? 0),
          n,
        });
      }
    }
    dimAggMap[varyingLetter] = byMetric;
    console.info(
      `[stats-across-dims] dim=${varyingLetter} query_ms=${dimQueryMs.toFixed(1)} rows=${rows.length} where="${whereParts.join(' AND ')}"`,
    );
  }

  const dimLettersVisible = dimLetters.filter((letter) => {
    const bm = dimAggMap[letter];
    if (!bm) return false;
    return metrics.some((m) => (bm[m] ?? []).some((r) => Number.isFinite(r.x)));
  });

  if (!dimLettersVisible.length) {
    container.innerHTML =
      '<div class="no-data">No dimension slices found with plottable data for the current filter.</div>';
    return;
  }

  // Group color legend — shared across all plots in this widget
  if (ctx.groups.length > 1) {
    const legendDiv = document.createElement('div');
    legendDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:12px;margin-bottom:10px;font-size:0.85rem;align-items:center';
    for (const g of ctx.groups) {
      const color = ctx.color.group(g);
      const item  = document.createElement('span');
      item.style.cssText = 'display:flex;align-items:center;gap:5px';
      item.innerHTML = `<span style="display:inline-block;width:12px;height:3px;border-radius:2px;background:${color}"></span>${escapeHtml(String(g))}`;
      legendDiv.appendChild(item);
    }
    container.appendChild(legendDiv);
  }

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
  const metricSelect = controls.querySelector('#stats-across-metric');

  const tableHost = document.createElement('div');
  container.appendChild(tableHost);

  const STATS_DIMS_LAYOUT = {
    ...LAYOUT,
    showlegend: false,
    xaxis: {
      showgrid: false,
      zeroline: false,
      showline: true,
      mirror: true,
      ticks: 'outside',
      title: null,
      tickmode: 'auto',
      nticks: 8,
      tickformat: 'd',
    },
    yaxis: { showgrid: false, zeroline: false, showline: true, mirror: true, ticks: 'outside', title: null },
  };

  function renderMetricGrid(selectedMetric) {
    tableHost.innerHTML = '';
    plotJobs.length = 0;
    const metricsToRender = selectedMetric === '__all__' ? metrics : [selectedMetric];

    const tableWrap = document.createElement('div');
    tableWrap.style.overflowX = 'auto';
    const table = document.createElement('table');
    table.style.cssText = `border-collapse:collapse;table-layout:fixed;width:100%;min-width:${dimLettersVisible.length * minColPx}px`;

    const thead = table.createTHead();
    const hRow = thead.insertRow();
    for (let ci = 0; ci < dimLettersVisible.length; ci++) {
      const th = document.createElement('th');
      th.style.cssText = `padding:8px 12px;text-align:left;border-bottom:2px solid #dee2e6;font-weight:600;background:${COL_BG[ci % 2]};${ci > 0 ? 'border-left:1px solid #dee2e6;' : ''}`;
      th.textContent = `Across '${dimLettersVisible[ci].toUpperCase()}' slices`;
      hRow.appendChild(th);
    }

    const tbody = table.createTBody();
    for (const metric of metricsToRender) {
      const titleRow = tbody.insertRow();
      const titleCell = titleRow.insertCell();
      titleCell.colSpan = dimLettersVisible.length;
      titleCell.style.cssText = 'font-weight:500;font-size:0.95rem;color:#343a40;padding:5px 10px;border-top:1px solid #e9ecef;background:#f8f9fa';
      titleCell.textContent = niceName(metric);

      const plotRow = tbody.insertRow();
      for (let ci = 0; ci < dimLettersVisible.length; ci++) {
        const varyingLetter = dimLettersVisible[ci];
        const cell = plotRow.insertCell();
        cell.style.cssText = `padding:4px;vertical-align:top;background:${COL_BG[ci % 2]};${ci > 0 ? 'border-left:1px solid #dee2e6;' : ''}`;
        const rows = (dimAggMap[varyingLetter]?.[metric] ?? [])
          .filter(r => Number.isFinite(r.x));
        if (!rows.length) {
          cell.innerHTML = '<div style="text-align:center;color:#6c757d;padding:15px">No data</div>';
          continue;
        }
        plotJobs.push({ cell, agg: rows });
      }
    }

    tableWrap.appendChild(table);
    tableHost.appendChild(tableWrap);
    for (const { cell, agg } of plotJobs) {
      renderAggScatter(cell, agg, ctx, STATS_DIMS_LAYOUT, appendPlot);
    }
  }

  metricSelect.addEventListener('change', () => {
    renderMetricGrid(metricSelect.value || metrics[0]);
  });
  renderMetricGrid('__all__');
  const totalMs = performance.now() - renderStartMs;
  console.info(
    `[stats-across-dims] total_render_ms=${totalMs.toFixed(1)} metrics=${metrics.length} dims_visible=${dimLettersVisible.length} initial_plots=${plotJobs.length}`,
  );
}

function makeAcrossDimsPlugin(id, label, info, filterMetric) {
  return {
    id, label, info, group: 'Dataset Stats',
    requires(schema) {
      return !!schema.isLongFormat && schema.metricCols.some(filterMetric);
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
      { type:'scatter', mode:'lines+markers', name:ctx.groupLabel(g), x:xVals, y:yMean, line:{width:2, color}, marker:{size:sizes, color, line:{width:1, color:'white'}}, hovertemplate:'%{text}<extra></extra>', text:hover },
    );
  }
  appendPlot(container, traces, { ...STATS_DIMS_LAYOUT, margin: { l:36, r:8, t:8, b:28 }, height: 140 });
}
