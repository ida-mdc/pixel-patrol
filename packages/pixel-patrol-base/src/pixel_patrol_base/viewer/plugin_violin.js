// Matches BasicStatsProcessor.OUTPUT_SCHEMA
const BASIC_METRIC_BASES = new Set([
  'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
]);

// Matches QualityMetricsProcessor.OUTPUT_SCHEMA + CompressionMetricsProcessor.OUTPUT_SCHEMA
const QUALITY_METRIC_BASES = new Set([
  'michelson_contrast', 'mscn_variance', 'texture_heterogeneity', 'laplacian_variance',
  'blocking_index', 'ringing_index',
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

// Plots with at most this many total points show the actual distribution shape
// (a true violin, fetched as raw rows); larger plots fall back to a summary box
// plot computed entirely in SQL (see buildRawViolinTraces / renderViolins).
const MAX_VIOLIN_POINTS = 5000;

const DISTRIBUTION_HELP = `Plots with ${MAX_VIOLIN_POINTS.toLocaleString()} or fewer datapoints show the actual ` +
  'distribution shape (a violin); larger plots switch to a summary box plot (quartiles, min/max, ' +
  'mean) computed directly in the database for performance.';

const GRANULARITY_HELP = 'For datasets with multiple dimensions (channels, Z-planes, timepoints, ' +
  'spatial tiles, …), use the **Slice by** toggles above to control what one datapoint represents - ' +
  'shown by the **per image** / **per slice** badge in the card header. With nothing toggled, each ' +
  'point is one whole-image aggregate (**per image**). Switching a toggle on stops that dimension ' +
  'from being aggregated away, so each point becomes one (image × that dimension) combination instead ' +
  '(**per slice**) - e.g. switching on "C" gives one point per C-slice per image. Switching on ' +
  'more dimensions multiplies the number of points, so check the sample size shown in the plot ' +
  'subtitle before trusting significance results.';

// Mirrors viewer/src/scopes.js SCOPES.image / SCOPES.slice - reused here via the shared
// .widget-scope-badge style so this plugin's dynamic badge matches other widgets'.
const SCOPE_BADGES = {
  image: { icon: '🖼️', color: '#0d6efd', label: 'per image', desc: 'Each datapoint here is a whole image.' },
  slice: { icon: '🧩', color: '#fd7e14', label: 'per slice', desc: 'Each datapoint here is a slice within an image (e.g. a channel, Z-plane, timepoint, or spatial tile).' },
};
function updateScopeBadge(el, splitDims) {
  const s = splitDims.size ? SCOPE_BADGES.slice : SCOPE_BADGES.image;
  el.style.setProperty('--scope-color', s.color);
  el.title = s.desc;
  el.textContent = `${s.icon} ${s.label}`;
}

const BASIC_INFO = [
  'Shows **per-image intensity statistics** across groups.',
  '', 'You can choose which statistic to plot and filter by image dimensions.',
  '', 'Each plot shows the distribution per group: median, quartiles, min/max, and mean.',
  '', DISTRIBUTION_HELP,
  '', GRANULARITY_HELP,
  '', SIGNIFICANCE_HELP,
].join('\n');

const QUALITY_INFO = [
  'Visualizes **image quality metrics** across groups.',
  '', 'Use these plots to quickly spot outliers, compare image sets, and detect quality differences.',
  '', DISTRIBUTION_HELP,
  '', GRANULARITY_HELP,
  '', '**Metrics**',
  '- **Michelson contrast** – Global contrast ratio; higher values indicate greater dynamic range.',
  '- **MSCN variance** – Mean Subtracted Contrast Normalized variance; sensitive to noise and blur.',
  '- **Texture heterogeneity** – Coefficient of variation of local standard deviations; captures spatial non-uniformity of texture.',
  '- **Laplacian variance** – Variance of the discrete Laplacian; higher values indicate a sharper image. Scale-dependent: values vary with bit depth.',
  '- **Blocking index** – Strength of blocky compression artifacts (e.g. JPEG blocking).',
  '- **Ringing index** – Edge oscillation artifacts around sharp boundaries, often due to compression.',
  '', SIGNIFICANCE_HELP,
].join('\n');

// _rollup has one row per subset of dims fixed (the power-set), with obs_level
// = subset size and every other dim_* NULL. obs_level = dimFilters.size +
// splitDims.size plus IS NULL/IS NOT NULL on the remaining dim_* columns
// pins down exactly the subset (dimFilters ∪ splitDims).
function buildViolinWhereParts(ctx, q, splitDims, dimFilters, activeDims) {
  const whereParts = [];
  const baseWhere = ctx.userWhere ? ctx.userWhere.replace(/^\s*WHERE\s+/i, '') : '';
  if (baseWhere) whereParts.push(baseWhere);

  const obsLevel = dimFilters.length + splitDims.size;
  whereParts.push(`obs_level = ${obsLevel}`);
  whereParts.push(...dimFilters);

  for (const col of ctx.schema?.dimCols ?? []) {
    const letter = col.slice(4);
    if (letter in activeDims) continue; // already constrained via dimFilters above
    whereParts.push(splitDims.has(letter) ? `${q(col)} IS NOT NULL` : `${q(col)} IS NULL`);
  }
  return whereParts;
}

async function renderViolins(plotRoot, ctx, filterMetric, splitDims) {
  const { q, groupCol: gcFn, groupExpr: geFn, andWhere } = ctx.sql;
  const { append: appendPlot, flexGrid: createFlexGrid, niceName, dataAvailabilityWarning } = ctx.plot;

  const metrics = resolveMetrics(ctx.schema, ctx.state.dimensions)
    .filter(filterMetric);

  if (!metrics.length) {
    plotRoot.innerHTML = '<div class="no-data">No numeric metric columns.</div>';
    return;
  }

  const metricSelects = metrics.map((m, i) => `COUNT(${q(m)}) AS c${i}`).join(', ');
  const [availRow] = await ctx.queryRows(
    `SELECT COUNT(*) AS total, ${metricSelects} FROM pp_data ${ctx.where}`
  );
  const tot = Number(availRow.total);
  const counts = metrics.map((m, i) => ({ label: niceName(m), present: Number(availRow[`c${i}`]) }));
  dataAvailabilityWarning(plotRoot, counts, tot, { unit: splitDims.size ? 'slices' : 'images' });

  const dimFilters = Object.entries(ctx.state.dimensions ?? {})
    .map(([letter, idxRaw]) => {
      const idx = Number(idxRaw);
      if (!Number.isFinite(idx)) return null;
      return `${q(`dim_${letter}`)} = ${idx}`;
    })
    .filter(Boolean);
  const activeDims = ctx.state.dimensions ?? {};

  const sourceTable = 'pp_all';
  const whereParts = buildViolinWhereParts(ctx, q, splitDims, dimFilters, activeDims);
  const combinedWhere = whereParts.length ? `WHERE ${whereParts.join(' AND ')}` : '';
  const gc = gcFn();

  // Per-group summary stats (quartiles, min/max, mean) computed entirely in SQL,
  // so the box plots scale to any number of underlying rows - same as the bar plots.
  const statCols = metrics.map(m => `
    COUNT(${q(m)}) AS "${m}__n",
    MIN(${q(m)}) AS "${m}__min",
    MAX(${q(m)}) AS "${m}__max",
    AVG(${q(m)}) AS "${m}__mean",
    approx_quantile(${q(m)}, 0.25) AS "${m}__q1",
    approx_quantile(${q(m)}, 0.5) AS "${m}__median",
    approx_quantile(${q(m)}, 0.75) AS "${m}__q3"`).join(',\n');

  const statRows = await ctx.queryRows(`
    SELECT ${geFn()}, ${statCols}
    FROM ${sourceTable} ${combinedWhere}
    GROUP BY ${gc}
  `);

  if (!statRows.length) {
    plotRoot.innerHTML += '<div class="no-data">No rows match the current filter.</div>';
    return;
  }

  const statsByGroup = new Map(statRows.map(r => [String(r.__group__), r]));
  const groups = ctx.groups.filter(g => statsByGroup.has(g));

  const toPlot = [], noVariance = [];
  for (const metric of metrics) {
    let total = 0, gmin = Infinity, gmax = -Infinity;
    for (const g of groups) {
      const r = statsByGroup.get(g);
      const n = Number(r[`${metric}__n`]);
      total += n;
      if (n > 0) {
        gmin = Math.min(gmin, Number(r[`${metric}__min`]));
        gmax = Math.max(gmax, Number(r[`${metric}__max`]));
      }
    }
    if (total === 0) continue;
    if (gmin === gmax) noVariance.push({ metric, value: gmin });
    else toPlot.push({ metric, total });
  }

  const numGroups   = groups.length;
  const plotsPerRow = numGroups <= 2 ? 3 : numGroups === 3 ? 2 : 1;
  const showSig = ctx.state.showSignificance && groups.length >= 2;

  // Pairwise Mann-Whitney rank sums, computed in SQL (one query per group pair,
  // covering all metrics at once via UNION ALL + window functions).
  const rankData = (showSig && toPlot.length)
    ? await fetchRankSums(ctx, q, andWhere, gc, combinedWhere, sourceTable, toPlot.map(t => t.metric), groups)
    : null;

  if (toPlot.length) {
    const { wrap, flexBasisPct } = createFlexGrid(plotRoot, plotsPerRow);

    const granularityDesc = splitDims.size
      ? `one point per (file × ${[...splitDims].map(l => l.toUpperCase()).join(' × ')})`
      : 'one point per file';

    for (const { metric, total } of toPlot) {
      const label = niceName(metric);
      const title = `Distribution of ${label}<br><sup>${granularityDesc}; n=${total.toLocaleString()}</sup>`;

      const traces = total <= MAX_VIOLIN_POINTS
        ? await fetchRawViolinTraces(ctx, q, andWhere, gc, combinedWhere, sourceTable, metric, groups)
        : groups.map(g => {
            const r = statsByGroup.get(g);
            return {
              type: 'box', name: ctx.groupLabel(g),
              x: [ctx.groupLabel(g)],
              q1: [Number(r[`${metric}__q1`])],
              median: [Number(r[`${metric}__median`])],
              q3: [Number(r[`${metric}__q3`])],
              lowerfence: [Number(r[`${metric}__min`])],
              upperfence: [Number(r[`${metric}__max`])],
              mean: [Number(r[`${metric}__mean`])],
              boxmean: true, boxpoints: false, opacity: 0.9,
              marker: { color: ctx.color.group(g), line: { width: 1, color: 'black' } },
              hovertemplate: '<b>Group:</b> %{x}' +
                '<br><b>Median:</b> %{median:.2f}<br><b>Q1:</b> %{q1:.2f}<br><b>Q3:</b> %{q3:.2f}' +
                '<br><b>Min:</b> %{lowerfence:.2f}<br><b>Max:</b> %{upperfence:.2f}<extra></extra>',
            };
          });

      const outerDiv = appendPlot(wrap, traces, {
        title: { text: title },
        yaxis: { title: label },
        xaxis: { title: ctx.plot.groupingLabel ? ctx.plot.groupingLabel('') : '', type: 'category' },
        violinmode: groups.length > 1 ? 'group' : undefined,
        showlegend: false,
      }, `flex:0 0 ${flexBasisPct}%;min-width:300px;margin-bottom:20px;box-sizing:border-box`);

      if (showSig && rankData) {
        const pairs = computeSignificancePairsFromRanks(groups, rankData[metric] ?? {});
        addSignificanceBrackets(outerDiv, pairs, groups);
      }
    }
  }

  if (noVariance.length) {
    const hr = document.createElement('hr');
    plotRoot.appendChild(hr);
    const h = document.createElement('h6');
    h.style.cssText = 'margin-top:20px;margin-bottom:12px';
    h.textContent = 'Metrics with No Variance across all files that report it';
    plotRoot.appendChild(h);
    const table = document.createElement('table');
    table.className = 'stat-table';
    table.innerHTML = `
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>${noVariance.map(({ metric, value }) => `<tr><td>${niceName(metric)}</td><td>${Number(value).toFixed(4)}</td></tr>`).join('')}</tbody>
    `;
    plotRoot.appendChild(table);
  }
}

// Raw per-row values for `metric`, one Plotly violin trace per group - used
// instead of the SQL-aggregate box summary when total rows <= MAX_VIOLIN_POINTS.
async function fetchRawViolinTraces(ctx, q, andWhere, gc, combinedWhere, sourceTable, metric, groups) {
  const rows = await ctx.queryRows(`
    SELECT ${gc} AS __group__, ${q(metric)} AS val
    FROM ${sourceTable} ${andWhere(combinedWhere, `${q(metric)} IS NOT NULL`)}
  `);
  return groups.map(g => {
    const vals = rows.filter(r => String(r.__group__) === g).map(r => Number(r.val));
    return {
      type: 'violin', name: ctx.groupLabel(g),
      x: vals.map(() => ctx.groupLabel(g)), y: vals,
      box: { visible: true }, meanline: { visible: true }, points: 'outliers',
      spanmode: 'hard', opacity: 0.9,
      marker: { color: ctx.color.group(g) },
      hovertemplate: '<b>Group:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>',
    };
  }).filter(t => t.y.length);
}

// Pairwise Mann-Whitney rank sums, computed in SQL: rankData[metric][`${g1}|${g2}`]
// = { [group]: { n, rankSum } }, one query per group pair covering all metrics.
async function fetchRankSums(ctx, q, andWhere, gc, combinedWhere, sourceTable, metrics, groups) {
  const esc = v => `'${String(v).replace(/'/g, "''")}'`;
  const rankData = {};
  for (const m of metrics) rankData[m] = {};

  for (let i = 0; i < groups.length; i++) {
    for (let j = i + 1; j < groups.length; j++) {
      const g1 = groups[i], g2 = groups[j];
      const pairWhere = andWhere(combinedWhere, `CAST(${gc} AS VARCHAR) IN (${esc(g1)}, ${esc(g2)})`);
      const unions = metrics.map(m =>
        `SELECT CAST(${gc} AS VARCHAR) AS grp, '${m.replace(/'/g, "''")}' AS metric, ${q(m)}::DOUBLE AS v
         FROM ${sourceTable} ${pairWhere} AND ${q(m)} IS NOT NULL`
      ).join('\nUNION ALL\n');

      const rows = await ctx.queryRows(`
        WITH unioned AS (${unions}),
        ranked AS (
          SELECT metric, grp, v, ROW_NUMBER() OVER (PARTITION BY metric ORDER BY v) AS rn
          FROM unioned
        ), avg_ranked AS (
          SELECT metric, grp, AVG(rn) OVER (PARTITION BY metric, v) AS rnk
          FROM ranked
        )
        SELECT metric, grp, COUNT(*) AS n, SUM(rnk) AS rank_sum
        FROM avg_ranked GROUP BY metric, grp
      `);

      const key = `${g1}|${g2}`;
      for (const r of rows) {
        const m = String(r.metric);
        rankData[m][key] = rankData[m][key] || {};
        rankData[m][key][String(r.grp)] = { n: Number(r.n), rankSum: Number(r.rank_sum) };
      }
    }
  }
  return rankData;
}

function makeViolinPlugin(id, label, info, filterMetric) {
  return {
    id, label, info, group: 'Dataset Stats', scope: 'image',
    requires(schema) {
      return !!schema.isLongFormat && schema.metricCols.some(filterMetric);
    },
    async render(container, ctx) {
      const dimCols = ctx.schema?.dimCols ?? [];
      const activeDims = ctx.state.dimensions ?? {};
      const splittable = dimCols.map(c => c.slice(4)).filter(letter => !(letter in activeDims));

      const splitDims = new Set();
      const plotRoot = document.createElement('div');

      // The card header already carries a "🖼️ per image" badge (from `scope: 'image'`
      // above) - keep it in sync with the toggles instead of adding a second badge.
      const headerBadge = container.closest('.widget-card')?.querySelector('.widget-scope-badge');
      if (headerBadge) updateScopeBadge(headerBadge, splitDims);

      const draw = async () => {
        plotRoot.innerHTML = '';
        try {
          await renderViolins(plotRoot, ctx, filterMetric, splitDims);
        } catch {
          plotRoot.innerHTML = '<div class="no-data">Failed to load data.</div>';
        }
      };

      if (splittable.length) {
        const controlRow = document.createElement('div');
        controlRow.className = 'violin-controls';
        controlRow.innerHTML = `<span class="violin-controls-label">Slice by:</span>`;

        for (const letter of splittable) {
          const sw = document.createElement('label');
          sw.className = 'dim-switch';
          sw.innerHTML = `<input type="checkbox"><span class="dim-switch-track"></span><span class="dim-switch-label">${letter.toUpperCase()}</span>`;
          sw.querySelector('input').addEventListener('change', e => {
            if (e.target.checked) splitDims.add(letter); else splitDims.delete(letter);
            if (headerBadge) updateScopeBadge(headerBadge, splitDims);
            draw();
          });
          controlRow.appendChild(sw);
        }

        container.appendChild(controlRow);
      }

      container.appendChild(plotRoot);
      await draw();
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

function mannWhitneyPFromRankSum(n1, n2, rankSum1) {
  if (n1 < 3 || n2 < 3) return 1.0;
  const U1  = rankSum1 - n1 * (n1 + 1) / 2;
  const mu  = n1 * n2 / 2;
  const sig = Math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12);
  if (sig === 0) return 1.0;
  return 2 * (1 - normalCDF(Math.abs(U1 - mu) / sig));
}

function computeSignificancePairsFromRanks(groups, metricRankData) {
  const pairs = [];
  for (let i = 0; i < groups.length; i++) {
    for (let j = i + 1; j < groups.length; j++) {
      const g1 = groups[i], g2 = groups[j];
      const d = metricRankData[`${g1}|${g2}`];
      const p = (d && d[g1] && d[g2]) ? mannWhitneyPFromRankSum(d[g1].n, d[g2].n, d[g1].rankSum) : 1.0;
      pairs.push({ g1, g2, p });
    }
  }
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
