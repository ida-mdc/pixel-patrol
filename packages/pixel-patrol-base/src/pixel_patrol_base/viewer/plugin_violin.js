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

// Mirrors plot-engine.js MAX_VIOLIN_POINTS - kept here only for the help text.
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

async function renderViolins(plotRoot, ctx, filterMetric, splitDims) {
  const { q, groupExpr: geFn, groupCol: gcFn } = ctx.sql;
  const { flexGrid: createFlexGrid, niceName, dataAvailabilityWarning, groupingLabel, engine } = ctx.plot;

  const metrics = resolveMetrics(ctx.schema, ctx.state.dimensions).filter(filterMetric);
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

  const fixed = {};
  for (const [letter, idxRaw] of Object.entries(ctx.state.dimensions ?? {})) {
    const idx = Number(idxRaw);
    if (Number.isFinite(idx)) fixed[letter] = idx;
  }

  const sourceTable = 'pp_all';
  // Pin the (fixed ∪ split) aggregation subset: each violin point is one image
  // aggregate, or one (image × split dim) when slicing is toggled on.
  const whereParts = ctx.sql.dimSubsetWhere({ fixed, split: splitDims });
  const combinedWhere = whereParts.length ? `WHERE ${whereParts.join(' AND ')}` : '';
  const gc = gcFn();

  // Per-group summary stats (quartiles, min/max, mean) computed entirely in SQL,
  // so the box-plot fallback scales to any number of underlying rows. Batched
  // across every metric in one query, then sliced per metric below.
  const statCols = metrics.map(m => `
    COUNT(${q(m)}) AS "${m}__n",
    MIN(${q(m)}) AS "${m}__min",
    MAX(${q(m)}) AS "${m}__max",
    AVG(${q(m)}) AS "${m}__mean",
    STDDEV(${q(m)}) AS "${m}__sd",
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
  const showSignificance = !!ctx.state.showSignificance;

  if (toPlot.length) {
    const { wrap, flexBasisPct } = createFlexGrid(plotRoot, plotsPerRow);

    const granularityDesc = splitDims.size
      ? `one point per (image × ${[...splitDims].map(l => l.toUpperCase()).join(' × ')})`
      : 'one point per image';

    for (const { metric, total } of toPlot) {
      const label = niceName(metric);
      const title = `Distribution of ${label}<br><sup>${granularityDesc}; n=${total.toLocaleString()}</sup>`;

      // Pre-sliced per-group stats for this metric, so renderDistribution can
      // draw the box-summary fallback without re-querying. catSql = the group
      // expression, so each group is its own colored violin (series.isCategory).
      const stats = groups.map(g => {
        const r = statsByGroup.get(g);
        return {
          __group__: g, __cat__: g,
          n: r[`${metric}__n`], mn: r[`${metric}__min`], mx: r[`${metric}__max`],
          mean: r[`${metric}__mean`], sd: r[`${metric}__sd`],
          q1: r[`${metric}__q1`], med: r[`${metric}__median`], q3: r[`${metric}__q3`],
        };
      });

      const cell = document.createElement('div');
      cell.style.cssText = `flex:0 0 ${flexBasisPct}%;min-width:300px;margin-bottom:20px;box-sizing:border-box`;
      wrap.appendChild(cell);

      await engine.renderDistribution(cell, ctx, {
        numCol: metric,
        source: { table: sourceTable, where: combinedWhere },
        catSql: gcFn(),
        catLabel: groupingLabel(''),
        yLabel: label,
        title,
        showSignificance,
        maxRawPoints: engine.CONSTANTS.MAX_VIOLIN_POINTS,
        series: { isCategory: true },
        categoriesOrder: groups,
        catLabelFn: ctx.groupLabel,
        stats,
      });
    }
  }

  if (noVariance.length) {
    ctx.plot.invariantTable(plotRoot, {
      title: 'Metrics with No Variance across all files that report it',
      headers: ['Metric', 'Value'],
      rows: noVariance.map(({ metric, value }) => [niceName(metric), Number(value).toFixed(4)]),
      hr: true,
    });
  }
}

// One simplified box plot per group, computed entirely in SQL (so it scales) -
// the condensed-mode tile preview for a violin widget. maxRawPoints:0 forces the
// box-summary path regardless of size.
async function violinCondensedPlot(ctx, container, metric) {
  if (!metric) return false;
  return ctx.plot.engine.renderDistribution(container, ctx, {
    numCol: metric,
    source: { table: 'pp_data', where: ctx.where },
    catSql: ctx.sql.groupCol(),
    yLabel: ctx.plot.niceName(metric),
    series: { isCategory: true },
    categoriesOrder: ctx.groups,
    catLabelFn: ctx.groupLabel,
    mini: true,
    maxRawPoints: 0,
  });
}

function makeViolinPlugin(id, label, info, filterMetric, condensedSummary, metricPref = [], shortLabel, inputMetrics) {
  const pickMetric = (ctx) =>
    metricPref.find(m => ctx.schema.allCols.includes(m)) ??
    ctx.schema.metricCols.find(m => filterMetric(m) && ctx.schema.allCols.includes(m)) ??
    null;
  return {
    id, label, info, shortLabel, group: 'Dataset Stats', scope: 'image', multiPlot: true,
    inputs: inputMetrics ? [...inputMetrics] : [],
    requires(schema) {
      return !!schema.isLongFormat && schema.metricCols.some(filterMetric);
    },
    ...(condensedSummary ? { condensedSummary } : {}),
    async condensedPlot(container, ctx) {
      return violinCondensedPlot(ctx, container, pickMetric(ctx));
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
      const syncBadge = () => ctx.plot.setScopeBadge(headerBadge, splitDims.size ? 'slice' : 'image');
      syncBadge();

      const draw = async () => {
        plotRoot.innerHTML = '';
        try {
          await renderViolins(plotRoot, ctx, filterMetric, splitDims);
        } catch {
          plotRoot.innerHTML = '<div class="no-data">Failed to load data.</div>';
        }
      };

      if (splittable.length) {
        container.appendChild(ctx.plot.sliceToggles(splittable, splitDims, () => { syncBadge(); draw(); }));
      }

      container.appendChild(plotRoot);
      await draw();
    },
  };
}

async function basicCondensedSummary(ctx) {
  try {
    if (!ctx.schema.allCols.includes('mean_intensity')) return null;
    // The one thing worth flagging up front is mixed pixel types, which make raw
    // intensities non-comparable across images. Otherwise stay neutral and just
    // tease what the widget lets you inspect.
    if (ctx.schema.allCols.includes('dtype')) {
      const [row] = await ctx.queryRows(`SELECT COUNT(DISTINCT "dtype") AS n_dtypes FROM pp_data ${ctx.where}`);
      if (Number(row?.n_dtypes ?? 1) > 1) {
        return { text: `Mixed pixel types — brightness not comparable.`, warning: true };
      }
    }
    return `Compare per-image intensity statistics across groups.`;
  } catch { return null; }
}

// Stay neutral: the old per-metric "varies a lot / looks consistent" verdicts were
// opaque (which metric, measured how?). Just tease what the widget lets you compare.
async function qualityCondensedSummary() {
  return `Spot quality differences and outliers across groups.`;
}

export default [
  makeViolinPlugin('violin-basic',   'Pixel Value Statistics', BASIC_INFO,   m => matchesBases(m, BASIC_METRIC_BASES), basicCondensedSummary,
    ['mean_intensity'], 'Intensity', BASIC_METRIC_BASES),
  makeViolinPlugin('violin-quality', 'Image Quality Metrics',  QUALITY_INFO, m => matchesBases(m, QUALITY_METRIC_BASES), qualityCondensedSummary,
    ['laplacian_variance', 'michelson_contrast', 'mscn_variance', 'texture_heterogeneity'], 'Image Quality', QUALITY_METRIC_BASES),
];

function resolveMetrics(schema, dimensions) {
  if (!schema.isLongFormat) return [];
  // Long format keeps metrics as base columns; dim filtering is row-based via dim_* + obs_level.
  return schema.metricCols;
}
