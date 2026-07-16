// Matches BasicStatsProcessor.OUTPUT_SCHEMA
const BASIC_METRIC_BASES = new Set([
  'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
]);

// Matches QualityMetricsProcessor.OUTPUT_SCHEMA + CompressionMetricsProcessor.OUTPUT_SCHEMA.
// One description per metric, used as each metric's own per-plot side note (and
// by plugin_stats_across_dims.js, the other quality-metric widget, so it isn't
// duplicated there). `hintUp`/`hintDown`/`goodDirection` are only set where the
// reading is a well-established, monotonic one - see the research this was based
// on before adding more: mscn_variance's distortion sensitivity isn't monotonic
// (noise raises it, blur lowers it) and michelson_contrast here isn't even the
// standard Michelson formula, so neither gets a direction claim.
export const QUALITY_METRIC_INFO = {
  michelson_contrast: {
    desc: 'Global contrast ratio; higher values indicate greater dynamic range.',
  },
  mscn_variance: {
    desc: 'Mean Subtracted Contrast Normalized variance; sensitive to noise and blur.',
  },
  texture_heterogeneity: {
    desc: 'Coefficient of variation of local standard deviations; captures spatial non-uniformity of texture.',
  },
  laplacian_variance: {
    desc: 'Variance of the discrete Laplacian; higher values indicate a sharper image. Scale-dependent: values vary with bit depth.',
    hintUp: 'sharper', hintDown: 'blurrier', goodDirection: 'up',
  },
  blocking_index: {
    desc: 'Strength of blocky compression artifacts (e.g. JPEG blocking).',
    hintUp: 'more blocking', hintDown: 'less blocking', goodDirection: 'down',
  },
  ringing_index: {
    desc: 'Edge oscillation artifacts around sharp boundaries, often due to compression.',
    hintUp: 'more ringing', hintDown: 'less ringing', goodDirection: 'down',
  },
};

const QUALITY_METRIC_BASES = new Set(Object.keys(QUALITY_METRIC_INFO));

function matchesBases(col, bases) {
  for (const base of bases) {
    if (col === base || col.startsWith(base + '_')) return true;
  }
  return false;
}

/** Look up a quality metric's shared metadata by column name, stripping any per-channel suffix. */
export function describeQualityMetric(col) {
  for (const [base, info] of Object.entries(QUALITY_METRIC_INFO)) {
    if (col === base || col.startsWith(base + '_')) return info;
  }
  return null;
}

/** Build a renderDistribution `sideInfo` object for a quality metric column, or null. */
function sideInfoFor(col) {
  const info = describeQualityMetric(col);
  return info && {
    text: info.desc, hintUp: info.hintUp, hintDown: info.hintDown, goodDirection: info.goodDirection,
  };
}

const SIGNIFICANCE_HELP = [
  '**Statistical Comparisons**',
  '',
  'If selected (side menu), pairwise group comparisons use the Mann–Whitney U test (a non-parametric test that makes no assumptions about the data distribution) with Bonferroni correction for multiple comparisons.',
  '',
  'Significance levels: `ns` not significant (p ≥ 0.05), `*` p < 0.05, `**` p < 0.01, `***` p < 0.001.',
].join('\n');

// Mirrors plot-engine.js MAX_VIOLIN_POINTS - kept here only for the help text.
const MAX_VIOLIN_POINTS = 5000;

const DISTRIBUTION_HELP = `Plots with ${MAX_VIOLIN_POINTS.toLocaleString()} or fewer datapoints show the actual ` +
  'distribution shape (a violin); larger plots switch to a summary box plot (quartiles, min/max, mean).';

const GRANULARITY_HELP = 'Use the **Slice by** toggles to control whether each datapoint is a whole-image ' +
  'aggregate (**per image**) or one (image × dimension) combination (**per slice**) - shown by the badge in the ' +
  'card header. More toggles mean more datapoints, so check the sample size in the plot subtitle before trusting significance.';

const BASIC_INFO = [
  'Shows **per-image intensity statistics** across groups.',
  '', DISTRIBUTION_HELP,
  '', GRANULARITY_HELP,
  '', SIGNIFICANCE_HELP,
].join('\n');

const QUALITY_INFO = [
  'Visualizes **image quality metrics** across groups.',
  '', DISTRIBUTION_HELP,
  '', GRANULARITY_HELP,
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
        sideInfo: sideInfoFor(metric),
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
// the overview-mode tile preview for a violin widget. maxRawPoints:0 forces the
// box-summary path regardless of size.
async function violinOverviewPlot(ctx, container, metric) {
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

function makeViolinPlugin(id, label, info, filterMetric, overviewMessage, metricPref = [], shortLabel, inputMetrics) {
  const pickMetric = (ctx) =>
    metricPref.find(m => ctx.schema.allCols.includes(m)) ??
    ctx.schema.metricCols.find(m => filterMetric(m) && ctx.schema.allCols.includes(m)) ??
    null;
  return {
    id, label, info, shortLabel, group: 'Dataset Stats', scope: 'image', multiPlot: true,
    inputs: inputMetrics ? [...inputMetrics] : [],
    requires(schema) {
      return schema.metricCols.some(filterMetric);
    },
    ...(overviewMessage ? { overviewMessage } : {}),
    async overviewPlot(container, ctx) {
      return violinOverviewPlot(ctx, container, pickMetric(ctx));
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

async function basicOverviewSummary(ctx) {
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
async function qualityOverviewSummary() {
  return `Spot quality differences and outliers across groups.`;
}

export default [
  makeViolinPlugin('violin-basic',   'Pixel Value Statistics', BASIC_INFO,   m => matchesBases(m, BASIC_METRIC_BASES), basicOverviewSummary,
    ['mean_intensity'], 'Intensity', BASIC_METRIC_BASES),
  makeViolinPlugin('violin-quality', 'Image Quality Metrics',  QUALITY_INFO, m => matchesBases(m, QUALITY_METRIC_BASES), qualityOverviewSummary,
    ['laplacian_variance', 'michelson_contrast', 'mscn_variance', 'texture_heterogeneity'], 'Image Quality', QUALITY_METRIC_BASES),
];

function resolveMetrics(schema, dimensions) {
  // Metrics are base columns; dim filtering is row-based via dim_* + obs_level.
  return schema.metricCols;
}
