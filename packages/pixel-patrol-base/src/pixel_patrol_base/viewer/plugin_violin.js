// Matches BasicStatsProcessor.OUTPUT_SCHEMA
const BASIC_METRIC_BASES = new Set([
  'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
]);

// Matches QualityMetricsProcessor.OUTPUT_SCHEMA.
// One description per metric, used as each metric's own per-plot side note (and
// by plugin_stats_across_dims.js, the other quality-metric widget, so it isn't
// duplicated there). `hintUp`/`hintDown`/`goodDirection` are only set where the
// reading is a well-established, monotonic one.
export const QUALITY_METRIC_INFO = {
  laplacian_variance: {
    desc: 'High-frequency content measure. Low values reliably indicate blur or heavy compression — ' +
          'both suppress high spatial frequencies. High values are ambiguous: genuine sharpness and ' +
          'fine detail produce high scores, but so do noise, quantization artifacts (false edges from ' +
          'too few color levels), and transmission errors. Clipped pixels create artificial edges at ' +
          'clip boundaries and inflate the score — interpret with caution whenever clipping fractions ' +
          'are elevated.',
    hintDown: 'blurrier / more compressed',
  },
  noise_mad: {
    desc: 'Noise standard deviation estimated from the diagonal detail subband of a single-level ' +
          'Haar wavelet (Donoho & Johnstone 1994 MAD estimator). The median discards the sparse ' +
          'large coefficients that represent real image structure, making this robust to texture — ' +
          'unlike mean-based estimators. For integer images an Anscombe transform is applied first ' +
          'to handle Poisson noise (dominant in photon-limited scientific imaging such as fluorescence ' +
          'microscopy). Comparable within a batch of similar images; the absolute value depends on ' +
          'bit depth and signal scale.',
    hintUp: 'noisier', hintDown: 'cleaner', goodDirection: 'down',
  },
  spectral_slope: {
    desc: 'Log-log slope of the radially averaged power spectrum, fit over the mid-frequency band ' +
          '(5–40% of Nyquist). Blur concentrates energy at low frequencies and steepens the slope ' +
          '(more negative values). Noise adds flat energy across all frequencies and flattens the ' +
          'slope (toward zero). This is the only metric that maps blur and noise to opposite ' +
          'directions on a single axis — laplacian_variance cannot make this distinction. ' +
          'Typical range for well-focused low-noise images: −2 to −4.',
    hintDown: 'blurrier / more compressed', hintUp: 'noisier / flatter spectrum',
  },
  dark_clipping_fraction: {
    desc: 'Fraction of pixels at the dtype\'s minimum representable value (0 for unsigned integers; ' +
          'NaN for float, which has no fixed lower bound). Detects underexposure, background, and ' +
          'nodata at the dark end. In scientific data this may be expected (ocean nodata in elevation ' +
          'rasters, dark background in fluorescence) or a sign of clipping below the dynamic range.',
    hintUp: 'more dark-clipped pixels', hintDown: 'fewer dark-clipped pixels', goodDirection: 'down',
  },
  bright_clipping_fraction: {
    desc: 'Fraction of pixels at this image\'s own observed maximum value (all dtypes, NaN excluded). ' +
          'Detects bright-end clipping regardless of storage format or bit depth: catches uint8 at 255, ' +
          'uint16 at 65535, and also sub-dtype clipping such as 12-bit data stored in uint16 (clips at ' +
          '4095, not 65535). For float images, where there is no fixed ceiling, this is the only ' +
          'available proxy for bright-end clipping.',
    hintUp: 'more bright-clipped pixels', hintDown: 'fewer bright-clipped pixels', goodDirection: 'down',
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

// QUALITY_METRIC_INFO's key order is curated by generalization/importance (sharpness,
// noise, exposure clipping, contrast/texture, then compression artifacts last - most
// niche/situational). Parquet column order is alphabetical (an internal processing
// detail unrelated to this), so widgets sort explicitly by this instead.
const QUALITY_METRIC_ORDER = Object.keys(QUALITY_METRIC_INFO);

/** Sort rank for a metric column by curated importance order; ties (e.g. non-quality
 * columns) sort last and preserve their original relative order (stable sort). */
export function qualityMetricRank(col) {
  const idx = QUALITY_METRIC_ORDER.findIndex(base => col === base || col.startsWith(base + '_'));
  return idx === -1 ? Number.MAX_SAFE_INTEGER : idx;
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

// Thresholds for fraction metrics where an absolute reading is meaningful - these
// are normalized 0-1 regardless of dtype or image content, so the numbers mean
// the same thing in every domain. Content-dependent metrics (laplacian_variance,
// estimated_noise_std, etc.) deliberately get no thresholds.
const FRACTION_WARNINGS = [
  {
    col: 'bright_clipping_fraction',
    threshold: 0.01,
    label: 'at their own image maximum',
    consequence: 'This may indicate clipping at the bright end. If the dtype is integer and these ' +
                 'images are NOT at the dtype ceiling (e.g. uint16 max pixel is 4095, not 65535), ' +
                 'the data is likely stored in a wider format than its true bit depth -- ' +
                 'clipping occurred at the sensor level, not the container level.',
  },
  {
    col: 'dark_clipping_fraction',
    threshold: 0.05,
    // label/consequence used only as fallback when dtype column is absent.
    label: 'at minimum value',
    consequence: 'This may indicate background, missing values, or underexposure.',
  },
];

/** Return the minimum representable integer value for a dtype string like 'uint8', 'int16'. */
function dtypeMinValue(dtype) {
  if (dtype.startsWith('uint')) return 0;
  const bits = parseInt(dtype.replace(/^[a-z]+/, ''), 10);
  return Number.isFinite(bits) ? -(2 ** (bits - 1)) : null;
}

/** Check fraction metrics against absolute thresholds and prepend inline warnings. */
async function prependFractionWarnings(plotRoot, ctx) {
  const { q, andWhere } = ctx.sql;
  const cols = ctx.schema.allCols;

  // Clipping inflates laplacian_variance via artificial clip-boundary edges.
  if (cols.includes('laplacian_variance')) {
    const clipParts = [];
    if (cols.includes('bright_clipping_fraction')) clipParts.push(`"bright_clipping_fraction" > 0.01`);
    if (cols.includes('dark_clipping_fraction')) clipParts.push(`"dark_clipping_fraction" > 0.05`);
    if (clipParts.length) {
      const clipWhere = andWhere(ctx.where, `(${clipParts.join(' OR ')})`);
      const [clipRow] = await ctx.queryRows(`SELECT COUNT(*) AS n FROM pp_data ${clipWhere}`);
      const nClip = Number(clipRow?.n ?? 0);
      if (nClip > 0) {
        ctx.plot.prependWarning(plotRoot, {
          level: 'yellow',
          html: `<strong>${nClip.toLocaleString()} image${nClip === 1 ? '' : 's'}</strong> ` +
                `have clipped pixels. <strong>laplacian_variance</strong> is inflated by ` +
                `artificial edges at clip boundaries — interpret with caution for those images.`,
        });
      }
    }
  }

  // Fraction metrics (saturated, min-value)
  for (const { col, threshold, label, consequence } of FRACTION_WARNINGS) {
    if (!cols.includes(col)) continue;
    const pct = (threshold * 100).toLocaleString(undefined, { maximumSignificantDigits: 2 });
    const where = andWhere(ctx.where, `${q(col)} > ${threshold}`);
    // min_value_pixel_fraction: split unsigned (zero = background/void/underexposure) from
    // signed (min = lower-bound clipping, signal loss) -- they mean different things.
    if (col === 'dark_clipping_fraction' && cols.includes('dtype')) {
      const uWhere = andWhere(where, `"dtype" LIKE 'uint%'`);
      const sWhere = andWhere(where, `"dtype" NOT LIKE 'uint%' AND "dtype" LIKE '%int%'`);
      const [[uRow], [sRow]] = await Promise.all([
        ctx.queryRows(`SELECT COUNT(*) AS n FROM pp_data ${uWhere}`),
        ctx.queryRows(`SELECT COUNT(*) AS n FROM pp_data ${sWhere}`),
      ]);
      const nu = Number(uRow?.n ?? 0), ns = Number(sRow?.n ?? 0);
      if (nu > 0) {
        ctx.plot.prependWarning(plotRoot, {
          level: 'yellow',
          html: `<strong>${nu.toLocaleString()} image${nu === 1 ? '' : 's'}</strong> ` +
                `have more than ${pct}% of pixels at value 0. ${consequence}`,
        });
      }
      if (ns > 0) {
        const dtypeRows = await ctx.queryRows(`SELECT DISTINCT "dtype" FROM pp_data ${sWhere}`);
        const vals = [...new Set(dtypeRows.map(r => dtypeMinValue(r.dtype)).filter(v => v !== null))];
        const valStr = vals.length ? ` (${vals.length === 1 ? 'value' : 'values'} ${vals.sort((a, b) => a - b).join(', ')})` : '';
        ctx.plot.prependWarning(plotRoot, {
          level: 'yellow',
          html: `<strong>${ns.toLocaleString()} image${ns === 1 ? '' : 's'}</strong> ` +
                `have more than ${pct}% of pixels clipped at the lower bound${valStr}. Signal in those pixels may have been lost.`,
        });
      }
      continue;
    }

    const [row] = await ctx.queryRows(`SELECT COUNT(*) AS n FROM pp_data ${where}`);
    const n = Number(row?.n ?? 0);
    if (n === 0) continue;
    ctx.plot.prependWarning(plotRoot, {
      level: 'yellow',
      html: `<strong>${n.toLocaleString()} image${n === 1 ? '' : 's'}</strong> ` +
            `have more than ${pct}% of pixels ${label}. ${consequence}`,
    });
  }
}

async function renderViolins(plotRoot, ctx, filterMetric, splitDims, fractionWarnings = false) {
  const { q, groupExpr: geFn, groupCol: gcFn } = ctx.sql;
  const { flexGrid: createFlexGrid, niceName, dataAvailabilityWarning, groupingLabel, engine } = ctx.plot;

  const metrics = resolveMetrics(ctx.schema, ctx.state.dimensions).filter(filterMetric);
  metrics.sort((a, b) => qualityMetricRank(a) - qualityMetricRank(b));
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
  if (fractionWarnings) await prependFractionWarnings(plotRoot, ctx);

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
  let plotsPerRow = numGroups <= 2 ? 3 : numGroups === 3 ? 2 : 1;
  const showSignificance = !!ctx.state.showSignificance;

  // Side-info text (description + optional direction badge) sits beside the
  // chart when there's room, but wraps above it when the cell is too narrow
  // (see appendSideInfoRow) - stacked across several narrow columns that looks
  // cramped and uneven. Cap at 2 per row whenever it'll actually render, so
  // each cell has enough width for text and chart side by side instead.
  const anySideInfo = ctx.state.showInfo && toPlot.some(({ metric }) => sideInfoFor(metric));
  if (anySideInfo) plotsPerRow = Math.min(plotsPerRow, 2);

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

function makeViolinPlugin(id, label, info, filterMetric, overviewMessage, metricPref = [], shortLabel, inputMetrics, { fractionWarnings = false } = {}) {
  const pickMetric = (ctx) =>
    metricPref.find(m => ctx.schema.allCols.includes(m)) ??
    ctx.schema.metricCols
      .filter(m => filterMetric(m) && ctx.schema.allCols.includes(m))
      .sort((a, b) => qualityMetricRank(a) - qualityMetricRank(b))[0] ??
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
          await renderViolins(plotRoot, ctx, filterMetric, splitDims, fractionWarnings);
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
    ['laplacian_variance', 'ringing_index', 'estimated_noise_std'], 'Image Quality', QUALITY_METRIC_BASES,
    { fractionWarnings: true }),
];

function resolveMetrics(schema, dimensions) {
  // Metrics are base columns; dim filtering is row-based via dim_* + obs_level.
  return schema.metricCols;
}
