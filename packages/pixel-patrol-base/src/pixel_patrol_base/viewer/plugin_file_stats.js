export const MIXED_GROUP_WARNING = 'Some files contain images from multiple groups - file count and size cannot be attributed to a single group.';

const SIZE_NUM_BINS      = 20;
const SIZE_LOG_THRESHOLD = 30;
const MAX_DAYS           = 20;
const FILE_STATS_MARGIN  = { l: 50, r: 80, t: 50, b: 80 };

const MS_SECOND = 1000;
const MS_MINUTE = 60 * MS_SECOND;
const MS_HOUR   = 60 * MS_MINUTE;
const MS_DAY    = 24 * MS_HOUR;

export default {
  id: 'file-stats',
  required_inputs: ['file_extension', 'size_bytes'],
  inputs: ['modification_date'],
  group: 'File Stats',
  scope: 'file',
  multiPlot: true,
  info: [
    'High-level **file statistics** for the dataset.',
    '',
    'If a property has **no variance** (e.g. all files share the same extension), it is summarized in the table instead of a chart.',
  ].join('\n'),
  label: 'File Statistics',
  shortLabel: 'File Metadata',

  requires(schema) {
    return schema.allCols.includes('file_extension') &&
           schema.allCols.includes('size_bytes');
  },

  async overviewMessage(ctx) {
    try {
      if (ctx.withinFileGroupVariation) {
        return { text: MIXED_GROUP_WARNING, warning: true };
      }
      const { groupCol: gcFn, fileCount } = ctx.sql;
      const { escapeHtml } = ctx.plot;
      const [extRows, groupRows] = await Promise.all([
        ctx.queryRows(`SELECT DISTINCT COALESCE("file_extension", '(none)') AS ext FROM pp_data ${ctx.where}`),
        ctx.queryRows(`SELECT ${gcFn()} AS g, ${fileCount()} AS c FROM pp_data ${ctx.where} GROUP BY 1`),
      ]);
      const exts   = [...new Set(extRows.map(r => String(r.ext)))];
      const counts = groupRows.map(r => Number(r.c)).filter(n => n > 0);

      const warnings = [];
      if (exts.length > 1) warnings.push('mixed file formats');
      if (counts.length > 1 && Math.max(...counts) / Math.min(...counts) >= 1.5) warnings.push('unequal file counts between conditions');
      if (warnings.length) return { text: `Inconsistencies: <strong>${warnings.join(', ')}</strong>.`, warning: true };

      return `All <strong>${escapeHtml(exts[0] ?? 'files')}</strong>.`;
    } catch { return null; }
  },

  async overviewPlot(container, ctx) {
    const { groupCol: gcFn } = ctx.sql;

    if (ctx.withinFileGroupVariation) {
      const groupRows = await ctx.queryRows(`
        SELECT ${gcFn()} AS g, COUNT(*) AS c
        FROM pp_data ${ctx.where} GROUP BY 1 ORDER BY 1
      `);
      if (!groupRows.length) return false;
      const groups = groupRows.map(r => String(r.g));
      ctx.plot.appendMini(container, [{
        type: 'bar',
        x: groups.map(g => ctx.groupLabel(g)),
        y: groupRows.map(r => Number(r.c)),
        marker: { color: groups.map(g => ctx.colorMap[g] ?? '#6c757d') },
        hoverinfo: 'skip',
      }], { xaxis: { type: 'category' }, bargap: 0.3 });
      return true;
    }

    const [extRows, sizeRange, dateRange] = await fetchFileStats(ctx);
    const exts = [...new Set(extRows.map(r => String(r.ext)))].sort();
    const date = datePlan(dateRange[0]);
    const miniBars = (cats, getValue) => ctx.plot.appendMini(container,
      ctx.plot.groupedBarTraces(cats, getValue, { mini: true }),
      { barmode: 'stack', xaxis: { type: 'category' }, bargap: 0.3 });

    // Show one mini-plot: the most informative varying property (ext > date > size).
    if (exts.length > 1) {
      miniBars(exts, pick(extRows, r => r.ext, 'count'));
      return true;
    }

    if (date.chart) {
      const { rows, cats } = await fetchDateBuckets(ctx, dateRange[0].span_ms);
      miniBars(cats, pick(rows, r => r.bucket, 'count'));
      return true;
    }

    const size = await fetchSizeBins(ctx, sizeRange);
    if (!size.invariant) {
      miniBars(size.labels, pick(size.rows, r => r.bin, 'count'));
      return true;
    }

    // Nothing varies: invariant summary table.
    const invariants = [];
    if (exts.length === 1) invariants.push(['File Extension', exts[0]]);
    invariants.push(['File Size', size.invariant]);
    if (date.invariant) invariants.push(['Modification Date', date.invariant]);
    ctx.plot.tilePreviewTable(container, ['Property', 'Value'], invariants);
    return true;
  },

  async render(container, ctx) {
    try {
      const invariants = [];
      const ungrouped = ctx.withinFileGroupVariation;
      if (ungrouped) ctx.plot.prependWarning(container, { level: 'yellow', html: MIXED_GROUP_WARNING });
      const [extRows, sizeRange, dateRange] = await fetchFileStats(ctx, { ungrouped });

      // Files per group, summed over extensions - the same numbers a separate
      // per-group count query returns, without the extra query.
      const byGroup = new Map();
      for (const r of extRows) byGroup.set(r.__group__, (byGroup.get(r.__group__) ?? 0) + Number(r.count));
      const total = [...byGroup.values()].reduce((a, b) => a + b, 0);

      const nullExtCount = extRows.filter(r => String(r.ext) === '(none)').reduce((s, r) => s + Number(r.count), 0);
      const availability = [
        { label: 'File Extension', present: total - nullExtCount },
        { label: 'File Size', present: total - Number(sizeRange[0]?.n_null ?? 0) },
      ];
      if (dateRange.length) availability.push({ label: 'Modification Date', present: total - Number(dateRange[0]?.n_null ?? 0) });
      ctx.plot.dataAvailabilityWarning(container, availability, total, { unit: 'files' });

      const counts = [...byGroup.values()].filter(n => n > 0);
      if (counts.length > 1 && Math.max(...counts) / Math.min(...counts) >= 1.5) {
        ctx.plot.prependWarning(container, {
          level: 'yellow',
          html: 'File counts differ significantly between conditions (ratio &gt;1.5×). This may indicate an imbalanced dataset.',
        });
      }

      // Each section draws a chart when the property varies, or adds an
      // invariant row when it's shared by every file.
      renderExtensions(container, ctx, extRows, invariants, { ungrouped });
      await renderSizeBins(container, ctx, sizeRange, invariants, { ungrouped });
      await renderModificationDates(container, ctx, dateRange, invariants, { ungrouped });

      if (invariants.length) ctx.plot.invariantTable(container, {
        title: 'Properties shared by all files that report it',
        headers: ['Property', 'Value'],
        rows: invariants,
      });

      if (!container.firstChild) {
        container.innerHTML = '<div class="no-data">No file statistics data available.</div>';
      }
    } catch {
      container.innerHTML = '<div class="no-data">Failed to load data.</div>';
    }
  },
};

// The three datasets the full view needs, fetched in parallel. size_bytes and
// modification_date are per-file values repeated across a file's image rows,
// so null counts use fileCount() (COUNT DISTINCT path) rather than COUNT(*) -
// otherwise a single null-valued file with many image rows would be counted
// once per row instead of once per file.
function fetchFileStats(ctx, { ungrouped = false } = {}) {
  const { perFile, fileCount } = ctx.sql;
  const gcExpr  = gcExprFor(ctx, ungrouped);
  const hasDate = ctx.schema.allCols.includes('modification_date');
  return Promise.all([
    ctx.queryRows(`
      SELECT COALESCE("file_extension", '(none)') AS ext, ${gcExpr} AS __group__,
             COUNT(*) AS count, SUM("size_bytes") AS total_bytes
      FROM ${perFile(ctx.where)}
      GROUP BY 1, 2 ORDER BY 1, 2
    `),
    ctx.queryRows(`
      SELECT MIN("size_bytes") AS min_s, MAX("size_bytes") AS max_s,
             COUNT(DISTINCT "size_bytes") AS n_unique,
             ${fileCount()} FILTER (WHERE "size_bytes" IS NULL) AS n_null
      FROM pp_data ${ctx.where}
    `),
    hasDate
      ? ctx.queryRows(`
          SELECT STRFTIME(MIN(TRY_CAST("modification_date" AS TIMESTAMP)), '%Y-%m-%d %H:%M:%S') AS min_fmt,
                 STRFTIME(MAX(TRY_CAST("modification_date" AS TIMESTAMP)), '%Y-%m-%d %H:%M:%S') AS max_fmt,
                 EPOCH_MS(MAX(TRY_CAST("modification_date" AS TIMESTAMP)))
                   - EPOCH_MS(MIN(TRY_CAST("modification_date" AS TIMESTAMP))) AS span_ms,
                 COUNT(DISTINCT TRY_CAST("modification_date" AS TIMESTAMP)) AS n_unique,
                 ${fileCount()} FILTER (WHERE "modification_date" IS NULL) AS n_null
          FROM pp_data ${ctx.where}
        `)
      : Promise.resolve([]),
  ]);
}

// One shared extension → invariant row; several → red warning + count/size bars.
function renderExtensions(container, ctx, extRows, invariants, { ungrouped = false } = {}) {
  const exts = [...new Set(extRows.map(r => String(r.ext)))].sort();
  if (!exts.length) return;
  if (exts.length === 1) {
    invariants.push(['File Extension', exts[0]]);
    return;
  }
  ctx.plot.prependWarning(container, {
    level: 'red',
    html: `This dataset contains files with more than one extension: ` +
      `${exts.map(e => ctx.plot.escapeHtml(e)).join(', ')}. ` +
      `Mixed file formats can mean a mixed dataset or even images that were saved twice - worth looking into.`,
  });
  renderGroupedBars(container, { categories: exts, getValue: pick(extRows, r => r.ext, 'count'),
    title: 'File Count by Extension', xLabel: 'Extension', yLabel: 'File count' }, ctx, { ungrouped });
  renderGroupedBars(container, { categories: exts, getValue: pick(extRows, r => r.ext, 'total_bytes'),
    title: 'Total Size by Extension', xLabel: 'Extension', yLabel: 'Total size (bytes)' }, ctx, { ungrouped });
}

// One distinct size (ignoring nulls) → invariant row; otherwise a count-per-size-bin
// chart, with a trailing '(no size)' bin when any file is missing size_bytes.
async function renderSizeBins(container, ctx, sizeRange, invariants, { ungrouped = false } = {}) {
  const { labels, rows, useLog, invariant } = await fetchSizeBins(ctx, sizeRange, { ungrouped });
  if (invariant) {
    invariants.push(['File Size', invariant]);
    return;
  }
  renderGroupedBars(container, {
    categories: labels, getValue: pick(rows, r => r.bin, 'count'),
    title: 'File Count by Size Bin',
    xLabel: useLog ? 'File size (log-spaced bins)' : 'File size bin', yLabel: 'File count', showLegend: true,
  }, ctx, { ungrouped });
}

// Size bins (labels + per-bin file counts), or an `invariant` text when sizes don't
// vary and a table row says it all. Labels drive both the chart categories and the
// CASE SQL, so the two never drift apart.
async function fetchSizeBins(ctx, sizeRange, { ungrouped = false } = {}) {
  const { min_s: minS, max_s: maxS, n_unique: nUniq, n_null: nNull } = sizeRange[0] ?? {};
  const { breaks, labels, useLog } = sizeBinsWithNull(Number(minS ?? 0), Number(maxS ?? 0), Number(nUniq ?? 0), Number(nNull ?? 0), ctx.plot.formatBytes);
  if (labels.length <= 1) return { labels, rows: [], useLog, invariant: labels[0] ?? ctx.plot.formatBytes(Number(minS ?? 0)) };
  const rows = await ctx.queryRows(`
    SELECT ${buildSizeCaseSQL(breaks, labels)} AS bin, ${gcExprFor(ctx, ungrouped)} AS __group__, ${ctx.sql.fileCount()} AS count
    FROM pp_data ${ctx.where}
    GROUP BY 1, 2
  `);
  return { labels, rows, useLog, invariant: null };
}

// One exact timestamp shared by every file → invariant row with full precision.
// Otherwise a timeline, bucketed at whatever granularity (day/hour/minute/second)
// actually shows spread - rolled up to months if there are too many distinct days,
// or collapsed to a compact range if the spread is sub-second and no bucket would help.
export async function renderModificationDates(container, ctx, dateRange, invariants, { ungrouped = false } = {}) {
  const { invariant, chart } = datePlan(dateRange[0]);
  if (invariant) invariants.push(['Modification Date', invariant]);
  if (!chart) return;

  const { rows, cats, label } = await fetchDateBuckets(ctx, dateRange[0].span_ms, { ungrouped });
  renderGroupedBars(container, {
    categories: cats, getValue: pick(rows, r => r.bucket, 'count'),
    title: 'File Count by Modification Date', xLabel: label, yLabel: 'File count', showLegend: true,
  }, ctx, { ungrouped });
}

// How to show modification dates: an `invariant` text when they barely vary,
// `chart: true` when a timeline is worth drawing (any file missing a date gets a
// '(no date)' bucket, so that alone makes it worth charting), neither when there are none.
function datePlan({ min_fmt: minFmt, max_fmt: maxFmt, span_ms: spanMs, n_unique: nUnique, n_null: nNull } = {}) {
  const missing = Number(nNull ?? 0) > 0;
  if (minFmt == null) return missing ? { invariant: '(no date)' } : {};
  if (missing) return { chart: true };
  if (Number(nUnique) <= 1) return { invariant: minFmt };
  if (Number(spanMs) < MS_SECOND) return { invariant: minFmt === maxFmt ? `${minFmt} (span < 1s)` : `${minFmt} – ${maxFmt} (span < 1s)` };
  return { chart: true };
}

// Date buckets at the granularity the span calls for, rolled up to months if there are too many days.
async function fetchDateBuckets(ctx, spanMs, { ungrouped = false } = {}) {
  spanMs = Number(spanMs);
  const [fmt, label] = spanMs >= MS_DAY    ? ['%Y-%m-%d', 'Date']
                     : spanMs >= MS_HOUR   ? ['%Y-%m-%d %H:00', 'Hour']
                     : spanMs >= MS_MINUTE ? ['%Y-%m-%d %H:%M', 'Minute']
                     :                       ['%Y-%m-%d %H:%M:%S', 'Second'];
  const buckets = await bucketByDateFmt(ctx, fmt, { ungrouped });
  if (fmt === '%Y-%m-%d' && buckets.cats.length > MAX_DAYS) {
    return { ...(await bucketByDateFmt(ctx, '%Y-%m', { ungrouped })), label: 'Month' };
  }
  return { ...buckets, label };
}

// Returns the group column SQL expression, or a constant when grouping is suppressed.
function gcExprFor(ctx, ungrouped) {
  return ungrouped ? "'_all_'" : ctx.sql.groupCol();
}

// Group modification_date into buckets of the given STRFTIME format. Files
// with no date fall into their own '(no date)' bucket instead of being dropped.
async function bucketByDateFmt(ctx, fmt, { ungrouped = false } = {}) {
  const gcExpr = gcExprFor(ctx, ungrouped);
  const rows = await ctx.queryRows(`
    SELECT COALESCE(STRFTIME(TRY_CAST("modification_date" AS TIMESTAMP), '${fmt}'), '(no date)') AS bucket,
           ${gcExpr} AS __group__, ${ctx.sql.fileCount()} AS count
    FROM pp_data ${ctx.where}
    GROUP BY 1, 2 ORDER BY 1, 2
  `);
  return { rows, cats: [...new Set(rows.map(r => String(r.bucket)))].sort() };
}

// (category, group) → a numeric field from long-format rows, 0 when absent.
function pick(rows, catOf, valueKey) {
  const m = new Map(rows.map(r => [`${catOf(r)}\x00${r.__group__}`, Number(r[valueKey] ?? 0)]));
  return (cat, g) => m.get(`${cat}\x00${g}`) ?? 0;
}

function renderGroupedBars(container, { categories, getValue, title, xLabel, yLabel, showLegend = true }, ctx, { ungrouped = false } = {}) {
  const traces = ungrouped
    ? [{ type: 'bar', x: categories, y: categories.map(c => getValue(c, '_all_')), marker: { color: '#6c757d' } }]
    : ctx.plot.groupedBarTraces(categories, getValue);
  const legend = !ungrouped && showLegend && ctx.groups.length > 1;
  ctx.plot.append(container, traces, {
    margin:     FILE_STATS_MARGIN,
    title:      { text: title },
    barmode:    'stack',
    bargap:     ctx.plot.bargap(categories.length),
    xaxis:      { title: xLabel, type: 'category' },
    yaxis:      { title: yLabel },
    height:     400,
    showlegend: legend,
    ...(legend ? { legend: ctx.plot.plotlyLegendConfig } : {}),
  }, 'margin-bottom:24px');
}

function computeSizeBins(minS, maxS, nUniq, fmt) {
  const effectiveBins = Math.min(SIZE_NUM_BINS, nUniq);
  if (effectiveBins <= 1 || maxS <= minS) return { breaks: [], labels: [], useLog: false };
  const minPositive = minS > 0 ? minS : 1;
  const useLog = (maxS / minPositive) >= SIZE_LOG_THRESHOLD;
  let breaks;
  if (useLog) {
    const logMin = Math.log10(minPositive);
    const logMax = Math.log10(maxS);
    if (logMax <= logMin) return { breaks: [], labels: [], useLog: false };
    const step = (logMax - logMin) / effectiveBins;
    breaks = Array.from({ length: effectiveBins - 1 }, (_, i) => 10 ** (logMin + step * (i + 1)));
  } else {
    const step = (maxS - minS) / effectiveBins;
    if (step <= 0) return { breaks: [], labels: [], useLog: false };
    breaks = Array.from({ length: effectiveBins - 1 }, (_, i) => minS + step * (i + 1));
  }
  const edges  = [minS, ...breaks, maxS];
  const labels = [];
  for (let i = 0; i < edges.length - 1; i++) labels.push(`${fmt(edges[i])}–${fmt(edges[i + 1])}`);
  return { breaks, labels, useLog };
}

// Non-null size bins/labels via computeSizeBins, plus a trailing '(no size)'
// label when any file is missing size_bytes. Labels drive both the chart
// categories and the CASE SQL below, so the two never drift apart.
function sizeBinsWithNull(minS, maxS, nUniq, nNull, fmt) {
  let breaks = [], labels = [], useLog = false;
  if (nUniq > 1) ({ breaks, labels, useLog } = computeSizeBins(minS, maxS, nUniq, fmt));
  else if (nUniq === 1) labels = [fmt(minS)];
  if (nNull > 0) labels = [...labels, '(no size)'];
  return { breaks, labels, useLog };
}

function buildSizeCaseSQL(breaks, labels) {
  let sql = `CASE WHEN "size_bytes" IS NULL THEN '(no size)'`;
  for (let i = 0; i < breaks.length; i++) sql += ` WHEN "size_bytes" < ${breaks[i]} THEN '${labels[i]}'`;
  sql += ` ELSE '${labels[breaks.length]}' END`;
  return sql;
}