const SIZE_NUM_BINS      = 20;
const SIZE_LOG_THRESHOLD = 30;
const MAX_DAYS           = 20;
const FILE_STATS_MARGIN  = { l: 50, r: 80, t: 50, b: 80 };

export default {
  id: 'file-stats',
  required_inputs: ['file_extension', 'size_bytes'],
  inputs: ['modification_date'],
  group: 'File Stats',
  scope: 'file',
  info: [
    'High-level **file statistics** for the dataset.',
    '',
    '**Charts**',
    '- File count by extension',
    '- Total size by extension',
    '- File count by size bin',
    '- File modification timeline',
    '',
    'If a property has **no variance** (e.g. all files share the same extension), it is summarized in the table instead of a chart.',
  ].join('\n'),
  label: 'File Statistics',
  shortLabel: 'File Metadata',

  requires(schema) {
    return schema.allCols.includes('file_extension') &&
           schema.allCols.includes('size_bytes');
  },

  async condensedSummary(ctx) {
    try {
      const { andWhere, groupCol: gcFn } = ctx.sql;
      const { escapeHtml } = ctx.plot;
      const [extRows, groupRows] = await Promise.all([
        ctx.queryRows(`SELECT DISTINCT "file_extension" AS ext FROM pp_data ${andWhere(ctx.where, '"file_extension" IS NOT NULL')}`),
        ctx.queryRows(`SELECT ${gcFn()} AS g, COUNT(*) AS c FROM pp_data ${ctx.where} GROUP BY 1`),
      ]);
      const exts   = [...new Set(extRows.map(r => String(r.ext)))];
      const counts = groupRows.map(r => Number(r.c)).filter(n => n > 0);

      const issues = [];
      if (exts.length > 1) issues.push('file type');
      if (counts.length > 1 && Math.max(...counts) / Math.min(...counts) >= 1.5) issues.push('file count');
      if (issues.length) return { text: `Inconsistencies: <strong>${issues.join(', ')}</strong>.`, warning: true };

      return `All <strong>${escapeHtml(exts[0] ?? 'files')}</strong>.`;
    } catch { return null; }
  },

  async condensedPlot(container, ctx) {
    const { andWhere, groupCol: gcFn } = ctx.sql;
    const gcExpr = gcFn();

    const extRows = await ctx.queryRows(`
      SELECT "file_extension" AS ext, ${gcExpr} AS __group__, COUNT(*) AS c
      FROM pp_data ${andWhere(ctx.where, '"file_extension" IS NOT NULL')}
      GROUP BY 1, 2
    `);
    const exts = [...new Set(extRows.map(r => String(r.ext)))];
    if (exts.length > 1) {
      const idx = new Map(extRows.map(r => [`${r.ext}\x00${r.__group__}`, Number(r.c)]));
      const extCount = (e, g) => idx.get(`${e}\x00${g}`) ?? 0;
      ctx.plot.appendMini(container, ctx.plot.groupedBarTraces(exts, extCount, { mini: true }),
        { barmode: 'stack', xaxis: { type: 'category' }, bargap: 0.3 });
      return true;
    }
    // Single format: file sizes are usually the interesting variation, so show
    // their distribution instead - unless they're all identical too, in which
    // case the full widget only shows an invariant table; mirror that here.
    const [sizeStats] = await ctx.queryRows(`
      SELECT MIN("size_bytes") AS min_s, MAX("size_bytes") AS max_s, COUNT(DISTINCT "size_bytes") AS n_unique
      FROM pp_data ${andWhere(ctx.where, '"size_bytes" IS NOT NULL')}
    `);
    const nUniq = Number(sizeStats?.n_unique ?? 0);
    if (nUniq <= 1) {
      const invariants = [['File Extension', exts[0] ?? '—']];
      if (sizeStats?.min_s != null) invariants.push(['File Size', ctx.plot.formatBytes(Number(sizeStats.min_s))]);
      ctx.plot.tilePreviewTable(container, ['Property', 'Value'], invariants);
      return true;
    }

    // Reuse the full widget's log/linear size bins so the preview reads like a
    // shrunk version of its "File Count by Size Bin" chart (shared bin edges
    // across groups, then stacked - instead of per-trace auto-binning).
    const minS = Number(sizeStats.min_s ?? 0), maxS = Number(sizeStats.max_s ?? 0);
    const { breaks, labels } = computeSizeBins(minS, maxS, nUniq, ctx.plot.formatBytes);
    if (!labels.length) return false;
    const binRows = await ctx.queryRows(`
      SELECT ${buildSizeCaseSQL(breaks, labels)} AS bin, ${gcExpr} AS __group__, COUNT(*) AS c
      FROM pp_data ${andWhere(ctx.where, '"size_bytes" IS NOT NULL')}
      GROUP BY 1, 2
    `);
    if (!binRows.length) return false;
    const idx = new Map(binRows.map(r => [`${r.bin}\x00${r.__group__}`, Number(r.c)]));
    const binCount = (b, g) => idx.get(`${b}\x00${g}`) ?? 0;
    ctx.plot.appendMini(container, ctx.plot.groupedBarTraces(labels, binCount, { mini: true }),
      { barmode: 'stack', xaxis: { type: 'category' }, bargap: 0.04 });
    return true;
  },

  async render(container, ctx) {
    try {
      const invariants = [];
      const [extRows, sizeRange, dateRows] = await fetchFileStats(ctx);

      // Each section draws a chart when the property varies, or adds an
      // invariant row when it's shared by every file.
      renderExtensions(container, ctx, extRows, invariants);
      await renderSizeBins(container, ctx, sizeRange, invariants);
      await renderModificationDates(container, ctx, dateRows, invariants);

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

// The three datasets the full view needs, fetched in parallel.
function fetchFileStats(ctx) {
  const { groupCol: gcFn, andWhere } = ctx.sql;
  const gcExpr  = gcFn();
  const hasDate = ctx.schema.allCols.includes('modification_date');
  return Promise.all([
    ctx.queryRows(`
      SELECT "file_extension" AS ext, ${gcExpr} AS __group__,
             COUNT(*) AS count, SUM("size_bytes") AS total_bytes
      FROM pp_data ${andWhere(ctx.where, '"file_extension" IS NOT NULL')}
      GROUP BY 1, 2 ORDER BY 1, 2
    `),
    ctx.queryRows(`
      SELECT MIN("size_bytes") AS min_s, MAX("size_bytes") AS max_s,
             COUNT(DISTINCT "size_bytes") AS n_unique
      FROM pp_data ${andWhere(ctx.where, '"size_bytes" IS NOT NULL')}
    `),
    hasDate
      ? ctx.queryRows(`
          SELECT STRFTIME(TRY_CAST("modification_date" AS TIMESTAMP), '%Y-%m-%d') AS day,
                 ${gcExpr} AS __group__, COUNT(*) AS count
          FROM pp_data ${andWhere(ctx.where, '"modification_date" IS NOT NULL')}
          GROUP BY 1, 2 ORDER BY 1, 2
        `)
      : Promise.resolve([]),
  ]);
}

// One shared extension → invariant row; several → red warning + count/size bars.
function renderExtensions(container, ctx, extRows, invariants) {
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
    title: 'File Count by Extension', xLabel: 'Extension', yLabel: 'File count' }, ctx);
  renderGroupedBars(container, { categories: exts, getValue: pick(extRows, r => r.ext, 'total_bytes'),
    title: 'Total Size by Extension', xLabel: 'Extension', yLabel: 'Total size (bytes)' }, ctx);
}

// One distinct size → invariant row; otherwise a count-per-size-bin chart.
async function renderSizeBins(container, ctx, sizeRange, invariants) {
  const minS  = Number(sizeRange[0]?.min_s ?? 0);
  const maxS  = Number(sizeRange[0]?.max_s ?? 0);
  const nUniq = Number(sizeRange[0]?.n_unique ?? 0);
  if (nUniq <= 1) {
    invariants.push(['File Size', ctx.plot.formatBytes(minS)]);
    return;
  }
  const { breaks, labels, useLog } = computeSizeBins(minS, maxS, nUniq, ctx.plot.formatBytes);
  if (!breaks.length) return;
  const rows = await ctx.queryRows(`
    SELECT ${buildSizeCaseSQL(breaks, labels)} AS bin, ${ctx.sql.groupCol()} AS __group__, COUNT(*) AS count
    FROM pp_data ${ctx.sql.andWhere(ctx.where, '"size_bytes" IS NOT NULL')}
    GROUP BY 1, 2
  `);
  renderGroupedBars(container, {
    categories: labels, getValue: pick(rows, r => r.bin, 'count'),
    title: 'File Count by Size Bin',
    xLabel: useLog ? 'File size (log-spaced bins)' : 'File size bin', yLabel: 'File count', showLegend: true,
  }, ctx);
}

// One distinct day → invariant row; otherwise a timeline, rolled up to months
// once there are too many distinct days to read.
async function renderModificationDates(container, ctx, dateRows, invariants) {
  if (!dateRows.length) return;
  const uniqueDays = [...new Set(dateRows.map(r => r.day))].sort();
  if (uniqueDays.length === 1) {
    invariants.push(['Modification Date (Day)', uniqueDays[0]]);
    return;
  }
  let rows = dateRows, dateLabel = 'Date';
  if (uniqueDays.length > MAX_DAYS) {
    rows = await ctx.queryRows(`
      SELECT STRFTIME(TRY_CAST("modification_date" AS TIMESTAMP), '%Y-%m') AS day,
             ${ctx.sql.groupCol()} AS __group__, COUNT(*) AS count
      FROM pp_data ${ctx.sql.andWhere(ctx.where, '"modification_date" IS NOT NULL')}
      GROUP BY 1, 2 ORDER BY 1, 2
    `);
    dateLabel = 'Month';
  }
  const cats = [...new Set(rows.map(r => String(r.day)))].sort();
  renderGroupedBars(container, {
    categories: cats, getValue: pick(rows, r => r.day, 'count'),
    title: 'File Count by Modification Date', xLabel: dateLabel, yLabel: 'File count', showLegend: true,
  }, ctx);
}

// (category, group) → a numeric field from long-format rows, 0 when absent.
function pick(rows, catOf, valueKey) {
  const m = new Map(rows.map(r => [`${catOf(r)}\x00${r.__group__}`, Number(r[valueKey] ?? 0)]));
  return (cat, g) => m.get(`${cat}\x00${g}`) ?? 0;
}

function renderGroupedBars(container, { categories, getValue, title, xLabel, yLabel, showLegend = true }, ctx) {
  const legend = showLegend && ctx.groups.length > 1;
  ctx.plot.append(container, ctx.plot.groupedBarTraces(categories, getValue), {
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

function buildSizeCaseSQL(breaks, labels) {
  let sql = `CASE`;
  for (let i = 0; i < breaks.length; i++) sql += ` WHEN "size_bytes" < ${breaks[i]} THEN '${labels[i]}'`;
  sql += ` ELSE '${labels[labels.length - 1]}' END`;
  return sql;
}

