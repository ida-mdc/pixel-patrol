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
      const { andWhere, groupCol: gcFn, fileCount } = ctx.sql;
      const { escapeHtml } = ctx.plot;
      const [extRows, groupRows] = await Promise.all([
        ctx.queryRows(`SELECT DISTINCT "file_extension" AS ext FROM pp_data ${andWhere(ctx.where, '"file_extension" IS NOT NULL')}`),
        ctx.queryRows(`SELECT ${gcFn()} AS g, ${fileCount()} AS c FROM pp_data ${ctx.where} GROUP BY 1`),
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

  async overviewPlot(container, ctx) {
    const { andWhere, groupCol: gcFn, fileCount } = ctx.sql;
    const gcExpr = gcFn();

    const [extRows, groupRows] = await Promise.all([
      ctx.queryRows(`
        SELECT "file_extension" AS ext, ${gcExpr} AS __group__, ${fileCount()} AS c
        FROM pp_data ${andWhere(ctx.where, '"file_extension" IS NOT NULL')}
        GROUP BY 1, 2
      `),
      ctx.queryRows(`SELECT ${gcExpr} AS g, ${fileCount()} AS c FROM pp_data ${ctx.where} GROUP BY 1`),
    ]);

    const exts   = [...new Set(extRows.map(r => String(r.ext)))];
    const counts = groupRows.map(r => Number(r.c)).filter(n => n > 0);
    const countImbalance = counts.length > 1 &&
      Math.max(...counts) / Math.min(...counts) >= 1.5;

    // One mini-plot per warning condition — same thresholds as overviewMessage,
    // capped at 2 (beyond that the tile becomes too cramped to read).
    const drawers = [];
    if (exts.length > 1) {
      drawers.push({ title: 'File type', draw: (host) => {
        const idx = new Map(extRows.map(r => [`${r.ext}\x00${r.__group__}`, Number(r.c)]));
        const extCount = (e, g) => idx.get(`${e}\x00${g}`) ?? 0;
        ctx.plot.appendMini(host, ctx.plot.groupedBarTraces(exts, extCount, { mini: true }),
          { barmode: 'stack', xaxis: { type: 'category' }, bargap: 0.3 });
        return true;
      }});
    }
    if (countImbalance) {
      drawers.push({ title: 'File count', draw: (host) => {
        const groups = groupRows.map(r => String(r.g));
        const idx = new Map(groupRows.map(r => [String(r.g), Number(r.c)]));
        ctx.plot.appendMini(host, ctx.plot.groupedBarTraces(groups, (g) => idx.get(g) ?? 0, { mini: true }),
          { barmode: 'stack', xaxis: { type: 'category' }, bargap: 0.3 });
        return true;
      }});
    }

    // Nothing varies: mirror the full widget, which lists every invariant property.
    if (!drawers.length) {
      const invariants = [];
      if (exts.length === 1) invariants.push(['File Extension', exts[0]]);

      const [sizeRange, dateRange] = await Promise.all([
        ctx.queryRows(`
          SELECT MIN("size_bytes") AS min_s, COUNT(DISTINCT "size_bytes") AS n_unique
          FROM pp_data ${andWhere(ctx.where, '"size_bytes" IS NOT NULL')}
        `),
        ctx.schema.allCols.includes('modification_date')
          ? ctx.queryRows(`
              SELECT STRFTIME(MIN(TRY_CAST("modification_date" AS TIMESTAMP)), '%Y-%m-%d %H:%M:%S') AS min_fmt,
                     COUNT(DISTINCT TRY_CAST("modification_date" AS TIMESTAMP)) AS n_unique
              FROM pp_data ${andWhere(ctx.where, '"modification_date" IS NOT NULL')}
            `)
          : Promise.resolve([]),
      ]);

      if (Number(sizeRange[0]?.n_unique ?? 0) <= 1) {
        invariants.push(['File Size', ctx.plot.formatBytes(Number(sizeRange[0]?.min_s ?? 0))]);
      }
      if (dateRange[0]?.min_fmt != null && Number(dateRange[0].n_unique) <= 1) {
        invariants.push(['Modification Date', dateRange[0].min_fmt]);
      }

      if (!invariants.length) return false;
      ctx.plot.tilePreviewTable(container, ['Property', 'Value'], invariants);
      return true;
    }

    if (drawers.length === 1) return drawers[0].draw(container);

    // Two issues: split the tile vertically, one labelled mini-plot each.
    container.style.cssText += ';display:flex;flex-direction:column;gap:2px';
    let drew = false;
    for (const { title, draw } of drawers) {
      const cell = document.createElement('div');
      cell.style.cssText = 'flex:1 1 0;min-height:0;display:flex;flex-direction:column';
      const cap = document.createElement('div');
      cap.textContent = title;
      cap.style.cssText = 'font-size:10px;font-weight:600;color:#6c757d;text-align:center;flex-shrink:0';
      const plotHost = document.createElement('div');
      plotHost.style.cssText = 'flex:1 1 0;min-height:0;position:relative';
      cell.append(cap, plotHost);
      container.appendChild(cell);
      drew = (await draw(plotHost)) !== false || drew;
    }
    return drew;
  },

  async render(container, ctx) {
    try {
      const invariants = [];
      const [extRows, sizeRange, dateRange] = await fetchFileStats(ctx);

      // Each section draws a chart when the property varies, or adds an
      // invariant row when it's shared by every file.
      renderExtensions(container, ctx, extRows, invariants);
      await renderSizeBins(container, ctx, sizeRange, invariants);
      await renderModificationDates(container, ctx, dateRange, invariants);

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
  const { perFile, andWhere } = ctx.sql;
  const hasDate = ctx.schema.allCols.includes('modification_date');
  return Promise.all([
    ctx.queryRows(`
      SELECT "file_extension" AS ext, __group__,
             COUNT(*) AS count, SUM("size_bytes") AS total_bytes
      FROM ${perFile({ where: andWhere(ctx.where, '"file_extension" IS NOT NULL') })}
      GROUP BY 1, 2 ORDER BY 1, 2
    `),
    ctx.queryRows(`
      SELECT MIN("size_bytes") AS min_s, MAX("size_bytes") AS max_s,
             COUNT(DISTINCT "size_bytes") AS n_unique
      FROM pp_data ${andWhere(ctx.where, '"size_bytes" IS NOT NULL')}
    `),
    hasDate
      ? ctx.queryRows(`
          SELECT STRFTIME(MIN(TRY_CAST("modification_date" AS TIMESTAMP)), '%Y-%m-%d %H:%M:%S') AS min_fmt,
                 STRFTIME(MAX(TRY_CAST("modification_date" AS TIMESTAMP)), '%Y-%m-%d %H:%M:%S') AS max_fmt,
                 EPOCH_MS(MAX(TRY_CAST("modification_date" AS TIMESTAMP)))
                   - EPOCH_MS(MIN(TRY_CAST("modification_date" AS TIMESTAMP))) AS span_ms,
                 COUNT(DISTINCT TRY_CAST("modification_date" AS TIMESTAMP)) AS n_unique
          FROM pp_data ${andWhere(ctx.where, '"modification_date" IS NOT NULL')}
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
    SELECT ${buildSizeCaseSQL(breaks, labels)} AS bin, ${ctx.sql.groupCol()} AS __group__, ${ctx.sql.fileCount()} AS count
    FROM pp_data ${ctx.sql.andWhere(ctx.where, '"size_bytes" IS NOT NULL')}
    GROUP BY 1, 2
  `);
  renderGroupedBars(container, {
    categories: labels, getValue: pick(rows, r => r.bin, 'count'),
    title: 'File Count by Size Bin',
    xLabel: useLog ? 'File size (log-spaced bins)' : 'File size bin', yLabel: 'File count', showLegend: true,
  }, ctx);
}

// One exact timestamp shared by every file → invariant row with full precision.
// Otherwise a timeline, bucketed at whatever granularity (day/hour/minute/second)
// actually shows spread - rolled up to months if there are too many distinct days,
// or collapsed to a compact range if the spread is sub-second and no bucket would help.
export async function renderModificationDates(container, ctx, dateRange, invariants) {
  const { min_fmt: minFmt, max_fmt: maxFmt, span_ms: spanMsRaw, n_unique: nUniqueRaw } = dateRange[0] ?? {};
  if (minFmt == null) return;

  const nUnique = Number(nUniqueRaw);
  if (nUnique <= 1) {
    invariants.push(['Modification Date', minFmt]);
    return;
  }

  const spanMs = Number(spanMsRaw);
  if (spanMs < MS_SECOND) {
    invariants.push(['Modification Date', minFmt === maxFmt
      ? `${minFmt} (span < 1s)`
      : `${minFmt} – ${maxFmt} (span < 1s)`]);
    return;
  }

  const [fmt, dateLabel] = spanMs >= MS_DAY    ? ['%Y-%m-%d', 'Date']
                         : spanMs >= MS_HOUR   ? ['%Y-%m-%d %H:00', 'Hour']
                         : spanMs >= MS_MINUTE ? ['%Y-%m-%d %H:%M', 'Minute']
                         :                       ['%Y-%m-%d %H:%M:%S', 'Second'];

  let { rows, cats } = await bucketByDateFmt(ctx, fmt);
  let finalLabel = dateLabel;
  if (fmt === '%Y-%m-%d' && cats.length > MAX_DAYS) {
    ({ rows, cats } = await bucketByDateFmt(ctx, '%Y-%m'));
    finalLabel = 'Month';
  }

  renderGroupedBars(container, {
    categories: cats, getValue: pick(rows, r => r.bucket, 'count'),
    title: 'File Count by Modification Date', xLabel: finalLabel, yLabel: 'File count', showLegend: true,
  }, ctx);
}

// Group modification_date into buckets of the given STRFTIME format.
async function bucketByDateFmt(ctx, fmt) {
  const rows = await ctx.queryRows(`
    SELECT STRFTIME(TRY_CAST("modification_date" AS TIMESTAMP), '${fmt}') AS bucket,
           ${ctx.sql.groupCol()} AS __group__, ${ctx.sql.fileCount()} AS count
    FROM pp_data ${ctx.sql.andWhere(ctx.where, '"modification_date" IS NOT NULL')}
    GROUP BY 1, 2 ORDER BY 1, 2
  `);
  return { rows, cats: [...new Set(rows.map(r => String(r.bucket)))].sort() };
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