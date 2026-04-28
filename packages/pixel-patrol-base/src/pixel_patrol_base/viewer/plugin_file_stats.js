const SIZE_NUM_BINS      = 20;
const SIZE_LOG_THRESHOLD = 30;
const MAX_DAYS           = 20;
const FILE_STATS_MARGIN  = { l: 50, r: 80, t: 50, b: 80 };

export default {
  id: 'file-stats',
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

  requires(schema) {
    return schema.allCols.includes('file_extension') &&
           schema.allCols.includes('size_bytes');
  },

  async render(container, ctx) {
    try {
      const { q, groupCol: gcFn, andWhere } = ctx.sql;
      const gcExpr  = gcFn();
      const hasDate = ctx.schema.allCols.includes('modification_date');
  
      const [extRows, sizeRange, dateRows] = await Promise.all([
        ctx.queryRows(`
          SELECT "file_extension" AS ext,
                 ${gcExpr}        AS __group__,
                 COUNT(*)         AS count,
                 SUM("size_bytes") AS total_bytes
          FROM pp_data ${andWhere(ctx.where, '"file_extension" IS NOT NULL')}
          GROUP BY 1, 2 ORDER BY 1, 2
        `),
        ctx.queryRows(`
          SELECT MIN("size_bytes")              AS min_s,
                 MAX("size_bytes")              AS max_s,
                 COUNT(DISTINCT "size_bytes")   AS n_unique
          FROM pp_data ${andWhere(ctx.where, '"size_bytes" IS NOT NULL')}
        `),
        hasDate
          ? ctx.queryRows(`
              SELECT STRFTIME(TRY_CAST("modification_date" AS TIMESTAMP), '%Y-%m-%d') AS day,
                     ${gcExpr} AS __group__,
                     COUNT(*)  AS count
              FROM pp_data ${andWhere(ctx.where, '"modification_date" IS NOT NULL')}
              GROUP BY 1, 2 ORDER BY 1, 2
            `)
          : Promise.resolve([]),
      ]);
  
      const invariants = [];
      const exts   = [...new Set(extRows.map(r => String(r.ext)))].sort();
      const groups = ctx.groups;
  
      if (exts.length === 0) {
        // No extension data
      } else if (exts.length === 1) {
        invariants.push({ Property: 'File Extension', Value: exts[0] });
      } else {
        renderGroupedBars(container, {
          categories: exts, groups,
          getValue: (ext, g) => { const r = extRows.find(r => String(r.ext) === ext && String(r.__group__) === g); return r ? Number(r.count) : 0; },
          title: 'File Count by Extension', xLabel: 'Extension', yLabel: 'File count',
        }, ctx);
  
        renderGroupedBars(container, {
          categories: exts, groups,
          getValue: (ext, g) => { const r = extRows.find(r => String(r.ext) === ext && String(r.__group__) === g); return r ? Number(r.total_bytes ?? 0) : 0; },
          title: 'Total Size by Extension', xLabel: 'Extension', yLabel: 'Total size (bytes)',
        }, ctx);
      }
  
      const minS  = Number(sizeRange[0]?.min_s   ?? 0);
      const maxS  = Number(sizeRange[0]?.max_s   ?? 0);
      const nUniq = Number(sizeRange[0]?.n_unique ?? 0);
  
      if (nUniq <= 1) {
        invariants.push({ Property: 'File Size', Value: formatBytes(minS) });
      } else {
        const { breaks, labels, useLog } = computeSizeBins(minS, maxS, nUniq);
        if (breaks.length) {
          const caseExpr    = buildSizeCaseSQL(breaks, labels);
          const sizeBinRows = await ctx.queryRows(`
            SELECT ${caseExpr} AS bin,
                   ${gcExpr}   AS __group__,
                   COUNT(*)    AS count
            FROM pp_data ${andWhere(ctx.where, '"size_bytes" IS NOT NULL')}
            GROUP BY 1, 2
          `);
          renderGroupedBars(container, {
            categories: labels, groups,
            getValue: (bin, g) => { const r = sizeBinRows.find(r => r.bin === bin && String(r.__group__) === g); return r ? Number(r.count) : 0; },
            title: 'File Count by Size Bin', xLabel: useLog ? 'File size (log-spaced bins)' : 'File size bin', yLabel: 'File count',
            showLegend: true,
          }, ctx);
        }
      }
  
      if (dateRows.length) {
        const uniqueDays = [...new Set(dateRows.map(r => r.day))].sort();
        if (uniqueDays.length === 1) {
          invariants.push({ Property: 'Modification Date (Day)', Value: uniqueDays[0] });
        } else {
          let displayRows = dateRows;
          let dateLabel   = 'Date';
          if (uniqueDays.length > MAX_DAYS) {
            displayRows = await ctx.queryRows(`
              SELECT STRFTIME(TRY_CAST("modification_date" AS TIMESTAMP), '%Y-%m') AS day,
                     ${gcExpr} AS __group__,
                     COUNT(*)  AS count
              FROM pp_data ${andWhere(ctx.where, '"modification_date" IS NOT NULL')}
              GROUP BY 1, 2 ORDER BY 1, 2
            `);
            dateLabel = 'Month';
          }
          const dateCats = [...new Set(displayRows.map(r => String(r.day)))].sort();
          renderGroupedBars(container, {
            categories: dateCats, groups,
            getValue: (d, g) => { const r = displayRows.find(r => String(r.day) === d && String(r.__group__) === g); return r ? Number(r.count) : 0; },
            title: 'File Count by Modification Date', xLabel: dateLabel, yLabel: 'File count',
            showLegend: true,
          }, ctx);
        }
      }
  
      if (invariants.length) {
        const h = document.createElement('h6');
        h.style.cssText = 'margin-top:20px;margin-bottom:10px';
        h.textContent = 'Properties shared between all files';
        container.appendChild(h);
        const table = document.createElement('table');
        table.className = 'stat-table';
        table.innerHTML = `
          <thead><tr><th>Property</th><th>Value</th></tr></thead>
          <tbody>${invariants.map(r => `<tr><td>${r.Property}</td><td>${r.Value}</td></tr>`).join('')}</tbody>
        `;
        container.appendChild(table);
      }
  
      if (!container.firstChild) {
        container.innerHTML = '<div class="no-data">No file statistics data available.</div>';
      }
    
    } catch {
      container.innerHTML = '<div class="no-data">Failed to load data.</div>';
    }
  },
};

function renderGroupedBars(container, { categories, groups, getValue, title, xLabel, yLabel, showLegend = false }, ctx) {
  const traces = groups.map(g => ({
    type:   'bar',
    name:   g,
    x:      categories,
    y:      categories.map(cat => getValue(cat, g)),
    marker: { color: ctx.color.group(g) },
  }));
  ctx.plot.append(container, traces, {
    margin:     FILE_STATS_MARGIN,
    title:      { text: title },
    barmode:    'stack',
    bargap:     ctx.plot.bargap(categories.length),
    xaxis:      { title: xLabel, type: 'category' },
    yaxis:      { title: yLabel },
    height:     400,
    showlegend: showLegend && groups.length > 1,
    ...(showLegend && groups.length > 1 ? { legend: ctx.plot.LEGEND } : {}),
  }, 'margin-bottom:24px');
}

function computeSizeBins(minS, maxS, nUniq) {
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
  for (let i = 0; i < edges.length - 1; i++) labels.push(`${formatBytes(edges[i])}–${formatBytes(edges[i + 1])}`);
  return { breaks, labels, useLog };
}

function buildSizeCaseSQL(breaks, labels) {
  let sql = `CASE`;
  for (let i = 0; i < breaks.length; i++) sql += ` WHEN "size_bytes" < ${breaks[i]} THEN '${labels[i]}'`;
  sql += ` ELSE '${labels[labels.length - 1]}' END`;
  return sql;
}

function formatBytes(v) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let val = Number(v), u = 0;
  while (val >= 1024 && u < units.length - 1) { val /= 1024; u++; }
  return val >= 10 ? `${Math.round(val)} ${units[u]}` : `${val.toFixed(1)} ${units[u]}`;
}
