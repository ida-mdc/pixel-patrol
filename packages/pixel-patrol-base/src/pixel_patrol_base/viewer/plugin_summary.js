export default {
  id: 'summary',
  label: 'File Data Summary',
  group: 'Summary',
  info: 'High-level overview of the dataset: total files, total size, file types present, ' +
    'and (when grouped) a per-group breakdown.',

  requires(schema) {
    return schema.allCols.includes('size_bytes') &&
           schema.allCols.includes('file_extension');
  },

  async condensedSummary(ctx) {
    try {
      const hasNImages = ctx.schema.allCols.includes('n_images');
      const [row] = await ctx.queryRows(`
        SELECT COUNT(*) AS file_count
               ${hasNImages ? ', SUM("n_images") AS n_images_sum' : ''}
        FROM pp_data ${ctx.where}
      `);
      const fileCount  = Number(row.file_count ?? 0);
      const imageCount = hasNImages ? Number(row.n_images_sum ?? 0) : null;

      let text = `Pixel Patrol scanned <strong>${fileCount.toLocaleString()} file${fileCount === 1 ? '' : 's'}</strong>`;
      if (imageCount != null && imageCount > 0 && imageCount !== fileCount) {
        text += ` containing <strong>${imageCount.toLocaleString()} image${imageCount === 1 ? '' : 's'}</strong> in total.`;
      } else {
        text += ` - one image per file.`;
      }
      return text;
    } catch { return null; }
  },

  async render(container, ctx) {
    try {
      const [rows, pathRow, basePathRow] = await fetchSummary(ctx);
      if (!rows.length) {
        container.innerHTML = '<div class="no-data">No data available after filtering.</div>';
        return;
      }
      const summary = summarize(ctx, rows, pathRow, basePathRow);

      // In condensed mode the summary is a bare header; the uneven-group warning
      // is surfaced in the File Metadata tile instead.
      if (summary.nGroups > 1 && !ctx.state.condensedMode) prependUnevenGroupWarning(container, ctx, rows);
      renderKpis(container, ctx, summary);
      renderMetaLines(container, ctx, summary);
      if (summary.nGroups > 1) renderGroupTable(container, ctx, rows, summary);
    } catch {
      container.innerHTML = '<div class="no-data">Failed to load data.</div>';
    }
  },
};

// Per-group totals, plus distinct-path and path-range probes for the base path.
function fetchSummary(ctx) {
  const { groupCol: gcFn, andWhere } = ctx.sql;
  const gcExpr     = gcFn();
  const hasPath    = ctx.schema.allCols.includes('path');
  const nImagesSql = ctx.schema.allCols.includes('n_images') ? ', SUM("n_images") AS n_images_sum' : '';
  return Promise.all([
    ctx.queryRows(`
      SELECT ${gcExpr} AS __group__, COUNT(*) AS file_count, SUM("size_bytes") AS total_bytes,
             LIST(DISTINCT "file_extension") AS file_types ${nImagesSql}
      FROM pp_data ${ctx.where} GROUP BY 1 ORDER BY 1
    `),
    hasPath
      ? ctx.queryRows(`SELECT COUNT(DISTINCT "path") AS n FROM pp_data ${ctx.where}`)
      : Promise.resolve([{ n: null }]),
    hasPath
      ? ctx.queryRows(`
          SELECT MIN("path")::VARCHAR AS lo, MAX("path")::VARCHAR AS hi
          FROM pp_data ${andWhere(ctx.where, '"path" IS NOT NULL')}
        `)
      : Promise.resolve([{ lo: null, hi: null }]),
  ]);
}

// Derive the headline numbers and labels the view needs from the grouped rows.
function summarize(ctx, rows, pathRow, basePathRow) {
  const hasNImages   = ctx.schema.allCols.includes('n_images');
  const totalRecords = rows.reduce((s, r) => s + Number(r.file_count), 0);
  const totalBytes   = rows.reduce((s, r) => s + Number(r.total_bytes ?? 0), 0);
  const extensions   = new Set();
  for (const r of rows) for (const ext of (r.file_types ?? [])) if (ext) extensions.add(ext);

  const basePath = (basePathRow[0]?.lo != null && basePathRow[0]?.hi != null)
    ? commonDirPath(String(basePathRow[0].lo), String(basePathRow[0].hi))
    : null;
  const groupCol            = ctx.state.groupCol ?? 'group';
  const isImportedPathShort = groupCol === 'imported_path_short';

  return {
    nGroups:       rows.length,
    totalRecords,
    totalBytes,
    extensions,
    totalFiles:    pathRow[0]?.n != null ? Number(pathRow[0].n) : null,
    totalImages:   hasNImages ? rows.reduce((s, r) => s + Number(r.n_images_sum ?? 0), 0) : null,
    basePath,
    isImportedPathShort,
    groupColLabel: isImportedPathShort ? groupCol : ctx.plot.niceName(groupCol),
  };
}

// Yellow banner when the largest group has ≥1.5× the files of the smallest.
function prependUnevenGroupWarning(container, ctx, rows) {
  const biggest  = rows.reduce((a, b) => Number(a.file_count) >= Number(b.file_count) ? a : b);
  const smallest = rows.reduce((a, b) => Number(a.file_count) <= Number(b.file_count) ? a : b);
  if (Number(biggest.file_count) / Number(smallest.file_count) < 1.5) return;
  const { escapeHtml } = ctx.plot;
  ctx.plot.prependWarning(container, {
    level: 'yellow',
    html: `Group sizes differ a fair bit: <code>${escapeHtml(ctx.groupLabel(String(biggest.__group__)))}</code> ` +
      `has ${Number(biggest.file_count).toLocaleString()} files, while ` +
      `<code>${escapeHtml(ctx.groupLabel(String(smallest.__group__)))}</code> has ` +
      `${Number(smallest.file_count).toLocaleString()}. That's not necessarily an issue, but uneven group ` +
      `sizes can affect statistics and significance tests - worth a quick check that it's what you expect.`,
  });
}

// The Files / Images / Total Size / Extensions tiles.
function renderKpis(container, ctx, s) {
  const { escapeHtml, formatBytes } = ctx.plot;
  const kpis = [{ label: 'Files', value: (s.totalFiles ?? s.totalRecords).toLocaleString() }];
  if (s.totalImages > 0) kpis.push({ label: 'Images', value: s.totalImages.toLocaleString() });
  kpis.push({ label: 'Total Size', value: formatBytes(s.totalBytes) });
  kpis.push({ label: 'File Extensions', value: formatFileTypes(s.extensions) });

  const kpiRow = document.createElement('div');
  kpiRow.className = 'kpi-row';
  kpiRow.innerHTML = kpis.map(k => `
    <div class="kpi-tile">
      <div class="kpi-value">${escapeHtml(k.value)}</div>
      <div class="kpi-label">${escapeHtml(k.label)}</div>
    </div>
  `).join('');
  container.appendChild(kpiRow);
}

// "Base path:" and "Grouped by:" context lines below the tiles.
function renderMetaLines(container, ctx, s) {
  const { escapeHtml } = ctx.plot;
  if (s.basePath) {
    const div = document.createElement('div');
    div.className = 'summary-meta-line';
    div.innerHTML = `<strong>Base path:</strong> <code>${escapeHtml(s.basePath)}</code>`;
    container.appendChild(div);
  }
  if (s.nGroups > 1) {
    const div = document.createElement('div');
    div.className = 'summary-meta-line';
    div.innerHTML = `<strong>Grouped by:</strong> <code>${escapeHtml(s.groupColLabel)}</code>` +
      (s.isImportedPathShort
        ? ' &mdash; the top-level imported folder each file came from, relative to the common base path of all imported folders'
        : '');
    container.appendChild(div);
  }
}

// Per-group breakdown table with inline bar cells for file count and size.
function renderGroupTable(container, ctx, rows, s) {
  const { escapeHtml, formatBytes } = ctx.plot;
  const showExtCol   = s.extensions.size > 1;
  const maxFileCount = Math.max(...rows.map(r => Number(r.file_count)));
  const maxBytes     = Math.max(...rows.map(r => Number(r.total_bytes ?? 0)));
  const table = document.createElement('table');
  table.className = 'stat-table';
  table.innerHTML = `
    <thead>
      <tr>
        <th>${escapeHtml(s.groupColLabel)}</th>
        <th>Files</th>
        <th>Size</th>
        ${showExtCol ? '<th>File Extension</th>' : ''}
      </tr>
    </thead>
    <tbody>
      ${rows.map(r => {
        const color     = ctx.color.group(String(r.__group__));
        const fileCount = Number(r.file_count);
        const bytes     = Number(r.total_bytes ?? 0);
        return `
        <tr>
          <td><span class="group-color-dot" style="background:${color}"></span>${escapeHtml(ctx.groupLabel(String(r.__group__)))}</td>
          <td>${barCell(fileCount, maxFileCount, color, fileCount.toLocaleString())}</td>
          <td>${barCell(bytes, maxBytes, color, formatBytes(bytes))}</td>
          ${showExtCol ? `<td>${escapeHtml(formatFileTypes(r.file_types))}</td>` : ''}
        </tr>
      `;
      }).join('')}
    </tbody>
  `;
  container.appendChild(table);
}

// LCP of the lexicographically smallest and largest path equals the LCP of all paths,
// trimmed back to the last path separator to yield a valid directory.
function commonDirPath(lo, hi) {
  let i = 0;
  const len = Math.min(lo.length, hi.length);
  while (i < len && lo[i] === hi[i]) i++;
  const prefix  = lo.slice(0, i);
  const lastSep = Math.max(prefix.lastIndexOf('/'), prefix.lastIndexOf('\\'));
  return lastSep > 0 ? prefix.slice(0, lastSep) : '';
}

function barCell(value, maxValue, color, label) {
  const pct = maxValue > 0 ? Math.max(0, Math.min(100, (value / maxValue) * 100)) : 0;
  return `<div class="bar-cell"><div class="bar-cell-fill" style="width:${pct}%;background:${color}"></div><span class="bar-cell-value">${label}</span></div>`;
}

function formatFileTypes(val) {
  if (!val) return '-';
  if (typeof val === 'string') return val;
  let arr;
  try {
    arr = [...val].filter(Boolean);
  } catch {
    return String(val);
  }
  return arr.length ? arr.sort().join(', ') : '-';
}
