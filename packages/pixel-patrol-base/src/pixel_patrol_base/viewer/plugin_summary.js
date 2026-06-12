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

  async render(container, ctx) {
    try {
      const { groupCol: gcFn } = ctx.sql;
      const { escapeHtml, niceName } = ctx.plot;
      const gcExpr = gcFn();
      const hasPath    = ctx.schema.allCols.includes('path');
      const hasNImages = ctx.schema.allCols.includes('n_images');
      const nImagesSql = hasNImages ? ', SUM("n_images") AS n_images_sum' : '';

      const [rows, pathRow, leafRow, basePathRow] = await Promise.all([
        ctx.queryRows(`
          SELECT ${gcExpr} AS __group__,
                 COUNT(*)                         AS file_count,
                 SUM("size_bytes")                AS total_bytes,
                 LIST(DISTINCT "file_extension")  AS file_types
                 ${nImagesSql}
          FROM pp_data ${ctx.where}
          GROUP BY 1
          ORDER BY 1
        `),
        hasPath
          ? ctx.queryRows(`SELECT COUNT(DISTINCT "path") AS n FROM pp_data ${ctx.where}`)
          : Promise.resolve([{ n: null }]),
        hasPath
          ? ctx.queryRows(`
              WITH leaves AS (
                SELECT obs_level, MAX(obs_level) OVER (PARTITION BY "path") AS max_level
                FROM pp_all
                WHERE "path" IN (SELECT DISTINCT "path" FROM pp_data ${ctx.where})
              )
              SELECT COUNT(*) AS n FROM leaves WHERE obs_level = max_level
            `)
          : Promise.resolve([{ n: null }]),
        hasPath
          ? ctx.queryRows(`
              SELECT MIN("path")::VARCHAR AS lo, MAX("path")::VARCHAR AS hi
              FROM pp_data ${ctx.sql.andWhere(ctx.where, '"path" IS NOT NULL')}
            `)
          : Promise.resolve([{ lo: null, hi: null }]),
      ]);

      if (!rows.length) {
        container.innerHTML = '<div class="no-data">No data available after filtering.</div>';
        return;
      }

      const nGroups = rows.length;

      if (nGroups > 1) {
        const UNEVEN_RATIO = 1.5;
        const biggest  = rows.reduce((a, b) => Number(a.file_count) >= Number(b.file_count) ? a : b);
        const smallest = rows.reduce((a, b) => Number(a.file_count) <= Number(b.file_count) ? a : b);
        if (Number(biggest.file_count) / Number(smallest.file_count) >= UNEVEN_RATIO) {
          ctx.plot.prependWarning(container, {
            level: 'yellow',
            html: `Group sizes differ a fair bit: <code>${escapeHtml(ctx.groupLabel(String(biggest.__group__)))}</code> ` +
              `has ${Number(biggest.file_count).toLocaleString()} files, while ` +
              `<code>${escapeHtml(ctx.groupLabel(String(smallest.__group__)))}</code> has ` +
              `${Number(smallest.file_count).toLocaleString()}. That's not necessarily an issue, but uneven group ` +
              `sizes can affect statistics and significance tests - worth a quick check that it's what you expect.`,
          });
        }
      }

      const totalRecords  = rows.reduce((sum, r) => sum + Number(r.file_count), 0);
      const totalBytes    = rows.reduce((sum, r) => sum + Number(r.total_bytes ?? 0), 0);
      const allExtensions = new Set();
      for (const r of rows) for (const ext of (r.file_types ?? [])) if (ext) allExtensions.add(ext);

      const totalFiles     = pathRow[0]?.n != null ? Number(pathRow[0].n) : null;
      const leafCount      = leafRow[0]?.n != null ? Number(leafRow[0].n) : null;
      const totalImagesSum = hasNImages
        ? rows.reduce((sum, r) => sum + Number(r.n_images_sum ?? 0), 0)
        : null;
      const basePath = (basePathRow[0]?.lo != null && basePathRow[0]?.hi != null)
        ? commonDirPath(String(basePathRow[0].lo), String(basePathRow[0].hi))
        : null;

      const groupCol            = ctx.state.groupCol ?? 'group';
      const isImportedPathShort = groupCol === 'imported_path_short';
      const groupColLabel       = isImportedPathShort ? groupCol : niceName(groupCol);

      const kpis = [
        { label: 'Files', value: (totalFiles ?? totalRecords).toLocaleString() },
      ];
      if (totalImagesSum != null && totalImagesSum > 0) {
        kpis.push({ label: 'Images', value: totalImagesSum.toLocaleString() });
      }
      if (leafCount != null && leafCount > totalRecords) {
        kpis.push({ label: 'Image Slices', value: leafCount.toLocaleString() });
      }
      kpis.push({ label: 'Total Size', value: formatBytes(totalBytes) });
      kpis.push({ label: 'File Extensions', value: formatFileTypes(allExtensions) });
      if (nGroups > 1) {
        kpis.push({ label: 'Groups', value: String(nGroups) });
      }

      const kpiRow = document.createElement('div');
      kpiRow.className = 'kpi-row';
      kpiRow.innerHTML = kpis.map(k => `
        <div class="kpi-tile">
          <div class="kpi-value">${escapeHtml(k.value)}</div>
          <div class="kpi-label">${escapeHtml(k.label)}</div>
        </div>
      `).join('');
      container.appendChild(kpiRow);

      if (basePath) {
        const basePathDiv = document.createElement('div');
        basePathDiv.className = 'summary-meta-line';
        basePathDiv.innerHTML = `<strong>Base path:</strong> <code>${escapeHtml(basePath)}</code>`;
        container.appendChild(basePathDiv);
      }

      if (nGroups > 1) {
        const groupedByDiv = document.createElement('div');
        groupedByDiv.className = 'summary-meta-line';
        groupedByDiv.innerHTML = `<strong>Grouped by:</strong> <code>${escapeHtml(groupColLabel)}</code>` +
          (isImportedPathShort
            ? ' &mdash; the top-level imported folder each file came from, relative to the common base path of all imported folders'
            : '');
        container.appendChild(groupedByDiv);
      }

      if (nGroups > 1) {
        const showExtCol  = allExtensions.size > 1;
        const maxFileCount = Math.max(...rows.map(r => Number(r.file_count)));
        const maxBytes     = Math.max(...rows.map(r => Number(r.total_bytes ?? 0)));
        const table = document.createElement('table');
        table.className = 'stat-table';
        table.innerHTML = `
          <thead>
            <tr>
              <th>${escapeHtml(groupColLabel)}</th>
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
                <td>${escapeHtml(ctx.groupLabel(String(r.__group__)))}</td>
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

    } catch {
      container.innerHTML = '<div class="no-data">Failed to load data.</div>';
    }
  },
};

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

function formatBytes(v) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let val = Number(v), u = 0;
  while (val >= 1024 && u < units.length - 1) { val /= 1024; u++; }
  return val >= 10 ? `${Math.round(val)} ${units[u]}` : `${val.toFixed(1)} ${units[u]}`;
}
