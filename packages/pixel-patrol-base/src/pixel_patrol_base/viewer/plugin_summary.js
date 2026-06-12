export default {
  id: 'summary',
  label: 'File Data Summary',
  group: 'Summary',
  scope: 'file',
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

      const rows = await ctx.queryRows(`
        SELECT ${gcExpr} AS __group__,
               COUNT(*)                         AS file_count,
               SUM("size_bytes")                AS total_bytes,
               LIST(DISTINCT "file_extension")  AS file_types
        FROM pp_data ${ctx.where}
        GROUP BY 1
        ORDER BY 1
      `);

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

      const totalFiles    = rows.reduce((sum, r) => sum + Number(r.file_count), 0);
      const totalBytes    = rows.reduce((sum, r) => sum + Number(r.total_bytes ?? 0), 0);
      const allExtensions = new Set();
      for (const r of rows) for (const ext of (r.file_types ?? [])) if (ext) allExtensions.add(ext);

      const kpis = [
        { label: 'Files', value: totalFiles.toLocaleString() },
        { label: 'Total Size', value: formatBytes(totalBytes) },
        { label: 'File Extensions', value: formatFileTypes(allExtensions) },
      ];
      if (nGroups > 1) {
        kpis.push({ label: `Groups (by '${niceName(ctx.state.groupCol ?? 'group')}')`, value: String(nGroups) });
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

      if (nGroups > 1) {
        const showExtCol = allExtensions.size > 1;
        const table = document.createElement('table');
        table.className = 'stat-table';
        table.innerHTML = `
          <thead>
            <tr>
              <th>${niceName(ctx.state.groupCol ?? 'group')}</th>
              <th>Files</th>
              <th>Size</th>
              ${showExtCol ? '<th>File Extension</th>' : ''}
            </tr>
          </thead>
          <tbody>
            ${rows.map(r => `
              <tr>
                <td>${escapeHtml(ctx.groupLabel(String(r.__group__)))}</td>
                <td>${Number(r.file_count).toLocaleString()}</td>
                <td>${formatBytes(Number(r.total_bytes ?? 0))}</td>
                ${showExtCol ? `<td>${escapeHtml(formatFileTypes(r.file_types))}</td>` : ''}
              </tr>
            `).join('')}
          </tbody>
        `;
        container.appendChild(table);
      }

    } catch {
      container.innerHTML = '<div class="no-data">Failed to load data.</div>';
    }
  },
};

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
