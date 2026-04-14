import { q, groupCol } from '../sql.js';
import { niceName } from '../plot-utils.js';
import { groupColor } from '../colors.js';
import { appendPlots, bargap } from '../plot-utils.js';

/**
 * File Data Summary — matches Dash FileSummaryWidget:
 *   - Intro text with per-group breakdown
 *   - Bar charts: file count + total size
 *   - Summary table: group / count / size / file types
 */
export default {
  id: 'summary',
  label: 'File Data Summary',
  info: 'Summarizes file counts, total size, and file types present in each group.',

  requires(schema) {
    return schema.allCols.includes('size_bytes') &&
           schema.allCols.includes('file_extension');
  },

  async render(container, ctx) {
    const groupLabel = ctx.state.groupCol ?? 'group';
    const gcExpr     = groupCol(ctx.state);

    const rows = await ctx.queryRows(`
      SELECT ${gcExpr} AS __group__,
             COUNT(*)                              AS file_count,
             SUM("size_bytes") / (1024.0 * 1024)  AS total_size_mb,
             LIST(DISTINCT "file_extension")       AS file_types
      FROM pp_data ${ctx.where}
      GROUP BY 1
      ORDER BY 1
    `);

    if (!rows.length) {
      container.innerHTML = '<div class="no-data">No data available after filtering.</div>';
      return;
    }

    const groups = rows.map(r => String(r.__group__));
    const colors = groups.map(g => groupColor(ctx.colorMap, g));

    // ── Intro text ────────────────────────────────────────────────────────
    const nGroups = rows.length;
    const intro   = document.createElement('div');
    intro.style.marginBottom = '16px';

    const prettiedGroupLabel = groupLabel.replace(/_/g, ' ');
    let introHtml = `<p>This summary focuses on file properties across ${nGroups} group(s) (by '${prettiedGroupLabel}').</p>`;
    for (const row of rows) {
      const ftStr = formatFileTypes(row.file_types);
      introHtml += `<p>Group '<strong>${row.__group__}</strong>' contains ` +
        `${Number(row.file_count).toLocaleString()} files ` +
        `(${Number(row.total_size_mb).toFixed(3)} MB) with types: ${ftStr}.</p>`;
    }
    intro.innerHTML = introHtml;
    container.appendChild(intro);

    // ── Bar charts ────────────────────────────────────────────────────────
    // Use appendPlots (not appendPlot) so both divs are in the DOM before
    // Plotly renders either one — ensures correct flex-box sizing for both.
    const barLayout = {
      height:     340,
      showlegend: false,
      margin:     { l: 50, r: 50, t: 50, b: 80 },
      bargap:     bargap(groups.length),
    };

    appendPlots(container, [
      {
        traces: [{ type: 'bar', x: groups, y: rows.map(r => Number(r.file_count)), marker: { color: colors } }],
        layout: { ...barLayout, title: { text: 'File Count per Group' }, xaxis: { title: prettiedGroupLabel, type: 'category' }, yaxis: { title: 'Number of files' } },
        divStyle: 'flex:1 1 320px',
      },
      {
        traces: [{ type: 'bar', x: groups, y: rows.map(r => Number(r.total_size_mb)), marker: { color: colors } }],
        layout: { ...barLayout, title: { text: 'Total Size per Group (MB)' }, xaxis: { title: prettiedGroupLabel, type: 'category' }, yaxis: { title: 'Size (MB)' } },
        divStyle: 'flex:1 1 320px',
      },
    ], 'display:flex;flex-wrap:wrap;gap:16px;margin-bottom:20px');

    // ── Summary table ─────────────────────────────────────────────────────
    const table = document.createElement('table');
    table.className = 'stat-table';
    const prettyGroup = niceName(groupLabel);
    table.innerHTML = `
      <thead>
        <tr>
          <th>${prettyGroup}</th>
          <th>Number of Files</th>
          <th>Size (MB)</th>
          <th>File Extension</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(r => `
          <tr>
            <td>${r.__group__}</td>
            <td>${Number(r.file_count).toLocaleString()}</td>
            <td>${Number(r.total_size_mb).toFixed(3)}</td>
            <td>${formatFileTypes(r.file_types)}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    container.appendChild(table);
  },
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * DuckDB LIST() returns either an Array or an Arrow List vector.
 * Normalise to a comma-separated string.
 */
function formatFileTypes(val) {
  if (!val) return '—';
  if (Array.isArray(val)) return val.filter(Boolean).sort().join(', ');
  if (typeof val === 'string') return val;
  // Arrow list: iterate
  try {
    const arr = [...val].filter(Boolean);
    return arr.sort().join(', ');
  } catch {
    return String(val);
  }
}
