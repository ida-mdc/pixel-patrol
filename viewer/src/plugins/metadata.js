import { q, groupCol, andWhere } from '../sql.js';
import { groupColor } from '../colors.js';
import { META_COLS } from '../schema.js';
import { LEGEND, appendPlot, niceName } from '../plot-utils.js';

// Categorical columns that get a distribution bar chart (stacked by group).
const DIST_COLS = ['dtype', 'dim_order'];

/**
 * Metadata plugin — combines two Dash widgets:
 *   1. Data Type Distribution (ColumnCountWithGroupingBarWidget for 'dtype')
 *   2. Dim Order Distribution (same pattern for 'dim_order')
 *   3. Invariant-properties table (constant values across all rows)
 *   4. Dimension range summary
 */
export default {
  id: 'metadata',
  label: 'Metadata',
  info: 'Shows the distribution of **pixel data types** and **dimension ordering** across groupings.\n\nAlso lists properties shared by all files, and available dimension ranges.',

  requires(schema) {
    const hasDist = DIST_COLS.some(c => schema.allCols.includes(c));
    const hasMeta = META_COLS.some(c => schema.allCols.includes(c));
    return hasDist || hasMeta || Object.keys(schema.dimensionInfo).length > 0;
  },

  async render(container, ctx) {
    // ── Distribution charts for categorical metadata columns ──────────────
    for (const col of DIST_COLS) {
      if (!ctx.schema.allCols.includes(col)) continue;

      const gcExpr = groupCol(ctx.state);
      const rows   = await ctx.queryRows(`
        SELECT ${q(col)} AS __cat__,
               ${gcExpr} AS __group__,
               COUNT(*)  AS count
        FROM pp_data ${andWhere(ctx.where, `${q(col)} IS NOT NULL`)}
        GROUP BY 1, 2
        ORDER BY 1, 2
      `);

      const totalRows = await ctx.queryRows(
        `SELECT COUNT(*) AS n FROM pp_data ${ctx.where}`,
      );
      const total = Number(totalRows[0]?.n ?? 0);
      const withData = rows.reduce((s, r) => s + Number(r.count), 0);
      const ratio = total > 0 ? ((withData / total) * 100).toFixed(2) : '0.00';

      const label = niceName(col);
      const ratioText = document.createElement('p');
      ratioText.style.marginBottom = '12px';
      ratioText.textContent =
        `${withData.toLocaleString()} of ${total.toLocaleString()} files ` +
        `(${ratio}%) have '${label}' information.`;
      container.appendChild(ratioText);

      if (!rows.length) continue;

      const cats   = [...new Set(rows.map(r => String(r.__cat__)))].sort();
      const groups = ctx.groups;

      // One trace per group, stacked bars.
      const traces = groups.map(g => ({
        type:   'bar',
        name:   g,
        x:      cats,
        y:      cats.map(cat => {
          const r = rows.find(r => String(r.__cat__) === cat && String(r.__group__) === g);
          return r ? Number(r.count) : 0;
        }),
        marker: { color: groupColor(ctx.colorMap, g) },
      }));

      appendPlot(container, traces, {
        title:      { text: `${label} Distribution` },
        barmode:    'stack',
        height:     500,
        xaxis:      { title: label, type: 'category' },
        yaxis:      { title: 'Count' },
        showlegend: groups.length > 1,
        ...(groups.length > 1 ? { legend: LEGEND } : {}),
      }, 'margin-bottom:24px');
    }

    // ── Invariant properties table ────────────────────────────────────────
    const available = META_COLS.filter(c => ctx.schema.allCols.includes(c) &&
                                           !DIST_COLS.includes(c));
    const invariants = [];

    if (available.length) {
      // Only show columns that are truly invariant across the *filtered* dataset.
      // The previous implementation used LIMIT 1 (single-row sample), which can
      // incorrectly claim varying columns are "shared".
      const colsToCheck = available;

      const exprs = colsToCheck.map(c => {
        const col = q(c);
        const baseInvariant = `(COUNT(*) = COUNT(${col}) AND COUNT(DISTINCT ${col}) = 1)`;
        // Dimension size columns should only be considered "shared" if every file truly has that
        // dimension (we treat <= 1 as effectively missing, matching other widgets).
        const invariant =
          c.endsWith('_size')
            ? `(${baseInvariant} AND MIN(${col}) > 1)`
            : baseInvariant;
        return `
          CASE
            WHEN ${invariant} THEN CAST(MIN(${col}) AS VARCHAR)
            ELSE NULL
          END AS ${col}
        `;
      }).join(',\n');

      const rows = await ctx.queryRows(`
        SELECT
          ${exprs}
        FROM pp_data ${ctx.where}
      `);

      const row = rows[0] ?? {};
      for (const col of colsToCheck) {
        const val = row[col];
        if (val != null && String(val) !== '') {
          invariants.push({ Property: col.replace(/_/g, ' '), Value: String(val) });
        }
      }
    }

    // Dimension ranges
    for (const [dim, indices] of Object.entries(ctx.schema.dimensionInfo)) {
      invariants.push({
        Property: `Dimension ${dim.toUpperCase()}`,
        Value:    `${indices[0]}–${indices[indices.length - 1]} (${indices.length} steps)`,
      });
    }

    if (invariants.length) {
      const h = document.createElement('h6');
      h.style.cssText = 'margin-top:16px;margin-bottom:10px';
      h.textContent = 'Properties shared between all files';
      container.appendChild(h);

      const table = document.createElement('table');
      table.className = 'stat-table';
      table.innerHTML = `
        <thead><tr><th>Property</th><th>Value</th></tr></thead>
        <tbody>
          ${invariants.map(r => `<tr><td>${r.Property}</td><td>${r.Value}</td></tr>`).join('')}
        </tbody>
      `;
      container.appendChild(table);
    }

    if (!container.firstChild) {
      container.innerHTML = '<div class="no-data">No metadata columns found.</div>';
    }
  },
};
