/**
 * Widget: NoData Value Frequency
 *
 * Bar plot showing how many images use each distinct no-data value.
 * Assumes `nodata_value` holds a single Optional[int|float] per image.
 */
import { pick, renderGroupedBars } from '../../extension/1/plugin_file_stats.js';

export default {
  id: 'nodata-value-count',
  label: 'NoData',
  group: 'no-data',
  scope: 'image',
  info: 'Counts how many images use each distinct no-data value.',
  requires(schema) {
    return schema.allCols.includes('nodata_value');
  },

  async render(container, ctx) {
    const { andWhere, groupCol: gcFn } = ctx.sql;
    const gcExpr = gcFn();

    // Aggregate: one row per (nodata_value, group) with its image count.
    const rows = await ctx.queryRows(`
      SELECT COALESCE(CAST("nodata_value" AS VARCHAR), 'not set') AS nodata_value,
             ${gcExpr} AS __group__, COUNT(*) AS count
      FROM pp_data ${ctx.where}
      GROUP BY 1, 2 ORDER BY 1, 2
    `);
    if (!rows.length) {
      container.innerHTML = '<div class="no-data">No no-data values available.</div>';
      return;
    }

    // Distinct category labels, numerically sorted (string sort would misorder 2 vs 10).
    const cats = [...new Set(rows.map(r => String(r.nodata_value)))]
      .sort((a, b) => {
        if (a === 'not set') return 1;
        if (b === 'not set') return -1;
        return Number(a) - Number(b);
      });

    renderGroupedBars(container, {
      categories: cats,
      getValue:   pick(rows, r => r.nodata_value, 'count'),
      title:      'Image Count by NoData Value',
      xLabel:     'NoData value',
      yLabel:     'Image count',
    }, ctx);
  },
};

