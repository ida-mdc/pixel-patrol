/**
 * Label count distribution plugin.
 *
 * Displays the number of distinct label IDs per record.
 * All label IDs are included, including zero.
 */

/**
 * Query label count frequencies directly in SQL.
 */
async function queryLabelCounts(ctx) {
  const { andWhere } = ctx.sql;

  return ctx.queryRows(`
    SELECT
      "label_count" AS label_count,
      COUNT(*) AS frequency
    FROM pp_data
    ${andWhere(ctx.where, '"label_count" IS NOT NULL')}
    GROUP BY "label_count"
    ORDER BY "label_count"
  `);
}

/**
 * Build a shared bar trace for the full plot and overview.
 */
function buildTrace(rows) {
  return {
    type: 'bar',
    x: rows.map(r => Number(r.label_count)),
    y: rows.map(r => Number(r.frequency)),
    marker: { color: '#3d7ea6' },
    hovertemplate:
      'Distinct labels: %{x:d}<br>' +
      'Number of records: %{y:d}<extra></extra>',
  };
}

/**
 * Use integer ticks with adaptive spacing.
 */
function integerAxis(title, min, max) {
  const step = Math.max(1, Math.ceil((max - min) / 10));

  return {
    title,
    tickformat: 'd',
    tickmode: 'linear',
    tick0: 0,
    dtick: step,
  };
}

/**
 * Build the table for invariant label counts.
 */
function invariantRows(rows) {
  return [[
    Number(rows[0].label_count).toLocaleString(),
    Number(rows[0].frequency).toLocaleString(),
  ]];
}

export default {
  id: 'label-count',
  label: 'Label Count Distribution',
  shortLabel: 'Label Count',
  group: 'Labels',
  scope: 'image',

  required_inputs: ['label_count'],

  info: [
    'Shows the distribution of distinct label counts across records.',
    'All label IDs are included, including zero.',
    '',
    'If all records with label data have the same label count, ' +
    'a table is displayed instead of a plot.',
  ].join('\n'),

  requires(schema) {
    return schema.allCols.includes('label_count');
  },

  async overviewMessage(ctx) {
    const rows = await queryLabelCounts(ctx);

    if (!rows.length) {
      return 'No label data available.';
    }

    const total = rows.reduce(
      (sum, r) => sum + Number(r.frequency), 0
    );

    const min = Number(rows[0].label_count);
    const max = Number(rows[rows.length - 1].label_count);

    if (rows.length === 1) {
      return (
        `All <strong>${total.toLocaleString()}</strong> records ` +
        'with label data contain exactly ' +
        `<strong>${min.toLocaleString()}</strong> distinct label IDs.`
      );
    }

    return (
      'Label counts range from ' +
      `<strong>${min.toLocaleString()}</strong> to ` +
      `<strong>${max.toLocaleString()}</strong> across ` +
      `${total.toLocaleString()} records with label data.`
    );
  },

  async overviewPlot(container, ctx) {
    const rows = await queryLabelCounts(ctx);

    if (!rows.length) {
      return false;
    }

    // No variance: show a compact table.
    if (rows.length === 1) {
      ctx.plot.tilePreviewTable(
        container,
        ['Distinct labels', 'Records'],
        invariantRows(rows)
      );

      return true;
    }

    const min = Number(rows[0].label_count);
    const max = Number(rows[rows.length - 1].label_count);

    ctx.plot.appendMini(container, [buildTrace(rows)], {
      xaxis: integerAxis('Label count', min, max),
      yaxis: {
        showticklabels: false,
        rangemode: 'tozero',
      },
    });

    return true;
  },

  async render(container, ctx) {
    
    const rows = await queryLabelCounts(ctx);

    // No data available.
    if (!rows.length) {
      container.innerHTML =
        '<div class="no-data">No label data available.</div>';
      return;
    }

    // No variance: reuse PixelPatrol's invariant table.
    if (rows.length === 1) {
      ctx.plot.invariantTable(container, {
        title: 'Label count with no variance',
        headers: ['Distinct labels', 'Number of records'],
        rows: invariantRows(rows),
      });

      return;
    }

    const min = Number(rows[0].label_count);
    const max = Number(rows[rows.length - 1].label_count);

    const maxFrequency = rows.reduce(
      (max, r) => Math.max(max, Number(r.frequency)), 0
    );

    // Reuse PixelPatrol's standard plot rendering.
    ctx.plot.append(container, [buildTrace(rows)], {
      title: { text: 'Label Count Distribution' },
      height: 420,

      xaxis: integerAxis(
        'Number of distinct labels', min, max
      ),

      yaxis: {
        ...integerAxis('Number of records', 0, maxFrequency),
        rangemode: 'tozero',
      },
    });
  },
};