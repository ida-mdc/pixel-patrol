/**
 * Label count distribution plugin.
 *
 * Displays the number of distinct label IDs per record.
 * All label IDs are included, including zero.
 * Supports PixelPatrol's global grouping and color scheme.
 */

/**
 * Query label count frequencies per group directly in SQL.
 */
async function queryLabelCounts(ctx) {
  const { andWhere, groupExpr } = ctx.sql;

  return ctx.queryRows(`
    SELECT
      ${groupExpr()},
      "label_count" AS label_count,
      COUNT(*) AS frequency
    FROM pp_data
    ${andWhere(ctx.where, '"label_count" IS NOT NULL')}
    GROUP BY __group__, "label_count"
    ORDER BY "label_count", __group__
  `);
}

/**
 * Calculate global statistics from aggregated frequencies.
 */
function getStats(rows) {
  let min = Infinity;
  let max = -Infinity;
  let total = 0;
  let maxFrequency = 0;

  for (const row of rows) {
    const count = Number(row.label_count);
    const frequency = Number(row.frequency);

    min = Math.min(min, count);
    max = Math.max(max, count);
    total += frequency;
    maxFrequency = Math.max(maxFrequency, frequency);
  }

  return { min, max, total, maxFrequency };
}

/**
 * Build one bar trace per group using PixelPatrol's
 * existing group labels and colors.
 */
function buildTraces(ctx, rows) {
  const grouped = new Map();

  for (const row of rows) {
    const group = String(row.__group__);

    if (!grouped.has(group)) {
      grouped.set(group, {
        x: [],
        y: [],
      });
    }

    const data = grouped.get(group);

    data.x.push(Number(row.label_count));
    data.y.push(Number(row.frequency));
  }

  return ctx.groups
    .filter(g => grouped.has(String(g)))
    .map(g => {
      const data = grouped.get(String(g));

      return {
        type: 'bar',
        name: ctx.groupLabel(g),

        x: data.x,
        y: data.y,

        marker: {
          color: ctx.color.group(g),
        },

        hovertemplate:
          'Distinct labels: %{x:d}<br>' +
          'Number of records: %{y:d}' +
          '<extra>%{fullData.name}</extra>',
      };
    });
}

/**
 * Use integer ticks with adaptive spacing.
 */
function integerAxis(title, min, max) {
  const step = Math.max(
    1,
    Math.ceil((max - min) / 10)
  );

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
function invariantRows(ctx, rows, stats) {
  if (ctx.groups.length <= 1) {
    return [[
      stats.min.toLocaleString(),
      stats.total.toLocaleString(),
    ]];
  }

  const frequencies = new Map(
    rows.map(row => [
      String(row.__group__),
      row,
    ])
  );

  return ctx.groups
    .filter(g => frequencies.has(String(g)))
    .map(g => {
      const row = frequencies.get(String(g));

      return [
        ctx.groupLabel(g),
        Number(row.label_count).toLocaleString(),
        Number(row.frequency).toLocaleString(),
      ];
    });
}

/**
 * Build table headers depending on active grouping.
 */
function invariantHeaders(ctx) {
  return ctx.groups.length > 1
    ? ['Group', 'Distinct labels', 'Records']
    : ['Distinct labels', 'Records'];
}

/**
 * Shared plot layout for the full plot and overview.
 */
function buildLayout(ctx, stats, mini = false) {
  return {
    barmode: 'group',

    xaxis: integerAxis(
      mini
        ? 'Label count'
        : 'Number of distinct labels',
      stats.min,
      stats.max
    ),

    yaxis: {
      ...integerAxis(
        mini ? '' : 'Number of records',
        0,
        stats.maxFrequency
      ),
      rangemode: 'tozero',
      showticklabels: !mini,
    },

    showlegend: !mini && ctx.groups.length > 1,

    ...(mini ? {} : {
      title: {
        text: 'Label Count Distribution',
      },
      height: 420,
    }),
  };
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
    'Distributions are displayed separately for active groups.',
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

    const stats = getStats(rows);

    if (stats.min === stats.max) {
      return (
        `All <strong>${stats.total.toLocaleString()}</strong> records ` +
        'with label data contain exactly ' +
        `<strong>${stats.min.toLocaleString()}</strong> distinct label IDs.`
      );
    }

    return (
      'Label counts range from ' +
      `<strong>${stats.min.toLocaleString()}</strong> to ` +
      `<strong>${stats.max.toLocaleString()}</strong> across ` +
      `${stats.total.toLocaleString()} records with label data.`
    );
  },

  async overviewPlot(container, ctx) {
    const rows = await queryLabelCounts(ctx);

    if (!rows.length) {
      return false;
    }

    const stats = getStats(rows);

    // No variance: show a compact table.
    if (stats.min === stats.max) {
      ctx.plot.tilePreviewTable(
        container,
        invariantHeaders(ctx),
        invariantRows(ctx, rows, stats)
      );

      return true;
    }

    // Use the same group traces and colors as the full plot.
    ctx.plot.appendMini(
      container,
      buildTraces(ctx, rows),
      buildLayout(ctx, stats, true)
    );

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

    const stats = getStats(rows);

    // No variance: reuse PixelPatrol's invariant table.
    if (stats.min === stats.max) {
      ctx.plot.invariantTable(container, {
        title: 'Label count with no variance',
        headers: invariantHeaders(ctx),
        rows: invariantRows(ctx, rows, stats),
      });

      return;
    }

    // Reuse PixelPatrol's standard plot rendering.
    ctx.plot.append(
      container,
      buildTraces(ctx, rows),
      buildLayout(ctx, stats)
    );
  },
};