const METRICS = [
  {
    col: 'label_min_size',
    label: 'Minimum Label Size',
  },
  {
    col: 'label_max_size',
    label: 'Maximum Label Size',
  },
  {
    col: 'label_mean_size',
    label: 'Mean Label Size',
  },
  {
    col: 'label_median_size',
    label: 'Median Label Size',
  },
  {
    col: 'label_size_std',
    label: 'Label Size Standard Deviation',
  },
];


// ---------------------------------------------------------------------------
// Available metrics
// ---------------------------------------------------------------------------

function availableMetrics(schema) {
  return METRICS.filter(
    metric => schema.allCols.includes(metric.col)
  );
}


// ---------------------------------------------------------------------------
// Summary statistics
// ---------------------------------------------------------------------------

async function queryStats(ctx, metrics) {
  const {
    q,
    groupExpr,
    groupCol,
  } = ctx.sql;

  const gc = groupCol();

  const statCols = metrics.map(({ col }) => `
    COUNT(${q(col)}) AS "${col}__n",
    MIN(${q(col)}) AS "${col}__min",
    MAX(${q(col)}) AS "${col}__max",
    AVG(${q(col)}) AS "${col}__mean",
    STDDEV(${q(col)}) AS "${col}__sd",
    approx_quantile(
      ${q(col)}, 0.25
    ) AS "${col}__q1",
    approx_quantile(
      ${q(col)}, 0.5
    ) AS "${col}__median",
    approx_quantile(
      ${q(col)}, 0.75
    ) AS "${col}__q3",
    approx_count_distinct(
      ${q(col)}
    ) AS "${col}__nd"
  `).join(',\n');

  return ctx.queryRows(`
    SELECT
      ${groupExpr()},
      GROUPING(${gc}) AS __is_total__,
      COUNT(*) AS __row_count__,
      ${statCols}
    FROM pp_data
    ${ctx.where}
    GROUP BY ROLLUP(${gc})
  `);
}


// ---------------------------------------------------------------------------
// Metric-specific statistics
// ---------------------------------------------------------------------------

function metricStats(rows, groups, col) {
  return groups
    .map(group => {
      const row = rows.get(String(group));

      if (!row || !Number(row[`${col}__n`])) {
        return null;
      }

      return {
        __group__: group,
        __cat__: group,

        n: row[`${col}__n`],

        mn: row[`${col}__min`],
        mx: row[`${col}__max`],

        mean: row[`${col}__mean`],
        sd: row[`${col}__sd`],

        q1: row[`${col}__q1`],
        med: row[`${col}__median`],
        q3: row[`${col}__q3`],
      };
    })
    .filter(Boolean);
}


function formatSize(value) {
  return Number(value).toLocaleString(
    undefined,
    { maximumFractionDigits: 4 }
  );
}


// ---------------------------------------------------------------------------
// Full widget
// ---------------------------------------------------------------------------

async function renderStatistics(container, ctx) {
  const metrics = availableMetrics(ctx.schema);

  if (!metrics.length) {
    container.innerHTML =
      '<div class="no-data">No label statistics available.</div>';

    return;
  }

  const {
    flexGrid,
    dataAvailabilityWarning,
    groupingLabel,
    engine,
  } = ctx.plot;

  // Calculate all metric summaries in one SQL query.

  const rows = await queryStats(ctx, metrics);

  const totalRow = rows.find(
    row => Number(row.__is_total__) === 1
  );

  const groupRows = rows.filter(
    row => Number(row.__is_total__) !== 1
  );

  const totalRecords = Number(
    totalRow?.__row_count__ ?? 0
  );

  if (!totalRecords) {
    container.innerHTML =
      '<div class="no-data">No rows match the current filter.</div>';

    return;
  }

  const statsByGroup = new Map(
    groupRows.map(row => [
      String(row.__group__),
      row,
    ])
  );

  const groups = ctx.groups.filter(
    group => statsByGroup.has(String(group))
  );

  // Report missing metric values.

  const counts = metrics.map(({ col, label }) => ({
    label,
    present: Number(totalRow[`${col}__n`] ?? 0),
  }));

  dataAvailabilityWarning(
    container,
    counts,
    totalRecords,
    { unit: 'records' }
  );

  // Separate varying and invariant metrics.

  const toPlot = [];
  const noVariance = [];

  for (const metric of metrics) {
    const { col } = metric;

    const total = Number(totalRow[`${col}__n`]);

    if (!total) {
      continue;
    }

    const min = Number(totalRow[`${col}__min`]);
    const max = Number(totalRow[`${col}__max`]);

    if (min === max) {
      noVariance.push({
        ...metric,
        value: min,
      });
    } else {
      toPlot.push({
        ...metric,
        total,
      });
    }
  }

  // Same responsive arrangement as the intensity widget.

  const numGroups = groups.length;

  const plotsPerRow =
    numGroups <= 2
      ? 3
      : numGroups === 3
        ? 2
        : 1;

  const showSignificance =
    !!ctx.state.showSignificance;

  if (toPlot.length) {
    const {
      wrap,
      flexBasisPct,
    } = flexGrid(container, plotsPerRow);

    for (const { col, label, total } of toPlot) {
      const stats = metricStats(
        statsByGroup,
        groups,
        col
      );

      const cell = document.createElement('div');

      cell.style.cssText = `
        flex: 0 0 ${flexBasisPct}%;
        min-width: 300px;
        margin-bottom: 20px;
        box-sizing: border-box;
      `;

      wrap.appendChild(cell);

      const title =
        `Distribution of ${label}<br>` +
        `<sup>one point per record; ` +
        `n=${total.toLocaleString()}</sup>`;

      await engine.renderDistribution(cell, ctx, {
        numCol: col,

        source: {
          table: 'pp_data',
          where: ctx.where,
        },

        catSql: ctx.sql.groupCol(),
        catLabel: groupingLabel(''),

        yLabel: `${label} (pixels / voxels)`,

        title,

        showSignificance,

        maxRawPoints:
          engine.CONSTANTS.MAX_VIOLIN_POINTS,

        series: {
          isCategory: true,
        },

        categoriesOrder: groups,

        catLabelFn: ctx.groupLabel,

        stats,

        distinctCount: Number(
          totalRow[`${col}__nd`]
        ),
      });
    }
  }

  // Metrics that have the same value across all records.

  if (noVariance.length) {
    ctx.plot.invariantTable(container, {
      title: 'Metrics with No Variance',

      headers: [
        'Metric',
        'Value (pixels / voxels)',
      ],

      rows: noVariance.map(({ label, value }) => [
        label,
        formatSize(value),
      ]),

      hr: true,
    });
  }

  if (!toPlot.length && !noVariance.length) {
    const message = document.createElement('div');

    message.className = 'no-data';
    message.textContent = 'No label statistics available.';

    container.appendChild(message);
  }
}


// ---------------------------------------------------------------------------
// Widget
// ---------------------------------------------------------------------------

export default {
  id: 'label-size-statistics',

  label: 'Label Size Statistics',

  shortLabel: 'Label Statistics',

  group: 'Labels',

  scope: 'image',

  multiPlot: true,

  required_inputs: METRICS.map(
    metric => metric.col
  ),

  info: [
    'Shows per-mask label size statistics across the dataset.',
    '',
    'Each data point represents one label mask record.',
    'All label IDs are included, including zero.',
    'Sizes are measured in pixels or voxels.',
    '',
    '**Metrics**',
    '',
    '- **Minimum:** Size of the smallest label.',
    '- **Maximum:** Size of the largest label.',
    '- **Mean:** Average label size.',
    '- **Median:** Median label size.',
    '- **Standard deviation:** Population standard deviation of label sizes within a mask.',
    '',
    'Distributions are displayed separately for active groups.',
    '',
    'Metrics with no variance across records are displayed in a table.',
    '',
    'The plot type is selected automatically according to the number of data points and distinct values.',
    '',
    '**Segmentation type**',
    '',
    'For instance segmentation, label sizes usually represent individual object sizes.',
    'For semantic segmentation, label sizes represent total class areas or volumes.',
    'Aggregating across classes may obscure class-specific differences.',
    '',
    '**Statistical comparisons**',
    '',
    'Group comparisons can be enabled through the global statistical comparisons setting.',
  ].join('\n'),

  requires(schema) {
    return availableMetrics(schema).length > 0;
  },


  // -------------------------------------------------------------------------
  // Overview message
  // -------------------------------------------------------------------------

  async overviewMessage() {
    return 'Compare per-mask label size statistics across groups.';
  },


  // -------------------------------------------------------------------------
  // Overview plot
  // -------------------------------------------------------------------------

  async overviewPlot(container, ctx) {
    const available = availableMetrics(ctx.schema);

    const metric =
      available.find(
        m => m.col === 'label_median_size'
      ) ??
      available.find(
        m => m.col === 'label_mean_size'
      ) ??
      available[0];

    if (!metric) {
      return false;
    }

    return ctx.plot.engine.renderDistribution(
      container,
      ctx,
      {
        numCol: metric.col,

        source: {
          table: 'pp_data',
          where: ctx.where,
        },

        catSql: ctx.sql.groupCol(),

        yLabel: metric.label,

        series: {
          isCategory: true,
        },

        categoriesOrder: ctx.groups,

        catLabelFn: ctx.groupLabel,

        mini: true,

        maxRawPoints: 0,
      }
    );
  },


  // -------------------------------------------------------------------------
  // Full widget
  // -------------------------------------------------------------------------

  async render(container, ctx) {
    try {
      await renderStatistics(container, ctx);
    } catch (error) {
      console.error(
        'Label size statistics:',
        error
      );

      container.innerHTML =
        '<div class="no-data">Failed to load label statistics.</div>';
    }
  },
};