const NUM_COL = 'label_size';

// ---------------------------------------------------------------------------
// SQL data source
// ---------------------------------------------------------------------------

function buildSource(ctx, relative = false, includeZero = true) {
  const { andWhere, groupExpr } = ctx.sql;

  const sizeExpr = relative
    ? 'label_area::DOUBLE / NULLIF(mask_size, 0)'
    : 'label_area';

  const table = `(
    SELECT
      file_row_number,
      __group__,
      label_id,
      ${sizeExpr} AS ${NUM_COL}
    FROM (
      SELECT
        file_row_number,
        ${groupExpr()},
        UNNEST("label_ids") AS label_id,
        UNNEST("label_areas") AS label_area,
        list_sum("label_areas") AS mask_size
      FROM pp_data
      ${andWhere(
        ctx.where,
        '"label_ids" IS NOT NULL AND "label_areas" IS NOT NULL'
      )}
    ) AS expanded_labels
    WHERE label_id IS NOT NULL
      AND label_area IS NOT NULL
      ${includeZero ? '' : 'AND label_id <> 0'}
  ) AS label_data`;

  return {
    table,
    where: '',
  };
}


// ---------------------------------------------------------------------------
// Summary statistics
// ---------------------------------------------------------------------------

async function queryStats(ctx, source) {
  return ctx.queryRows(`
    SELECT
      __group__,
      CAST(label_id AS VARCHAR) AS __cat__,
      COUNT(label_size) AS n,
      MIN(label_size) AS mn,
      MAX(label_size) AS mx,
      AVG(label_size) AS mean,
      STDDEV(label_size) AS sd,
      approx_quantile(label_size, 0.25) AS q1,
      approx_quantile(label_size, 0.5) AS med,
      approx_quantile(label_size, 0.75) AS q3
    FROM ${source.table}
    WHERE label_size IS NOT NULL
    GROUP BY 1, 2
  `);
}


function getLabelIds(stats) {
  return [...new Set(stats.map(row => String(row.__cat__)))]
    .sort((a, b) => {
      const x = BigInt(a);
      const y = BigInt(b);

      return x < y ? -1 : x > y ? 1 : 0;
    });
}


function getSummary(stats) {
  let total = 0;
  let min = Infinity;
  let max = -Infinity;

  for (const row of stats) {
    total += Number(row.n);
    min = Math.min(min, Number(row.mn));
    max = Math.max(max, Number(row.mx));
  }

  return {
    total,
    min,
    max,
    ids: getLabelIds(stats),
  };
}


// ---------------------------------------------------------------------------
// Grouping
// ---------------------------------------------------------------------------

function buildSeries(ctx) {
  return {
    isCategory: false,
    sql: '__group__',

    build(rows) {
      const present = new Set(
        rows.map(row => String(row.__group__))
      );

      return {
        groups: ctx.groups
          .map(String)
          .filter(group => present.has(group)),

        colorFn: group => ctx.color.group(group),
        labelFn: group => ctx.groupLabel(group),
      };
    },
  };
}


// ---------------------------------------------------------------------------
// Formatting and invariant tables
// ---------------------------------------------------------------------------

function sizeLabel(relative) {
  return relative
    ? 'Relative label size (%)'
    : 'Label size (pixels / voxels)';
}


function formatSize(value, relative) {
  const number = Number(value);

  return relative
    ? `${(number * 100).toFixed(2)}%`
    : number.toLocaleString();
}


function renderInvariantTable(container, ctx, stats, relative) {
  const grouped = ctx.groups.length > 1;

  const headers = [
    'Label ID',
    ...(grouped ? ['Group'] : []),
    sizeLabel(relative),
    'Occurrences',
  ];

  const rows = [...stats]
    .sort((a, b) => {
      const x = BigInt(a.__cat__);
      const y = BigInt(b.__cat__);

      return x < y ? -1 : x > y ? 1 : 0;
    })
    .map(row => [
      String(row.__cat__),

      ...(grouped
        ? [ctx.groupLabel(String(row.__group__))]
        : []),

      formatSize(row.mn, relative),
      Number(row.n).toLocaleString(),
    ]);

  ctx.plot.invariantTable(container, {
    title: 'Label sizes with no variance',
    headers,
    rows,
  });
}


// ---------------------------------------------------------------------------
// Distribution plot
// ---------------------------------------------------------------------------

async function renderDistribution(
  container,
  ctx,
  source,
  stats,
  relative
) {
  const summary = getSummary(stats);

  if (!summary.total) {
    container.textContent = 'No label data available.';
    return;
  }

  if (summary.min === summary.max) {
    renderInvariantTable(container, ctx, stats, relative);
    return;
  }

  await ctx.plot.engine.renderDistribution(container, ctx, {
    numCol: NUM_COL,
    source,

    catSql: 'CAST(label_id AS VARCHAR)',
    catLabel: 'Label ID',

    yLabel: sizeLabel(relative),
    title: 'Label Size Distribution per ID',

    series: buildSeries(ctx),

    stats,

    showSignificance: false,

    maxRawPoints:
      ctx.plot.engine.CONSTANTS.MAX_VIOLIN_POINTS,

    layout: {
      xaxis: {
        title: 'Label ID',
        type: 'category',
        categoryorder: 'array',
        categoryarray: summary.ids,
      },

      yaxis: {
        title: sizeLabel(relative),
        rangemode: 'tozero',
        tickformat: relative ? '.1%' : '~s',
      },

      violinmode: 'group',
      boxmode: 'group',

      height: 420,
    },
  });
}


// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

function buildControls(container) {
  const controls = document.createElement('div');

  controls.style.cssText = `
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 20px;
    margin-bottom: 14px;
    font-size: 0.88em;
  `;

  // Absolute / Relative size toggle

  const sizeWrap = document.createElement('div');

  sizeWrap.style.cssText = `
    display: flex;
    align-items: center;
    gap: 10px;
  `;

  const absoluteLabel = document.createElement('span');
  absoluteLabel.textContent = 'Absolute size';

  const switchWrap = document.createElement('div');
  switchWrap.className = 'form-check form-switch mb-0';

  const relativeToggle = document.createElement('input');

  relativeToggle.className = 'form-check-input';
  relativeToggle.type = 'checkbox';
  relativeToggle.checked = false;

  relativeToggle.setAttribute('role', 'switch');
  relativeToggle.setAttribute('aria-label', 'Relative size');

  switchWrap.appendChild(relativeToggle);

  const relativeLabel = document.createElement('span');
  relativeLabel.textContent = 'Relative size';

  sizeWrap.append(
    absoluteLabel,
    switchWrap,
    relativeLabel
  );

  // Include label ID 0

  const zeroWrap = document.createElement('label');

  zeroWrap.className = 'form-check mb-0';
  zeroWrap.style.cursor = 'pointer';

  const zeroToggle = document.createElement('input');

  zeroToggle.className = 'form-check-input';
  zeroToggle.type = 'checkbox';
  zeroToggle.checked = true;

  const zeroLabel = document.createElement('span');

  zeroLabel.className = 'form-check-label';
  zeroLabel.textContent = 'Include label ID 0';

  zeroWrap.append(zeroToggle, zeroLabel);

  controls.append(sizeWrap, zeroWrap);

  container.appendChild(controls);

  return {
    relativeToggle,
    zeroToggle,
  };
}


// ---------------------------------------------------------------------------
// Widget
// ---------------------------------------------------------------------------

export default {
  id: 'label-size-distribution-per-id',

  label: 'Label Size Distribution per ID',

  shortLabel: 'Label Sizes per ID',

  group: 'Labels',

  scope: 'image',

  required_inputs: [
    'label_ids',
    'label_areas',
  ],

  info: [
    'Shows the distribution of label sizes per label ID across masks.',
    '',
    'Each data point represents the size of a specific label ID within one mask.',
    '',
    'Only masks containing the respective label ID are included.',
    'Missing label IDs are not treated as zero-sized labels.',
    '',
    '**Display mode**',
    '',
    '- **Absolute size** – Number of pixels or voxels per label.',
    '- **Relative size** – Percentage of the total mask area or volume.',
    '',
    '**Include label ID 0**',
    '',
    'Includes or excludes label ID 0 from the visualization.',
    'Label ID 0 is not necessarily the background.',
    'Excluding ID 0 does not change the total mask size used for relative size calculations.',
    '',
    '**Grouping**',
    '',
    'Distributions are displayed separately for active groups.',
    '',
    '**Segmentation type**',
    '',
    'This widget is particularly useful for semantic segmentation, where label IDs have a consistent meaning across masks.',
    '',
    'For instance segmentation, comparisons across masks are only meaningful if label IDs are consistent across those masks.',
  ].join('\n'),

  requires(schema) {
    return (
      schema.allCols.includes('label_ids') &&
      schema.allCols.includes('label_areas')
    );
  },


  // -------------------------------------------------------------------------
  // Overview message
  // -------------------------------------------------------------------------

  async overviewMessage(ctx) {
    const source = buildSource(ctx, false, true);

    const [row] = await ctx.queryRows(`
      SELECT
        COUNT(DISTINCT label_id) AS label_count,
        COUNT(*) AS occurrence_count
      FROM ${source.table}
    `);

    const labelCount = Number(row?.label_count ?? 0);
    const occurrenceCount = Number(row?.occurrence_count ?? 0);

    if (!occurrenceCount) {
      return 'No label data available.';
    }

    return `
      <strong>${labelCount.toLocaleString()}</strong>
      distinct label IDs across
      <strong>${occurrenceCount.toLocaleString()}</strong>
      label occurrences.
    `;
  },


  // -------------------------------------------------------------------------
  // Overview plot
  // -------------------------------------------------------------------------

  async overviewPlot(container, ctx) {
  const source = buildSource(ctx, false, true);

  const stats = await queryStats(ctx, source);
  const summary = getSummary(stats);

  if (!summary.total) {
    return false;
  }

  if (summary.min === summary.max) {
    ctx.plot.tilePreviewTable(
      container,
      ['Label IDs', 'Label size', 'Occurrences'],
      [[
        summary.ids.length.toLocaleString(),
        formatSize(summary.min, false),
        summary.total.toLocaleString(),
      ]]
    );

    return true;
  }

  await ctx.plot.engine.renderDistribution(container, ctx, {
    numCol: NUM_COL,
    source,

    catSql: 'CAST(label_id AS VARCHAR)',
    catLabel: 'Label ID',

    yLabel: 'Label size',

    series: buildSeries(ctx),
    categoriesOrder: summary.ids,

    stats,

    showSignificance: false,

    mini: true,
    maxRawPoints: 0,
  });

  return true;
},


  // -------------------------------------------------------------------------
  // Full widget
  // -------------------------------------------------------------------------

  async render(container, ctx) {
    const {
      relativeToggle,
      zeroToggle,
    } = buildControls(container);

    const plotRoot = document.createElement('div');

    container.appendChild(plotRoot);

    const statsCache = new Map();

    async function draw() {
      const relative = relativeToggle.checked;
      const includeZero = zeroToggle.checked;

      const cacheKey = `${relative}:${includeZero}`;

      relativeToggle.disabled = true;
      zeroToggle.disabled = true;

      try {
        const source = buildSource(
          ctx,
          relative,
          includeZero
        );

        if (!statsCache.has(cacheKey)) {
          statsCache.set(
            cacheKey,
            await queryStats(ctx, source)
          );
        }

        const stats = statsCache.get(cacheKey);

        for (const plot of plotRoot.querySelectorAll(
          '.js-plotly-plot'
        )) {
          Plotly.purge(plot);
        }

        plotRoot.replaceChildren();

        await renderDistribution(
          plotRoot,
          ctx,
          source,
          stats,
          relative
        );

      } catch (error) {
        console.error(
          'Label size distribution per ID:',
          error
        );

        plotRoot.textContent =
          'Could not calculate label size distribution per ID.';

      } finally {
        relativeToggle.disabled = false;
        zeroToggle.disabled = false;
      }
    }

    relativeToggle.addEventListener('change', draw);
    zeroToggle.addEventListener('change', draw);

    await draw();
  },
};