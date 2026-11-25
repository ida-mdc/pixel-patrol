/**
 * Global label size distribution plugin.
 *
 * Displays the pooled size distribution of individual labels.
 * Label ID 0 can optionally be excluded.
 */

const MAX_BINS = 100;

/**
 * Select label areas, optionally excluding label ID 0.
 *
 * Label IDs and areas are aligned by position.
 */
function areasExpr(includeZero) {
  if (includeZero) {
    return '"label_areas"';
  }

  return `
    list_filter(
      "label_areas",
      (area, i) -> "label_ids"[i] <> 0
    )
  `;
}

/**
 * Calculate global statistics without unnesting the lists.
 */
async function querySizeStats(ctx, includeZero) {
  const { andWhere } = ctx.sql;

  const [row] = await ctx.queryRows(`
    WITH masks AS (
      SELECT ${areasExpr(includeZero)} AS areas
      FROM pp_data
      ${andWhere(ctx.where, '"label_areas" IS NOT NULL')}
    )
    SELECT
      MIN(list_min(areas)) AS min_size,
      MAX(list_max(areas)) AS max_size,
      SUM(len(areas)) AS label_total
    FROM masks
  `);

  return {
    min: Number(row?.min_size ?? 0),
    max: Number(row?.max_size ?? 0),
    total: Number(row?.label_total ?? 0),
  };
}

/**
 * Adaptive bin count based on the square-root rule (overview).
 */
function getBinCount(total) {
  return Math.min(
    MAX_BINS,
    Math.max(20, Math.ceil(Math.sqrt(total)))
  );
}

/**
 * Calculate histogram frequencies directly in DuckDB.
 */
async function queryHistogram(
  ctx, stats, logarithmic, includeZero,
  nBins = getBinCount(stats.total)
) {
  const { andWhere, groupExpr } = ctx.sql;

  const min = logarithmic
    ? Math.log10(stats.min)
    : stats.min;

  const max = logarithmic
    ? Math.log10(stats.max)
    : stats.max;

  const step = (max - min) / nBins;

  const valueExpr = logarithmic
    ? 'LOG10(label_size)'
    : 'label_size';

  const rows = await ctx.queryRows(`
    WITH labels AS (
      SELECT
        ${groupExpr()},
        UNNEST(${areasExpr(includeZero)}) AS label_size
      FROM pp_data
      ${andWhere(ctx.where, '"label_areas" IS NOT NULL')}
    ),
    binned AS (
      SELECT
        __group__,
        LEAST(
          ${nBins - 1},
          GREATEST(
            0,
            CAST(
              FLOOR((${valueExpr} - ${min}) / ${step})
              AS INTEGER
            )
          )
        ) AS bin
      FROM labels
    )
    SELECT
      __group__,
      bin,
      COUNT(*) AS frequency
    FROM binned
    GROUP BY __group__, bin
    ORDER BY __group__, bin
  `);

  return { rows, min, step, nBins };
}

/**
 * Build filled step histograms using shared bin boundaries.
 */
function buildTraces(ctx, histogram, logarithmic) {
  const { rows, min, step, nBins } = histogram;

  const frequencies = new Map();

  for (const row of rows) {
    const group = String(row.__group__);

    if (!frequencies.has(group)) {
      frequencies.set(group, new Array(nBins).fill(0));
    }

    frequencies.get(group)[Number(row.bin)] =
      Number(row.frequency);
  }

  const edges = Array.from(
    { length: nBins + 1 },
    (_, i) => {
      const value = min + i * step;
      return logarithmic ? 10 ** value : value;
    }
  );

  return ctx.groups
    .filter(g => frequencies.has(String(g)))
    .map(g => {
      const counts = frequencies.get(String(g));

      const x = [];
      const y = [];
      const customdata = [];

      for (let i = 0; i < nBins; i++) {
        const left = edges[i];
        const right = edges[i + 1];
        const count = counts[i];

        x.push(left, right);
        y.push(count, count);

        customdata.push(
          [left, right],
          [left, right]
        );
      }

      return {
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        opacity: 0.6,

        name: ctx.groupLabel(g),

        x,
        y,
        customdata,

        line: {
          color: ctx.color.group(g),
          width: 1.5,
          shape: 'linear',
        },

        hovertemplate:
          'Size range: %{customdata[0]:.4g} – ' +
          '%{customdata[1]:.4g}<br>' +
          'Label occurrences: %{y:d}' +
          '<extra>%{fullData.name}</extra>',
      };
    });
}

/**
 * Shared plot layout.
 */
function buildLayout(ctx, logarithmic, mini = false) {
  return {
    xaxis: {
      title: mini
        ? 'Label size'
        : 'Label size (pixels / voxels)',

      type: logarithmic ? 'log' : 'linear',
      tickformat: '~s',

      ...(mini
        ? {
            tickmode: 'auto',
            nticks: 4,
          }
        : logarithmic
          ? {
              dtick: 'D2',
            }
          : {
              nticks: 8,
            }),
    },

    yaxis: {
      title: mini ? '' : 'Number of label occurrences',
      rangemode: 'tozero',
      tickformat: 'd',
      showticklabels: !mini,
    },

    showlegend: !mini && ctx.groups.length > 1,

    ...(mini ? {} : {
      title: { text: 'Global Label Size Distribution' },
      height: 420,
    }),
  };
}

/**
 * Table rows for invariant label sizes.
 */
function invariantRows(stats) {
  return [[
    stats.min.toLocaleString(),
    stats.total.toLocaleString(),
  ]];
}

/**
 * Build controls using PixelPatrol's existing form styles.
 */
function buildControls(container) {
  const controls = document.createElement('div');

  controls.style.cssText =
    'display:flex;align-items:center;flex-wrap:wrap;' +
    'gap:20px;margin-bottom:14px;font-size:0.88em';

  // Linear / logarithmic switch.
  const scaleWrap = document.createElement('div');

  scaleWrap.style.cssText =
    'display:flex;align-items:center;gap:10px';

  const linearLabel = document.createElement('span');
  linearLabel.textContent = 'Linear bins';

  const switchWrap = document.createElement('div');
  switchWrap.className = 'form-check form-switch mb-0';

  const logToggle = document.createElement('input');
  logToggle.className = 'form-check-input';
  logToggle.type = 'checkbox';
  logToggle.checked = true;
  logToggle.setAttribute('role', 'switch');
  logToggle.setAttribute('aria-label', 'Logarithmic bins');

  switchWrap.appendChild(logToggle);

  const logLabel = document.createElement('span');
  logLabel.textContent = 'Logarithmic bins';

  scaleWrap.append(linearLabel, switchWrap, logLabel);

  // Include / exclude label ID 0.
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
  controls.append(scaleWrap, zeroWrap);

  // Interactive bin count.
  const binWrap = document.createElement('label');

  binWrap.style.cssText =
    'display:flex;align-items:center;gap:8px';

  const binLabel = document.createElement('span');
  binLabel.textContent = 'Bins:';

  const binSlider = document.createElement('input');
  binSlider.type = 'range';
  binSlider.className = 'form-range';
  binSlider.min = '5';
  binSlider.max = String(MAX_BINS);
  binSlider.step = '1';
  binSlider.value = '20';
  binSlider.style.width = '120px';
  binSlider.setAttribute('aria-label', 'Number of bins');

  const binValue = document.createElement('span');
  binValue.textContent = binSlider.value;

  binSlider.addEventListener('input', () => {
    binValue.textContent = binSlider.value;
  });

  binWrap.append(binLabel, binSlider, binValue);
  controls.appendChild(binWrap);

  container.appendChild(controls);

  return { logToggle, zeroToggle, binSlider };
}

export default {
  id: 'label-size-distribution',
  label: 'Global Label Size Distribution',
  shortLabel: 'Label Sizes',
  group: 'Labels',
  scope: 'image',

  required_inputs: ['label_areas', 'label_ids'],

  info: [
    'Shows the pooled size distribution of individual labels.',
    'Each label occurrence contributes one size value.',
    'Sizes are measured in pixels or voxels.',
    '',
    '**Display mode**',
    '- **Linear** – equally spaced size intervals.',
    '- **Logarithmic** – logarithmically spaced size intervals.',
    '',
    '**Include label ID 0**',
    'Includes or excludes label ID 0.',
    'Label ID 0 is not necessarily the background.',
    '',
    'The number of bins can be adjusted from 5 to 100.',
    'The overview uses an adaptive bin count.',
    'Group histograms use shared bin boundaries.',
    'If all labels have the same size, a table is shown instead.',
  ].join('\n'),

  requires(schema) {
    return (
      schema.allCols.includes('label_areas') &&
      schema.allCols.includes('label_ids')
    );
  },

  async overviewMessage(ctx) {
    const stats = await querySizeStats(ctx, true);

    if (!stats.total) {
      return 'No label data available.';
    }

    if (stats.min === stats.max) {
      return (
        `All <strong>${stats.total.toLocaleString()}</strong> ` +
        'labels have a size of ' +
        `<strong>${stats.min.toLocaleString()}</strong> ` +
        'pixels / voxels.'
      );
    }

    return (
      `<strong>${stats.total.toLocaleString()}</strong> labels ` +
      'with sizes ranging from ' +
      `<strong>${stats.min.toLocaleString()}</strong> to ` +
      `<strong>${stats.max.toLocaleString()}</strong> ` +
      'pixels / voxels.'
    );
  },

  async overviewPlot(container, ctx) {
    const stats = await querySizeStats(ctx, true);

    if (!stats.total) {
      return false;
    }

    if (stats.min === stats.max) {
      ctx.plot.tilePreviewTable(
        container,
        ['Label size', 'Labels'],
        invariantRows(stats)
      );

      return true;
    }

    const histogram = await queryHistogram(
      ctx, stats, true, true
    );

    ctx.plot.appendMini(
      container,
      buildTraces(ctx, histogram, true),
      buildLayout(ctx, true, true)
    );

    return true;
  },

  async render(container, ctx) {
    const { logToggle, zeroToggle, binSlider } =
      buildControls(container);

    const plotRoot = document.createElement('div');
    container.appendChild(plotRoot);

    // Cache statistics for each inclusion mode.
    const statsCache = new Map();

    // Cache histograms for each combination of controls.
    const histogramCache = new Map();

    async function draw() {
      const logarithmic = logToggle.checked;
      const includeZero = zeroToggle.checked;
      const nBins = Number(binSlider.value);

      const key = `${logarithmic}:${includeZero}:${nBins}`;

      logToggle.disabled = true;
      zeroToggle.disabled = true;
      binSlider.disabled = true;

      try {
        if (!statsCache.has(includeZero)) {
          statsCache.set(
            includeZero,
            await querySizeStats(ctx, includeZero)
          );
        }

        const stats = statsCache.get(includeZero);

        // Remove previous Plotly plots.
        for (const plot of plotRoot.querySelectorAll(
          '.js-plotly-plot'
        )) {
          Plotly.purge(plot);
        }

        plotRoot.replaceChildren();

        if (!stats.total) {
          plotRoot.textContent = 'No label data available.';
          return;
        }

        if (stats.min === stats.max) {
          ctx.plot.invariantTable(plotRoot, {
            title: 'Label sizes with no variance',
            headers: [
              'Label size (pixels / voxels)',
              'Number of label occurrences',
            ],
            rows: invariantRows(stats),
          });

          return;
        }

        if (!histogramCache.has(key)) {
          histogramCache.set(
            key,
            await queryHistogram(
              ctx, stats, logarithmic, includeZero, nBins
            )
          );

          // Bound the cache to eight aggregated histograms.
          if (histogramCache.size > 8) {
            const oldestKey = histogramCache.keys().next().value;
            histogramCache.delete(oldestKey);
          }
        }

        const histogram = histogramCache.get(key);

        ctx.plot.append(
          plotRoot,
          buildTraces(ctx, histogram, logarithmic),
          buildLayout(ctx, logarithmic)
        );
      } catch (error) {
        console.error('Label size distribution:', error);

        plotRoot.textContent =
          'Could not calculate label size distribution.';
      } finally {
        logToggle.disabled = false;
        zeroToggle.disabled = false;
        binSlider.disabled = false;
      }
    }

    logToggle.addEventListener('change', draw);
    zeroToggle.addEventListener('change', draw);
    binSlider.addEventListener('change', draw);

    await draw();
  },
};