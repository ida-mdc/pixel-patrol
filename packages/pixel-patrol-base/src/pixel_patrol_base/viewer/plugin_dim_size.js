const MAX_XY_BUBBLE = 50_000;  // above this many files, switch the X/Y scatter to a heatmap
const HEATMAP_BINS  = 70;      // target bin count per axis for the X/Y heatmap

export default {
  id: 'dim-size',
  group: 'Metadata',
  scope: 'image',
  label: 'Dimensionality',
  info: [
    'Shows how **image dimensions** (X, Y, Z, T, …) vary across the dataset.',
    '',
    'Includes an X/Y size scatter plot and per-dimension strip plots.',
    '',
    '**Use this to identify**',
    '- unexpected dimension sizes',
    '- mismatched shapes between groupings',
  ].join('\n'),

  requires(schema) {
    return schema.allCols.some(c => c.endsWith('_size') && !c.startsWith('__'));
  },

  async condensedSummary(ctx) {
    try {
      const { q, andWhere } = ctx.sql;
      const sizeCols = ctx.schema.allCols.filter(c => c.endsWith('_size') && !c.startsWith('__'));
      const hasXY    = sizeCols.includes('X_size') && sizeCols.includes('Y_size');
      const KNOWN_DIMS = { Z_size: 'Z slices', T_size: 'timepoints', C_size: 'channels', S_size: 'series/tiles' };
      const extraDimCols = sizeCols.filter(c => c !== 'X_size' && c !== 'Y_size' && c !== 'num_pixels');

      const [{ total }] = await ctx.queryRows(`SELECT COUNT(*) AS total FROM pp_data ${ctx.where}`);
      const totalN = Number(total ?? 0);

      let xySizeVaries = false;
      if (hasXY) {
        const [row] = await ctx.queryRows(`
          SELECT COUNT(DISTINCT "X_size") AS ndx, COUNT(DISTINCT "Y_size") AS ndy
          FROM pp_data ${andWhere(ctx.where, '"X_size" > 1 AND "Y_size" > 1')}
        `);
        xySizeVaries = Number(row?.ndx ?? 0) > 1 || Number(row?.ndy ?? 0) > 1;
      }

      const present     = [];
      const raggedDims  = [];
      for (const col of extraDimCols) {
        const [r] = await ctx.queryRows(`SELECT MAX(${q(col)}) AS m, COUNT(*) AS n FROM pp_data ${andWhere(ctx.where, `${q(col)} > 1`)}`);
        const m = Number(r?.m ?? 0);
        if (m <= 1) continue;
        const label = KNOWN_DIMS[col] ?? col;
        present.push(label);
        if (Number(r?.n ?? 0) < totalN) raggedDims.push(label);
      }

      if (xySizeVaries || raggedDims.length) {
        const issues = [];
        if (xySizeVaries) issues.push('image size');
        issues.push(...raggedDims);
        return { text: `Inconsistencies: <strong>${issues.join(', ')}</strong>.`, warning: true };
      }

      if (present.length) return `All same size, with ${ctx.plot.humanList(present)}.`;
      return `All 2D, same size.`;
    } catch { return null; }
  },

  async condensedPlot(container, ctx) {
    const { andWhere, sample, groupCol: gcFn } = ctx.sql;
    const sizeCols = ctx.schema.allCols.filter(c => c.endsWith('_size') && !c.startsWith('__'));
    if (!(sizeCols.includes('X_size') && sizeCols.includes('Y_size'))) return false;

    const rows = await ctx.queryRows(`
      SELECT "X_size" AS x, "Y_size" AS y, ${gcFn()} AS __group__
      FROM pp_data ${andWhere(ctx.where, '"X_size" > 1 AND "Y_size" > 1')} ${sample(3000)}
    `);
    if (!rows.length) return false;

    const traces = ctx.groups.map(g => {
      const gr = rows.filter(r => String(r.__group__) === g);
      if (!gr.length) return null;
      return {
        type: 'scatter', mode: 'markers', name: ctx.groupLabel(g),
        x: gr.map(r => Number(r.x)), y: gr.map(r => Number(r.y)),
        marker: { color: ctx.color.group(g), size: 5, opacity: 0.5 }, hoverinfo: 'skip',
      };
    }).filter(Boolean);
    if (!traces.length) return false;

    ctx.plot.appendMini(container, traces, { xaxis: { title: 'X (px)' }, yaxis: { title: 'Y (px)' } });
    return true;
  },

  async render(container, ctx) {
    try {
      const sizeCols = ctx.schema.allCols.filter(c => c.endsWith('_size') && !c.startsWith('__'));
      await renderSizeAvailability(container, ctx, sizeCols);
      await renderXYDistribution(container, ctx, sizeCols);
      await renderDimDistributions(container, ctx, sizeCols);
    } catch {
      container.innerHTML = '<div class="no-data">Failed to load data.</div>';
    }
  },
};

// How many images actually report each *_size dimension.
async function renderSizeAvailability(container, ctx, sizeCols) {
  const { q, andWhere } = ctx.sql;
  const [{ n: total }] = await ctx.queryRows(`SELECT COUNT(*) AS n FROM pp_data ${ctx.where}`);
  const availability = await Promise.all(sizeCols.map(async col => {
    const [{ n }] = await ctx.queryRows(`SELECT COUNT(*) AS n FROM pp_data ${andWhere(ctx.where, `${q(col)} > 1`)}`);
    return { label: ctx.plot.niceName(col), present: Number(n ?? 0) };
  }));
  ctx.plot.dataAvailabilityWarning(container, availability, Number(total ?? 0), { unit: 'images' });
}

// X vs Y size: a per-file bubble scatter, or a log-binned heatmap once there are
// too many files to show individually. Skipped when X and Y are both constant.
async function renderXYDistribution(container, ctx, sizeCols) {
  if (!(sizeCols.includes('X_size') && sizeCols.includes('Y_size'))) return;
  const xyWhere = ctx.sql.andWhere(ctx.where, '"X_size" > 1 AND "Y_size" > 1');

  const [{ ndx, ndy }] = await ctx.queryRows(`
    SELECT COUNT(DISTINCT "X_size") AS ndx, COUNT(DISTINCT "Y_size") AS ndy FROM pp_data ${xyWhere}
  `);
  if (Number(ndx ?? 0) <= 1 && Number(ndy ?? 0) <= 1) return;

  const [stats = {}] = await ctx.queryRows(`
    SELECT COUNT(*)::BIGINT AS n,
           MIN("X_size")::DOUBLE AS min_x, MAX("X_size")::DOUBLE AS max_x,
           MIN("Y_size")::DOUBLE AS min_y, MAX("Y_size")::DOUBLE AS max_y
    FROM pp_data ${xyWhere}
  `);
  if (Number(stats.n ?? 0) <= MAX_XY_BUBBLE) await renderXYBubble(container, ctx, xyWhere);
  else await renderXYHeatmap(container, ctx, xyWhere, stats);
}

// One marker per distinct (X, Y) size, sized by how many files share it.
async function renderXYBubble(container, ctx, xyWhere) {
  const rows = await ctx.queryRows(`
    SELECT "X_size" AS x, "Y_size" AS y, ${ctx.sql.groupCol()} AS __group__, COUNT(*) AS bubble_size
    FROM pp_data ${xyWhere} GROUP BY 1, 2, 3 ORDER BY 1, 2
  `);
  if (!rows.length) return;
  const maxCount = Math.max(...rows.map(r => Number(r.bubble_size)));
  const sizeref  = (2 * maxCount) / (20 ** 2);
  const traces = ctx.groups.map(g => {
    const gRows = rows.filter(r => String(r.__group__) === g);
    if (!gRows.length) return null;
    return {
      type: 'scatter', mode: 'markers', name: ctx.groupLabel(g),
      x: gRows.map(r => Number(r.x)), y: gRows.map(r => Number(r.y)),
      marker: { size: gRows.map(r => Number(r.bubble_size)), sizeref, sizemode: 'area', color: ctx.color.group(g), line: { width: 1, color: 'white' } },
      hovertemplate: `<b>${g}</b><br>X: %{x} px<br>Y: %{y} px<br>Count: %{text}<extra></extra>`,
      text: gRows.map(r => String(r.bubble_size)),
    };
  }).filter(Boolean);
  ctx.plot.append(container, traces, {
    title: { text: 'Distribution of X and Y Dimension Sizes' },
    xaxis: { title: 'X size (pixels)' }, yaxis: { title: 'Y size (pixels)' },
    showlegend: ctx.groups.length > 1,
    legend: { ...ctx.plot.plotlyLegendConfig, itemsizing: 'constant' },
  }, 'margin-bottom:24px');
}

// Log-scaled 2D histogram of (X, Y) sizes for datasets too large to bubble-plot.
async function renderXYHeatmap(container, ctx, xyWhere, stats) {
  const minX = Number(stats.min_x), minY = Number(stats.min_y);
  const spanX = Number(stats.max_x) - minX, spanY = Number(stats.max_y) - minY;
  const binX = Math.max(1, Math.ceil(spanX / HEATMAP_BINS));
  const binY = Math.max(1, Math.ceil(spanY / HEATMAP_BINS));

  const rows = await ctx.queryRows(`
    SELECT (FLOOR(("X_size" - ${minX}) / ${binX}) * ${binX} + ${minX})::INTEGER AS xbin,
           (FLOOR(("Y_size" - ${minY}) / ${binY}) * ${binY} + ${minY})::INTEGER AS ybin,
           COUNT(*)::INTEGER AS n
    FROM pp_data ${xyWhere} GROUP BY 1, 2 ORDER BY 1, 2
  `);
  if (!rows.length) return;

  const xs = Array.from({ length: Math.ceil(spanX / binX) + 1 }, (_, i) => minX + i * binX);
  const ys = Array.from({ length: Math.ceil(spanY / binY) + 1 }, (_, i) => minY + i * binY);
  const xi = new Map(xs.map((v, i) => [v, i]));
  const yi = new Map(ys.map((v, i) => [v, i]));
  const z  = Array.from({ length: ys.length }, () => Array(xs.length).fill(null));
  let zMax = 0;
  for (const r of rows) {
    const cx = xi.get(Number(r.xbin)), cy = yi.get(Number(r.ybin));
    if (cx != null && cy != null) { const lv = Math.log10(Number(r.n)); z[cy][cx] = lv; if (lv > zMax) zMax = lv; }
  }
  const tickVals = [], tickText = [];
  for (let i = 0; i <= Math.ceil(zMax); i++) { tickVals.push(i); tickText.push(String(10 ** i)); }
  ctx.plot.append(container, [{
    type: 'heatmap', x: xs, y: ys, z, colorscale: 'Viridis', zmin: 0, zmax: zMax,
    colorbar: { title: { text: 'count', side: 'right' }, tickvals: tickVals, ticktext: tickText },
    hovertemplate: 'X: %{x} px<br>Y: %{y} px<br>count: %{text}<extra></extra>',
    text: ys.map((_, cy) => xs.map((_, cx) => { const v = z[cy][cx]; return v != null ? String(Math.round(10 ** v)) : ''; })),
  }], {
    title: { text: 'Distribution of X and Y Dimension Sizes (log scale)' },
    xaxis: { title: 'X size (pixels)' }, yaxis: { title: 'Y size (pixels)' },
  }, 'margin-bottom:24px');
}

// Per-dimension size spread: a violin per dimension that varies, plus an
// invariant table for dimensions that are the same size in every file.
async function renderDimDistributions(container, ctx, sizeCols) {
  const { q } = ctx.sql;
  const dims = sizeCols.filter(c => c !== 'num_pixels');
  if (!dims.length) return;

  const dimStats = await Promise.all(dims.map(async col => {
    const [{ nd, val }] = await ctx.queryRows(`
      SELECT COUNT(DISTINCT ${q(col)}) AS nd, MIN(${q(col)}) AS val
      FROM pp_data ${ctx.sql.andWhere(ctx.where, `${q(col)} > 1`)}
    `);
    return { col, nd: Number(nd ?? 0), val };
  }));
  const variant   = dimStats.filter(d => d.nd > 1).map(d => d.col);
  const invariant = dimStats.filter(d => d.nd === 1);

  if (variant.length) await renderDimViolins(container, ctx, variant);
  if (invariant.length) ctx.plot.invariantTable(container, {
    title: 'Dimension Sizes - same across all files that report it',
    headers: ['Dimension', 'Size (pixels)'],
    rows: invariant.map(d => [ctx.plot.niceName(d.col), String(d.val)]),
    hr: true,
  });
}

// One small violin per varying dimension, laid out in a flex grid. Uses the
// shared distribution engine (same as the violin widget) so these render
// identically to it apart from the compact grid size, and get the large-data
// box fallback for free.
async function renderDimViolins(container, ctx, dims) {
  const { q, andWhere, groupCol: gcFn } = ctx.sql;

  const heading = document.createElement('h6');
  heading.style.cssText = 'margin-top:16px;margin-bottom:12px';
  heading.textContent = 'Dimension Size Distribution';
  container.appendChild(heading);
  const wrap = document.createElement('div');
  wrap.style.cssText = 'display:flex;flex-wrap:wrap;gap:12px';
  container.appendChild(wrap);

  for (const col of dims) {
    await ctx.plot.engine.renderDistribution(wrap, ctx, {
      numCol:          col,
      source:          { table: 'pp_data', where: andWhere(ctx.where, `${q(col)} > 1`) },
      catSql:          gcFn(),
      catLabel:        ctx.plot.groupingLabel(''),
      yLabel:          'pixels',
      title:           ctx.plot.niceName(col),
      series:          { isCategory: true },
      categoriesOrder: ctx.groups,
      catLabelFn:      ctx.groupLabel,
      // Only the size differs from the violin widget - everything else is the engine default.
      layout:          { height: 280, margin: { l: 44, r: 10, t: 36, b: 40 }, title: { font: { size: 12 } } },
      divStyle:        'flex:0 0 280px;min-width:220px;margin-bottom:16px',
    });
  }
}
