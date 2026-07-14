/**
 * Shared plotting engine for viewer plugins.
 *
 * Two responsibilities:
 *   1. `choosePlotKind` - maps a pair of columns (+ their cardinalities) to a
 *      plot kind.
 *   2. `renderDistribution` and the `render*` family - draw a given kind, with
 *      consistent large-data scaling and significance brackets.
 *
 * Everything takes the plugin `ctx` and draws through `ctx.plot.*` /
 * `ctx.sql.*` / `ctx.queryRows`. The module is bundled into the viewer app and
 * exposed as `ctx.plot.engine` (see renderer.js) so it is available in both the
 * served and the fully-offline (data-URL-inlined) report modes; a sibling module
 * imported by the package plugins would not be, because relative imports can't
 * resolve from a data: URL.
 */

// plot-utils is bundled into the same app chunk (not a plugin), so importing it
// here is safe in offline mode - only package plugins can't use sibling imports.
import { statTable } from './plot-utils.js';
export { statTable };

// ── Constants ───────────────────────────────────────────────────────────────
export const COUNT_Y     = '(count)';
export const NULL_LABEL  = '(missing)';
export const MAX_CAT     = 30;     // max distinct values on a categorical axis
export const MAX_HUE     = 12;     // max distinct values to color-by
export const MAX_SAMPLE  = 5_000;  // reservoir-sample cap (dates / fallbacks)
// Distributions with at most this many total points show the real shape (a raw
// violin); larger ones fall back to a SQL-computed box summary that scales.
export const MAX_VIOLIN_POINTS = 5_000;
// A violin with fewer than this many points reads better showing every (jittered)
// datapoint than a sparse shape. Default for all category violins so the violin,
// dimension-size, and custom-plot widgets render distributions identically.
export const VIOLIN_ALL_POINTS_BELOW = 500;

// Columns holding real timestamps - plotted on a date axis, never as categories,
// and never summarised with approx_quantile (use a sampled raw violin instead).
// Populated from the detected schema via setDateCols() - see renderer.js buildCtx.
export const DATE_COLS = new Set();
export function setDateCols(cols) {
  DATE_COLS.clear();
  for (const c of cols) DATE_COLS.add(c);
}
const DATE_FMT = "'%Y-%m-%d %H:%M:%S'";

export const CONSTANTS = {
  COUNT_Y, NULL_LABEL, MAX_CAT, MAX_HUE, MAX_SAMPLE, MAX_VIOLIN_POINTS,
  VIOLIN_ALL_POINTS_BELOW, DATE_COLS,
};

// ── Small SQL / formatting helpers ────────────────────────────────────────────
function selectExpr(q, col, alias) {
  return DATE_COLS.has(col)
    ? `STRFTIME(${q(col)}, ${DATE_FMT}) AS ${alias}`
    : `${q(col)} AS ${alias}`;
}
const DATE_TICK_FMT = '%Y-%m-%d %H:%M:%S';

// Tick label format scales with zoom (dtick), so a wide axis reads as bare
// dates and only grows a time-of-day component once ticks are close enough
// for it to matter. hoverformat stays fixed to full precision - it's always
// a single value, so there's no clutter risk there.
const DATE_TICKFORMATSTOPS = [
  { dtickrange: [null, 1000],        value: '%H:%M:%S.%L' },
  { dtickrange: [1000, 60000],       value: '%H:%M:%S' },
  { dtickrange: [60000, 3600000],    value: '%H:%M' },
  { dtickrange: [3600000, 86400000], value: '%b %d, %H:%M' },
  { dtickrange: [86400000, null],    value: '%Y-%m-%d' },
];

// Plotly's date-axis auto-dtick is correct except for a near-zero span: below
// ~4s it drops sub-second (dtick 0.2 -> "13:40:20.0001") and no longer reads as
// a date. Below the floor we pin a 1-second dtick, padding the range when the
// span is under one tick so a tick lands in view; above it we defer to Plotly.
const DATE_FLOOR_DTICK = 1000; // 1 second - never tick finer than this
const DATE_FLOOR_SPAN = 4000;  // below this, Plotly's auto dtick goes sub-second
export function niceDateAxis(values, title) {
  const base = { title, type: 'date', tickformatstops: DATE_TICKFORMATSTOPS, hoverformat: DATE_TICK_FMT };
  const times = (values ?? []).map(v => new Date(v).getTime()).filter(Number.isFinite);
  if (!times.length) return base;
  const min = Math.min(...times), max = Math.max(...times);
  const span = max - min;
  if (span >= DATE_FLOOR_SPAN) return base;
  const range = span < DATE_FLOOR_DTICK ? [min - DATE_FLOOR_DTICK * 3, max + DATE_FLOOR_DTICK * 3] : undefined;
  return { ...base, dtick: DATE_FLOOR_DTICK, ...(range ? { range } : {}) };
}
function axisCfg(col, title, values) {
  return DATE_COLS.has(col) ? niceDateAxis(values, title) : { title };
}
function valueOf(col, v) {
  return DATE_COLS.has(col) ? v : Number(v);
}
// Date-aware single aggregate, e.g. aggExpr(q, 'modification_date', 'MIN', 'xmin').
function aggExpr(q, col, fn, alias) {
  return DATE_COLS.has(col)
    ? `STRFTIME(${fn}(${q(col)}), ${DATE_FMT}) AS ${alias}`
    : `${fn}(${q(col)}) AS ${alias}`;
}
// Date-aware MIN/MAX/AVG(/STDDEV) aggregate block for table summaries.
function aggExprs(q, col) {
  return DATE_COLS.has(col)
    ? `STRFTIME(MIN(${q(col)}), ${DATE_FMT}) AS min_val, STRFTIME(MAX(${q(col)}), ${DATE_FMT}) AS max_val,
       STRFTIME(AVG(${q(col)}), ${DATE_FMT}) AS mean_val`
    : `AVG(${q(col)}) AS mean_val, STDDEV(${q(col)}) AS std_val,
       MIN(${q(col)}) AS min_val,  MAX(${q(col)}) AS max_val`;
}
function fmtStat(col, v) {
  if (v == null) return '—';
  return DATE_COLS.has(col) ? v : (Number.isFinite(Number(v)) ? Number(v).toPrecision(4) : '—');
}
// Categorical axis value: NULLs become their own "(missing)" bucket.
function catExprOf(q, col) {
  return `COALESCE(CAST(${q(col)} AS VARCHAR), '${NULL_LABEL}')`;
}
// Sort categories with the "(missing)" bucket last.
export function sortCats(cats) {
  const real = cats.filter(c => c !== NULL_LABEL).sort();
  return cats.includes(NULL_LABEL) ? [...real, NULL_LABEL] : real;
}

function legendCfg(ctx, nSeries) {
  return nSeries > 1
    ? { showlegend: true, legend: ctx.plot.plotlyLegendConfig }
    : { showlegend: false };
}

// ── Plot-kind decision ────────────────────────────────────────────────────────
/**
 * Decide how to plot columns x vs y. Async because some branches need a
 * cardinality probe, but query-free otherwise - inject `catCount`/`distinct`
 * so it stays unit-testable with fakes.
 *
 * @param {object} o
 * @param {string} o.xCol
 * @param {string} o.yCol                 may be COUNT_Y
 * @param {Set<string>} o.numericSet      columns treated as numeric
 * @param {(c:string)=>Promise<number>} o.catCount    distinct incl. "(missing)" bucket
 * @param {(c:string)=>Promise<number>} o.distinct    COUNT(DISTINCT) excl. NULLs
 * @param {number} [o.maxCat=MAX_CAT]
 * @param {string} [o.countY=COUNT_Y]
 * @returns {Promise<object>} descriptor `{ kind, ... }`. kind is one of:
 *   'message' (with `message`), 'countTable', 'countBar', 'invalid',
 *   'numNumTable', 'scatter', 'heatmap', 'catNumTable',
 *   'distribution' ({ catCol, numCol, flipped, isDate }).
 */
export async function choosePlotKind({
  xCol, yCol, numericSet, catCount, distinct, maxCat = MAX_CAT, countY = COUNT_Y,
}) {
  if (yCol === countY) {
    const n = await catCount(xCol);
    if (n > maxCat) {
      return { kind: 'message', message: `"${xCol}" has ${n} unique values — too many for count bar (max ${maxCat}).` };
    }
    return n === 1 ? { kind: 'countTable', x: xCol } : { kind: 'countBar', x: xCol };
  }

  if (xCol === yCol) {
    return { kind: 'invalid', message: 'X and Y must be different columns.' };
  }

  const xNum = numericSet.has(xCol);
  const yNum = numericSet.has(yCol);

  if (xNum && yNum) {
    const yUniq = await distinct(yCol);
    return yUniq <= 1
      ? { kind: 'numNumTable', x: xCol, y: yCol }
      : { kind: 'scatter', x: xCol, y: yCol };
  }

  if (!xNum && !yNum) {
    const [xCats, yCats] = await Promise.all([catCount(xCol), catCount(yCol)]);
    if (xCats > maxCat || yCats > maxCat) {
      const bad = xCats > maxCat ? xCol : yCol;
      const badN = xCats > maxCat ? xCats : yCats;
      return { kind: 'message', message: `"${bad}" has ${badN} unique values — too many for heatmap (max ${maxCat}).` };
    }
    return { kind: 'heatmap', x: xCol, y: yCol };
  }

  // Mixed: one categorical, one numeric.
  const catCol = xNum ? yCol : xCol;
  const numCol = xNum ? xCol : yCol;
  const flipped = xNum;            // user put the numeric column on X
  const isDate = DATE_COLS.has(numCol);
  const [n, numUniq] = await Promise.all([catCount(catCol), distinct(numCol)]);
  if (n > maxCat) {
    return { kind: 'message', message: `"${catCol}" has ${n} unique values — too many for a categorical axis (max ${maxCat}).` };
  }
  if (n === 1 || numUniq <= 1) {
    return { kind: 'catNumTable', catCol, numCol, flipped, isDate };
  }
  return { kind: 'distribution', catCol, numCol, flipped, isDate };
}

// ── Distribution rendering (violin / box / bar) ───────────────────────────────
const STAT_SELECT = (q, num) => `
  COUNT(${q(num)}) AS n,
  MIN(${q(num)}) AS mn, MAX(${q(num)}) AS mx, AVG(${q(num)}) AS mean, STDDEV(${q(num)}) AS sd,
  approx_quantile(${q(num)}, 0.25) AS q1,
  approx_quantile(${q(num)}, 0.5)  AS med,
  approx_quantile(${q(num)}, 0.75) AS q3`;

/**
 * Draw one distribution plot of `numCol`, split across an X category axis and
 * (optionally) a color series. With `series.isCategory` each X category is its
 * own colored violin/box; otherwise traces are split by the color series and
 * each spans the X categories.
 *
 * spec:
 *   numCol
 *   source:    { table, where }
 *   catSql:    SQL expression producing the X category value
 *   catLabel:  X axis title
 *   yLabel, title
 *   force:     'auto' | 'bar'
 *   isDate:    numCol is a timestamp (sampled raw violin only - no box summary)
 *   mini:      compact condensed-preview mode (no title/legend/significance)
 *   showSignificance
 *   maxRawPoints  (default MAX_VIOLIN_POINTS)
 *   series:
 *     isCategory: true  → one violin per X category, colored per category
 *                          (catLabelFn maps value→label, color via ctx.color.group)
 *     else        → { sql, build(rows)→{groups,colorFn,labelFn} }
 *   categoriesOrder?  explicit category ordering (else sorted, "(missing)" last)
 *   catLabelFn?       value→display label (category mode)
 *   stats?            precomputed per-category stats (category mode, avoids a query)
 *   allPointsBelow?   category-mode violins with fewer than this many points show
 *                     every (jittered) datapoint instead of just outliers (default 0)
 *   layout?           extra Plotly layout merged over the computed one (height,
 *                     margin, title font, …) - for compact grid cells
 *   divStyle?         wrapper element style string (e.g. flex sizing in a grid)
 *
 * Returns true if anything was drawn, false otherwise.
 */
export async function renderDistribution(container, ctx, spec) {
  const { q, andWhere } = ctx.sql;
  const {
    numCol, source, catSql, catLabel = '', yLabel = '', title = '',
    force = 'auto', isDate = false, mini = false, showSignificance = false,
    maxRawPoints = MAX_VIOLIN_POINTS, series, categoriesOrder = null,
    catLabelFn = (v) => v, stats: precomputed = null, isStale = () => false,
    allPointsBelow = VIOLIN_ALL_POINTS_BELOW, layout: layoutOverride = {}, divStyle = '',
  } = spec;
  const { table, where } = source;

  // Per-category (and per-series) summary stats - skip the query when the caller
  // supplies them precomputed. Date columns can't be summarised with
  // approx_quantile, so they skip stats and always draw a sampled raw violin.
  let statRows;
  if (precomputed) {
    statRows = precomputed;
  } else if (isDate) {
    statRows = null;
  } else {
    // series.sql already carries its own "AS __group__"; catSql is a bare
    // expression, so it gets aliased here (as __group__ in category mode, and
    // always as __cat__).
    const groupSel = series.isCategory ? `${catSql} AS __group__` : series.sql;
    statRows = await ctx.queryRows(`
      SELECT ${groupSel}, ${catSql} AS __cat__, ${STAT_SELECT(q, numCol)}
      FROM ${table} ${andWhere(where, `${q(numCol)} IS NOT NULL`)}
      GROUP BY 1, 2
    `);
  }

  const total = statRows ? statRows.reduce((s, r) => s + Number(r.n || 0), 0) : null;
  if (statRows && !total) return false;

  const mode = force === 'bar' ? 'bar'
             : (statRows && total > maxRawPoints) ? 'box'
             : 'violin';

  let traces, categories, categoryValues;
  if (series.isCategory) {
    ({ traces, categories, categoryValues } = await buildCategoryTraces(
      ctx, { numCol, table, where, catSql, mode, statRows, categoriesOrder, catLabelFn, allPointsBelow }));
  } else {
    ({ traces, categories, categoryValues } = await buildSeriesTraces(
      ctx, { numCol, table, where, catSql, mode, statRows, series, isDate }));
  }
  if (!traces.length || isStale()) return false;

  const layout = {
    yaxis: axisCfg(numCol, yLabel, traces.flatMap(t => t.y ?? [])),
    xaxis: { title: catLabel, type: 'category', ...(categories ? { categoryarray: categories } : {}) },
    // Each violin/box owns its own x category; default 'overlay' keeps them
    // centred. 'group' would reserve an empty sub-slot per trace per category.
    violinmode: 'overlay', boxmode: 'overlay',
  };

  if (mini) {
    // The X categories are the groups, and their colour mapping now lives in the
    // summary table at the top of the report - so hide the (redundant) category
    // tick labels and surface what's plotted on the Y axis instead.
    ctx.plot.appendMini(container, traces, {
      xaxis: { type: 'category', showticklabels: false },
      yaxis: { zeroline: false, ...(yLabel ? { title: yLabel } : {}) },
      ...(yLabel ? { margin: { l: 52 } } : {}),
    });
    return true;
  }

  const nSeries = series.isCategory ? 1 : traces.length;
  const finalLayout = {
    ...layout,
    title: title ? { text: title } : undefined,
    ...legendCfg(ctx, nSeries),
    ...layoutOverride,
  };
  // Keep the computed title text when the caller only overrides its styling (font, …).
  if (title && layoutOverride.title) finalLayout.title = { text: title, ...layoutOverride.title };
  const plotDiv = ctx.plot.append(container, traces, finalLayout, divStyle);

  // Significance only makes sense with one violin/box per X category: either
  // category mode, or a single color series. Keyed on raw category values;
  // brackets are positioned by category index, which matches the rendered axis
  // order (trace order / categoryarray).
  const sigOk = showSignificance && (series.isCategory || nSeries <= 1) && categoryValues && categoryValues.length >= 2;
  if (sigOk) {
    const rankData = await fetchCategoryRankSums(ctx, {
      table, where: andWhere(where, `${q(numCol)} IS NOT NULL`), catSql, numCol, categories: categoryValues,
    });
    const pairs = computeSignificancePairs(categoryValues, rankData);
    // Awaited so renderDistribution only resolves once the brackets are actually
    // drawn - callers that capture/measure the plot right after (e.g. the baked
    // HTML export) then see the finished plot, not a one-frame-early version.
    await addSignificanceBrackets(plotDiv, pairs, categoryValues);
  }
  return true;
}

// One violin/box trace per X category, each its own color. Category mode is
// never used with date columns, so no date-axis / sampling handling is needed.
async function buildCategoryTraces(ctx, { numCol, table, where, catSql, mode, statRows, categoriesOrder, catLabelFn, allPointsBelow = 0 }) {
  const { q, andWhere } = ctx.sql;
  const byCat = new Map(statRows.map(r => [String(r.__cat__ ?? r.__group__), r]));
  const present = (categoriesOrder ?? sortCats([...byCat.keys()])).filter(c => byCat.has(c) && Number(byCat.get(c).n) > 0);
  if (!present.length) return { traces: [], categories: [] };

  if (mode === 'violin') {
    const rows = await ctx.queryRows(`
      SELECT ${catSql} AS __cat__, ${q(numCol)} AS val
      FROM ${table} ${andWhere(where, `${q(numCol)} IS NOT NULL`)}
    `);
    const byG = new Map(present.map(g => [g, []]));
    for (const r of rows) byG.get(String(r.__cat__))?.push(Number(r.val));
    const traces = present.map(g => {
      const y = byG.get(g);
      // Small samples read better as every (jittered) point than as a sparse violin.
      const allPoints = allPointsBelow > 0 && y.length < allPointsBelow;
      return {
        // spanmode 'soft' (default) gives smooth tapering tails; 'hard' clips the
        // density flat at the data min/max, which reads as an ugly "cropped" violin.
        type: 'violin', name: catLabelFn(g), x: y.map(() => catLabelFn(g)), y,
        box: { visible: true }, meanline: { visible: true },
        points: allPoints ? 'all' : 'outliers', ...(allPoints ? { pointpos: 0, jitter: 0.3 } : {}),
        spanmode: 'soft', opacity: 0.9, marker: { color: ctx.color.group(g) },
        hovertemplate: '<b>Group:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>',
      };
    }).filter(t => t.y.length);
    return { traces, categories: present.map(catLabelFn), categoryValues: present };
  }

  // box / bar from precomputed stats
  const traces = present.map(g => boxOrBarTrace(ctx, byCat.get(g), {
    name: catLabelFn(g), x: catLabelFn(g), color: ctx.color.group(g), mode,
  }));
  return { traces, categories: present.map(catLabelFn), categoryValues: present };
}

// One trace per color series; each spans the X categories.
async function buildSeriesTraces(ctx, { numCol, table, where, catSql, mode, statRows, series, isDate }) {
  const { q, andWhere } = ctx.sql;

  if (mode === 'violin') {
    // Raw rows (no statRows needed - also covers date columns, which skip stats).
    // Dates are reservoir-sampled to keep the payload bounded.
    const inner = `SELECT ${catSql} AS cat, ${selectExpr(q, numCol, 'val')}, ${series.sql}
      FROM ${table} ${andWhere(where, `${q(numCol)} IS NOT NULL`)}`;
    const rows = await ctx.queryRows(
      isDate ? `SELECT * FROM (${inner}) USING SAMPLE ${MAX_SAMPLE} ROWS (reservoir, 42)` : inner);
    const { groups, colorFn, labelFn } = series.build(rows);
    const cats = sortCats([...new Set(rows.map(r => String(r.cat)))]);
    const traces = groups.map(g => {
      const gr = rows.filter(r => String(r.__group__) === g);
      return {
        type: 'violin', name: labelFn(g),
        x: gr.map(r => String(r.cat)), y: gr.map(r => valueOf(numCol, r.val)),
        box: { visible: true }, meanline: { visible: true }, points: 'outliers',
        spanmode: 'soft', marker: { color: colorFn(g) },  // smooth tails, not clipped flat
      };
    }).filter(t => t.y.length);
    return { traces, categories: cats, categoryValues: cats };
  }

  // box / bar: per series, arrays over categories (statRows always present here)
  const { groups, colorFn, labelFn } = series.build(statRows);
  const cats = sortCats([...new Set(statRows.map(r => String(r.__cat__)))]);
  const statByKey = new Map(statRows.map(r => [`${r.__group__}\x00${r.__cat__}`, r]));
  const traces = groups.map(g => {
    const present = cats.filter(c => statByKey.has(`${g}\x00${c}`) && Number(statByKey.get(`${g}\x00${c}`).n) > 0);
    if (!present.length) return null;
    if (mode === 'bar') {
      return {
        type: 'bar', name: labelFn(g), x: present,
        y: present.map(c => Number(statByKey.get(`${g}\x00${c}`).mean)),
        error_y: { type: 'data', visible: true, array: present.map(c => Number(statByKey.get(`${g}\x00${c}`).sd)) },
        marker: { color: colorFn(g) },
      };
    }
    return {
      type: 'box', name: labelFn(g), x: present,
      q1:         present.map(c => Number(statByKey.get(`${g}\x00${c}`).q1)),
      median:     present.map(c => Number(statByKey.get(`${g}\x00${c}`).med)),
      q3:         present.map(c => Number(statByKey.get(`${g}\x00${c}`).q3)),
      lowerfence: present.map(c => Number(statByKey.get(`${g}\x00${c}`).mn)),
      upperfence: present.map(c => Number(statByKey.get(`${g}\x00${c}`).mx)),
      mean:       present.map(c => Number(statByKey.get(`${g}\x00${c}`).mean)),
      boxmean: true, boxpoints: false, marker: { color: colorFn(g) },
    };
  }).filter(Boolean);
  return { traces, categories: cats, categoryValues: cats };
}

// Single-category box or bar trace from one precomputed stat row.
function boxOrBarTrace(ctx, r, { name, x, color, mode }) {
  if (mode === 'bar') {
    return {
      type: 'bar', name, x: [x], y: [Number(r.mean)],
      error_y: { type: 'data', visible: true, array: [Number(r.sd)] },
      marker: { color },
    };
  }
  return {
    type: 'box', name, x: [x],
    q1: [Number(r.q1)], median: [Number(r.med)], q3: [Number(r.q3)],
    lowerfence: [Number(r.mn)], upperfence: [Number(r.mx)], mean: [Number(r.mean)],
    boxmean: true, boxpoints: false, opacity: 0.9,
    marker: { color, line: { width: 1, color: 'rgba(0,0,0,0.35)' } },
    hovertemplate: '<b>Group:</b> %{x}<br><b>Median:</b> %{median:.2f}' +
      '<br><b>Q1:</b> %{q1:.2f}<br><b>Q3:</b> %{q3:.2f}' +
      '<br><b>Min:</b> %{lowerfence:.2f}<br><b>Max:</b> %{upperfence:.2f}<extra></extra>',
  };
}

// ── Other plot kinds ──────────────────────────────────────────────────────────
// These take a resolved color config and a {table,where} source. They return
// true when an exportable plot was drawn, false otherwise (tables, stale calls).
// `isStale` lets a superseded call bail before mutating the DOM.

export async function renderScatter(container, ctx, spec) {
  const { q, andWhere } = ctx.sql;
  const { niceName } = ctx.plot;
  const { x, y, source, continuous, colorBy, colorScale, series, isStale = () => false } = spec;
  const { table, where } = source;

  if (continuous) {
    const wh = andWhere(where, `${q(x)} IS NOT NULL AND ${q(y)} IS NOT NULL AND ${q(colorBy)} IS NOT NULL`);
    const rows = await ctx.queryRows(`
      SELECT * FROM (
        SELECT ${selectExpr(q, x, 'x')}, ${selectExpr(q, y, 'y')}, ${q(colorBy)} AS c
        FROM ${table} ${wh}
      ) USING SAMPLE ${MAX_SAMPLE} ROWS (reservoir, 42)
    `);
    if (isStale()) return false;
    const sampled = rows.length >= MAX_SAMPLE;
    ctx.plot.append(container, [{
      type: 'scatter', mode: 'markers',
      x: rows.map(r => valueOf(x, r.x)), y: rows.map(r => valueOf(y, r.y)),
      marker: { color: rows.map(r => Number(r.c)), colorscale: colorScale, showscale: true,
        colorbar: { title: { text: niceName(colorBy) } }, size: 5, opacity: 0.7 },
    }], {
      title: { text: `${niceName(x)} vs ${niceName(y)}` + (sampled ? '<br><sup>up to 5,000 points shown</sup>' : '') },
      xaxis: axisCfg(x, niceName(x), rows.map(r => r.x)), yaxis: axisCfg(y, niceName(y), rows.map(r => r.y)), showlegend: false,
    });
    return true;
  }

  const wh = andWhere(where, `${q(x)} IS NOT NULL AND ${q(y)} IS NOT NULL`);
  const rows = await ctx.queryRows(`
    SELECT * FROM (
      SELECT ${selectExpr(q, x, 'x')}, ${selectExpr(q, y, 'y')}, ${series.sql}
      FROM ${table} ${wh}
    ) USING SAMPLE ${MAX_SAMPLE} ROWS (reservoir, 42)
  `);
  if (isStale()) return false;
  const { groups, colorFn, labelFn } = series.build(rows);
  const sampled = rows.length >= MAX_SAMPLE;
  const traces = groups.map(g => {
    const gr = rows.filter(r => String(r.__group__) === g);
    return {
      type: 'scatter', mode: 'markers', name: labelFn(g),
      x: gr.map(r => valueOf(x, r.x)), y: gr.map(r => valueOf(y, r.y)),
      marker: { color: colorFn(g), size: 5, opacity: 0.7 },
    };
  }).filter(t => t.x.length);
  ctx.plot.append(container, traces, {
    title: { text: `${niceName(x)} vs ${niceName(y)}` + (sampled ? '<br><sup>up to 5,000 points shown</sup>' : '') },
    xaxis: axisCfg(x, niceName(x), rows.map(r => r.x)), yaxis: axisCfg(y, niceName(y), rows.map(r => r.y)), ...legendCfg(ctx, groups.length),
  });
  return true;
}

export async function renderCountBar(container, ctx, spec) {
  const { q } = ctx.sql;
  const { niceName } = ctx.plot;
  const { x, source, series, isStale = () => false } = spec;
  const { table, where } = source;
  const rows = await ctx.queryRows(`
    SELECT ${catExprOf(q, x)} AS cat, ${series.sql}, COUNT(*) AS n
    FROM ${table} ${where}
    GROUP BY 1, 2 ORDER BY 1
  `);
  if (isStale()) return false;
  const { groups, colorFn, labelFn } = series.build(rows);
  const cats = sortCats([...new Set(rows.map(r => String(r.cat)))]);
  const traces = groups.map(g => {
    const gr = rows.filter(r => String(r.__group__) === g);
    return { type: 'bar', name: labelFn(g), x: gr.map(r => String(r.cat)), y: gr.map(r => Number(r.n)), marker: { color: colorFn(g) } };
  }).filter(t => t.x.length);
  ctx.plot.append(container, traces, {
    title: { text: `Count by ${niceName(x)}` },
    xaxis: { title: niceName(x), type: 'category', categoryarray: cats },
    yaxis: { title: 'Count' }, barmode: 'stack', ...legendCfg(ctx, groups.length),
  });
  return true;
}

export async function renderHeatmap(container, ctx, spec) {
  const { q } = ctx.sql;
  const { niceName } = ctx.plot;
  const { x, y, source, heatColor, heatInvert, isStale = () => false } = spec;
  const { table, where } = source;
  const rows = await ctx.queryRows(`
    SELECT ${catExprOf(q, x)} AS x, ${catExprOf(q, y)} AS y, COUNT(*) AS n
    FROM ${table} ${where}
    GROUP BY 1, 2 ORDER BY 1, 2
  `);
  if (isStale()) return false;
  const xs = sortCats([...new Set(rows.map(r => String(r.x)))]);
  const ys = sortCats([...new Set(rows.map(r => String(r.y)))]);
  if (xs.length <= 1 || ys.length <= 1) {
    container.appendChild(statTable([niceName(x), niceName(y), 'Count'],
      rows.map(r => [String(r.x), String(r.y), Number(r.n).toLocaleString()])));
    return false;
  }
  const counts = new Map(rows.map(r => [`${r.x}\x00${r.y}`, Number(r.n)]));
  const z = ys.map(yv => xs.map(xv => counts.get(`${xv}\x00${yv}`) ?? 0));
  ctx.plot.append(container, [{
    type: 'heatmap', x: xs, y: ys, z, colorscale: [[0, '#ffffff'], [1, heatColor]],
    reversescale: !!heatInvert, showscale: true,
  }], {
    title: { text: `Count: ${niceName(x)} × ${niceName(y)}` },
    xaxis: { title: niceName(x), type: 'category' },
    yaxis: { title: niceName(y), type: 'category' }, showlegend: false,
  });
  return true;
}

export async function renderCountTable(container, ctx, spec) {
  const { q } = ctx.sql;
  const { niceName } = ctx.plot;
  const { x, source, series, isStale = () => false } = spec;
  const { table, where } = source;
  const rows = await ctx.queryRows(`
    SELECT ${catExprOf(q, x)} AS cat, ${series.sql}, COUNT(*) AS n
    FROM ${table} ${where}
    GROUP BY 1, 2 ORDER BY 1
  `);
  if (isStale()) return false;
  const { labelFn, header } = series.build(rows);
  const headers = header ? [niceName(x), header, 'Count'] : [niceName(x), 'Count'];
  container.appendChild(statTable(headers, rows.map(r => header
    ? [String(r.cat), labelFn(String(r.__group__)), Number(r.n).toLocaleString()]
    : [String(r.cat), Number(r.n).toLocaleString()])));
  return false;
}

export async function renderCatNumTable(container, ctx, spec) {
  const { q } = ctx.sql;
  const { niceName } = ctx.plot;
  const { catCol, numCol, source, series, isDate, isStale = () => false } = spec;
  const { table, where } = source;
  const rows = await ctx.queryRows(`
    SELECT ${catExprOf(q, catCol)} AS cat, ${series.sql},
           COUNT(${q(numCol)}) AS n, COUNT(*) - COUNT(${q(numCol)}) AS n_null, ${aggExprs(q, numCol)}
    FROM ${table} ${where}
    GROUP BY 1, 2 ORDER BY 1
  `);
  if (isStale()) return false;
  const { labelFn, header } = series.build(rows);
  const hasNulls = rows.some(r => Number(r.n_null) > 0);
  const fmt = v => fmtStat(numCol, v);
  const cols = isDate
    ? [niceName(catCol), 'n', `Mean ${niceName(numCol)}`, 'Min', 'Max']
    : [niceName(catCol), 'n', `Mean ${niceName(numCol)}`, 'SD', 'Min', 'Max'];
  if (hasNulls) cols.push(`Null ${niceName(numCol)}`);
  const headers = header ? [cols[0], header, ...cols.slice(1)] : cols;
  container.appendChild(statTable(headers, rows.map(r => {
    const vals = isDate
      ? [Number(r.n).toLocaleString(), fmt(r.mean_val), fmt(r.min_val), fmt(r.max_val)]
      : [Number(r.n).toLocaleString(), fmt(r.mean_val), fmt(r.std_val), fmt(r.min_val), fmt(r.max_val)];
    if (hasNulls) vals.push(Number(r.n_null).toLocaleString());
    const row = [String(r.cat), ...vals];
    return header ? [row[0], labelFn(String(r.__group__)), ...row.slice(1)] : row;
  })));
  return false;
}

// Numeric X vs near-constant numeric Y: one row per distinct Y value.
export async function renderNumNumTable(container, ctx, spec) {
  const { q } = ctx.sql;
  const { niceName } = ctx.plot;
  const { x, y, source, isStale = () => false } = spec;
  const { table, where } = source;
  const rows = await ctx.queryRows(`
    SELECT ${aggExpr(q, y, 'MIN', 'yval')},
           COUNT(${q(x)}) AS n, COUNT(*) - COUNT(${q(x)}) AS n_null,
           ${aggExpr(q, x, 'MIN', 'xmin')}, ${aggExpr(q, x, 'MAX', 'xmax')}
    FROM ${table} ${where}
    GROUP BY ${q(y)}
    ORDER BY (${q(y)} IS NULL), 1
  `);
  if (isStale()) return false;
  const hasNullX = rows.some(r => Number(r.n_null) > 0);
  const headers = [niceName(y), 'n', `${niceName(x)} range`];
  if (hasNullX) headers.push(`Null ${niceName(x)}`);
  container.appendChild(statTable(headers, rows.map(r => {
    const yLabel = r.yval == null ? NULL_LABEL : fmtStat(y, r.yval);
    const xMinTxt = fmtStat(x, r.xmin), xMaxTxt = fmtStat(x, r.xmax);
    const xRange = xMinTxt === xMaxTxt ? xMinTxt : `${xMinTxt} – ${xMaxTxt}`;
    const row = [yLabel, Number(r.n).toLocaleString(), xRange];
    if (hasNullX) row.push(Number(r.n_null).toLocaleString());
    return row;
  })));
  return null;
}

// ── Statistical significance (Mann-Whitney U, Bonferroni corrected) ───────────
const THRESHOLDS = [[0.001, '***'], [0.01, '**'], [0.05, '*']];
function sigSymbol(p) {
  for (const [t, s] of THRESHOLDS) if (p < t) return s;
  return 'ns';
}
function erf(x) {
  const a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((a[4]*t + a[3])*t + a[2])*t + a[1])*t + a[0]) * t * Math.exp(-x * x);
  return sign * y;
}
const normalCDF = z => 0.5 * (1 + erf(z / Math.SQRT2));

function mannWhitneyPFromRankSum(n1, n2, rankSum1) {
  if (n1 < 3 || n2 < 3) return 1.0;
  const U1  = rankSum1 - n1 * (n1 + 1) / 2;
  const mu  = n1 * n2 / 2;
  const sig = Math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12);
  if (sig === 0) return 1.0;
  return 2 * (1 - normalCDF(Math.abs(U1 - mu) / sig));
}

/**
 * Pairwise Mann-Whitney rank sums across category values, computed in SQL.
 * Returns rankData[`${c1}\x00${c2}`] = { [cat]: { n, rankSum } } - one query per pair.
 */
async function fetchCategoryRankSums(ctx, { table, where, catSql, numCol, categories }) {
  const { q, andWhere } = ctx.sql;
  const esc = v => `'${String(v).replace(/'/g, "''")}'`;
  // The pairwise queries are independent, so fire them together and let DuckDB
  // pipeline them rather than blocking on each in turn (the pair count grows
  // quadratically with the category count).
  const pairs = [];
  for (let i = 0; i < categories.length; i++) {
    for (let j = i + 1; j < categories.length; j++) pairs.push([categories[i], categories[j]]);
  }
  const results = await Promise.all(pairs.map(([c1, c2]) => {
    const pairWhere = andWhere(where, `CAST(${catSql} AS VARCHAR) IN (${esc(c1)}, ${esc(c2)})`);
    return ctx.queryRows(`
      WITH base AS (
        SELECT CAST(${catSql} AS VARCHAR) AS cat, ${q(numCol)}::DOUBLE AS v
        FROM ${table} ${pairWhere}
      ), ranked AS (
        SELECT cat, v, ROW_NUMBER() OVER (ORDER BY v) AS rn FROM base
      ), avg_ranked AS (
        SELECT cat, AVG(rn) OVER (PARTITION BY v) AS rnk FROM ranked
      )
      SELECT cat, COUNT(*) AS n, SUM(rnk) AS rank_sum FROM avg_ranked GROUP BY cat
    `);
  }));
  const rankData = {};
  pairs.forEach(([c1, c2], idx) => {
    const key = `${c1}\x00${c2}`;
    rankData[key] = {};
    for (const r of results[idx]) rankData[key][String(r.cat)] = { n: Number(r.n), rankSum: Number(r.rank_sum) };
  });
  return rankData;
}

function computeSignificancePairs(categories, rankData) {
  const pairs = [];
  for (let i = 0; i < categories.length; i++) {
    for (let j = i + 1; j < categories.length; j++) {
      const c1 = categories[i], c2 = categories[j];
      const d = rankData[`${c1}\x00${c2}`];
      const p = (d && d[c1] && d[c2]) ? mannWhitneyPFromRankSum(d[c1].n, d[c2].n, d[c1].rankSum) : 1.0;
      pairs.push({ c1, c2, p });
    }
  }
  const n = pairs.length;
  return pairs
    .map(p => ({ ...p, symbol: sigSymbol(Math.min(p.p * n, 1.0)) }))
    .sort((a, b) => Math.abs(categories.indexOf(a.c1) - categories.indexOf(a.c2)) - Math.abs(categories.indexOf(b.c1) - categories.indexOf(b.c2)));
}

function addSignificanceBrackets(plotDiv, pairs, categories) {
  // Show every compared pair, including non-significant ones: a greyed-out "ns"
  // bracket tells the reader the comparison *was* run and came back not
  // significant, rather than silently omitting it (which reads as "no test").
  const sigPairs = pairs;
  if (!sigPairs.length) return Promise.resolve();
  // Defer to the next frame: unlike main (static box-plot cards), the beginner-mode
  // tile-expand flow can resize/re-layout the plot after the initial draw, so a
  // synchronous read of yaxis.range can be stale and the bracket headroom we add
  // gets discarded. Reading after the frame settles - and pinning autorange off -
  // makes the extended range (and thus the brackets) stick. Returns a promise so
  // renderDistribution can await the actual draw.
  return new Promise(resolve => requestAnimationFrame(() => {
    if (!plotDiv.isConnected || !plotDiv._fullLayout) { resolve(); return; }
    const renderedRange = plotDiv._fullLayout?.yaxis?.range;
    const yBottom = renderedRange?.[0] ?? 0;
    const yTop    = renderedRange?.[1] ?? 1;
    const span    = Math.abs(yTop - yBottom) || 1;
    const gap     = span * 0.06, tickH = span * 0.04;
    const xPos    = Object.fromEntries(categories.map((c, i) => [c, i]));
    const shapes = [], annotations = [];
    let currentY = yTop + gap;
    for (const { c1, c2, symbol } of sigPairs) {
      const x1 = Math.min(xPos[c1], xPos[c2]);
      const x2 = Math.max(xPos[c1], xPos[c2]);
      // Non-significant pairs are drawn in muted grey so the significant ones
      // (black) still read as the headline result.
      const ns    = symbol === 'ns';
      const color = ns ? '#adb5bd' : 'black';
      shapes.push(
        { type:'line', x0:x1, x1:x1, y0:currentY-tickH, y1:currentY, xref:'x', yref:'y', line:{color,width:1} },
        { type:'line', x0:x1, x1:x2, y0:currentY,       y1:currentY, xref:'x', yref:'y', line:{color,width:1} },
        { type:'line', x0:x2, x1:x2, y0:currentY-tickH, y1:currentY, xref:'x', yref:'y', line:{color,width:1} },
      );
      annotations.push({ x:(x1+x2)/2, y:currentY+tickH*0.5, text:symbol, showarrow:false, font:{size:ns?10:12,color}, xref:'x', yref:'y' });
      currentY += gap + tickH;
    }
    Plotly.relayout(plotDiv, { shapes, annotations, 'yaxis.range': [yBottom, currentY + gap], 'yaxis.autorange': false })
      .then(resolve, resolve);
  }));
}
