/**
 * Widget: NoData / non-finite pixel statistics
 *
 * 1. Bar plot of how many images use each distinct no-data value.
 * 2. Histogram of the % of nodata pixels per image (nodata_count / num_pixels).
 * 3. Histogram of non-finite pixel count for float images
 *    (num_pixels - finite_pixel_count), with a summary line.
 */
export default {
  id: 'nodata-value-count',
  label: 'NoData',
  group: 'no-data',
  scope: 'image',
  info: 'NoData value frequency, nodata-pixel percentage, and non-finite pixels (float images).',
  requires(schema) {
    return schema.allCols.includes('nodata_value');
  },

  async condensedPlot(container, ctx) {
    const { groupCol: gcFn } = ctx.sql;
    const rows = await ctx.queryRows(`
      SELECT COALESCE(CAST("nodata_value" AS VARCHAR), 'not set') AS nodata_value,
             ${gcFn()} AS __group__, COUNT(*) AS count
      FROM pp_data ${ctx.where}
      GROUP BY 1, 2
    `);
    if (!rows.length) return false;
    const cats = [...new Set(rows.map(r => String(r.nodata_value)))];
    ctx.plot.appendMini(
      container,
      ctx.plot.groupedBarTraces(cats, pick(rows, r => r.nodata_value, 'count'), { mini: true }),
      { xaxis: { type: 'category' }, yaxis: { showticklabels: false } },
    );
    return true;
  },

  async render(container, ctx) {
    try {
      await renderNodataValueBar(container, ctx);
      await renderNodataPercentHist(container, ctx);
      await renderNonFiniteHist(container, ctx);

      if (!container.firstChild) {
        container.innerHTML = '<div class="no-data">No no-data statistics available.</div>';
      }
    } catch {
      container.innerHTML = '<div class="no-data">Failed to load data.</div>';
    }
  },
};

// --- Section 1: distinct no-data values -------------------------------------
async function renderNodataValueBar(container, ctx) {
  const { groupCol: gcFn } = ctx.sql;
  const gcExpr = gcFn();

  const rows = await ctx.queryRows(`
    SELECT COALESCE(CAST("nodata_value" AS VARCHAR), 'not set') AS nodata_value,
           ${gcExpr} AS __group__, COUNT(*) AS count
    FROM pp_data ${ctx.where}
    GROUP BY 1, 2 ORDER BY 1, 2
  `);
  if (!rows.length) return;

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
}

// --- Section 2: % nodata pixels per image ----------------------------------
async function renderNodataPercentHist(container, ctx) {
  if (!ctx.schema.allCols.includes('num_pixels') ||
      !ctx.schema.allCols.includes('nodata_count')) return;

  const { andWhere, groupCol: gcFn } = ctx.sql;
  const gcExpr  = gcFn();
  const pctExpr = '("nodata_count" * 100.0 / "num_pixels")';

  // Fixed 0–100% bins in 10-point steps. breaks = interior edges, one fewer than labels.
  const breaks = [10, 20, 30, 40, 50, 60, 70, 80, 90];
  const labels = ['0–10%','10–20%','20–30%','30–40%','40–50%',
                  '50–60%','60–70%','70–80%','80–90%','90–100%'];
  const ZERO_LABEL = 'None (0%)';
  const binCase = buildBinCaseSQL(pctExpr, breaks, labels);

  const rows = await ctx.queryRows(`
    SELECT CASE WHEN "nodata_count" = 0 THEN '${ZERO_LABEL}' ELSE ${binCase} END AS bin,
           ${gcExpr} AS __group__, COUNT(*) AS count
    FROM pp_data ${andWhere(ctx.where, '"num_pixels" > 0 AND "nodata_count" IS NOT NULL')}
    GROUP BY 1, 2
  `);
  if (!rows.length) return;

  renderGroupedBars(container, {
    categories: [ZERO_LABEL, ...labels],
    getValue:   pick(rows, r => r.bin, 'count'),
    title:      'NoData Pixel Percentage per Image',
    xLabel:     '% nodata pixels',
    yLabel:     'Image count',
  }, ctx);
}

// --- Section 3: non-finite pixel count for float images --------------------
async function renderNonFiniteHist(container, ctx) {
  if (!ctx.schema.allCols.includes('finite_pixel_count') ||
      !ctx.schema.allCols.includes('num_pixels') ||
      !ctx.schema.allCols.includes('dtype')) return;

  const { andWhere, groupCol: gcFn } = ctx.sql;
  const gcExpr    = gcFn();
  const floatPred = `"dtype" LIKE 'float%' AND "finite_pixel_count" IS NOT NULL AND "num_pixels" > 0`;
  const pctExpr   = '(("num_pixels" - "finite_pixel_count") * 100.0 / "num_pixels")';

  // Summary line: how many images are float.
  const [meta] = await ctx.queryRows(`
    SELECT COUNT(*) FILTER (WHERE "dtype" LIKE 'float%') AS n_float,
           COUNT(*) AS n_total
    FROM pp_data ${ctx.where}
  `);
  const nFloat = Number(meta?.n_float ?? 0);
  const nTotal = Number(meta?.n_total ?? 0);

  const note = document.createElement('div');
  note.style.cssText = 'font-size:13px;color:#495057;margin:8px 0';
  note.textContent = `${nFloat} of ${nTotal} (all) images are float and are expressed here.`;
  container.appendChild(note);
  if (!nFloat) return;

  // Fixed 0–100% bins, plus an exact-zero bucket.
  const breaks = [10, 20, 30, 40, 50, 60, 70, 80, 90];
  const labels = ['0–10%','10–20%','20–30%','30–40%','40–50%',
                  '50–60%','60–70%','70–80%','80–90%','90–100%'];

  const ZERO_LABEL = 'None (0%)';
  const binCase = buildBinCaseSQL(pctExpr, breaks, labels);

  const rows = await ctx.queryRows(`
    SELECT CASE WHEN ("num_pixels" - "finite_pixel_count") = 0 THEN '${ZERO_LABEL}'
                ELSE ${binCase} END AS bin,
           ${gcExpr} AS __group__, COUNT(*) AS count
    FROM pp_data ${andWhere(ctx.where, floatPred)}
    GROUP BY 1, 2
  `);
  if (!rows.length) return;

  renderGroupedBars(container, {
    categories: [ZERO_LABEL, ...labels],
    getValue:   pick(rows, r => r.bin, 'count'),
    title:      'Non-Finite Pixel Percentage per Float Image',
    xLabel:     '% non-finite pixels',
    yLabel:     'Image count',
  }, ctx);
}

// --- local helpers (kept self-contained rather than importing from the base
//     package's plugin_file_stats.js, whose relative extension path is not
//     stable across extension install order) --------------------------------

// (category, group) → a numeric field from long-format rows, 0 when absent.
function pick(rows, catOf, valueKey) {
  const m = new Map(rows.map(r => [`${catOf(r)}\x00${r.__group__}`, Number(r[valueKey] ?? 0)]));
  return (cat, g) => m.get(`${cat}\x00${g}`) ?? 0;
}

function renderGroupedBars(container, { categories, getValue, title, xLabel, yLabel, showLegend = true }, ctx) {
  const legend = showLegend && ctx.groups.length > 1;
  ctx.plot.append(container, ctx.plot.groupedBarTraces(categories, getValue), {
    title:      { text: title },
    barmode:    'stack',
    bargap:     ctx.plot.bargap(categories.length),
    xaxis:      { title: xLabel, type: 'category' },
    yaxis:      { title: yLabel },
    height:     400,
    showlegend: legend,
    ...(legend ? { legend: ctx.plot.plotlyLegendConfig } : {}),
  }, 'margin-bottom:24px');
}

function buildBinCaseSQL(expr, breaks, labels) {
  let sql = `CASE`;
  for (let i = 0; i < breaks.length; i++) sql += ` WHEN ${expr} < ${breaks[i]} THEN '${labels[i]}'`;
  sql += ` ELSE '${labels[labels.length - 1]}' END`;
  return sql;
}