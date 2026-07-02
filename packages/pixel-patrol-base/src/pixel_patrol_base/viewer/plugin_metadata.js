// Columns considered metadata (shared with schema.js META_COLS constant).
const META_COLS = [
  'dim_order', 'dtype', 'ndim', 'pixel_size_X', 'pixel_size_Y', 'pixel_size_Z',
];

const DIST_COLS = ['dtype', 'dim_order'];

const DTYPE_INCONSISTENCY_NOTE =
  ' Mixed <code>dtype</code> is worth a particularly close look: a pixel value of 100 sits near the ' +
  'bottom of the <code>uint16</code> range but the middle of the <code>uint8</code> range, so the ' +
  'same number can mean very different things.';

// Distinct categories of `col` plus a (cat, group) → count lookup. Shared by the
// condensed tile preview and the full distribution bars so both render from one
// query instead of each rebuilding the per-group counts.
async function distCounts(ctx, col) {
  const { q, groupCol: gcFn, andWhere } = ctx.sql;
  const rows = await ctx.queryRows(`
    SELECT ${q(col)} AS cat, ${gcFn()} AS __group__, COUNT(*) AS c
    FROM pp_data ${andWhere(ctx.where, `${q(col)} IS NOT NULL`)}
    GROUP BY 1, 2
  `);
  const cats  = [...new Set(rows.map(r => String(r.cat)))].sort();
  const table = new Map(rows.map(r => [`${r.cat}\x00${r.__group__}`, Number(r.c)]));
  const count = (cat, g) => table.get(`${cat}\x00${g}`) ?? 0;
  const withData = rows.reduce((s, r) => s + Number(r.c), 0);
  return { cats, count, withData };
}

// One categorical distribution: emit a data-availability warning, then either a
// grouped bar chart (varies across groups) or a classification telling render()
// to list it as an invariant property. Returns { kind: 'empty'|'invariant'|'chart' }.
async function distributionCard(container, ctx, col) {
  const { cats, count, withData } = await distCounts(ctx, col);
  const [{ n: total }] = await ctx.queryRows(`SELECT COUNT(*) AS n FROM pp_data ${ctx.where}`);
  const label = ctx.plot.niceName(col);
  ctx.plot.dataAvailabilityWarning(container, [{ label, present: withData }], Number(total ?? 0), { unit: 'images' });

  if (!cats.length)        return { kind: 'empty' };
  if (cats.length === 1)   return { kind: 'invariant', label, value: cats[0] };

  ctx.plot.append(container, ctx.plot.groupedBarTraces(cats, count), {
    title:      { text: `${label} Distribution` },
    barmode:    'stack',
    height:     500,
    xaxis:      { title: label, type: 'category' },
    yaxis:      { title: 'Count' },
    showlegend: ctx.groups.length > 1,
    ...(ctx.groups.length > 1 ? { legend: ctx.plot.plotlyLegendConfig } : {}),
  }, 'margin-bottom:24px');
  return { kind: 'chart', cats };
}

// Red banner naming every categorical metadata column that varies across the dataset.
function prependInconsistencyWarning(container, ctx, varied) {
  const { escapeHtml, humanList } = ctx.plot;
  const list = humanList(varied.map(({ col, cats }) =>
    `<code>${col}</code> (${cats.map(c => `<code>${escapeHtml(c)}</code>`).join(', ')})`));
  const dtypeNote = varied.some(v => v.col === 'dtype') ? DTYPE_INCONSISTENCY_NOTE : '';
  ctx.plot.prependWarning(container, {
    level: 'red',
    html: `This report contains more than one value for ${list}. ` +
      `That can point to inconsistent acquisition - different instruments, software, or settings - ` +
      `and is worth checking before comparing groups.${dtypeNote}`,
  });
}

// Scalar META_COLS that are constant across the whole dataset → invariant rows.
// A *_size of 1 means the dimension is absent, not a shared property, so skip it.
async function collectScalarInvariants(ctx, invariants) {
  const cols = META_COLS.filter(c => ctx.schema.allCols.includes(c) && !DIST_COLS.includes(c));
  for (const { col, value } of await ctx.constantValues(cols)) {
    if (col.endsWith('_size') && Number(value) <= 1) continue;
    invariants.push([col.replace(/_/g, ' '), value]);
  }
}

export default {
  id: 'metadata',
  inputs: ['dim_order', 'dtype', 'ndim', 'pixel_size_X', 'pixel_size_Y', 'pixel_size_Z'],
  group: 'Metadata',
  scope: 'image',
  multiPlot: true,
  label: 'Metadata',
  shortLabel: 'Image Metadata',
  info: 'Shows the distribution of **pixel data types** and **dimension ordering** across groupings.\n\nAlso lists properties shared by all files, and available dimension ranges.',

  requires(schema) {
    const hasDist = DIST_COLS.some(c => schema.allCols.includes(c));
    const hasMeta = META_COLS.some(c => schema.allCols.includes(c));
    return hasDist || hasMeta || Object.keys(schema.dimensionInfo).length > 0;
  },

  async condensedMessage(ctx) {
    try {
      const { q, andWhere } = ctx.sql;
      const { escapeHtml } = ctx.plot;
      const LABELS = { dtype: 'pixel type', dim_order: 'dimension order' };

      // Mixed dtype / dim_order across the dataset is the most important thing to flag.
      const issues = [];
      for (const col of DIST_COLS) {
        if (!ctx.schema.allCols.includes(col)) continue;
        const [row] = await ctx.queryRows(
          `SELECT COUNT(DISTINCT ${q(col)}) AS n FROM pp_data ${andWhere(ctx.where, `${q(col)} IS NOT NULL`)}`,
        );
        if (Number(row?.n ?? 0) > 1) issues.push(LABELS[col] ?? col);
      }
      if (issues.length) return { text: `Inconsistencies: <strong>${issues.join(', ')}</strong>.`, warning: true };

      const hasDtype = ctx.schema.allCols.includes('dtype');
      if (!hasDtype) return null;
      const [row] = await ctx.queryRows(`SELECT MODE("dtype") AS dtype FROM pp_data ${ctx.where}`);
      if (!row?.dtype) return null;
      return `All <strong>${escapeHtml(String(row.dtype))}</strong> pixels.`;
    } catch { return null; }
  },

  async condensedPlot(container, ctx) {
    const { q, andWhere } = ctx.sql;
    const cols = ctx.schema.allCols;

    // One mini-plot per DIST_COL that actually varies — matching exactly the
    // columns condensedMessage flags as inconsistencies. Capped at 2 (beyond
    // that the tile becomes too cramped to read).
    const varying = [];
    let firstPresent = null;
    for (const c of DIST_COLS) {
      if (!cols.includes(c)) continue;
      firstPresent ??= c;
      const [r] = await ctx.queryRows(`SELECT COUNT(DISTINCT ${q(c)}) AS n FROM pp_data ${andWhere(ctx.where, `${q(c)} IS NOT NULL`)}`);
      if (Number(r?.n ?? 0) > 1) varying.push(c);
      if (varying.length === 2) break;
    }
    if (!firstPresent) return false;

    if (varying.length === 0) {
      // Nothing varies → mirror the invariant table the full widget would show.
      const invariants = [];
      for (const c of DIST_COLS) {
        if (!cols.includes(c)) continue;
        const { cats } = await distCounts(ctx, c);
        if (cats.length === 1) invariants.push([ctx.plot.niceName(c), cats[0]]);
      }
      await collectScalarInvariants(ctx, invariants);
      if (!invariants.length) return false;
      ctx.plot.tilePreviewTable(container, ['Property', 'Value'], invariants);
      return true;
    }

    if (varying.length === 1) {
      const { cats, count } = await distCounts(ctx, varying[0]);
      if (!cats.length) return false;
      ctx.plot.appendMini(container, ctx.plot.groupedBarTraces(cats, count, { mini: true }),
        { barmode: 'stack', xaxis: { type: 'category' }, bargap: 0.3 });
      return true;
    }

    // Two vary → stack both, one labelled mini-plot each.
    container.style.cssText += ';display:flex;flex-direction:column;gap:2px';
    let drew = false;
    for (const col of varying) {
      const { cats, count } = await distCounts(ctx, col);
      if (!cats.length) continue;
      const cell = document.createElement('div');
      cell.style.cssText = 'flex:1 1 0;min-height:0;display:flex;flex-direction:column';
      const cap = document.createElement('div');
      cap.textContent = ctx.plot.niceName(col);
      cap.style.cssText = 'font-size:10px;font-weight:600;color:#6c757d;text-align:center;flex-shrink:0';
      const plotHost = document.createElement('div');
      plotHost.style.cssText = 'flex:1 1 0;min-height:0;position:relative';
      cell.append(cap, plotHost);
      container.appendChild(cell);
      ctx.plot.appendMini(plotHost, ctx.plot.groupedBarTraces(cats, count, { mini: true }),
        { barmode: 'stack', xaxis: { type: 'category' }, bargap: 0.3 });
      drew = true;
    }
    return drew;
  },

  async render(container, ctx) {
    try {
      const invariants = [];               // [property, value] rows shared by all files
      const varied     = [];               // categorical cols that differ across the dataset

      // 1. A grouped bar chart per categorical column (or an invariant row if constant).
      for (const col of DIST_COLS) {
        if (!ctx.schema.allCols.includes(col)) continue;
        const r = await distributionCard(container, ctx, col);
        if (r.kind === 'invariant') invariants.push([r.label, r.value]);
        else if (r.kind === 'chart') varied.push({ col, cats: r.cats });
      }

      // 2. Flag inconsistent acquisition, 3. add constant scalar metadata, 4. list invariants.
      if (varied.length) prependInconsistencyWarning(container, ctx, varied);
      await collectScalarInvariants(ctx, invariants);
      if (invariants.length) ctx.plot.invariantTable(container, {
        title: 'Properties shared by all files that report it',
        headers: ['Property', 'Value'],
        rows: invariants,
      });

      if (!container.firstChild) {
        container.innerHTML = '<div class="no-data">No metadata columns found.</div>';
      }
    } catch {
      container.innerHTML = '<div class="no-data">Failed to load data.</div>';
    }
  },
};
