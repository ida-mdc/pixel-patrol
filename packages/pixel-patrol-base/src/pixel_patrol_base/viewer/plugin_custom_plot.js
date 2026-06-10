// Columns whose name contains any of these substrings are excluded from dropdowns.
const EXCLUDED_SUBSTRINGS = ['thumbnail', 'histogram', 'obs_level', 'channel_names', 'file_row_number'];

const EXTRA_NUMERIC = new Set([
  'Y_size', 'X_size', 'Z_size', 'T_size', 'C_size', 'S_size',
  'n_images', 'ndim', 'num_pixels', 'depth', 'size_bytes',
  'pixel_size_X', 'pixel_size_Y', 'pixel_size_Z',
]);

const MAX_SCATTER  = 5_000;
const MAX_CAT      = 30;
const MAX_HUE      = 12;
const COUNT_Y      = '(count)';
const GLOBAL_COLOR = '(global group)';
const NO_COLOR     = '(none)';
const HEATMAP_DEFAULT_COLOR = '#2171b5';

// Label used for a categorical bucket of NULL values, shown as its own
// category on bar/violin/heatmap axes.
const NULL_LABEL = '(missing)';

// Continuous colormaps offered for "color by" when the chosen column is
// numeric and the plot is a scatter (per-point coloring, no legend/groups).
// These are Plotly built-in colorscale names - passed straight through to
// marker.colorscale, no palette lookup needed.
const CONTINUOUS_PALETTES = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis', 'Turbo', 'Blues', 'YlOrRd'];

// Columns holding real timestamps - shown as date axes, never as categories.
const DATE_COLS = new Set(['modification_date']);

const dateFmt = "'%Y-%m-%d %H:%M:%S'";

// Date columns are formatted to readable strings in SQL and plotted on a
// Plotly 'date' axis instead of a raw numeric/category axis.
function selectExpr(q, col, alias) {
  return DATE_COLS.has(col)
    ? `STRFTIME(${q(col)}, ${dateFmt}) AS ${alias}`
    : `${q(col)} AS ${alias}`;
}

function axisCfg(col, title) {
  return DATE_COLS.has(col) ? { title, type: 'date' } : { title };
}

function valueOf(col, v) {
  return DATE_COLS.has(col) ? v : Number(v);
}

// Build an id from column names + a suffix, for exported plugin filenames.
function idFor(...parts) {
  return parts.map(p => String(p).replace(/\W/g, '_')).join('-');
}

// Date-aware MIN/MAX/AVG/STDDEV aggregate expressions for a numeric column.
function aggExprs(q, col) {
  return DATE_COLS.has(col)
    ? `STRFTIME(MIN(${q(col)}), ${dateFmt}) AS min_val, STRFTIME(MAX(${q(col)}), ${dateFmt}) AS max_val,
       STRFTIME(AVG(${q(col)}), ${dateFmt}) AS mean_val`
    : `AVG(${q(col)}) AS mean_val, STDDEV(${q(col)}) AS std_val,
       MIN(${q(col)}) AS min_val,  MAX(${q(col)}) AS max_val`;
}

// Date-aware single aggregate expression, e.g. aggExpr(q, 'modification_date', 'MIN', 'xmin').
function aggExpr(q, col, fn, alias) {
  return DATE_COLS.has(col)
    ? `STRFTIME(${fn}(${q(col)}), ${dateFmt}) AS ${alias}`
    : `${fn}(${q(col)}) AS ${alias}`;
}

// Format a stat value for display: dates pass through, numbers get 4 sig figs, nulls show '—'.
function fmtStat(col, v) {
  if (v == null) return '—';
  return DATE_COLS.has(col) ? v : (Number.isFinite(Number(v)) ? Number(v).toPrecision(4) : '—');
}

// Fallback palettes — used when viewer hasn't been rebuilt to expose ctx.color.getColors.
// Colors match the viewer's colors.js exactly so the fallback is visually identical.
const LOCAL_PALETTES = {
  tab10:  ['#9467bd','#ff7f0e','#17becf','#2ca02c','#8c564b','#7f7f7f','#bcbd22','#d62728','#1f77b4','#e377c2'],
  tab20:  ['#9467bd','#c5b0d5','#2ca02c','#98df8a','#e377c2','#f7b6d2','#17becf','#9edae5','#d62728','#ff9896',
           '#bcbd22','#dbdb8d','#1f77b4','#aec7e8','#ff7f0e','#ffbb78','#8c564b','#c49c94','#7f7f7f','#c7c7c7'],
  Accent: ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666'],
  Dark2:  ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666'],
  Paired: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a'],
  Set1:   ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999'],
  Set2:   ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'],
  Set3:   ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd'],
  rainbow: null, // generated
};

function getColorsLocal(name, n) {
  const base = LOCAL_PALETTES[name];
  if (base) {
    if (n <= base.length) return base.slice(0, n);
    return Array.from({ length: n }, (_, i) => base[i % base.length]);
  }
  // Simple HSL rainbow fallback
  return Array.from({ length: n }, (_, i) => `hsl(${Math.round(360 * i / n)},70%,50%)`);
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function lbl(text) {
  const el = Object.assign(document.createElement('span'), { textContent: text });
  el.style.cssText = 'font-size:13px;font-weight:600;white-space:nowrap';
  return el;
}

function mkSelect(opts) {
  const sel = document.createElement('select');
  sel.style.cssText = 'padding:4px 8px;border:1px solid #dee2e6;border-radius:4px;font-size:13px;background:#fff';
  for (const [text, value] of opts) {
    sel.appendChild(Object.assign(document.createElement('option'), { textContent: text, value }));
  }
  return sel;
}

function makeCombobox(cols, placeholder) {
  const wrap  = document.createElement('div');
  wrap.style.cssText = 'position:relative;display:inline-block';

  const input = Object.assign(document.createElement('input'), { type: 'text', placeholder });
  input.style.cssText = 'padding:4px 8px;border:1px solid #dee2e6;border-radius:4px;font-size:13px;width:190px';

  const dropdown = document.createElement('div');
  dropdown.style.cssText = [
    'position:absolute;top:100%;left:0;width:230px;z-index:200',
    'max-height:200px;overflow-y:auto;display:none',
    'background:#fff;border:1px solid #dee2e6;border-top:none',
    'border-radius:0 0 4px 4px;box-shadow:0 4px 8px rgba(0,0,0,0.1)',
  ].join(';');

  let selected = null;
  let onSelect = null;

  function renderDropdown(filter) {
    dropdown.innerHTML = '';
    const f       = filter.toLowerCase();
    const matches = f ? cols.filter(c => c.toLowerCase().includes(f)) : cols;
    if (!matches.length) {
      const d = Object.assign(document.createElement('div'), { textContent: 'No matches' });
      d.style.cssText = 'padding:6px 10px;color:#aaa;font-size:13px';
      dropdown.appendChild(d);
      return;
    }
    for (const col of matches) {
      const item    = Object.assign(document.createElement('div'), { textContent: col });
      const special = col === COUNT_Y || col === GLOBAL_COLOR;
      item.style.cssText = 'padding:5px 10px;cursor:pointer;font-size:13px;' +
        'background:' + (col === selected ? '#e8f0fe' : '') +
        (special ? ';font-style:italic;color:#0d6efd;border-bottom:1px solid #dee2e6' : '');
      item.addEventListener('mouseover', () => { item.style.background = '#f0f4ff'; });
      item.addEventListener('mouseout',  () => { item.style.background = col === selected ? '#e8f0fe' : ''; });
      item.addEventListener('mousedown', e => {
        e.preventDefault();
        selected = col;
        input.value = col;
        dropdown.style.display = 'none';
        onSelect?.(col);
      });
      dropdown.appendChild(item);
    }
  }

  // Always show all options on focus so a pre-set value doesn't collapse the list.
  input.addEventListener('focus',   ()  => { renderDropdown(''); dropdown.style.display = 'block'; });
  input.addEventListener('input',   ()  => { selected = null; renderDropdown(input.value); dropdown.style.display = 'block'; });
  input.addEventListener('keydown', e  => { if (e.key === 'Escape') { dropdown.style.display = 'none'; input.blur(); } });
  input.addEventListener('blur',    ()  => {
    setTimeout(() => {
      dropdown.style.display = 'none';
      if (!cols.includes(input.value)) input.value = selected ?? '';
    }, 150);
  });

  wrap.append(input, dropdown);
  return {
    el:       wrap,
    onSelect: cb  => { onSelect = cb; },
    get:      ()  => selected,
    set:      col => { selected = col; input.value = col ?? ''; },
  };
}

// ── Plugin code generator ─────────────────────────────────────────────────────

function generatePluginCode({ plotType, x, y, catCol, numCol, colorBy, continuous, palette, noneColor, heatColor, heatInvert }) {
  const qc   = c => '"' + String(c).replace(/"/g, '""') + '"';
  const nice = c => (c ?? '').replace(/_/g, ' ').replace(/\b\w/g, ch => ch.toUpperCase());
  const cid  = c => (c ?? '').replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();

  const yId = y && y !== COUNT_Y ? cid(y) : 'count';
  const id  = cid(x) + '-' + yId + '-' + plotType;

  // Only show this plugin on datasets that actually have the columns it plots.
  const requiredCols = [...new Set([x, y, catCol, numCol, colorBy].filter(c => c && c !== COUNT_Y && c !== 'none'))];

  const TITLE = plotType === 'scatter'  ? nice(x) + ' vs ' + nice(y)
              : plotType === 'violin'   ? nice(numCol) + ' by ' + nice(catCol)
              : plotType === 'bar'      ? 'Mean ' + nice(numCol) + ' by ' + nice(catCol)
              : plotType === 'countBar' ? 'Count by ' + nice(x)
              : plotType === 'heatmap'  ? 'Count: ' + nice(x) + ' × ' + nice(y)
              : 'Custom Plot';

  const isCustomColor = colorBy && colorBy !== 'none' && !continuous;

  const gExpr = colorBy === 'none'
    ? "'__all__' AS __group__"
    : isCustomColor
      ? 'CAST(' + qc(colorBy) + ' AS VARCHAR) AS __group__'
      : '${ctx.sql.groupExpr()}';

  const groupSetup = colorBy === 'none'
    ? [
        "      const groups  = ['__all__'];",
        '      const colorFn = () => ' + JSON.stringify(noneColor) + ';',
        "      const labelFn = () => '';",
      ].join('\n')
    : isCustomColor
      ? [
          '      const allG    = [...new Set(rows.map(r => String(r.__group__)))].sort();',
          '      const colors  = (ctx.color.getColors ?? _fallbackColors)(' + JSON.stringify(palette) + ', allG.length);',
          '      const idxM    = new Map(allG.map((g, i) => [g, i]));',
          '      const groups  = allG;',
          '      const colorFn = g => colors[idxM.get(g)] ?? \'#888\';',
          '      const labelFn = g => g;',
        ].join('\n')
      : [
          '      const dg      = new Set(rows.map(r => String(r.__group__)));',
          '      const groups  = ctx.groups.filter(g => dg.has(g));',
          '      const colorFn = g => ctx.color.group(g);',
          '      const labelFn = g => ctx.groupLabel(g);',
        ].join('\n');

  const legDecl = [
    '      const leg = groups.length > 1',
    '        ? { showlegend: true, legend: ctx.plot.plotlyLegendConfig }',
    '        : { showlegend: false };',
  ].join('\n');

  // Fallback color helper embedded in generated file (in case older viewer)
  const fallbackFn = isCustomColor ? [
    '',
    '  function _fallbackColors(name, n) {',
    '    const P = ' + JSON.stringify(LOCAL_PALETTES[palette] ?? LOCAL_PALETTES.tab10) + ';',
    '    return Array.from({length:n}, (_, i) => P[i % P.length]);',
    '  }',
  ].join('\n') : '';

  // Date columns are formatted to readable strings in SQL and plotted on a 'date' axis.
  const sel     = (col, alias) => DATE_COLS.has(col) ? `STRFTIME(${qc(col)}, ${dateFmt}) AS ${alias}` : `${qc(col)} AS ${alias}`;
  const valExpr = (col, v)     => DATE_COLS.has(col) ? v : `Number(${v})`;
  const axisObj = (col, title) => DATE_COLS.has(col)
    ? `{ title: ${JSON.stringify(title)}, type: 'date' }`
    : `{ title: ${JSON.stringify(title)} }`;

  // NULLs become their own "(missing)" category, sorted last.
  const catExprG = col => `COALESCE(CAST(${qc(col)} AS VARCHAR), '${NULL_LABEL}')`;
  const usesCatExpr = ['violin', 'bar', 'countBar', 'heatmap'].includes(plotType);
  const sortCatsFn = usesCatExpr ? [
    '',
    '  function _sortCats(cats) {',
    '    const real = cats.filter(c => c !== ' + JSON.stringify(NULL_LABEL) + ').sort();',
    '    return cats.includes(' + JSON.stringify(NULL_LABEL) + ') ? [...real, ' + JSON.stringify(NULL_LABEL) + '] : real;',
    '  }',
  ].join('\n') : '';

  let body = '';

  if (plotType === 'scatter' && continuous) {
    body = [
      '      const wh = andWhere(ctx.where, `' + qc(x) + ' IS NOT NULL AND ' + qc(y) + ' IS NOT NULL AND ' + qc(colorBy) + ' IS NOT NULL`);',
      '      const rows = await ctx.queryRows(`',
      '        SELECT * FROM (',
      '          SELECT ' + sel(x, 'x') + ', ' + sel(y, 'y') + ', ' + qc(colorBy) + ' AS c',
      '          FROM pp_data ${wh}',
      '        ) USING SAMPLE 5000 ROWS (reservoir, 42)',
      '      `);',
      '      const sampled = rows.length >= 5000;',
      '      const trace = {',
      '        type: \'scatter\', mode: \'markers\',',
      '        x: rows.map(r => ' + valExpr(x, 'r.x') + '),',
      '        y: rows.map(r => ' + valExpr(y, 'r.y') + '),',
      '        marker: {',
      '          color: rows.map(r => Number(r.c)),',
      '          colorscale: ' + JSON.stringify(palette) + ',',
      '          showscale: true,',
      '          colorbar: { title: { text: ' + JSON.stringify(nice(colorBy)) + ' } },',
      '          size: 5, opacity: 0.7,',
      '        },',
      '      };',
      '      append(container, [trace], {',
      '        title: { text: ' + JSON.stringify(TITLE) + ' + (sampled ? \'<br><sup>sampled to 5,000 pts</sup>\' : \'\') },',
      '        xaxis: ' + axisObj(x, nice(x)) + ',',
      '        yaxis: ' + axisObj(y, nice(y)) + ',',
      '        showlegend: false,',
      '      });',
    ].join('\n');
  } else if (plotType === 'scatter') {
    body = [
      '      const wh = andWhere(ctx.where, `' + qc(x) + ' IS NOT NULL AND ' + qc(y) + ' IS NOT NULL`);',
      '      const rows = await ctx.queryRows(`',
      '        SELECT * FROM (',
      '          SELECT ' + sel(x, 'x') + ', ' + sel(y, 'y') + ', ' + gExpr,
      '          FROM pp_data ${wh}',
      '        ) USING SAMPLE 5000 ROWS (reservoir, 42)',
      '      `);',
      groupSetup,
      '      const sampled = rows.length >= 5000;',
      '      const traces = groups.map(g => ({',
      '        type: \'scatter\', mode: \'markers\', name: labelFn(g),',
      '        x: rows.filter(r => String(r.__group__) === g).map(r => ' + valExpr(x, 'r.x') + '),',
      '        y: rows.filter(r => String(r.__group__) === g).map(r => ' + valExpr(y, 'r.y') + '),',
      '        marker: { color: colorFn(g), size: 5, opacity: 0.7 },',
      '      })).filter(t => t.x.length);',
      legDecl,
      '      append(container, traces, {',
      '        title: { text: ' + JSON.stringify(TITLE) + ' + (sampled ? \'<br><sup>sampled to 5,000 pts</sup>\' : \'\') },',
      '        xaxis: ' + axisObj(x, nice(x)) + ',',
      '        yaxis: ' + axisObj(y, nice(y)) + ',',
      '        ...leg,',
      '      });',
    ].join('\n');
  } else if (plotType === 'violin') {
    body = [
      '      const wh = andWhere(ctx.where, `' + qc(numCol) + ' IS NOT NULL`);',
      '      const rows = await ctx.queryRows(`',
      '        SELECT * FROM (',
      '          SELECT ' + catExprG(catCol) + ' AS cat, ' + sel(numCol, 'val') + ', ' + gExpr,
      '          FROM pp_data ${wh}',
      '        ) USING SAMPLE 5000 ROWS (reservoir, 42)',
      '      `);',
      groupSetup,
      '      const cats = _sortCats([...new Set(rows.map(r => String(r.cat)))]);',
      '      const traces = groups.map(g => ({',
      '        type: \'violin\', name: labelFn(g),',
      '        x: rows.filter(r => String(r.__group__) === g).map(r => String(r.cat)),',
      '        y: rows.filter(r => String(r.__group__) === g).map(r => ' + valExpr(numCol, 'r.val') + '),',
      '        box: { visible: true }, meanline: { visible: true }, points: \'outliers\',',
      '        spanmode: \'hard\', marker: { color: colorFn(g) },',
      '      })).filter(t => t.y.length);',
      legDecl,
      '      append(container, traces, {',
      '        title: { text: ' + JSON.stringify(TITLE) + ' },',
      '        xaxis: { title: ' + JSON.stringify(nice(catCol)) + ', type: \'category\', categoryarray: cats },',
      '        yaxis: ' + axisObj(numCol, nice(numCol)) + ',',
      '        violinmode: groups.length > 1 ? \'group\' : undefined,',
      '        ...leg,',
      '      });',
    ].join('\n');
  } else if (plotType === 'bar') {
    body = [
      '      const wh = andWhere(ctx.where, `' + qc(numCol) + ' IS NOT NULL`);',
      '      const rows = await ctx.queryRows(`',
      '        SELECT ' + catExprG(catCol) + ' AS cat, ' + gExpr + ',',
      '               AVG(' + qc(numCol) + ') AS mean_val, STDDEV(' + qc(numCol) + ') AS std_val',
      '        FROM pp_data ${wh}',
      '        GROUP BY 1, 2 ORDER BY 1',
      '      `);',
      groupSetup,
      '      const cats = _sortCats([...new Set(rows.map(r => String(r.cat)))]);',
      '      const traces = groups.map(g => {',
      '        const gr = rows.filter(r => String(r.__group__) === g);',
      '        return {',
      '          type: \'bar\', name: labelFn(g),',
      '          x: gr.map(r => String(r.cat)), y: gr.map(r => Number(r.mean_val)),',
      '          error_y: { type: \'data\', visible: true, array: gr.map(r => Number(r.std_val)) },',
      '          marker: { color: colorFn(g) },',
      '        };',
      '      }).filter(t => t.x.length);',
      legDecl,
      '      append(container, traces, {',
      '        title: { text: ' + JSON.stringify(TITLE) + ' },',
      '        xaxis: { title: ' + JSON.stringify(nice(catCol)) + ', type: \'category\', categoryarray: cats },',
      '        yaxis: { title: ' + JSON.stringify('Mean ' + nice(numCol)) + ' },',
      '        barmode: groups.length > 1 ? \'group\' : undefined,',
      '        ...leg,',
      '      });',
    ].join('\n');
  } else if (plotType === 'countBar') {
    body = [
      '      const rows = await ctx.queryRows(`',
      '        SELECT ' + catExprG(x) + ' AS cat, ' + gExpr + ', COUNT(*) AS n',
      '        FROM pp_data ${ctx.where}',
      '        GROUP BY 1, 2 ORDER BY 1',
      '      `);',
      groupSetup,
      '      const cats = _sortCats([...new Set(rows.map(r => String(r.cat)))]);',
      '      const traces = groups.map(g => ({',
      '        type: \'bar\', name: labelFn(g),',
      '        x: rows.filter(r => String(r.__group__) === g).map(r => String(r.cat)),',
      '        y: rows.filter(r => String(r.__group__) === g).map(r => Number(r.n)),',
      '        marker: { color: colorFn(g) },',
      '      })).filter(t => t.x.length);',
      legDecl,
      '      append(container, traces, {',
      '        title: { text: ' + JSON.stringify(TITLE) + ' },',
      '        xaxis: { title: ' + JSON.stringify(nice(x)) + ', type: \'category\', categoryarray: cats },',
      '        yaxis: { title: \'Count\' },',
      '        barmode: \'stack\',',
      '        ...leg,',
      '      });',
    ].join('\n');
  } else if (plotType === 'heatmap') {
    body = [
      '      const rows = await ctx.queryRows(`',
      '        SELECT ' + catExprG(x) + ' AS x, ' + catExprG(y) + ' AS y, COUNT(*) AS n',
      '        FROM pp_data ${ctx.where}',
      '        GROUP BY 1, 2 ORDER BY 1, 2',
      '      `);',
      '      const xs = _sortCats([...new Set(rows.map(r => String(r.x)))]);',
      '      const ys = _sortCats([...new Set(rows.map(r => String(r.y)))]);',
      '      const z  = ys.map(yv => xs.map(xv => {',
      '        const r = rows.find(r => String(r.x) === xv && String(r.y) === yv);',
      '        return r ? Number(r.n) : 0;',
      '      }));',
      '      const colorscale = [[0, \'#ffffff\'], [1, ' + JSON.stringify(heatColor) + ']];',
      '      append(container,',
      '        [{ type: \'heatmap\', x: xs, y: ys, z, colorscale, reversescale: ' + JSON.stringify(!!heatInvert) + ', showscale: true }], {',
      '        title: { text: ' + JSON.stringify(TITLE) + ' },',
      '        xaxis: { title: ' + JSON.stringify(nice(x)) + ', type: \'category\' },',
      '        yaxis: { title: ' + JSON.stringify(nice(y)) + ', type: \'category\' },',
      '        showlegend: false,',
      '      });',
    ].join('\n');
  }

  return [
    '// Generated by Pixel Patrol Custom Plot widget',
    '// Drop into your viewer plugins directory and add to extension.json',
    fallbackFn,
    sortCatsFn,
    '',
    'export default {',
    '  id: ' + JSON.stringify(id) + ',',
    '  label: ' + JSON.stringify(TITLE) + ',',
    '  group: \'Custom\',',
    '  requires: s => s.isLongFormat && ' + JSON.stringify(requiredCols) + '.every(c => s.allCols.includes(c)),',
    '  async render(container, ctx) {',
    '    try {',
    '      const { andWhere } = ctx.sql;',
    '      const { append } = ctx.plot;',
    '',
    body,
    '    } catch (e) {',
    '      container.innerHTML = \'<div class="no-data">Failed to load.</div>\';',
    '      console.error(e);',
    '    }',
    '  },',
    '};',
  ].join('\n');
}

// ── Plugin ────────────────────────────────────────────────────────────────────

export default {
  id:    'custom-plot',
  label: 'Custom Plot',
  group: 'Explore',
  info:  'Build your own plot from the columns in your current data.\n\n' +
         '- **Two numerics** → scatter\n' +
         '- **Categorical × numeric** → violin or bar (mean ± sd)\n' +
         '- **Any column × (count)** → count bar\n' +
         '- **Two categoricals** → count heatmap\n\n' +
         'Single unique value → table instead of plot. For scatter, this applies when the Y ' +
         'column is constant across all rows (regardless of X).\n\n' +
         '**Color by:**\n' +
         '- **(none)** → a single, ungrouped trace (no legend), in a color you pick with the swatch ' +
         'next to "Color by".\n' +
         '- **(global group)** (default) → colors/splits by whatever column the app-wide grouping is ' +
         'currently set to.\n' +
         '- **any other column** → colors/splits by that column instead, using the chosen palette.\n\n' +
         '**Scatter plots** additionally support coloring by a *numeric* column: each point is ' +
         'colored individually on a continuous colormap (with a colorbar) instead of being split ' +
         'into groups — pick the colormap from the palette dropdown.\n\n' +
         `Color-by is automatically switched to **(none)** when:\n` +
         '- the chosen X/Y column is already the active global-grouping column (avoids "double ' +
         'grouping", e.g. the same category split into both the x-axis and the legend), or\n' +
         `- the chosen color-by column has more than ${MAX_HUE} unique values and isn't used as a ` +
         'continuous colormap (too many to color distinctly) — a warning is shown when this happens.\n\n' +
         '**Heatmaps** (two categoricals) never use color-by/hue grouping — instead, pick a single ' +
         'base color with the swatch; cell counts are shown on a white-to-color scale (white = 0, ' +
         'your color = the maximum). Check **Invert** to reverse this (your color = 0, white = the maximum).\n\n' +
         '**Dates**: `modification_date` is always plotted as a date axis (never as a category), ' +
         'shown as a readable timestamp. "Bar (mean ± sd)" is unavailable for it — use violin instead.\n\n' +
         `**Missing values**: for violin/bar/heatmap, rows where a *categorical* axis is null are ` +
         `grouped into their own "${NULL_LABEL}" category (shown last). For scatter, rows where X or Y ` +
         `is null can't be placed on a numeric axis and are excluded — a warning shows how many.\n\n` +
         '**＋ Add plot** adds an independent plot below.',

  requires(schema) {
    return schema.isLongFormat;
  },

  async render(container, ctx) {
    const { q, andWhere } = ctx.sql;
    const { append: appendPlot, niceName, plotlyLegendConfig } = ctx.plot;

    const available  = ctx.schema.allCols.filter(c => !EXCLUDED_SUBSTRINGS.some(s => c.includes(s))).sort();
    const numericSet = new Set([
      ...ctx.schema.metricCols,
      ...available.filter(c => EXTRA_NUMERIC.has(c) || DATE_COLS.has(c)),
    ]);

    // Use viewer's palette functions if available (requires rebuilt viewer),
    // otherwise fall back to local copies.
    const paletteNames = typeof ctx.color.getPaletteNames === 'function'
      ? ctx.color.getPaletteNames()
      : Object.keys(LOCAL_PALETTES);
    const getColors = typeof ctx.color.getColors === 'function'
      ? (name, n) => ctx.color.getColors(name, n)
      : getColorsLocal;

    // ── Slot management ───────────────────────────────────────────────────────
    const slotsDiv = document.createElement('div');
    container.appendChild(slotsDiv);

    const addBtn = Object.assign(document.createElement('button'), { textContent: '＋ Add plot' });
    addBtn.style.cssText = 'margin-top:12px;padding:4px 12px;border:1px solid #0d6efd;border-radius:4px;background:#fff;cursor:pointer;font-size:13px;color:#0d6efd';
    container.appendChild(addBtn);

    const slotRefs = []; // { slotEl, removeBtn }

    function syncSlotUi() {
      const n = slotRefs.length;
      slotRefs.forEach(({ slotEl, removeBtn }, i) => {
        removeBtn.style.display = i > 0 ? '' : 'none';
        slotEl.style.borderTop  = i > 0 ? '1px solid #dee2e6' : 'none';
        slotEl.style.paddingTop = i > 0 ? '16px' : '0';
        slotEl.style.marginTop  = i > 0 ? '16px' : '0';
      });
    }

    function createSlot() {
      const slotEl = document.createElement('div');

      // ── Controls ────────────────────────────────────────────────────────────
      const controls  = document.createElement('div');
      controls.style.cssText = 'display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap';

      const xBox       = makeCombobox(available, 'X column…');
      const yBox       = makeCombobox([COUNT_Y, ...available], 'Y column…');
      const typeSel    = mkSelect([['Violin', 'violin'], ['Bar (mean ± sd)', 'bar']]);
      const colorBySel = makeCombobox([NO_COLOR, GLOBAL_COLOR, ...available], 'Color by…');
      const paletteSel = mkSelect(paletteNames.map(n => [n, n]));
      const noneColor  = Object.assign(document.createElement('input'), { type: 'color', value: getColors('tab10', 1)[0] });

      // Heatmaps don't support color-by grouping - instead let the user pick a
      // single-hue colorscale (white → color, optionally inverted).
      const heatColor  = Object.assign(document.createElement('input'), { type: 'color', value: HEATMAP_DEFAULT_COLOR });
      const heatInvert = Object.assign(document.createElement('input'), { type: 'checkbox' });
      const heatInvertLabel = document.createElement('label');
      heatInvertLabel.style.cssText = 'font-size:13px;display:flex;align-items:center;gap:4px;cursor:pointer';
      heatInvertLabel.append(heatInvert, document.createTextNode('Invert'));

      const clearBtn  = Object.assign(document.createElement('button'), { textContent: '✕ Clear' });
      const exportBtn = Object.assign(document.createElement('button'), { textContent: '↓ Export plugin' });
      const removeBtn = Object.assign(document.createElement('button'), { textContent: '✕ Remove' });

      typeSel.style.display    = 'none';
      paletteSel.style.display = 'none';
      exportBtn.style.display  = 'none';
      removeBtn.style.display  = 'none';

      typeSel.style.cssText    = 'padding:4px 8px;border:1px solid #dee2e6;border-radius:4px;font-size:13px;background:#fff;display:none';
      paletteSel.style.cssText = 'padding:4px 8px;border:1px solid #dee2e6;border-radius:4px;font-size:13px;background:#fff;display:none';
      noneColor.style.cssText  = 'width:32px;height:28px;padding:1px;border:1px solid #dee2e6;border-radius:4px;cursor:pointer;display:none';
      heatColor.style.cssText  = 'width:32px;height:28px;padding:1px;border:1px solid #dee2e6;border-radius:4px;cursor:pointer';
      clearBtn.style.cssText   = 'padding:3px 10px;border:1px solid #dee2e6;border-radius:4px;background:#fff;cursor:pointer;font-size:12px';
      exportBtn.style.cssText  = 'padding:3px 10px;border:1px solid #0d6efd;border-radius:4px;background:#fff;cursor:pointer;font-size:12px;color:#0d6efd;display:none';
      removeBtn.style.cssText  = 'padding:3px 10px;border:1px solid #dc3545;border-radius:4px;background:#fff;cursor:pointer;font-size:12px;color:#dc3545;display:none';

      // Color-by group (hidden for heatmaps) vs heatmap colorscale group.
      const colorGroup = document.createElement('div');
      colorGroup.style.cssText = 'display:flex;align-items:center;gap:10px';
      colorGroup.append(lbl('Color:'), colorBySel.el, paletteSel, noneColor);

      const heatGroup = document.createElement('div');
      heatGroup.style.cssText = 'display:none;align-items:center;gap:10px';
      heatGroup.append(lbl('Color:'), heatColor, heatInvertLabel);

      controls.append(
        lbl('X:'), xBox.el,
        lbl('Y:'), yBox.el,
        typeSel,
        colorGroup, heatGroup,
        clearBtn, exportBtn, removeBtn,
      );
      slotEl.appendChild(controls);

      colorBySel.set(GLOBAL_COLOR);

      // The palette dropdown shows different option lists depending on
      // whether color-by resolved to a discrete (qualitative) or continuous
      // (numeric, scatter-only) column. Remember the last choice per kind so
      // switching back and forth doesn't lose the user's pick.
      let paletteKind     = 'qualitative';
      let lastQualPalette = paletteNames[0];
      let lastContPalette = CONTINUOUS_PALETTES[0];

      function setPaletteKind(kind) {
        if (kind === paletteKind) return;
        paletteKind = kind;
        const list      = kind === 'continuous' ? CONTINUOUS_PALETTES : paletteNames;
        const remembered = kind === 'continuous' ? lastContPalette : lastQualPalette;
        paletteSel.innerHTML = '';
        for (const n of list) {
          paletteSel.appendChild(Object.assign(document.createElement('option'), { textContent: n, value: n }));
        }
        paletteSel.value = list.includes(remembered) ? remembered : list[0];
      }

      const plotArea = document.createElement('div');
      slotEl.appendChild(plotArea);
      slotsDiv.appendChild(slotEl);

      const ref = { slotEl, removeBtn };
      slotRefs.push(ref);
      syncSlotUi();

      removeBtn.addEventListener('click', () => {
        const idx = slotRefs.indexOf(ref);
        if (idx >= 0) slotRefs.splice(idx, 1);
        slotEl.remove();
        syncSlotUi();
      });

      // ── Export ─────────────────────────────────────────────────────────────
      let activeConfig = null;

      function setConfig(cfg) {
        activeConfig = cfg;
        exportBtn.style.display = '';
      }

      exportBtn.addEventListener('click', () => {
        if (!activeConfig) return;
        const code = generatePluginCode(activeConfig);
        const name = 'plugin_' + activeConfig.id + '.js';
        const url  = URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
        const a    = Object.assign(document.createElement('a'), { href: url, download: name });
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      });

      // ── Group config ────────────────────────────────────────────────────────
      // colorMode is one of: 'none' (no grouping/legend), 'global' (the app's
      // global group), or a column name (custom categorical color-by).
      function groupSqlExpr(colorMode) {
        if (colorMode === 'none')   return `'__all__' AS __group__`;
        if (colorMode === 'global') return ctx.sql.groupExpr();
        return `CAST(${q(colorMode)} AS VARCHAR) AS __group__`;
      }

      function buildGroupConfig(rows, colorMode) {
        if (colorMode === 'none') {
          return {
            groups:  ['__all__'],
            colorFn: () => noneColor.value,
            labelFn: () => '',
            header:  '',
          };
        }
        if (colorMode === 'global') {
          const dg = new Set(rows.map(r => String(r.__group__)));
          return {
            groups:  ctx.groups.filter(g => dg.has(g)),
            colorFn: g => ctx.color.group(g),
            labelFn: g => ctx.groupLabel(g),
            header:  ctx.state.groupCol ? niceName(ctx.state.groupCol) : 'Group',
          };
        }
        const groups = [...new Set(rows.map(r => String(r.__group__)))].sort();
        const colors = getColors(paletteSel.value, groups.length);
        const idxMap = new Map(groups.map((g, i) => [g, i]));
        return {
          groups,
          colorFn: g => colors[idxMap.get(g)] ?? '#888',
          labelFn: g => g,
          header:  niceName(colorMode),
        };
      }

      // Decide how to color/group this plot, given the columns it uses.
      // Falls back to 'none' when the chosen color-by would just duplicate an
      // axis (double grouping) or has too many unique values to color by.
      async function resolveColorMode(relevantCols, { scatter = false } = {}) {
        const c = colorBySel.get();
        if (c === NO_COLOR) return { mode: 'none', note: null, continuous: false };
        if (!c || c === GLOBAL_COLOR) {
          if (ctx.state.groupCol && relevantCols.includes(ctx.state.groupCol)) {
            return {
              mode: 'none',
              note: `Coloring by "${niceName(ctx.state.groupCol)}" (the global group) is hidden because it's already shown on this plot.`,
              continuous: false,
            };
          }
          return { mode: 'global', note: null, continuous: false };
        }
        // Scatter only: a numeric color-by column is shown with a continuous
        // colormap (one trace, per-point color) instead of discrete groups.
        if (scatter && numericSet.has(c) && !DATE_COLS.has(c)) {
          return { mode: c, note: null, continuous: true };
        }
        const [{ n }] = await cardinalityCheck(c);
        if (n > MAX_HUE) {
          return {
            mode: 'none',
            note: `"${niceName(c)}" has ${n} unique values — too many to color by (max ${MAX_HUE}). Showing without color grouping.`,
            continuous: false,
          };
        }
        return { mode: c, note: null, continuous: false };
      }

      function showColorNote(note) {
        if (!note) return;
        const div = document.createElement('div');
        div.style.cssText = 'font-size:12px;color:#856404;background:#fff3cd;border:1px solid #ffc107;border-radius:4px;padding:4px 10px;margin-bottom:8px';
        div.textContent = `⚠ ${note}`;
        plotArea.appendChild(div);
      }

      function legend(groups) {
        return groups.length > 1 ? { showlegend: true, legend: plotlyLegendConfig } : { showlegend: false };
      }

      async function cardinalityCheck(...cols) {
        const exprs = cols.map((c, i) => `COUNT(DISTINCT ${q(c)}) AS n${i}`).join(', ');
        const [row]  = await ctx.queryRows(`SELECT ${exprs} FROM pp_data ${ctx.where}`);
        return cols.map((c, i) => ({ col: c, n: Number(row[`n${i}`]) }));
      }

      // Categorical axis values: NULLs become their own "(missing)" category
      // instead of being dropped.
      function catExpr(col) {
        return `COALESCE(CAST(${q(col)} AS VARCHAR), '${NULL_LABEL}')`;
      }

      async function categoryCount(col) {
        const [row] = await ctx.queryRows(`SELECT COUNT(DISTINCT ${catExpr(col)}) AS n FROM pp_data ${ctx.where}`);
        return Number(row.n);
      }

      // Sort categories with the "(missing)" bucket placed last.
      function sortCats(cats) {
        const real = cats.filter(c => c !== NULL_LABEL).sort();
        return cats.includes(NULL_LABEL) ? [...real, NULL_LABEL] : real;
      }

      // Count NULLs in numeric columns - these can't be placed on a numeric
      // axis (e.g. scatter), so callers show a warning instead.
      async function nullCounts(...cols) {
        const exprs = cols.map((c, i) => `COUNT(*) FILTER (WHERE ${q(c)} IS NULL) AS n${i}`).join(', ');
        const [row]  = await ctx.queryRows(`SELECT ${exprs} FROM pp_data ${ctx.where}`);
        return cols.map((c, i) => ({ col: c, n: Number(row[`n${i}`]) }));
      }

      // Warning text for numeric column(s) that contain NULLs and so can't be
      // placed on a numeric axis. Returns null if none of the columns have NULLs.
      async function nullWarningNote(...cols) {
        const nc    = await nullCounts(...cols);
        const parts = nc.filter(c => c.n > 0).map(c => `${c.n.toLocaleString()} null "${niceName(c.col)}" value${c.n === 1 ? '' : 's'}`);
        return parts.length ? `${parts.join(' and ')} can't be shown on a numeric axis — those points are excluded.` : null;
      }

      function statTable(headers, rowData) {
        const t = document.createElement('table');
        t.className = 'stat-table';
        const hRow = headers.map(h => `<th>${h}</th>`).join('');
        const body = rowData.map(cells => `<tr>${cells.map(c => `<td>${c}</td>`).join('')}</tr>`).join('');
        t.innerHTML = `<thead><tr>${hRow}</tr></thead><tbody>${body}</tbody>`;
        return t;
      }

      // ── Plot functions ──────────────────────────────────────────────────────
      async function plotScatter(x, y, colorMode, continuous) {
        if (continuous) {
          const fullWh = andWhere(ctx.where, `${q(x)} IS NOT NULL AND ${q(y)} IS NOT NULL AND ${q(colorMode)} IS NOT NULL`);
          const rows = await ctx.queryRows(`
            SELECT * FROM (
              SELECT ${selectExpr(q, x, 'x')}, ${selectExpr(q, y, 'y')}, ${q(colorMode)} AS c
              FROM pp_data ${fullWh}
            ) USING SAMPLE ${MAX_SCATTER} ROWS (reservoir, 42)
          `);
          const sampled = rows.length >= MAX_SCATTER;
          const trace = {
            type: 'scatter', mode: 'markers',
            x: rows.map(r => valueOf(x, r.x)), y: rows.map(r => valueOf(y, r.y)),
            marker: {
              color: rows.map(r => Number(r.c)),
              colorscale: paletteSel.value,
              showscale: true,
              colorbar: { title: { text: niceName(colorMode) } },
              size: 5, opacity: 0.7,
            },
          };
          appendPlot(plotArea, [trace], {
            title:  { text: `${niceName(x)} vs ${niceName(y)}` +
                            (sampled ? '<br><sup>sampled to 5,000 points</sup>' : '') },
            xaxis:  axisCfg(x, niceName(x)),
            yaxis:  axisCfg(y, niceName(y)),
            showlegend: false,
          });
          setConfig({ plotType: 'scatter', x, y, catCol: null, numCol: null,
            colorBy: colorMode, continuous: true,
            palette: paletteSel.value,
            noneColor: noneColor.value,
            id: idFor(x, y, 'scatter'),
          });
          return;
        }

        const fullWh = andWhere(ctx.where, `${q(x)} IS NOT NULL AND ${q(y)} IS NOT NULL`);
        const rows = await ctx.queryRows(`
          SELECT * FROM (
            SELECT ${selectExpr(q, x, 'x')}, ${selectExpr(q, y, 'y')}, ${groupSqlExpr(colorMode)}
            FROM pp_data ${fullWh}
          ) USING SAMPLE ${MAX_SCATTER} ROWS (reservoir, 42)
        `);
        const { groups, colorFn, labelFn } = buildGroupConfig(rows, colorMode);
        const sampled = rows.length >= MAX_SCATTER;
        const traces  = groups.map(g => {
          const gr = rows.filter(r => String(r.__group__) === g);
          return {
            type: 'scatter', mode: 'markers', name: labelFn(g),
            x: gr.map(r => valueOf(x, r.x)), y: gr.map(r => valueOf(y, r.y)),
            marker: { color: colorFn(g), size: 5, opacity: 0.7 },
          };
        }).filter(t => t.x.length);
        appendPlot(plotArea, traces, {
          title:  { text: `${niceName(x)} vs ${niceName(y)}` +
                          (sampled ? '<br><sup>sampled to 5,000 points</sup>' : '') },
          xaxis:  axisCfg(x, niceName(x)),
          yaxis:  axisCfg(y, niceName(y)),
          ...legend(groups),
        });
        setConfig({ plotType: 'scatter', x, y, catCol: null, numCol: null,
          colorBy: colorMode === 'global' ? null : colorMode, continuous: false,
          palette: paletteSel.value,
          noneColor: noneColor.value,
          id: idFor(x, y, 'scatter'),
        });
      }

      async function plotViolin(catCol, numCol, colorMode) {
        const fullWh = andWhere(ctx.where, `${q(numCol)} IS NOT NULL`);
        const rows = await ctx.queryRows(`
          SELECT * FROM (
            SELECT ${catExpr(catCol)} AS cat, ${selectExpr(q, numCol, 'val')}, ${groupSqlExpr(colorMode)}
            FROM pp_data ${fullWh}
          ) USING SAMPLE ${MAX_SCATTER} ROWS (reservoir, 42)
        `);
        const { groups, colorFn, labelFn } = buildGroupConfig(rows, colorMode);
        const cats = sortCats([...new Set(rows.map(r => String(r.cat)))]);
        const traces = groups.map(g => {
          const gr = rows.filter(r => String(r.__group__) === g);
          return {
            type: 'violin', name: labelFn(g),
            x: gr.map(r => String(r.cat)), y: gr.map(r => valueOf(numCol, r.val)),
            box: { visible: true }, meanline: { visible: true }, points: 'outliers',
            spanmode: 'hard', marker: { color: colorFn(g) },
          };
        }).filter(t => t.y.length);
        appendPlot(plotArea, traces, {
          title:      { text: `${niceName(numCol)} by ${niceName(catCol)}` },
          xaxis:      { title: niceName(catCol), type: 'category', categoryarray: cats },
          yaxis:      axisCfg(numCol, niceName(numCol)),
          violinmode: groups.length > 1 ? 'group' : undefined,
          ...legend(groups),
        });
        setConfig({ plotType: 'violin', x: catCol, y: numCol, catCol, numCol,
          colorBy: colorMode === 'global' ? null : colorMode,
          palette: paletteSel.value,
          noneColor: noneColor.value,
          id: idFor(catCol, numCol, 'violin'),
        });
      }

      async function plotBar(catCol, numCol, colorMode) {
        const fullWh = andWhere(ctx.where, `${q(numCol)} IS NOT NULL`);
        const rows = await ctx.queryRows(`
          SELECT ${catExpr(catCol)} AS cat, ${groupSqlExpr(colorMode)},
                 AVG(${q(numCol)}) AS mean_val, STDDEV(${q(numCol)}) AS std_val
          FROM pp_data ${fullWh}
          GROUP BY 1, 2 ORDER BY 1
        `);
        const { groups, colorFn, labelFn } = buildGroupConfig(rows, colorMode);
        const cats = sortCats([...new Set(rows.map(r => String(r.cat)))]);
        const traces = groups.map(g => {
          const gr = rows.filter(r => String(r.__group__) === g);
          return {
            type: 'bar', name: labelFn(g),
            x: gr.map(r => String(r.cat)), y: gr.map(r => Number(r.mean_val)),
            error_y: { type: 'data', visible: true, array: gr.map(r => Number(r.std_val)) },
            marker: { color: colorFn(g) },
          };
        }).filter(t => t.x.length);
        appendPlot(plotArea, traces, {
          title:   { text: `Mean ${niceName(numCol)} by ${niceName(catCol)}` },
          xaxis:   { title: niceName(catCol), type: 'category', categoryarray: cats },
          yaxis:   { title: `Mean ${niceName(numCol)}` },
          barmode: groups.length > 1 ? 'group' : undefined,
          ...legend(groups),
        });
        setConfig({ plotType: 'bar', x: catCol, y: numCol, catCol, numCol,
          colorBy: colorMode === 'global' ? null : colorMode,
          palette: paletteSel.value,
          noneColor: noneColor.value,
          id: idFor(catCol, numCol, 'bar'),
        });
      }

      async function plotCountBar(x, colorMode) {
        const rows = await ctx.queryRows(`
          SELECT ${catExpr(x)} AS cat, ${groupSqlExpr(colorMode)}, COUNT(*) AS n
          FROM pp_data ${ctx.where}
          GROUP BY 1, 2 ORDER BY 1
        `);
        const { groups, colorFn, labelFn } = buildGroupConfig(rows, colorMode);
        const cats = sortCats([...new Set(rows.map(r => String(r.cat)))]);
        const traces = groups.map(g => {
          const gr = rows.filter(r => String(r.__group__) === g);
          return {
            type: 'bar', name: labelFn(g),
            x: gr.map(r => String(r.cat)), y: gr.map(r => Number(r.n)),
            marker: { color: colorFn(g) },
          };
        }).filter(t => t.x.length);
        appendPlot(plotArea, traces, {
          title:   { text: `Count by ${niceName(x)}` },
          xaxis:   { title: niceName(x), type: 'category', categoryarray: cats },
          yaxis:   { title: 'Count' },
          barmode: 'stack',
          ...legend(groups),
        });
        setConfig({ plotType: 'countBar', x, y: COUNT_Y, catCol: null, numCol: null,
          colorBy: colorMode === 'global' ? null : colorMode,
          palette: paletteSel.value,
          noneColor: noneColor.value,
          id: idFor(x, 'count', 'countBar'),
        });
      }

      async function plotCountTable(x, colorMode) {
        const rows = await ctx.queryRows(`
          SELECT ${catExpr(x)} AS cat, ${groupSqlExpr(colorMode)}, COUNT(*) AS n
          FROM pp_data ${ctx.where}
          GROUP BY 1, 2 ORDER BY 1
        `);
        const { labelFn, header } = buildGroupConfig(rows, colorMode);
        const headers = header ? [niceName(x), header, 'Count'] : [niceName(x), 'Count'];
        plotArea.appendChild(statTable(
          headers,
          rows.map(r => header
            ? [String(r.cat), labelFn(String(r.__group__)), Number(r.n).toLocaleString()]
            : [String(r.cat), Number(r.n).toLocaleString()])
        ));
      }

      async function plotCatNumTable(catCol, numCol, colorMode) {
        const isDate = DATE_COLS.has(numCol);
        const rows = await ctx.queryRows(`
          SELECT ${catExpr(catCol)} AS cat, ${groupSqlExpr(colorMode)},
                 COUNT(${q(numCol)}) AS n, COUNT(*) - COUNT(${q(numCol)}) AS n_null, ${aggExprs(q, numCol)}
          FROM pp_data ${ctx.where}
          GROUP BY 1, 2 ORDER BY 1
        `);
        const { labelFn, header } = buildGroupConfig(rows, colorMode);
        const hasNulls = rows.some(r => Number(r.n_null) > 0);
        const fmt = v => fmtStat(numCol, v);
        const cols = isDate
          ? [niceName(catCol), 'n', `Mean ${niceName(numCol)}`, 'Min', 'Max']
          : [niceName(catCol), 'n', `Mean ${niceName(numCol)}`, 'SD', 'Min', 'Max'];
        if (hasNulls) cols.push(`Null ${niceName(numCol)}`);
        const headers = header ? [cols[0], header, ...cols.slice(1)] : cols;
        plotArea.appendChild(statTable(
          headers,
          rows.map(r => {
            const vals = isDate
              ? [Number(r.n).toLocaleString(), fmt(r.mean_val), fmt(r.min_val), fmt(r.max_val)]
              : [Number(r.n).toLocaleString(), fmt(r.mean_val), fmt(r.std_val), fmt(r.min_val), fmt(r.max_val)];
            if (hasNulls) vals.push(Number(r.n_null).toLocaleString());
            const row = [String(r.cat), ...vals];
            return header ? [row[0], labelFn(String(r.__group__)), ...row.slice(1)] : row;
          })
        ));
      }

      async function plotHeatmap(x, y) {
        const rows = await ctx.queryRows(`
          SELECT ${catExpr(x)} AS x, ${catExpr(y)} AS y, COUNT(*) AS n
          FROM pp_data ${ctx.where}
          GROUP BY 1, 2 ORDER BY 1, 2
        `);
        const xs = sortCats([...new Set(rows.map(r => String(r.x)))]);
        const ys = sortCats([...new Set(rows.map(r => String(r.y)))]);

        if (xs.length <= 1 || ys.length <= 1) {
          plotArea.appendChild(statTable(
            [niceName(x), niceName(y), 'Count'],
            rows.map(r => [String(r.x), String(r.y), Number(r.n).toLocaleString()])
          ));
          return;
        }

        const z = ys.map(yv => xs.map(xv => {
          const r = rows.find(r => String(r.x) === xv && String(r.y) === yv);
          return r ? Number(r.n) : 0;
        }));
        const colorscale = [[0, '#ffffff'], [1, heatColor.value]];
        appendPlot(plotArea, [{
          type: 'heatmap', x: xs, y: ys, z, colorscale, reversescale: heatInvert.checked, showscale: true,
        }], {
          title:      { text: `Count: ${niceName(x)} × ${niceName(y)}` },
          xaxis:      { title: niceName(x), type: 'category' },
          yaxis:      { title: niceName(y), type: 'category' },
          showlegend: false,
        });
        setConfig({ plotType: 'heatmap', x, y, catCol: null, numCol: null,
          colorBy: null, palette: paletteSel.value,
          heatColor: heatColor.value, heatInvert: heatInvert.checked,
          id: idFor(x, y, 'heatmap'),
        });
      }

      // ── Main entry ────────────────────────────────────────────────────────
      async function doPlot() {
        const x = xBox.get(), y = yBox.get();
        plotArea.innerHTML = '';
        activeConfig = null;
        exportBtn.style.display = 'none';

        if (!x || !y) {
          plotArea.innerHTML = '<div class="no-data">Select X and Y columns above.</div>';
          return;
        }

        // Palette/swatch are only shown once we know how color-by resolved.
        paletteSel.style.display = 'none';
        noneColor.style.display  = 'none';
        // Heatmaps don't support color-by grouping - show the colorscale
        // controls instead. Default to the color-by group; the heatmap
        // branch below switches this.
        colorGroup.style.display = 'flex';
        heatGroup.style.display  = 'none';

        function applyColorMode({ mode, note, continuous }) {
          paletteSel.style.display = (mode !== 'none' && mode !== 'global') ? '' : 'none';
          noneColor.style.display  = mode === 'none' ? '' : 'none';
          setPaletteKind(continuous ? 'continuous' : 'qualitative');
          showColorNote(note);
          return mode;
        }

        plotArea.innerHTML = '<div class="no-data">Loading…</div>';
        try {
          if (y === COUNT_Y) {
            typeSel.style.display = 'none';
            const n = await categoryCount(x);
            if (n > MAX_CAT) {
              plotArea.innerHTML = `<div class="no-data">"${niceName(x)}" has ${n} unique values — too many for count bar (max ${MAX_CAT}).</div>`;
              return;
            }
            const cm = await resolveColorMode([x]);
            plotArea.innerHTML = '';
            applyColorMode(cm);
            if (n === 1) await plotCountTable(x, cm.mode);
            else         await plotCountBar(x, cm.mode);
            return;
          }

          if (x === y) {
            plotArea.innerHTML = '<div class="no-data">X and Y must be different columns.</div>';
            return;
          }

          const xNum = numericSet.has(x);
          const yNum = numericSet.has(y);

          if (xNum && yNum) {
            typeSel.style.display = 'none';

            const [{ n: yUniq }] = await cardinalityCheck(y);
            if (yUniq <= 1) {
              colorGroup.style.display = 'none';
              // One row per distinct Y value (NULL Y becomes its own "(missing)"
              // row), with the range of X and how many X values are null in it.
              const rows = await ctx.queryRows(`
                SELECT ${aggExpr(q, y, 'MIN', 'yval')},
                       COUNT(${q(x)}) AS n, COUNT(*) - COUNT(${q(x)}) AS n_null,
                       ${aggExpr(q, x, 'MIN', 'xmin')}, ${aggExpr(q, x, 'MAX', 'xmax')}
                FROM pp_data ${ctx.where}
                GROUP BY ${q(y)}
                ORDER BY (${q(y)} IS NULL), 1
              `);
              const hasNullX = rows.some(r => Number(r.n_null) > 0);
              const headers  = [niceName(y), 'n', `${niceName(x)} range`];
              if (hasNullX) headers.push(`Null ${niceName(x)}`);
              plotArea.innerHTML = '';
              plotArea.appendChild(statTable(headers, rows.map(r => {
                const yLabel  = r.yval == null ? NULL_LABEL : fmtStat(y, r.yval);
                const xMinTxt = fmtStat(x, r.xmin), xMaxTxt = fmtStat(x, r.xmax);
                const xRange  = xMinTxt === xMaxTxt ? xMinTxt : `${xMinTxt} – ${xMaxTxt}`;
                const row = [yLabel, Number(r.n).toLocaleString(), xRange];
                if (hasNullX) row.push(Number(r.n_null).toLocaleString());
                return row;
              })));
              return;
            }
            const cm = await resolveColorMode([x, y], { scatter: true });
            plotArea.innerHTML    = '';
            applyColorMode(cm);
            // Numeric axes can't place NULLs - warn how many points are
            // excluded because X and/or Y is null.
            showColorNote(await nullWarningNote(x, y));
            await plotScatter(x, y, cm.mode, cm.continuous);

          } else if (!xNum && !yNum) {
            typeSel.style.display    = 'none';
            colorGroup.style.display = 'none';
            heatGroup.style.display  = 'flex';
            const xCats = await categoryCount(x);
            const yCats = await categoryCount(y);
            if (xCats > MAX_CAT || yCats > MAX_CAT) {
              const bad  = xCats > MAX_CAT ? x : y;
              const badN = xCats > MAX_CAT ? xCats : yCats;
              plotArea.innerHTML = `<div class="no-data">"${niceName(bad)}" has ${badN} unique values — too many for heatmap (max ${MAX_CAT}).</div>`;
              return;
            }
            plotArea.innerHTML = '';
            await plotHeatmap(x, y);

          } else {
            const catCol  = xNum ? y : x;
            const numCol  = xNum ? x : y;
            const flipped = xNum; // user put the numeric col on X; we swap internally
            const isDate  = DATE_COLS.has(numCol);
            const n = await categoryCount(catCol);
            if (n > MAX_CAT) {
              typeSel.style.display = 'none';
              plotArea.innerHTML    = `<div class="no-data">"${niceName(catCol)}" has ${n} unique values — too many for a categorical axis (max ${MAX_CAT}).</div>`;
              return;
            }
            // A single category, or a numeric axis with at most one distinct
            // (non-null) value, can't usefully be plotted - show a table instead.
            const [{ n: numUniq }] = await cardinalityCheck(numCol);
            const showTable = n === 1 || numUniq <= 1;
            // Bar (mean ± sd) doesn't make sense for a date-valued axis - violin only.
            typeSel.style.display = (showTable || isDate) ? 'none' : '';
            const cm = await resolveColorMode([catCol, numCol]);
            plotArea.innerHTML    = '';
            applyColorMode(cm);
            if (flipped && !showTable) {
              showColorNote(`X and Y were swapped: "${x}" is numeric so it becomes the Y axis, "${y}" is categorical on X.`);
            }
            if (showTable) {
              await plotCatNumTable(catCol, numCol, cm.mode);
            } else {
              showColorNote(await nullWarningNote(numCol));
              if (typeSel.value === 'bar' && !isDate) await plotBar(catCol, numCol, cm.mode);
              else                                     await plotViolin(catCol, numCol, cm.mode);
            }
          }
        } catch (e) {
          plotArea.innerHTML = '<div class="no-data">Failed to render plot.</div>';
          console.error('[custom-plot]', e);
        }
      }

      xBox.onSelect(doPlot);
      yBox.onSelect(doPlot);
      typeSel.addEventListener('change', doPlot);
      colorBySel.onSelect(doPlot);
      paletteSel.addEventListener('change', () => {
        if (paletteKind === 'continuous') lastContPalette = paletteSel.value;
        else                               lastQualPalette = paletteSel.value;
        doPlot();
      });
      noneColor.addEventListener('input', doPlot);
      heatColor.addEventListener('input', doPlot);
      heatInvert.addEventListener('change', doPlot);
      clearBtn.addEventListener('click', () => {
        xBox.set(null);
        yBox.set(null);
        colorBySel.set(GLOBAL_COLOR);
        paletteSel.style.display = 'none';
        noneColor.style.display  = 'none';
        typeSel.style.display    = 'none';
        colorGroup.style.display = 'flex';
        heatGroup.style.display  = 'none';
        exportBtn.style.display  = 'none';
        activeConfig             = null;
        plotArea.innerHTML = '<div class="no-data">Select X and Y columns above.</div>';
      });

      plotArea.innerHTML = '<div class="no-data">Select X and Y columns above.</div>';
    }

    createSlot();
    addBtn.addEventListener('click', createSlot);
  },
};
