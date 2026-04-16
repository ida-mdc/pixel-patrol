import { q, sample, groupCol } from '../sql.js';
import { groupColor } from '../colors.js';
import { LEGEND, appendPlot, niceName } from '../plot-utils.js';

/**
 * Dimension Size Distribution — matches Dash DimSizeWidget:
 *   - X/Y scatter (bubble size = count) to spot resolution mismatches
 *   - Strip plots per *_size column (shown as violin+all-points)
 *   - Data availability ratios per dimension
 */
export default {
  id: 'dim-size',
  label: 'Dimension Size Distribution',
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

  async render(container, ctx) {
    const sizeCols = ctx.schema.allCols.filter(
      c => c.endsWith('_size') && !c.startsWith('__'),
    );
    const gcExpr = groupCol(ctx.state);

    // Total rows for ratio calculation.
    const totalRes = await ctx.queryRows(
      `SELECT COUNT(*) AS n FROM pp_data ${ctx.where}`,
    );
    const totalRows = Number(totalRes[0]?.n ?? 0);

    // ── Availability ratios ───────────────────────────────────────────────
    const ratioLines = await Promise.all(sizeCols.map(async col => {
      const colCond = `${q(col)} > 1`;
      const whereClause = ctx.where ? `${ctx.where} AND ${colCond}` : `WHERE ${colCond}`;
      const res = await ctx.queryRows(
        `SELECT COUNT(*) AS n FROM pp_data ${whereClause}`,
      );
      const present = Number(res[0]?.n ?? 0);
      const label   = niceName(col);
      const pct     = totalRows > 0 ? ((present / totalRows) * 100).toFixed(2) : '0.00';
      return `${label}: ${present.toLocaleString()} of ${totalRows.toLocaleString()} files (${pct}%).`;
    }));

    const ratioDiv = document.createElement('div');
    ratioDiv.style.marginBottom = '16px';
    ratioDiv.innerHTML = `
      <p><strong>Data Availability by Dimension:</strong></p>
      ${ratioLines.map(l => `<p class="mb-1 small">${l}</p>`).join('')}
    `;
    container.appendChild(ratioDiv);

    // ── X/Y plot: bubble scatter for small datasets, log heatmap for large ──
    const MAX_XY_BUBBLE = 50_000;
    const HEATMAP_BINS  = 70;
    const hasXY = sizeCols.includes('X_size') && sizeCols.includes('Y_size');
    if (hasXY) {
      const xyCond  = `"X_size" > 1 AND "Y_size" > 1`;
      const xyWhere = ctx.where ? `${ctx.where} AND ${xyCond}` : `WHERE ${xyCond}`;

      const statsRow = (await ctx.queryRows(`
        SELECT COUNT(*)::BIGINT         AS n,
               MIN("X_size")::DOUBLE   AS min_x, MAX("X_size")::DOUBLE AS max_x,
               MIN("Y_size")::DOUBLE   AS min_y, MAX("Y_size")::DOUBLE AS max_y
        FROM pp_data ${xyWhere}
      `))[0] ?? {};
      const xyFiles = Number(statsRow.n ?? 0);

      if (xyFiles <= MAX_XY_BUBBLE) {
        // ── Bubble scatter (mirrors Dash px.scatter with size='bubble_size') ──
        const xyRows = await ctx.queryRows(`
          SELECT "X_size"  AS x,
                 "Y_size"  AS y,
                 ${gcExpr} AS __group__,
                 COUNT(*)  AS bubble_size
          FROM pp_data ${xyWhere}
          GROUP BY 1, 2, 3
          ORDER BY 1, 2
        `);
        if (xyRows.length) {
          // sizeref = 2 * max / size_max² replicates Plotly Express area scaling
          const maxCount = Math.max(...xyRows.map(r => Number(r.bubble_size)));
          const sizeref  = (2 * maxCount) / (20 ** 2);
          const traces = ctx.groups.map(g => {
            const gRows = xyRows.filter(r => String(r.__group__) === g);
            if (!gRows.length) return null;
            return {
              type:   'scatter',
              mode:   'markers',
              name:   g,
              x:      gRows.map(r => Number(r.x)),
              y:      gRows.map(r => Number(r.y)),
              marker: {
                size:     gRows.map(r => Number(r.bubble_size)),
                sizeref,
                sizemode: 'area',
                color:    groupColor(ctx.colorMap, g),
                line:     { width: 1, color: 'white' },
              },
              hovertemplate:
                `<b>${g}</b><br>X: %{x} px<br>Y: %{y} px<br>Count: %{text}<extra></extra>`,
              text: gRows.map(r => String(r.bubble_size)),
            };
          }).filter(Boolean);
          appendPlot(container, traces, {
            title:      { text: 'Distribution of X and Y Dimension Sizes' },
            xaxis:      { title: 'X size (pixels)' },
            yaxis:      { title: 'Y size (pixels)' },
            showlegend: ctx.groups.length > 1,
            legend:     LEGEND,
          }, 'margin-bottom:24px');
        }
      } else {
        // ── Log-scale heatmap for large datasets ─────────────────────────────
        const spanX = Number(statsRow.max_x) - Number(statsRow.min_x);
        const spanY = Number(statsRow.max_y) - Number(statsRow.min_y);
        const binX  = Math.max(1, Math.ceil(spanX / HEATMAP_BINS));
        const binY  = Math.max(1, Math.ceil(spanY / HEATMAP_BINS));
        const minX  = Number(statsRow.min_x);
        const minY  = Number(statsRow.min_y);

        const binRows = await ctx.queryRows(`
          SELECT (FLOOR(("X_size" - ${minX}) / ${binX}) * ${binX} + ${minX})::INTEGER AS xbin,
                 (FLOOR(("Y_size" - ${minY}) / ${binY}) * ${binY} + ${minY})::INTEGER AS ybin,
                 COUNT(*)::INTEGER AS n
          FROM pp_data ${xyWhere}
          GROUP BY 1, 2
          ORDER BY 1, 2
        `);
        if (binRows.length) {
          const xs = [...new Set(binRows.map(r => Number(r.xbin)))].sort((a, b) => a - b);
          const ys = [...new Set(binRows.map(r => Number(r.ybin)))].sort((a, b) => a - b);
          const xi = new Map(xs.map((v, i) => [v, i]));
          const yi = new Map(ys.map((v, i) => [v, i]));
          // z = log10(count) for filled bins; null for empty cells (rendered as
          // plot background, clearly distinct from the darkest Viridis colour).
          const z = Array.from({ length: ys.length }, () => Array(xs.length).fill(null));
          let zMax = 0;
          for (const r of binRows) {
            const cx = xi.get(Number(r.xbin));
            const cy = yi.get(Number(r.ybin));
            if (cx != null && cy != null) {
              const logVal = Math.log10(Number(r.n));
              z[cy][cx] = logVal;
              if (logVal > zMax) zMax = logVal;
            }
          }

          // Colorbar ticks at every integer power of 10 up to the data max.
          const tickVals = [], tickText = [];
          for (let i = 0; i <= Math.ceil(zMax); i++) {
            tickVals.push(i);
            tickText.push(String(10 ** i));
          }

          appendPlot(container, [{
            type:          'heatmap',
            x:             xs,
            y:             ys,
            z,
            colorscale:    'Viridis',
            zmin:          0,          // anchors colour scale at count = 1
            zmax:          zMax,
            colorbar: {
              title:    { text: 'count', side: 'right' },
              tickvals: tickVals,
              ticktext: tickText,
            },
            hovertemplate: 'X: %{x} px<br>Y: %{y} px<br>count: %{text}<extra></extra>',
            text: ys.map((_, cy) => xs.map((_, cx) => {
              const v = z[cy][cx];
              return v != null ? String(Math.round(10 ** v)) : '';
            })),
          }], {
            title:  { text: 'Distribution of X and Y Dimension Sizes (log scale)' },
            xaxis:  { title: 'X size (pixels)' },
            yaxis:  { title: 'Y size (pixels)' },
          }, 'margin-bottom:24px');
        }
      }
    }

    // ── Per-dimension strip plots (violin + all points) ───────────────────
    const dimsToPlot = sizeCols.filter(c => c !== 'num_pixels');

    if (dimsToPlot.length) {
      const sampleSQL = `
        SELECT ${gcExpr} AS __group__,
               ${dimsToPlot.map(q).join(', ')}
        FROM pp_data ${ctx.where}
        ${sample(5000)}
      `;
      const dimRows = await ctx.queryRows(sampleSQL);

      const titleDiv = document.createElement('h6');
      titleDiv.style.cssText = 'margin-top:8px;margin-bottom:12px';
      titleDiv.textContent = 'Individual Dimension Sizes per Dataset';
      container.appendChild(titleDiv);

      const wrap = document.createElement('div');
      wrap.style.cssText = 'display:flex;flex-wrap:wrap;gap:12px';
      container.appendChild(wrap);

      for (const col of dimsToPlot) {
        const colLabel = niceName(col);
        const validRows = dimRows.filter(r => r[col] != null && Number(r[col]) > 1);
        if (!validRows.length) continue;

        const traces = ctx.groups.map(g => {
          const vals = validRows
            .filter(r => String(r.__group__) === g)
            .map(r => Number(r[col]));
          if (!vals.length) return null;
          return {
            type:     'violin',
            y:        vals,
            name:     g,
            box:      { visible: true },
            meanline: { visible: false },
            points:   vals.length < 500 ? 'all' : 'outliers',
            pointpos: 0,
            jitter:   0.3,
            marker:   { color: groupColor(ctx.colorMap, g), size: 3 },
            line:     { color: groupColor(ctx.colorMap, g) },
            showlegend: false,
          };
        }).filter(Boolean);

        if (!traces.length) continue;

        appendPlot(wrap, traces, {
          title:  { text: colLabel, font: { size: 12 } },
          yaxis:  { title: 'pixels' },
          xaxis:  { title: '' },
          height: 280,
          margin: { l: 44, r: 10, t: 36, b: 40 },
        }, 'flex:0 0 280px;min-width:220px;margin-bottom:16px');
      }
    }
  },
};

