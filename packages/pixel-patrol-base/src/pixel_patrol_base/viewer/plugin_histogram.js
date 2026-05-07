const SAMPLE_N         = 2000;
const MAX_FILE_OPTIONS = 500;
const MODE_ID          = 'hist-mode-radio';
const GROUP_SEL_ID     = 'hist-group-select';
const FILE_SEL_ID      = 'hist-file-select';

export default {
  id: 'histogram',
  group: 'Dataset Stats',
  label: 'Pixel Value Histograms',
  info: [
    'Histograms are computed **per image** and grouped based on your groupings.',
    'They are normalized to sum to **1**, and the **mean histogram per group** is shown as a bold line.',
    '',
    '**Modes**',
    '- **Fixed 0–255 bins (shape)** — uses 256 fixed bins regardless of the actual pixel range.',
    '- **Native pixel range** — bins are defined using the actual min/max pixel values across the selected images.',
  ].join('\n'),

  requires(schema) {
    return schema.blobCols.includes('histogram_counts');
  },

  async render(container, ctx) {
    try {
      const { escapeHtml } = ctx.plot;
      const hasRange = ctx.schema.allCols.includes('histogram_min') && ctx.schema.allCols.includes('histogram_max');
      const hasNames = ctx.schema.allCols.includes('name');
  
      const controlsDiv = document.createElement('div');
      controlsDiv.style.cssText = 'display:flex;flex-wrap:wrap;gap:20px;margin-bottom:20px';
  
      controlsDiv.innerHTML += `
        <div>
          <div style="font-weight:600;margin-bottom:6px">Histogram plot mode:</div>
          <label style="margin-right:16px;cursor:pointer">
            <input type="radio" name="${MODE_ID}" value="shape" checked> Fixed 0–255 bins (Shape)
          </label>
          <label style="cursor:pointer">
            <input type="radio" name="${MODE_ID}" value="native"> Native pixel range (Absolute)
          </label>
        </div>
      `;
  
      const groupOpts = ctx.groups.map(g => `<option value="${g}">${ctx.groupLabel(g)}</option>`).join('');
      controlsDiv.innerHTML += `
        <div style="max-width:360px;flex:1 1 240px">
          <div style="font-weight:600;margin-bottom:6px">Select specific groups to compare (optional):</div>
          <select id="${GROUP_SEL_ID}" class="form-select form-select-sm" multiple style="height:80px">${groupOpts}</select>
          <small class="text-muted">Hold Ctrl/Cmd to multi-select. Empty = all groups.</small>
        </div>
      `;
  
      if (hasNames) {
        const nameRows = await ctx.queryRows(`SELECT DISTINCT "name" FROM pp_data ${ctx.where} ORDER BY 1 LIMIT ${MAX_FILE_OPTIONS}`);
        const nameOpts = nameRows.map(r => `<option value="${escapeHtml(String(r.name))}">${escapeHtml(String(r.name))}</option>`).join('');
        const limitNote = nameRows.length === MAX_FILE_OPTIONS ? `<small class="text-muted">Showing first ${MAX_FILE_OPTIONS} files.</small>` : '';
        controlsDiv.innerHTML += `
          <div style="max-width:400px;flex:1 1 240px">
            <div style="font-weight:600;margin-bottom:6px">Overlay specific file (optional):</div>
            <select id="${FILE_SEL_ID}" class="form-select form-select-sm">
              <option value="">— none —</option>${nameOpts}
            </select>${limitNote}
          </div>
        `;
      }
  
      container.appendChild(controlsDiv);
  
      const plotDiv = document.createElement('div');
      container.appendChild(plotDiv);
  
      const render = async () => {
        const mode           = container.querySelector(`input[name="${MODE_ID}"]:checked`)?.value ?? 'shape';
        const groupSelEl     = document.getElementById(GROUP_SEL_ID);
        const fileSelEl      = document.getElementById(FILE_SEL_ID);
        const selectedGroups = groupSelEl ? [...groupSelEl.selectedOptions].map(o => o.value).filter(Boolean) : [];
        const selectedFile   = fileSelEl?.value ?? '';
        plotDiv.innerHTML = '';
        await renderHistogram(plotDiv, ctx, { mode, selectedGroups, selectedFile, hasRange, hasNames });
      };
  
      container.querySelectorAll(`input[name="${MODE_ID}"]`).forEach(el => el.addEventListener('change', render));
      const groupSelEl = document.getElementById(GROUP_SEL_ID);
      if (groupSelEl) groupSelEl.addEventListener('change', render);
      const fileSelEl = document.getElementById(FILE_SEL_ID);
      if (fileSelEl) fileSelEl.addEventListener('change', render);
  
      await render();
    
    } catch {
      container.innerHTML = '<div class="no-data">Failed to load data.</div>';
    }
  },
};

async function renderHistogram(container, ctx, { mode, selectedGroups, selectedFile, hasRange }) {
  const { q, sample, groupExpr: geFn } = ctx.sql;
  const { append: appendPlot, plotlyLegendConfig } = ctx.plot;
  const gcExpr   = geFn();
  const rangeSel = hasRange ? ', "histogram_min", "histogram_max"' : '';

  let extraWhere = ctx.where;
  if (selectedGroups.length && ctx.state.groupCol) {
    const list      = selectedGroups.map(g => `'${g.replace(/'/g, "''")}'`).join(', ');
    const connector = extraWhere ? 'AND' : 'WHERE';
    extraWhere += ` ${connector} ${q(ctx.state.groupCol)} IN (${list})`;
  }

  const result    = await ctx.query(`SELECT ${gcExpr}, "histogram_counts"${rangeSel} FROM pp_data ${extraWhere} ${sample(SAMPLE_N)}`);
  const arrowRows = result.toArray();

  if (!arrowRows.length) {
    container.innerHTML = '<div class="no-data">No rows match the current filter.</div>';
    return;
  }

  const groupData = {};
  for (const row of arrowRows) {
    const group  = String(row.__group__);
    const counts = extractBinary(row.histogram_counts);
    if (!counts?.length) continue;
    if (!groupData[group]) groupData[group] = { sums: new Float64Array(counts.length), count: 0, min: hasRange ? Number(row.histogram_min) : 0, max: hasRange ? Number(row.histogram_max) : 255 };
    const gd = groupData[group];
    for (let i = 0; i < counts.length; i++) gd.sums[i] += counts[i];
    gd.count++;
    if (hasRange) { gd.min = Math.min(gd.min, Number(row.histogram_min)); gd.max = Math.max(gd.max, Number(row.histogram_max)); }
  }

  const visibleGroups = (selectedGroups.length ? selectedGroups : ctx.groups).filter(g => groupData[g]);
  const traces = visibleGroups.map(g => {
    const { sums, count, min, max } = groupData[g];
    const total = sums.reduce((s, v) => s + v, 0);
    const nBins = sums.length;
    const ys    = Array.from(sums, v => total > 0 ? v / total : 0);
    const xs    = histXAxis(mode, nBins, min, max);
    return { type: 'scatter', mode: 'lines', name: ctx.groupLabel(String(g)), x: xs, y: ys, fill: 'tozeroy', opacity: 0.6, line: { color: ctx.color.group(g), width: 2 } };
  });

  if (selectedFile && hasRange) {
    const fileResult = await ctx.query(`SELECT "histogram_counts", "histogram_min", "histogram_max" FROM pp_data WHERE "name" = '${selectedFile.replace(/'/g, "''")}' LIMIT 1`);
    const fileRows   = fileResult.toArray();
    if (fileRows.length) {
      const row    = fileRows[0];
      const counts = extractBinary(row.histogram_counts);
      if (counts?.length) {
        const total = counts.reduce((s, v) => s + v, 0);
        const minV  = Number(row.histogram_min), maxV = Number(row.histogram_max);
        const nBins = counts.length;
        const ys    = Array.from(counts, v => total > 0 ? v / total : 0);
        const xs    = histXAxis(mode, nBins, minV, maxV);
        const width = xs[1] - xs[0];
        traces.unshift({ type: 'bar', name: `File: ${selectedFile.split('/').pop()}`, x: xs, y: ys, width: Array(nBins).fill(width), marker: { color: 'black' }, opacity: 0.3 });
      }
    }
  }

  const showLegend = visibleGroups.length > 1 || !!selectedFile;
  appendPlot(container, traces, {
    title: { text: 'Intensity Histograms (averaged per group)' },
    xaxis: { title: mode === 'native' ? 'Pixel value' : 'Intensity (0–255)' },
    yaxis: { title: 'Normalized Count' },
    bargap: 0, height: 500, showlegend: showLegend,
    ...(showLegend ? { legend: plotlyLegendConfig } : {}),
  });
}

function histXAxis(mode, nBins, min, max) {
  return mode === 'native'
    ? Array.from({ length: nBins }, (_, i) => min + (i / nBins) * (max - min))
    : Array.from({ length: nBins }, (_, i) => (i / nBins) * 255);
}

function extractBinary(val) {
  if (!val) return null;
  if (val instanceof Uint8Array)    return val;
  if (val instanceof Int32Array)    return val;
  if (val instanceof Float32Array)  return val;
  if (val instanceof Float64Array)  return val;
  if (val instanceof BigInt64Array) return Array.from(val, v => Number(v));
  if (val instanceof BigUint64Array) return Array.from(val, v => Number(v));
  if (Array.isArray(val)) return val;
  if (typeof val.toArray === 'function') {
    const arr = val.toArray();
    if (arr instanceof BigInt64Array || arr instanceof BigUint64Array) return Array.from(arr, v => Number(v));
    return arr;
  }
  if (val.values) return val.values;
  return null;
}
