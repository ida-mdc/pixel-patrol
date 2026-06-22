const DEFAULT_SAMPLES_PER_GROUP = 2000;
const MAX_FILE_OPTIONS          = 500;
const MODE_ID                   = 'hist-mode-radio';
const GROUP_SEL_ID              = 'hist-group-select';
const FILE_SEL_ID               = 'hist-file-select';
const SAMPLE_INPUT_ID           = 'hist-sample-input';

export default {
  id: 'histogram',
  group: 'Dataset Stats',
  scope: 'image',
  label: 'Pixel Value Histograms',
  shortLabel: 'Histograms',
  info: [
    'Histograms are computed **per image** and grouped based on your groupings.',
    'They are normalized to sum to **1**, and the **mean histogram per group** is shown as a bold line.',
    '',
    '**Modes**',
    '- **Fixed 0–255 bins (shape)** - uses 256 fixed bins regardless of the actual pixel range.',
    '- **Native pixel range** - bins are defined using the actual min/max pixel values across the selected images.',
  ].join('\n'),

  requires(schema) {
    return schema.blobCols.includes('histogram_counts');
  },

  async condensedSummary(ctx) {
    try {
      const [{ total, n }] = await ctx.queryRows(
        `SELECT COUNT(*) AS total, COUNT("histogram_counts") AS n FROM pp_data ${ctx.where}`,
      );
      const totalN   = Number(total ?? 0);
      const withHist = Number(n ?? 0);
      if (withHist < totalN) {
        return { text: `Only <strong>${withHist.toLocaleString()}</strong>/${totalN.toLocaleString()} have histograms.`, warning: true };
      }
      // The plot samples for larger collections, so don't imply a fixed file count -
      // just tease what the widget shows.
      return `Compare pixel intensity distributions across groups.`;
    } catch { return null; }
  },

  async condensedPlot(container, ctx) {
    const { groupExpr: geFn } = ctx.sql;
    const { extractBinary } = ctx.data;
    const result = await ctx.query(`
      SELECT ${geFn()}, "histogram_counts"
      FROM pp_data ${ctx.where}
      QUALIFY ROW_NUMBER() OVER (PARTITION BY __group__ ORDER BY random()) <= 300
    `);
    const arrowRows = result.toArray();
    if (!arrowRows.length) return false;

    const groupData = {};
    for (const row of arrowRows) {
      const g = String(row.__group__);
      const counts = extractBinary(row.histogram_counts);
      if (!counts?.length) continue;
      const total = histTotal(counts);
      if (total <= 0) continue;
      if (!groupData[g]) groupData[g] = { sums: new Float64Array(counts.length), count: 0 };
      const gd = groupData[g];
      for (let i = 0; i < counts.length; i++) gd.sums[i] += counts[i] / total;
      gd.count++;
    }
    const groups = ctx.groups.filter(g => groupData[g]?.count);
    if (!groups.length) return false;

    const traces = groups.map(g => {
      const { sums, count } = groupData[g];
      const n = sums.length;
      return {
        type: 'scatter', mode: 'lines', fill: 'tozeroy', opacity: 0.6,
        x: Array.from({ length: n }, (_, i) => (i / n) * 255),
        // Mean of the per-image (area-1) histograms, so every image counts equally.
        y: Array.from(sums, v => v / count),
        line: { color: ctx.color.group(g), width: 1.5 }, hoverinfo: 'skip',
      };
    });
    ctx.plot.appendMini(container, traces, {
      xaxis: { title: 'intensity' }, yaxis: { showticklabels: false },
    });
    return true;
  },

  async render(container, ctx) {
    try {
      const hasRange = ctx.schema.allCols.includes('histogram_min') && ctx.schema.allCols.includes('histogram_max');
      const hasNames = ctx.schema.allCols.includes('name');

      const [availRow] = await ctx.queryRows(
        `SELECT COUNT(*) AS total, COUNT("histogram_counts") AS n FROM pp_data ${ctx.where}`
      );
      ctx.plot.dataAvailabilityWarning(container,
        [{ label: 'Pixel Value Histograms', present: Number(availRow.n) }], Number(availRow.total), { unit: 'images' });

      const nameRows = hasNames
        ? await ctx.queryRows(`SELECT DISTINCT "name" FROM pp_data ${ctx.where} ORDER BY 1 LIMIT ${MAX_FILE_OPTIONS}`)
        : [];
      container.appendChild(buildControls(ctx, { hasNames, nameRows }));

      const plotDiv = document.createElement('div');
      container.appendChild(plotDiv);

      const draw = async () => {
        plotDiv.innerHTML = '';
        await renderHistogram(plotDiv, ctx, { ...readControls(container), hasRange, hasNames });
      };
      wireControls(container, draw);
      await draw();
    } catch {
      container.innerHTML = '<div class="no-data">Failed to load data.</div>';
    }
  },
};

// All histogram controls (plot mode, group multiselect, optional file overlay,
// max-samples cap) as one flex row.
function buildControls(ctx, { hasNames, nameRows }) {
  const { escapeHtml } = ctx.plot;
  const groupOpts = ctx.groups.map(g => `<option value="${g}">${ctx.groupLabel(g)}</option>`).join('');

  let fileBlock = '';
  if (hasNames) {
    const nameOpts  = nameRows.map(r => `<option value="${escapeHtml(String(r.name))}">${escapeHtml(String(r.name))}</option>`).join('');
    const limitNote = nameRows.length === MAX_FILE_OPTIONS ? `<small class="text-muted">Showing first ${MAX_FILE_OPTIONS} files.</small>` : '';
    fileBlock = `
      <div style="max-width:400px;flex:1 1 240px">
        <div style="font-weight:600;margin-bottom:6px">Overlay specific file (optional):</div>
        <select id="${FILE_SEL_ID}" class="form-select form-select-sm">
          <option value="">- none -</option>${nameOpts}
        </select>${limitNote}
      </div>`;
  }

  const controls = document.createElement('div');
  controls.style.cssText = 'display:flex;flex-wrap:wrap;gap:20px;margin-bottom:20px';
  controls.innerHTML = `
    <div>
      <div style="font-weight:600;margin-bottom:6px">Histogram plot mode:</div>
      <label style="margin-right:16px;cursor:pointer"><input type="radio" name="${MODE_ID}" value="shape" checked> Fixed 0–255 bins (Shape)</label>
      <label style="cursor:pointer"><input type="radio" name="${MODE_ID}" value="native"> Native pixel range (Absolute)</label>
    </div>
    <div style="max-width:360px;flex:1 1 240px">
      <div style="font-weight:600;margin-bottom:6px">Select specific groups to compare (optional):</div>
      <select id="${GROUP_SEL_ID}" class="form-select form-select-sm" multiple style="height:80px">${groupOpts}</select>
      <small class="text-muted">Hold Ctrl/Cmd to multi-select. Empty = all groups.</small>
    </div>
    ${fileBlock}
    <div style="align-self:flex-end">
      <label style="font-weight:600">Max samples per group:
        <input type="number" id="${SAMPLE_INPUT_ID}" value="${DEFAULT_SAMPLES_PER_GROUP}" min="50" step="100"
               style="width:90px;margin-left:8px;padding:2px 6px;border:1px solid #ccc;border-radius:4px"></label>
    </div>
  `;
  return controls;
}

// Current selections from the controls.
function readControls(container) {
  const groupSel = container.querySelector(`#${GROUP_SEL_ID}`);
  const sampleRaw = container.querySelector(`#${SAMPLE_INPUT_ID}`)?.value ?? DEFAULT_SAMPLES_PER_GROUP;
  return {
    mode:            container.querySelector(`input[name="${MODE_ID}"]:checked`)?.value ?? 'shape',
    selectedGroups:  groupSel ? [...groupSel.selectedOptions].map(o => o.value).filter(Boolean) : [],
    selectedFile:    container.querySelector(`#${FILE_SEL_ID}`)?.value ?? '',
    samplesPerGroup: Math.max(50, parseInt(sampleRaw, 10) || DEFAULT_SAMPLES_PER_GROUP),
  };
}

// Re-draw whenever any control changes.
function wireControls(container, onChange) {
  container.querySelectorAll(`input[name="${MODE_ID}"]`).forEach(el => el.addEventListener('change', onChange));
  for (const id of [GROUP_SEL_ID, FILE_SEL_ID, SAMPLE_INPUT_ID]) {
    container.querySelector(`#${id}`)?.addEventListener('change', onChange);
  }
}

async function renderHistogram(container, ctx, { mode, selectedGroups, selectedFile, hasRange, samplesPerGroup }) {
  const { q, groupExpr: geFn } = ctx.sql;
  const { append: appendPlot, plotlyLegendConfig } = ctx.plot;
  const { extractBinary } = ctx.data;
  const gcExpr   = geFn();
  const rangeSel = hasRange ? ', "histogram_min", "histogram_max"' : '';

  let extraWhere = ctx.where;
  if (selectedGroups.length && ctx.state.groupCol) {
    const list      = selectedGroups.map(g => `'${g.replace(/'/g, "''")}'`).join(', ');
    const connector = extraWhere ? 'AND' : 'WHERE';
    extraWhere += ` ${connector} ${q(ctx.state.groupCol)} IN (${list})`;
  }

  // Per-group reservoir sample: QUALIFY runs after SELECT so __group__ alias is visible.
  const result    = await ctx.query(
    `SELECT ${gcExpr}, "histogram_counts"${rangeSel}
     FROM pp_data ${extraWhere}
     QUALIFY ROW_NUMBER() OVER (PARTITION BY __group__ ORDER BY random()) <= ${samplesPerGroup}`
  );
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
    const total = histTotal(counts);
    if (total <= 0) continue;
    if (!groupData[group]) groupData[group] = { sums: new Float64Array(counts.length), count: 0, min: hasRange ? Number(row.histogram_min) : 0, max: hasRange ? Number(row.histogram_max) : 255 };
    const gd = groupData[group];
    // Normalize each image to area 1 before accumulating, so the group curve is
    // the mean per-image histogram (every image weighted equally, not by size).
    for (let i = 0; i < counts.length; i++) gd.sums[i] += counts[i] / total;
    gd.count++;
    if (hasRange) { gd.min = Math.min(gd.min, Number(row.histogram_min)); gd.max = Math.max(gd.max, Number(row.histogram_max)); }
  }

  const visibleGroups = (selectedGroups.length ? selectedGroups : ctx.groups).filter(g => groupData[g]?.count);
  if (!visibleGroups.length) {
    container.innerHTML = '<div class="no-data">No histogram data available for the current filter.</div>';
    return;
  }
  const traces = visibleGroups.map(g => {
    const { sums, count, min, max } = groupData[g];
    const nBins = sums.length;
    const ys    = Array.from(sums, v => v / count);
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
        const ys    = Array.from(counts, v => total > 0 ? Math.max(0, v / total) : 0);
        const xs    = histXAxis(mode, nBins, minV, maxV);
        const width = xs[1] - xs[0];
        traces.unshift({ type: 'bar', name: `File: ${selectedFile.split('/').pop()}`, x: xs, y: ys, width: Array(nBins).fill(width), marker: { color: 'black' }, opacity: 0.3 });
      }
    }
  }

  const totalFetched = arrowRows.length;
  const maxPossible  = samplesPerGroup * visibleGroups.length;
  if (totalFetched < maxPossible) {
    // All groups fit within the cap - no note needed (sampling had no effect)
  } else {
    const note = document.createElement('p');
    note.style.cssText = 'font-size:0.82em;color:#888;margin:4px 0 0';
    note.textContent   = `Showing up to ${samplesPerGroup.toLocaleString()} samples per group.`;
    container.appendChild(note);
  }

  const showLegend = visibleGroups.length > 1 || !!selectedFile;
  appendPlot(container, traces, {
    title: { text: 'Intensity Histograms (averaged per group)' },
    xaxis: {
      title: mode === 'native'
        ? 'Pixel value'
        : 'Normalized intensity (0–255)',
    },
    yaxis: { title: 'Normalized Count' },
    bargap: 0, height: 500, showlegend: showLegend,
    ...(showLegend ? { legend: plotlyLegendConfig } : {}),
  });
}

// Total pixel count of one image's histogram, used to normalize it to area 1.
function histTotal(counts) {
  let total = 0;
  for (let i = 0; i < counts.length; i++) total += counts[i];
  return total;
}

function histXAxis(mode, nBins, min, max) {
  return mode === 'native'
    ? Array.from({ length: nBins }, (_, i) => min + (i / nBins) * (max - min))
    : Array.from({ length: nBins }, (_, i) => (i / nBins) * 255);
}

