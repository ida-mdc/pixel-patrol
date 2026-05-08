const INFO = [
  'Shows **channel co-localisation metrics** for every pair of channels (C axis).',
  '',
  '**Pearson r matrix** — upper-triangle heatmap. **SSIM matrix** — same layout, Blues',
  'colorscale. Both support group dropdown and side-by-side comparison.',
  '',
  '**Pearson r plot** — one line per group, pairs sorted by overall Pearson r.',
  '**SSIM plot** — one line per group, pairs sorted by overall SSIM composite.',
  'Shaded bands show ±1 std dev when n > 1.',
  '',
  '**Metrics** (one value per pair in combinations order)',
  '- **Pearson r** — Co-localisation score [−1, 1], invariant to offset and scale.',
  '- **SSIM structure** ≈ Pearson r (structural component).',
  '- **SSIM contrast** — Proportional intensity spread between channels.',
  '- **SSIM luminance** — Mean-intensity similarity (bleed-through indicator).',
  '- **SSIM** — Composite of luminance × contrast × structure.',
  '',
  'Ref: Manders et al. (1993) J. Microscopy 169(3):375–382',
  '  Wang et al. (2004) IEEE TIP 13(4):600–612',
].join('\n');

const METRIC_KEYS = [
  'coloc_pearson_r', 'coloc_ssim_structure', 'coloc_ssim_contrast',
  'coloc_ssim_luminance', 'coloc_ssim',
];

function allPairs(nC) {
  const pairs = [];
  for (let i = 0; i < nC; i++)
    for (let j = i + 1; j < nC; j++)
      pairs.push([i, j]);
  return pairs;
}

function pairLabel(ci, cj) { return `C${ci} × C${cj}`; }

function nanMean(vals) {
  const fin = vals.filter(Number.isFinite);
  return fin.length ? fin.reduce((s, v) => s + v, 0) / fin.length : null;
}

function nanStd(vals) {
  const fin = vals.filter(Number.isFinite);
  if (fin.length < 2) return null;
  const m = fin.reduce((s, v) => s + v, 0) / fin.length;
  return Math.sqrt(fin.reduce((s, v) => s + (v - m) ** 2, 0) / fin.length);
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Plugin export ─────────────────────────────────────────────────────────────

export default {
  id: 'channel-colocalization',
  label: 'Channel Co-localisation',
  group: 'Dataset Stats',
  info: INFO,

  requires(schema) {
    return schema.isLongFormat && schema.blobCols.includes('coloc_pearson_r');
  },

  async render(container, ctx) {
    try {
      await renderColoc(container, ctx);
    } catch (e) {
      console.error('plugin_colocalization', e);
      container.innerHTML = '<div class="no-data">Failed to load co-localisation data.</div>';
    }
  },
};

// ── Main render ───────────────────────────────────────────────────────────────

async function renderColoc(container, ctx) {
  const { groupExpr: geFn } = ctx.sql;
  const { extractBinary } = ctx.data;

  const baseWhere = ctx.userWhere
    ? ctx.userWhere.replace(/^\s*WHERE\s+/i, '') + ' AND "coloc_pearson_r" IS NOT NULL'
    : '"coloc_pearson_r" IS NOT NULL';

  const result = await ctx.query(`
    SELECT ${geFn()}, "coloc_n_channels",
           "coloc_pearson_r", "coloc_ssim",
           "coloc_ssim_luminance", "coloc_ssim_contrast", "coloc_ssim_structure"
    FROM pp_data WHERE ${baseWhere}
  `);
  const arrowRows = result.toArray();

  if (!arrowRows.length) {
    container.innerHTML =
      '<div class="no-data">No co-localisation data found. ' +
      'Process with <code>ChannelColocalizationProcessor</code>.</div>';
    return;
  }

  const maxNC = Math.max(...arrowRows.map(r => Number(r.coloc_n_channels) || 0));
  if (maxNC < 2) {
    container.innerHTML = '<div class="no-data">Need at least 2 channels.</div>';
    return;
  }

  const pairs    = allPairs(maxNC);
  const chLabels = Array.from({ length: maxNC }, (_, i) => `C${i}`);
  const groups   = ctx.groups.filter(g => arrowRows.some(r => String(r.__group__) === g));

  // metricAcc[metric][group][pairIdx] = [per-file values]
  const metricAcc = Object.fromEntries(METRIC_KEYS.map(mk => [
    mk, Object.fromEntries(groups.map(g => [g, pairs.map(() => [])])),
  ]));

  for (const row of arrowRows) {
    const g = String(row.__group__);
    for (const mk of METRIC_KEYS) {
      const acc = metricAcc[mk][g];
      if (!acc) continue;
      const vals = extractBinary(row[mk]);
      if (!vals) continue;
      for (let pi = 0; pi < Math.min(vals.length, pairs.length); pi++) {
        const v = vals[pi];
        if (Number.isFinite(v)) acc[pi].push(v);
      }
    }
  }

  const pearsonAcc = metricAcc['coloc_pearson_r'];
  const ssimAcc    = metricAcc['coloc_ssim'];

  const pearsonGlobal = pairs.map((_, pi) => nanMean(groups.flatMap(g => pearsonAcc[g][pi])));
  const ssimGlobal    = pairs.map((_, pi) => nanMean(groups.flatMap(g => ssimAcc[g][pi])));

  const pearsonSortIdx = pairs.map((_, i) => i)
    .sort((a, b) => (pearsonGlobal[b] ?? -Infinity) - (pearsonGlobal[a] ?? -Infinity));
  const ssimSortIdx = pairs.map((_, i) => i)
    .sort((a, b) => (ssimGlobal[b] ?? -Infinity) - (ssimGlobal[a] ?? -Infinity));

  // ── Heatmaps ──────────────────────────────────────────────────────────────
  renderSection(container, 'Pearson r — Channel Pair Matrix');
  renderHeatmapSection(container, ctx, groups, pairs, chLabels, maxNC,
    pearsonAcc, pearsonGlobal, HEATMAP_PEARSON);

  renderSection(container, 'SSIM — Channel Pair Matrix');
  renderHeatmapSection(container, ctx, groups, pairs, chLabels, maxNC,
    ssimAcc, ssimGlobal, HEATMAP_SSIM);

  // ── Strength plots ────────────────────────────────────────────────────────
  renderSection(container, 'Pearson r by Channel Pair');
  renderStrengthPlot(container, ctx, groups, pearsonAcc, pearsonSortIdx,
    pairs, "Pearson r", [-1, 1], true);

  renderSection(container, 'SSIM by Channel Pair');
  renderStrengthPlot(container, ctx, groups, ssimAcc, ssimSortIdx,
    pairs, 'SSIM', null, true);
}

// ── Strength plot ─────────────────────────────────────────────────────────────

function renderStrengthPlot(container, ctx, groups, acc, sortIdx, pairs, yTitle, yRange, zeroline) {
  const xLabels   = sortIdx.map(pi => pairLabel(...pairs[pi]));
  const tickAngle = xLabels.length > 6 ? -45 : 0;
  const traces    = [];

  for (const g of groups) {
    const color    = ctx.color.group(g);
    const fillRgba = hexToRgba(color, 0.15);
    const means = sortIdx.map(pi => nanMean(acc[g][pi]));
    const stds  = sortIdx.map(pi => nanStd(acc[g][pi]));

    if (stds.some(s => s !== null)) {
      const lower = means.map((m, i) => m != null && stds[i] != null ? m - stds[i] : null);
      const upper = means.map((m, i) => m != null && stds[i] != null ? m + stds[i] : null);
      traces.push({ x: xLabels, y: lower, type: 'scatter', mode: 'lines',
        line: { width: 0 }, hoverinfo: 'skip', showlegend: false, legendgroup: g });
      traces.push({ x: xLabels, y: upper, type: 'scatter', mode: 'lines',
        fill: 'tonexty', fillcolor: fillRgba,
        line: { width: 0 }, hoverinfo: 'skip', showlegend: false, legendgroup: g });
    }
    traces.push({
      x: xLabels, y: means, name: String(g),
      type: 'scatter', mode: 'lines+markers', legendgroup: g,
      line: { color, width: 2 }, marker: { color, size: 7 },
      hovertemplate: `<b>${escHtml(String(g))}</b><br>%{x}: %{y:.3f}<extra></extra>`,
    });
  }

  ctx.plot.append(container, traces, {
    yaxis: { title: yTitle, ...(yRange ? { range: yRange } : {}), zeroline, zerolinecolor: '#ccc' },
    xaxis: { title: 'Channel pair', tickangle: tickAngle },
    legend: { orientation: 'h', y: -0.25 },
    margin: { t: 20, b: tickAngle ? 100 : 60 },
  }, 'margin-bottom:20px');
}

// ── Heatmap helpers ───────────────────────────────────────────────────────────

const HEATMAP_PEARSON = {
  colorscale: 'RdBu', reversescale: true, zmin: -1, zmax: 1,
  colorbar: { thickness: 12, len: 0.75, title: { text: 'r', side: 'right' } },
  hoverLabel: 'r',
};
const HEATMAP_SSIM = {
  colorscale: 'RdBu', reversescale: true, zmin: -1, zmax: 1,
  colorbar: { thickness: 12, len: 0.75, title: { text: 'SSIM', side: 'right' } },
  hoverLabel: 'SSIM',
};

function buildHeatmapZ(pairs, maxNC, pairMeans) {
  const z    = Array.from({ length: maxNC }, () => Array(maxNC).fill(null));
  const text = Array.from({ length: maxNC }, () => Array(maxNC).fill(''));
  pairs.forEach(([ci, cj], pi) => {
    const m = pairMeans[pi];
    z[ci][cj]    = m;
    text[ci][cj] = m != null ? m.toFixed(2) : '–';
  });
  return { z, text };
}

function heatmapTrace(pairs, maxNC, chLabels, means, opts) {
  const { z, text } = buildHeatmapZ(pairs, maxNC, means);
  return {
    type: 'heatmap', z, text, x: chLabels, y: chLabels,
    colorscale: opts.colorscale, reversescale: opts.reversescale,
    zmin: opts.zmin, zmax: opts.zmax,
    showscale: true, hoverongaps: false, colorbar: opts.colorbar,
    texttemplate: '%{text}',
    hovertemplate: `%{y} × %{x}<br>${opts.hoverLabel} = %{z:.3f}<extra></extra>`,
  };
}

function heatmapLayout(title) {
  return {
    title: { text: title, font: { size: 13 } },
    margin: { t: 36, l: 44, r: 60, b: 44 },
    xaxis: { constrain: 'domain', side: 'bottom' },
    yaxis: { constrain: 'domain', scaleanchor: 'x', autorange: 'reversed' },
  };
}

function renderHeatmapSection(container, ctx, groups, pairs, chLabels, maxNC, acc, gMeans, opts) {
  const meansFor = g => pairs.map((_, pi) => nanMean(acc[g][pi]));

  if (groups.length > 1) {
    const ctrl = document.createElement('div');
    ctrl.style.cssText = 'display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:12px';
    const sel = document.createElement('select');
    sel.className = 'form-select form-select-sm';
    sel.style.cssText = 'width:auto;min-width:160px';
    sel.innerHTML = '<option value="__all__">All groups (mean)</option>' +
      groups.map(g => `<option value="${escVal(g)}">${escHtml(String(g))}</option>`).join('');
    const toggleWrap = document.createElement('label');
    toggleWrap.style.cssText = 'display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none';
    const chk = document.createElement('input');
    chk.type = 'checkbox';
    toggleWrap.append(chk, 'Compare side by side');
    ctrl.append(sel, toggleWrap);
    container.appendChild(ctrl);
    const matDiv = document.createElement('div');
    container.appendChild(matDiv);
    const refresh = () => {
      matDiv.innerHTML = '';
      sel.disabled = chk.checked;
      if (chk.checked) {
        const cols = Math.min(groups.length, 3);
        const pct  = Math.floor(100 / cols) - 2;
        const wrap = document.createElement('div');
        wrap.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px';
        matDiv.appendChild(wrap);
        for (const g of groups)
          ctx.plot.append(wrap, [heatmapTrace(pairs, maxNC, chLabels, meansFor(g), opts)],
            heatmapLayout(String(g)),
            `flex:0 0 ${pct}%;min-width:200px;box-sizing:border-box;margin-bottom:20px`);
      } else {
        const g     = sel.value === '__all__' ? null : sel.value;
        const means = g ? meansFor(g) : gMeans;
        ctx.plot.append(matDiv, [heatmapTrace(pairs, maxNC, chLabels, means, opts)],
          heatmapLayout(g ? String(g) : 'All groups (mean)'),
          'max-width:440px;margin-bottom:20px');
      }
    };
    sel.addEventListener('change', refresh);
    chk.addEventListener('change', refresh);
    refresh();
  } else {
    const means  = groups.length === 1 ? meansFor(groups[0]) : gMeans;
    const title  = groups.length === 1 ? String(groups[0]) : 'All';
    const matDiv = document.createElement('div');
    container.appendChild(matDiv);
    ctx.plot.append(matDiv, [heatmapTrace(pairs, maxNC, chLabels, means, opts)],
      heatmapLayout(title), 'max-width:440px;margin-bottom:20px');
  }
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

function renderSection(container, title) {
  const h = document.createElement('h5');
  h.textContent = title;
  h.style.cssText = 'margin:24px 0 10px;font-size:14px';
  container.appendChild(h);
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function escVal(s) {
  return String(s).replace(/"/g, '&quot;');
}
