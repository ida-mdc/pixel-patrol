const INFO = [
  'Shows **channel co-localisation metrics** for every pair of channels (C axis).',
  '',
  'Values are first averaged **within each group** (ignoring NaNs), then summarised across groups.',
  'Channel pairs in the line plots are ordered by the pooled mean across groups.',
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

  const pearsonAcc  = metricAcc['coloc_pearson_r'];
  const ssimAcc     = metricAcc['coloc_ssim'];
  const structAcc   = metricAcc['coloc_ssim_structure'];
  const contrastAcc = metricAcc['coloc_ssim_contrast'];
  const lumAcc      = metricAcc['coloc_ssim_luminance'];

  const pearsonGlobal  = pairs.map((_, pi) => nanMean(groups.flatMap(g => pearsonAcc[g][pi])));
  const ssimGlobal     = pairs.map((_, pi) => nanMean(groups.flatMap(g => ssimAcc[g][pi])));
  const structGlobal   = pairs.map((_, pi) => nanMean(groups.flatMap(g => structAcc[g][pi])));
  const contrastGlobal = pairs.map((_, pi) => nanMean(groups.flatMap(g => contrastAcc[g][pi])));
  const lumGlobal      = pairs.map((_, pi) => nanMean(groups.flatMap(g => lumAcc[g][pi])));

  const pearsonSortIdx = pairs.map((_, i) => i)
    .sort((a, b) => (pearsonGlobal[b] ?? -Infinity) - (pearsonGlobal[a] ?? -Infinity));
  const ssimSortIdx = pairs.map((_, i) => i)
    .sort((a, b) => (ssimGlobal[b] ?? -Infinity) - (ssimGlobal[a] ?? -Infinity));
  const structSortIdx = pairs.map((_, i) => i)
    .sort((a, b) => (structGlobal[b] ?? -Infinity) - (structGlobal[a] ?? -Infinity));
  const contrastSortIdx = pairs.map((_, i) => i)
    .sort((a, b) => (contrastGlobal[b] ?? -Infinity) - (contrastGlobal[a] ?? -Infinity));
  const lumSortIdx = pairs.map((_, i) => i)
    .sort((a, b) => (lumGlobal[b] ?? -Infinity) - (lumGlobal[a] ?? -Infinity));

  renderWidgetIntro(container);

  renderSection(container, 'Pearson r — Channel Pair Matrix',
    'Pearson correlation measures whether two channels vary together within the same tiles. '
    + '+1 indicates strong positive co-variation, −1 indicates strong inverse co-variation, '
    + 'and 0 indicates no linear relationship. Values are aggregated by taking the mean of '
    + 'finite per-row values within each group.');
  renderHeatmapSection(container, ctx, groups, pairs, chLabels, maxNC,
    pearsonAcc, pearsonGlobal, HEATMAP_PEARSON);

  renderSection(container, 'Pearson r',
    'Pairs are ordered by the mean Pearson r pooled across all groups (finite values only). '
    + 'Within each group, the plotted value for a pair is the mean of finite per-row values.');
  renderStrengthPlot(container, ctx, groups, pearsonAcc, pearsonSortIdx,
    pairs, 'Pearson r', [-1, 1], true);

  renderSection(container, 'SSIM — Channel Pair Matrix',
    'SSIM (structural similarity index) is a composite score in [0, 1] combining luminance, '
    + 'contrast, and structure terms. 1 means the channels look structurally identical within '
    + 'tiles; values near 0 indicate strong dissimilarity. Values are aggregated by taking the '
    + 'mean of finite per-row values within each group.');
  renderHeatmapSection(container, ctx, groups, pairs, chLabels, maxNC,
    ssimAcc, ssimGlobal, HEATMAP_SSIM);

  renderSection(container, 'SSIM (composite)',
    'Pairs are ordered by the mean SSIM pooled across all groups (finite values only). '
    + 'Within each group, the plotted value for a pair is the mean of finite per-row values.');
  renderStrengthPlot(container, ctx, groups, ssimAcc, ssimSortIdx,
    pairs, 'SSIM', [0, 1], false);

  renderSection(container, 'SSIM structure',
    'Structure is the correlation-like component of SSIM (often similar to Pearson r). '
    + 'Pairs are ordered by the mean structure pooled across all groups. Within each group, '
    + 'values are the mean of finite per-row structure scores.');
  renderStrengthPlot(container, ctx, groups, structAcc, structSortIdx,
    pairs, 'SSIM structure', [-1, 1], true);

  renderSection(container, 'SSIM contrast',
    'Contrast measures whether the channels have a similar intensity spread (variance) within tiles. '
    + 'Pairs are ordered by the mean contrast pooled across all groups. Within each group, values '
    + 'are the mean of finite per-row contrast scores.');
  renderStrengthPlot(container, ctx, groups, contrastAcc, contrastSortIdx,
    pairs, 'SSIM contrast', [0, 1], false);

  renderSection(container, 'SSIM luminance',
    'Luminance measures whether channels have a similar mean intensity within tiles. In fluorescence data '
    + 'this can reflect bleed-through or shared background, but it is not a general “quality” measure because '
    + 'different labels can have inherently different brightness. Pairs are ordered by the mean luminance pooled '
    + 'across all groups. Within each group, values are the mean of finite per-row luminance scores.');
  renderStrengthPlot(container, ctx, groups, lumAcc, lumSortIdx,
    pairs, 'SSIM luminance', [0, 1], false);

  renderSection(container, 'Largest between-group differences',
    'To highlight where groups disagree most, we rank channel pairs by the spread of group means '
    + '(max − min across groups). Each group mean is computed as the mean of finite per-row values. '
    + 'The tables list the top pairs by Pearson r spread and by SSIM spread, and include the SSIM '
    + 'components per group for context.');
  renderDiffTables(container, ctx, groups, pairs, pearsonAcc, ssimAcc, structAcc, contrastAcc, lumAcc, 5);
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

// ── Difference tables ─────────────────────────────────────────────────────────

function groupMeanForPair(acc, g, pi) {
  return nanMean(acc[g]?.[pi] ?? []);
}

function groupRangeOfMeans(acc, groups, pi) {
  const vals = groups.map(g => groupMeanForPair(acc, g, pi)).filter(Number.isFinite);
  if (!vals.length) return null;
  return Math.max(...vals) - Math.min(...vals);
}

function fmtNum(v, digits = 3) {
  return Number.isFinite(v) ? Number(v).toFixed(digits) : '–';
}

function renderDiffTables(container, ctx, groups, pairs, pearsonAcc, ssimAcc, structAcc, contrastAcc, lumAcc, topN = 5) {
  if (groups.length < 2) {
    const p = document.createElement('p');
    p.className = 'small text-muted';
    p.style.cssText = 'margin:0 0 12px;max-width:52rem';
    p.textContent = 'Between-group difference tables require at least two groups.';
    container.appendChild(p);
    return;
  }

  container.appendChild(buildColorLegend());

  const rankBy = (acc) => pairs
    .map((_, pi) => ({ pi, delta: groupRangeOfMeans(acc, groups, pi) }))
    .filter(o => o.delta != null)
    .sort((a, b) => (b.delta ?? -Infinity) - (a.delta ?? -Infinity))
    .slice(0, topN);

  const pearsonTop = rankBy(pearsonAcc);
  const ssimTop    = rankBy(ssimAcc);

  container.appendChild(buildDiffTable(
    'Top pairs by Pearson r spread (max − min across groups)',
    pearsonTop,
    { primaryAcc: pearsonAcc, primaryLabel: 'r' },
  ));
  container.appendChild(buildDiffTable(
    'Top pairs by SSIM spread (max − min across groups)',
    ssimTop,
    { primaryAcc: ssimAcc, primaryLabel: 'SSIM' },
  ));

  function buildColorLegend() {
    const wrap = document.createElement('div');
    wrap.className = 'small text-muted';
    wrap.style.cssText = 'display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:0 0 10px';

    const title = document.createElement('span');
    title.style.fontWeight = '600';
    title.textContent = 'Cell color:';
    wrap.appendChild(title);

    wrap.appendChild(legendItem('rgba(25, 135, 84, 0.35)', 'Above across-group average'));
    wrap.appendChild(legendItem('rgba(220, 53, 69, 0.35)', 'Below across-group average'));

    return wrap;
  }

  function legendItem(color, label) {
    const item = document.createElement('div');
    item.style.cssText = 'display:flex;align-items:center;gap:8px';
    const sw = document.createElement('span');
    sw.style.cssText =
      `display:inline-block;width:14px;height:14px;border-radius:3px;` +
      `border:1px solid #ddd;background:${color}`;
    const t = document.createElement('span');
    t.textContent = label;
    item.append(sw, t);
    return item;
  }

  function buildDiffTable(title, rows, primary) {
    const card = document.createElement('div');
    card.style.cssText = 'margin:0 0 14px';
    const h = document.createElement('div');
    h.className = 'small';
    h.style.cssText = 'font-weight:600;margin:0 0 6px';
    h.textContent = title;
    card.appendChild(h);

    if (!rows.length) {
      const p = document.createElement('p');
      p.className = 'small text-muted';
      p.style.margin = '0 0 12px';
      p.textContent = 'No finite values available for ranking with the current filter.';
      card.appendChild(p);
      return card;
    }

    const tableWrap = document.createElement('div');
    tableWrap.style.cssText = 'overflow:auto;border:1px solid #eee;border-radius:6px';
    const tbl = document.createElement('table');
    tbl.className = 'table table-sm mb-0';
    tbl.style.cssText = 'min-width:760px';

    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    const cols = ['Pair', 'Δ', 'Group', 'r', 'SSIM', 'structure', 'contrast', 'luminance'];
    cols.forEach(c => {
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = c;
      trh.appendChild(th);
    });
    thead.appendChild(trh);
    tbl.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (const { pi, delta } of rows) {
      const pair = pairLabel(...pairs[pi]);
      const perMetricStats = metricStatsForPair(pi);
      for (let gi = 0; gi < groups.length; gi++) {
        const g = groups[gi];
        const tr = document.createElement('tr');
        // Dark (not thick) separators so they stay visible on colored cells.
        if (gi === 0) tr.style.borderTop = '1px solid #888';
        if (gi === groups.length - 1) tr.style.borderBottom = '1px solid #888';
        if (gi === 0) {
          const tdPair = document.createElement('td');
          tdPair.rowSpan = groups.length;
          tdPair.style.fontWeight = '600';
          tdPair.textContent = pair;
          tr.appendChild(tdPair);

          const tdD = document.createElement('td');
          tdD.rowSpan = groups.length;
          tdD.textContent = fmtNum(delta, 3);
          tr.appendChild(tdD);
        }

        // Don't color the group name cell (keep it readable).
        const tdG = document.createElement('td');
        tdG.textContent = ctx.groupLabel ? ctx.groupLabel(g) : String(g);
        tr.appendChild(tdG);

        const rVal   = groupMeanForPair(pearsonAcc, g, pi);
        const sVal   = groupMeanForPair(ssimAcc, g, pi);
        const stVal  = groupMeanForPair(structAcc, g, pi);
        const cVal   = groupMeanForPair(contrastAcc, g, pi);
        const lVal   = groupMeanForPair(lumAcc, g, pi);

        tr.appendChild(signTd(fmtNum(rVal, 3),  rVal,  perMetricStats.r.avg));
        tr.appendChild(signTd(fmtNum(sVal, 3),  sVal,  perMetricStats.SSIM.avg));
        tr.appendChild(signTd(fmtNum(stVal, 3), stVal, perMetricStats.structure.avg));
        tr.appendChild(signTd(fmtNum(cVal, 3),  cVal,  perMetricStats.contrast.avg));
        tr.appendChild(signTd(fmtNum(lVal, 3),  lVal,  perMetricStats.luminance.avg));

        tbody.appendChild(tr);
      }
    }
    tbl.appendChild(tbody);
    tableWrap.appendChild(tbl);
    card.appendChild(tableWrap);
    return card;
  }

  function td(text) {
    const el = document.createElement('td');
    el.textContent = text;
    return el;
  }

  function signTd(text, v, avg) {
    const el = document.createElement('td');
    el.textContent = text;
    if (!Number.isFinite(v) || !Number.isFinite(avg)) return el;

    const delta = v - avg;
    el.style.background = delta >= 0
      ? 'rgba(25, 135, 84, 0.14)'   // green
      : 'rgba(220, 53, 69, 0.14)';  // red
    return el;
  }

  function metricStatsForPair(pi) {
    const statsFor = (acc) => {
      const vals = groups.map(g => groupMeanForPair(acc, g, pi)).filter(Number.isFinite);
      if (!vals.length) return { avg: null, range: null };
      const avg = vals.reduce((s, x) => s + x, 0) / vals.length;
      return { avg };
    };
    return {
      r: statsFor(pearsonAcc),
      SSIM: statsFor(ssimAcc),
      structure: statsFor(structAcc),
      contrast: statsFor(contrastAcc),
      luminance: statsFor(lumAcc),
    };
  }
}

// ── Heatmap helpers ───────────────────────────────────────────────────────────

/** Diverging Pearson r: −1 strong cool → 0 neutral → +1 strong warm (fixed domain). */
const COLORSCALE_PEARSON_DIVERGING = [
  [0, 'rgb(49, 54, 149)'],
  [0.35, 'rgb(146, 197, 222)'],
  [0.5, 'rgb(247, 247, 247)'],
  [0.65, 'rgb(252, 174, 97)'],
  [1, 'rgb(179, 0, 49)'],
];

const HEATMAP_PEARSON = {
  colorscale: COLORSCALE_PEARSON_DIVERGING,
  reversescale: false,
  zmin: -1,
  zmax: 1,
  colorbar: {
    thickness: 12,
    len: 0.75,
    title: { text: 'r', side: 'right' },
    tickmode: 'array',
    tickvals: [-1, 0, 1],
    ticktext: ['−1', '0', '+1'],
  },
  hoverLabel: 'r',
};

/** Sequential SSIM composite: 0 = dissimilar → 1 = identical. */
const HEATMAP_SSIM = {
  colorscale: 'Viridis',
  reversescale: false,
  zmin: 0,
  zmax: 1,
  colorbar: {
    thickness: 12,
    len: 0.75,
    title: { text: 'SSIM', side: 'right' },
    tickmode: 'array',
    tickvals: [0, 1],
    ticktext: ['0', '1'],
  },
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

function renderWidgetIntro(container) {
  const p = document.createElement('p');
  p.className = 'small text-muted';
  p.style.cssText = 'margin:0 0 18px;line-height:1.45;max-width:52rem';
  p.textContent =
    'Visualisations below summarise channel co-localisation for the current report rows '
    + 'and filters. Metrics were computed between channel pairs from image tiles and '
    + 'stored as arrays on each row; the viewer averages finite values across matching '
    + 'observations (per group or pooled). A difference table highlights channel pairs '
    + 'with the largest between-group spread (max − min across group means).';
  container.appendChild(p);
}

function renderSection(container, title, caption = null) {
  const h = document.createElement('h5');
  h.textContent = title;
  h.style.cssText = 'margin:24px 0 8px;font-size:14px';
  container.appendChild(h);
  if (caption) {
    const p = document.createElement('p');
    p.className = 'small text-muted';
    p.style.cssText = 'margin:-4px 0 12px;line-height:1.45;max-width:52rem';
    p.textContent = caption;
    container.appendChild(p);
  }
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function escVal(s) {
  return String(s).replace(/"/g, '&quot;');
}
