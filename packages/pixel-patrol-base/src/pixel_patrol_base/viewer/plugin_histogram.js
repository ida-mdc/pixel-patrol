const NBINS           = 256;
const DEFAULT_SAMPLES = 2000;
const MAX_FILE_OPTIONS = 500;
const GROUP_SEL_ID    = 'hist-group-select';
const FILE_SEL_ID     = 'hist-file-select';
const SAMPLE_INPUT_ID = 'hist-sample-input';
const SPREAD_ID       = 'hist-spread-cb';
const INDIV_ID        = 'hist-indiv-cb';
const INDIV_LIMIT     = 50;

// Normalization divisor per dtype for the dtype-range display mode.
// uint: width of the representable range (dtype_max + 1), so [0, 256) → [0, 1].
// int:  abs(dtype_min) = 2^(bits-1), so [-128, 128) → [-1, 1].
export const DTYPE_NORM = {
  uint8: 256, uint16: 65536, uint32: 4294967296, uint64: 2 ** 64,
  int8:  128, int16:  32768, int32:  2147483648, int64: 2 ** 63,
};

const KIND_LABEL = { uint: 'Unsigned integer', int: 'Signed integer', float: 'Floating point' };

export function dtypeKind(dtype) {
  if (dtype?.startsWith('float')) return 'float';
  if (dtype?.startsWith('uint'))  return 'uint';
  if (dtype?.startsWith('int'))   return 'int';
  return null;
}

function histTotal(counts) {
  let t = 0;
  for (let i = 0; i < counts.length; i++) t += counts[i];
  return t;
}

// Left-edge x positions for NBINS bins spanning [lo, hi].
function binXs(lo, hi) {
  const step = (hi - lo) / NBINS || 1 / NBINS;
  return Array.from({ length: NBINS }, (_, i) => lo + i * step);
}

// Value-space extent of the non-zero bins of a [lo, hi)-spanning series, or null if all zero.
// Used to zoom the plotted x-axis onto real data instead of the (possibly dtype-widened) bin window.
export function populatedExtent(lo, hi, ys) {
  const step = (hi - lo) / ys.length || 1 / ys.length;
  let first = -1, last = -1;
  for (let i = 0; i < ys.length; i++) {
    if (ys[i] > 0) { if (first < 0) first = i; last = i; }
  }
  if (first < 0) return null;
  return [lo + first * step, lo + (last + 1) * step];
}

// Convert a color string (hex or rgb) to rgba with the given alpha.
function withAlpha(color, alpha) {
  if (color.startsWith('#') && color.length === 7) {
    const r = parseInt(color.slice(1, 3), 16);
    const g = parseInt(color.slice(3, 5), 16);
    const b = parseInt(color.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }
  if (color.startsWith('rgb(')) return color.replace('rgb(', 'rgba(').replace(')', `,${alpha})`);
  return color;
}

// Uses the same widget-warning-yellow class other widgets share (see plot-utils.js's prependWarning) -
// built by hand instead of via prependWarning since these banners are inserted at a specific position,
// not always as the container's first child.
function warningBanner(text, marginBottom = 16) {
  const el = document.createElement('div');
  el.className = 'widget-warning widget-warning-yellow';
  if (marginBottom !== 16) el.style.marginBottom = `${marginBottom}px`;
  const icon = document.createElement('span');
  icon.className = 'widget-warning-icon';
  icon.textContent = '⚠️';
  const content = document.createElement('div');
  content.textContent = text;
  el.append(icon, content);
  return el;
}

function isClose(a, b, rtol = 1e-9, atol = 1e-9) {
  return Math.abs(a - b) <= atol + rtol * Math.abs(b);
}

// Redistributes a value array from its own [srcLo,srcHi) axis onto a shared [tgtLo,tgtHi) axis via bin-centre remap.
export function remapToAxis(values, srcLo, srcHi, tgtLo, tgtHi) {
  if (isClose(srcLo, tgtLo) && isClose(srcHi, tgtHi)) return values;
  const B = values.length;
  const out = new Float64Array(B);
  if (tgtLo === tgtHi) {
    let total = 0; for (let i = 0; i < B; i++) total += values[i];
    out[0] = total;
    return out;
  }
  const tgtStep = (tgtHi - tgtLo) / B;
  if (srcLo === srcHi) {
    const idx = Math.min(Math.max(Math.floor((srcLo - tgtLo) / tgtStep), 0), B - 1);
    for (let i = 0; i < B; i++) out[idx] += values[i];
    return out;
  }
  const srcStep = (srcHi - srcLo) / B;
  for (let i = 0; i < B; i++) {
    if (!values[i]) continue;
    const center = srcLo + (i + 0.5) * srcStep;
    let idx = Math.floor((center - tgtLo) / tgtStep);
    if (idx < 0) idx = 0; else if (idx >= B) idx = B - 1;
    out[idx] += values[i];
  }
  return out;
}

// Classifies the dataset's distinct dtypes into the kinds this widget can plot, plus
// any dtype it can't (e.g. bool) so callers can warn instead of silently dropping them.
// dtypeRows: rows with a `dtype` field, as returned by `SELECT DISTINCT dtype`.
export function classifyDtypes(dtypeRows, hasDtype) {
  if (!hasDtype) return { presentKinds: ['uint'], multiKind: false, unsupportedDtypes: [] };
  const presentKinds = ['uint', 'int', 'float'].filter(k => dtypeRows.some(r => dtypeKind(r.dtype) === k));
  const unsupportedDtypes = [...new Set(dtypeRows.map(r => r.dtype).filter(d => !dtypeKind(d)))];
  return { presentKinds, multiKind: presentKinds.length > 1, unsupportedDtypes };
}

// Combines per-image histograms into per-(kind,group) means on a shared axis, instead of summing raw bin indices.
// records: [{ kind, group, counts, total, aMin, aMax, nMin, nMax }].
export function accumulateGroupHistograms(records, { computeNormalized = true } = {}) {
  const buckets = {};
  for (const r of records) {
    (buckets[r.kind] ??= {});
    (buckets[r.kind][r.group] ??= []).push(r);
  }

  const result = {};
  for (const kind of Object.keys(buckets)) {
    result[kind] = {};
    const doNorm = computeNormalized && kind !== 'float';
    for (const group of Object.keys(buckets[kind])) {
      const items = buckets[kind][group];
      let aLo = Infinity, aHi = -Infinity, nLo = Infinity, nHi = -Infinity;
      for (const it of items) {
        if (it.aMin < aLo) aLo = it.aMin;
        if (it.aMax > aHi) aHi = it.aMax;
        if (doNorm) {
          if (it.nMin < nLo) nLo = it.nMin;
          if (it.nMax > nHi) nHi = it.nMax;
        }
      }

      const sums      = new Float64Array(NBINS);
      const sumSq     = new Float64Array(NBINS);
      const sumsNorm  = doNorm ? new Float64Array(NBINS) : null;
      const sumSqNorm = doNorm ? new Float64Array(NBINS) : null;

      for (const it of items) {
        const inv = 1 / it.total;
        const v = new Float64Array(it.counts.length);
        for (let i = 0; i < v.length; i++) v[i] = it.counts[i] * inv;

        const vA = remapToAxis(v, it.aMin, it.aMax, aLo, aHi);
        for (let i = 0; i < NBINS; i++) { sums[i] += vA[i]; sumSq[i] += vA[i] * vA[i]; }

        if (doNorm) {
          const vN = remapToAxis(v, it.nMin, it.nMax, nLo, nHi);
          for (let i = 0; i < NBINS; i++) { sumsNorm[i] += vN[i]; sumSqNorm[i] += vN[i] * vN[i]; }
        }
      }

      result[kind][group] = {
        count: items.length,
        aMin: aLo, aMax: aHi,
        nMin: doNorm ? nLo : aLo, nMax: doNorm ? nHi : aHi,
        sums, sumSq,
        sumsNorm: sumsNorm ?? sums, sumSqNorm: sumSqNorm ?? sumSq,
      };
    }
  }
  return result;
}

export default {
  id: 'histogram',
  required_inputs: ['histogram_counts'],
  inputs: ['histogram_min', 'histogram_max', 'dtype', 'name'],
  group: 'Dataset Stats',
  scope: 'image',
  label: 'Pixel Value Histograms',
  shortLabel: 'Histograms',
  info: [
    'Histograms are computed per image (256 bins) and averaged per group.',
    'Each image is area-normalized before averaging, so image size does not affect the result.',
    '',
    'If the dataset mixes unsigned integer, signed integer, and/or floating-point images, each kind gets its own plot.',
    '',
    '**Display mode** (integer panels only)',
    '- **Actual values** – x-axis shows real pixel values.',
    '- **Dtype range** – x-axis normalized to the representable range of the pixel type (unsigned integers: 0 to 1, signed integers: -1 to 1). Useful for comparing images across bit depths.',
    '',
    'Float images always show actual values.',
    '',
    '**Spread** – shaded +/-1 std band around the group mean.',
    'A wider band means more variation between images in that group.',
    '',
    '**Show individual images** – visible when one group is selected with 50 or fewer images.',
    'Replaces the group mean with one line per image.',
  ].join('\n'),

  requires(schema) {
    return schema.blobCols.includes('histogram_counts');
  },

  async condensedMessage(ctx) {
    try {
      const [{ total, n }] = await ctx.queryRows(
        `SELECT COUNT(*) AS total, COUNT("histogram_counts") AS n FROM pp_data ${ctx.where}`,
      );
      const totalN = Number(total ?? 0), withHist = Number(n ?? 0);
      if (withHist < totalN)
        return { text: `Only <strong>${withHist.toLocaleString()}</strong>/${totalN.toLocaleString()} have histograms.`, warning: true };
      return 'Compare pixel intensity distributions across groups.';
    } catch { return null; }
  },

  async condensedPlot(container, ctx) {
    const { groupExpr: geFn } = ctx.sql;
    const { extractBinary }   = ctx.data;
    const hasRange  = ctx.schema.allCols.includes('histogram_min') && ctx.schema.allCols.includes('histogram_max');
    const hasDtype  = ctx.schema.allCols.includes('dtype');
    const rangeCols = hasRange ? ', "histogram_min", "histogram_max"' : '';
    const dtypeCol  = hasDtype ? ', "dtype"' : '';

    const result = await ctx.query(`
      SELECT ${geFn()}, "histogram_counts"${rangeCols}${dtypeCol}
      FROM pp_data ${ctx.where}
      QUALIFY ROW_NUMBER() OVER (PARTITION BY __group__ ORDER BY random()) <= 300
    `);
    const rows = result.toArray();
    if (!rows.length) return false;

    const records = [];
    const kindCounts = {};
    for (const row of rows) {
      const kind = hasDtype ? dtypeKind(row.dtype) : 'uint';
      if (!kind) continue;
      const counts = extractBinary(row.histogram_counts);
      if (!counts?.length) continue;
      const total = histTotal(counts);
      if (total <= 0) continue;
      const aMin = hasRange ? Number(row.histogram_min) : 0;
      const aMax = hasRange ? Number(row.histogram_max) : NBINS - 1;
      records.push({ kind, group: String(row.__group__), counts, total, aMin, aMax });
      kindCounts[kind] = (kindCounts[kind] || 0) + 1;
    }
    if (!records.length) return false;

    // Mini preview is a single combined chart — plot only the dataset's dominant pixel kind.
    const dominantKind = Object.keys(kindCounts).sort((a, b) => kindCounts[b] - kindCounts[a])[0];
    const kindData = accumulateGroupHistograms(records.filter(r => r.kind === dominantKind), { computeNormalized: false });
    const gdByGroup = kindData[dominantKind] ?? {};

    const groups = ctx.groups.filter(g => gdByGroup[g]?.count);
    if (!groups.length) return false;

    ctx.plot.appendMini(container, groups.map(g => {
      const gd = gdByGroup[g];
      return {
        type: 'scatter', mode: 'lines', fill: 'tozeroy', opacity: 0.6,
        x: binXs(gd.aMin, gd.aMax), y: Array.from(gd.sums, v => v / gd.count),
        line: { color: ctx.color.group(g), width: 1.5 }, hoverinfo: 'skip',
      };
    }), { xaxis: { title: 'intensity' }, yaxis: { showticklabels: false } });
    return true;
  },

  async render(container, ctx) {
    const hasRange = ctx.schema.allCols.includes('histogram_min') && ctx.schema.allCols.includes('histogram_max');
    const hasDtype = ctx.schema.allCols.includes('dtype');
    const hasNames = ctx.schema.allCols.includes('name');

    try {
      const [availRow] = await ctx.queryRows(
        `SELECT COUNT(*) AS total, COUNT("histogram_counts") AS n FROM pp_data ${ctx.where}`
      );
      ctx.plot.dataAvailabilityWarning(container,
        [{ label: 'Pixel Value Histograms', present: Number(availRow.n) }],
        Number(availRow.total), { unit: 'images' });

      const { groupExpr: geFn } = ctx.sql;

      // Parallel setup queries.
      const [dtypeRows, nameRows] = await Promise.all([
        hasDtype ? ctx.queryRows(`SELECT DISTINCT "dtype" FROM pp_data ${ctx.where}`) : [],
        hasNames ? ctx.queryRows(`SELECT DISTINCT "name" FROM pp_data ${ctx.where} ORDER BY 1 LIMIT ${MAX_FILE_OPTIONS}`) : [],
      ]);

      const { presentKinds, multiKind, unsupportedDtypes } = classifyDtypes(dtypeRows, hasDtype);

      container.appendChild(buildControls(ctx, { hasNames, nameRows }));

      const samplingWarning = warningBanner('', 16);
      samplingWarning.style.display = 'none';
      container.appendChild(samplingWarning);

      if (multiKind) {
        container.appendChild(warningBanner(
          'This dataset mixes different kinds of pixel types (e.g. integer and float). '
          + 'Their value scales are not directly comparable, so each kind is shown in its own plot below.',
          20,
        ));
      }

      if (unsupportedDtypes.length) {
        container.appendChild(warningBanner(
          `Images with pixel type(s) ${unsupportedDtypes.join(', ')} are not supported by this widget and are excluded.`,
          20,
        ));
      }

      if (!presentKinds.length) {
        const noData = document.createElement('div');
        noData.className = 'no-data';
        noData.textContent = 'No supported pixel types to display.';
        container.appendChild(noData);
        return;
      }

      // One panel per kind — int/uint get a per-panel mode toggle, float does not.
      const panels = {};
      for (const kind of presentKinds) {
        const wrap = document.createElement('div');
        wrap.style.marginBottom = '32px';

        if (multiKind) {
          const h = document.createElement('div');
          h.style.cssText = 'font-weight:600;font-size:1.05em;margin-bottom:8px';
          h.textContent   = `${KIND_LABEL[kind]} images`;
          wrap.appendChild(h);
        }

        let toggleEl = null, labelLeft = null, labelRight = null;
        if (kind !== 'float') {
          const normLabel  = kind === 'int' ? 'Dtype range (−1 to 1, 256 bins)' : 'Dtype range (0 to 1, 256 bins)';
          const toggleWrap = document.createElement('div');
          toggleWrap.style.cssText = 'display:flex;align-items:center;gap:10px;margin-bottom:14px;font-size:0.88em;color:#444';

          const switchWrap = document.createElement('div');
          switchWrap.className = 'form-check form-switch mb-0';
          toggleEl = document.createElement('input');
          toggleEl.className = 'form-check-input';
          toggleEl.type      = 'checkbox';
          toggleEl.setAttribute('role', 'switch');
          toggleEl.style.cursor = 'pointer';
          switchWrap.appendChild(toggleEl);

          labelLeft  = document.createElement('span');
          labelLeft.textContent = 'Actual values (256 bins)';
          labelRight = document.createElement('span');
          labelRight.textContent = normLabel;

          toggleWrap.append(labelLeft, switchWrap, labelRight);
          wrap.appendChild(toggleWrap);
        } else if (multiKind) {
          const note = document.createElement('div');
          note.style.cssText = 'font-size:0.82em;color:#888;margin-bottom:10px';
          note.textContent = 'Float images always show actual pixel values — there is no fixed representable range.';
          wrap.appendChild(note);
        }

        const plotDiv = document.createElement('div');
        wrap.appendChild(plotDiv);
        container.appendChild(wrap);
        panels[kind] = { plotDiv, toggleEl, labelLeft, labelRight };
      }

      // Cached across mode/spread toggles — re-populated on every other control change.
      let kindData      = null; // { [kind]: { [group]: { sums, sumSq, sumsNorm, sumSqNorm, count, aMin, aMax, nMin, nMax } } }
      let fileOverlay   = null; // { ys, aMin, aMax, nMin, nMax, kind, label }
      let indivFiles    = null; // { rows: [...], overflow: bool } | null

      const fetchAndAccumulate = async () => {
        const { q } = ctx.sql;
        const { extractBinary } = ctx.data;
        const ctrl = readControls(container);

        let where = ctx.where;
        if (ctrl.selectedGroups.length && ctx.state.groupCol) {
          const list = ctrl.selectedGroups.map(g => `'${g.replace(/'/g, "''")}'`).join(', ');
          where += ` ${where ? 'AND' : 'WHERE'} ${q(ctx.state.groupCol)} IN (${list})`;
        }

        const rangeSel = hasRange ? ', "histogram_min", "histogram_max"' : '';
        const dtypeSel = hasDtype ? ', "dtype"'                          : '';
        const result   = await ctx.query(
          `SELECT ${geFn()}, "histogram_counts"${rangeSel}${dtypeSel}
           FROM pp_data ${where}
           QUALIFY ROW_NUMBER() OVER (PARTITION BY __group__ ORDER BY random()) <= ${ctrl.samplesPerGroup}`
        );
        const rows = result.toArray();

        const records = [];
        for (const row of rows) {
          const kind = hasDtype ? dtypeKind(row.dtype) : 'uint';
          if (!kind) continue;
          const counts = extractBinary(row.histogram_counts);
          if (!counts?.length) continue;
          const total = histTotal(counts);
          if (total <= 0) continue;

          const aMin  = hasRange ? Number(row.histogram_min) : 0;
          const aMax  = hasRange ? Number(row.histogram_max) : NBINS - 1;
          const dNorm = DTYPE_NORM[row.dtype];
          const nMin  = dNorm != null ? aMin / dNorm : aMin;
          const nMax  = dNorm != null ? aMax / dNorm : aMax;

          records.push({ kind, group: String(row.__group__), counts, total, aMin, aMax, nMin, nMax });
        }
        kindData = accumulateGroupHistograms(records);

        const rowsPerGroup = {};
        for (const r of rows) {
          const g = String(r.__group__);
          rowsPerGroup[g] = (rowsPerGroup[g] || 0) + 1;
        }
        const cappedGroups = Object.keys(rowsPerGroup)
          .filter(g => rowsPerGroup[g] >= ctrl.samplesPerGroup);
        if (cappedGroups.length === 0) {
          samplingWarning.style.display = 'none';
        } else {
          const allCapped = cappedGroups.length === Object.keys(rowsPerGroup).length;
          let msg = `Results are sampled: showing up to ${ctrl.samplesPerGroup.toLocaleString()} images per group`;
          if (!allCapped)
            msg += `. Groups at limit: ${cappedGroups.map(g => ctx.groupLabel(g)).join(', ')}`;
          samplingWarning.querySelector('div').textContent = msg + '. Increase "Max samples" for full accuracy.';
          samplingWarning.style.display = '';
        }

        // Individual file overlay — only when exactly one group is active and few enough files.
        indivFiles = null;
        const isSingleGroup = ctrl.selectedGroups.length === 1 ||
          (ctrl.selectedGroups.length === 0 && ctx.groups.length === 1);
        let singleGroupCount = 0;
        if (isSingleGroup) {
          for (const kd of Object.values(kindData))
            for (const gd of Object.values(kd)) singleGroupCount += gd.count;
        }
        const indivWrap = container.querySelector('#hist-indiv-wrap');
        if (indivWrap) indivWrap.style.display = (isSingleGroup && singleGroupCount <= INDIV_LIMIT) ? '' : 'none';

        if (container.querySelector(`#${INDIV_ID}`)?.checked && isSingleGroup && singleGroupCount <= INDIV_LIMIT) {
          const nameSel2  = hasNames ? ', "name"' : '';
          const iResult   = await ctx.query(
            `SELECT "histogram_counts"${rangeSel}${dtypeSel}${nameSel2}
             FROM pp_data ${where}
             LIMIT ${INDIV_LIMIT + 1}`
          );
          const iRows    = iResult.toArray();
          const overflow = iRows.length > INDIV_LIMIT;
          indivFiles     = { rows: [], overflow };
          for (const row of iRows.slice(0, INDIV_LIMIT)) {
            const counts = extractBinary(row.histogram_counts);
            if (!counts?.length) continue;
            const total  = histTotal(counts);
            if (total <= 0) continue;
            const aMin   = hasRange ? Number(row.histogram_min) : 0;
            const aMax   = hasRange ? Number(row.histogram_max) : NBINS - 1;
            const dNorm  = DTYPE_NORM[row.dtype];
            indivFiles.rows.push({
              ys: Array.from(counts, c => c / total),
              aMin, aMax,
              nMin: dNorm != null ? aMin / dNorm : aMin,
              nMax: dNorm != null ? aMax / dNorm : aMax,
              kind: hasDtype ? dtypeKind(row.dtype) : 'uint',
              label: hasNames ? String(row.name ?? '').split('/').pop() : '',
            });
          }
        }

        // File overlay — separate query, doesn't pollute the group sample.
        fileOverlay = null;
        if (ctrl.selectedFile && hasRange) {
          const safe       = ctrl.selectedFile.replace(/'/g, "''");
          const fileWhere  = ctx.where ? `${ctx.where} AND "name" = '${safe}'` : `WHERE "name" = '${safe}'`;
          const fileResult = await ctx.query(
            `SELECT "histogram_counts", "histogram_min", "histogram_max"${dtypeSel}
             FROM pp_data ${fileWhere} LIMIT 1`
          );
          const fr = fileResult.toArray()[0];
          if (fr) {
            const counts = extractBinary(fr.histogram_counts);
            if (counts?.length) {
              const total = histTotal(counts);
              const aMin  = Number(fr.histogram_min), aMax = Number(fr.histogram_max);
              const dNorm = DTYPE_NORM[fr.dtype];
              fileOverlay = {
                ys:    Array.from(counts, v => total > 0 ? v / total : 0),
                aMin,  aMax,
                nMin:  dNorm != null ? aMin / dNorm : aMin,
                nMax:  dNorm != null ? aMax / dNorm : aMax,
                kind:  hasDtype ? dtypeKind(fr.dtype) : 'uint',
                label: ctrl.selectedFile.split('/').pop(),
              };
            }
          }
        }
      };

      const renderPanel = (kind) => {
        if (!kindData) return;
        const { plotDiv, toggleEl, labelLeft, labelRight } = panels[kind];
        const useNorm                = (toggleEl?.checked ?? false) && kind !== 'float';
        const spreadChecked          = container.querySelector(`#${SPREAD_ID}`)?.checked ?? false;
        const ctrl                   = readControls(container);
        const activeGroups           = ctrl.selectedGroups.length ? ctrl.selectedGroups : ctx.groups;
        const { append: appendPlot, plotlyLegendConfig } = ctx.plot;

        if (labelLeft) {
          labelLeft.style.fontWeight  = useNorm ? 'normal' : '600';
          labelRight.style.fontWeight = useNorm ? '600' : 'normal';
        }

        plotDiv.innerHTML = '';
        const kd         = kindData[kind] ?? {};
        const visGroups  = activeGroups.filter(g => kd[g]?.count);
        const hasOverlay = fileOverlay?.kind === kind;

        if (!visGroups.length && !hasOverlay) {
          plotDiv.innerHTML = '<div class="no-data" style="color:#aaa;padding:8px 0">No data for the current filter.</div>';
          return;
        }

        const traces = [];
        let yMax = 0;
        let xLo = Infinity, xHi = -Infinity;
        const trackExtent = (lo, hi, ys) => {
          const ext = populatedExtent(lo, hi, ys);
          if (!ext) return;
          if (ext[0] < xLo) xLo = ext[0];
          if (ext[1] > xHi) xHi = ext[1];
        };

        // Individual file traces replace the group mean when active.
        const kindRows  = (indivFiles && visGroups.length === 1)
          ? indivFiles.rows.filter(r => r.kind === kind)
          : [];
        const showIndiv = kindRows.length > 0;

        if (!showIndiv) {
          for (const g of visGroups) {
            const gd        = kd[g];
            const color     = ctx.color.group(g);
            const lo        = useNorm ? gd.nMin : gd.aMin;
            const hi        = useNorm ? gd.nMax : gd.aMax;
            const sumsArr   = useNorm ? gd.sumsNorm  : gd.sums;
            const sumSqArr  = useNorm ? gd.sumSqNorm : gd.sumSq;
            const xs        = binXs(lo, hi);
            const meanYs    = Array.from(sumsArr, v => v / gd.count);
            const showSpread = spreadChecked && gd.count >= 2;

            for (let i = 0; i < NBINS; i++) if (meanYs[i] > yMax) yMax = meanYs[i];
            trackExtent(lo, hi, meanYs);

            if (showSpread) {
              // ±1 std band as a closed polygon behind the mean line.
              const upper = new Array(NBINS), lower = new Array(NBINS);
              for (let i = 0; i < NBINS; i++) {
                const m = meanYs[i];
                const s = Math.sqrt(Math.max(0, sumSqArr[i] / gd.count - m * m));
                upper[i] = m + s;
                lower[i] = Math.max(0, m - s);
              }
              const xRev = xs.slice().reverse();
              traces.push({
                x: [...xs, ...xRev], y: [...upper, ...lower.slice().reverse()],
                fill: 'toself', mode: 'none', line: { width: 0 },
                fillcolor: withAlpha(color, 0.18),
                showlegend: false, hoverinfo: 'skip',
              });
            }

            traces.push({
              type: 'scatter', mode: 'lines', name: ctx.groupLabel(String(g)),
              x: xs, y: meanYs,
              fill: showSpread ? 'none' : 'tozeroy',
              opacity: 0.6, line: { color, width: 2 },
            });
          }
        } else {
          const color = ctx.color.group(visGroups[0]);
          for (const row of kindRows) {
            const lo = useNorm ? row.nMin : row.aMin;
            const hi = useNorm ? row.nMax : row.aMax;
            for (let i = 0; i < row.ys.length; i++) if (row.ys[i] > yMax) yMax = row.ys[i];
            trackExtent(lo, hi, row.ys);
            traces.push({
              type: 'scatter', mode: 'lines',
              x: binXs(lo, hi), y: row.ys,
              name: row.label || 'file',
              line: { color, width: 1.5 },
              opacity: 0.6,
              showlegend: false,
              hovertemplate: `%{y:.4f}<br>${row.label}<extra></extra>`,
            });
          }
        }

        if (hasOverlay) {
          const fo = fileOverlay;
          const lo = useNorm ? fo.nMin : fo.aMin;
          const hi = useNorm ? fo.nMax : fo.aMax;
          const xs = binXs(lo, hi);
          trackExtent(lo, hi, fo.ys);
          traces.unshift({
            type: 'bar', name: `File: ${fo.label}`, x: xs, y: fo.ys,
            width: Array(NBINS).fill(xs.length > 1 ? xs[1] - xs[0] : 1),
            marker: { color: 'black' }, opacity: 0.3,
          });
        }

        const xTitle     = useNorm
          ? (kind === 'int' ? 'Dtype range (−1 to 1)' : 'Dtype range (0 to 1)')
          : 'Pixel value';
        const showLegend = ctx.groups.length > 1 || hasOverlay;
        const yRange     = yMax > 0 ? [0, yMax * 1.12] : undefined;
        // Zoom to where the data actually is instead of the full (possibly dtype-widened) bin window,
        // which would otherwise pad narrow-range integer data out with a wall of empty bins.
        const xPad       = xHi > xLo ? (xHi - xLo) * 0.04 : 0;
        const xRange     = xHi > xLo ? [xLo - xPad, xHi + xPad] : undefined;

        appendPlot(plotDiv, traces, {
          title:  { text: multiKind ? `${KIND_LABEL[kind]} — Intensity Histograms` : 'Intensity Histograms (averaged per group)' },
          xaxis:  { title: xTitle, exponentformat: 'e', ...(xRange ? { range: xRange } : {}) },
          yaxis:  { title: 'Normalized count', ...(yRange ? { range: yRange } : {}) },
          bargap: 0, height: 500, showlegend: showLegend, hovermode: 'x unified',
          ...(showLegend ? { legend: plotlyLegendConfig } : {}),
        });
      };

      const renderPanels = () => { for (const kind of presentKinds) renderPanel(kind); };
      const draw         = async () => { await fetchAndAccumulate(); renderPanels(); };

      // Per-panel mode toggles re-render only their panel; spread and group controls re-render all.
      for (const kind of presentKinds)
        panels[kind].toggleEl?.addEventListener('change', () => renderPanel(kind));
      container.querySelector(`#${SPREAD_ID}`)?.addEventListener('change', renderPanels);
      wireControls(container, draw);

      await draw();

    } catch (e) {
      console.error('Histogram widget error', e);
      container.innerHTML = '<div class="no-data">Failed to load data.</div>';
    }
  },
};

function buildControls(ctx, { hasNames, nameRows }) {
  const { escapeHtml } = ctx.plot;

  const groupOpts = ctx.groups.map(g =>
    `<option value="${escapeHtml(String(g))}">${escapeHtml(ctx.groupLabel(g))}</option>`).join('');

  const fileBlock = hasNames ? `
    <div style="max-width:400px;flex:1 1 240px">
      <div style="font-weight:600;margin-bottom:6px">Overlay specific file (optional):</div>
      <select id="${FILE_SEL_ID}" class="form-select form-select-sm">
        <option value="">- none -</option>
        ${nameRows.map(r => `<option value="${escapeHtml(String(r.name))}">${escapeHtml(String(r.name))}</option>`).join('')}
      </select>
      ${nameRows.length === MAX_FILE_OPTIONS ? `<small class="text-muted">Showing first ${MAX_FILE_OPTIONS} files.</small>` : ''}
    </div>` : '';

  const el = document.createElement('div');
  el.style.cssText = 'display:flex;flex-wrap:wrap;gap:20px;margin-bottom:20px';
  el.innerHTML = `
    <div style="max-width:360px;flex:1 1 240px">
      <div style="font-weight:600;margin-bottom:6px">Select groups (optional):</div>
      <select id="${GROUP_SEL_ID}" class="form-select form-select-sm" multiple style="height:80px">
        <option value="" selected>All groups</option>
        ${groupOpts}
      </select>
      <small class="text-muted">Ctrl/Cmd to multi-select.</small>
    </div>
    ${fileBlock}
    <div style="align-self:flex-start">
      <div class="form-check" style="margin-bottom:6px">
        <input class="form-check-input" type="checkbox" id="${SPREAD_ID}">
        <label class="form-check-label" for="${SPREAD_ID}" style="font-size:0.9em">
          Show spread (±1 std)
        </label>
      </div>
      <div id="hist-indiv-wrap" style="display:none;margin-bottom:8px">
        <div class="form-check">
          <input class="form-check-input" type="checkbox" id="${INDIV_ID}">
          <label class="form-check-label" for="${INDIV_ID}" style="font-size:0.9em">
            Show individual images
          </label>
        </div>
      </div>
      <label style="font-weight:600">Max samples per group:
        <input type="number" id="${SAMPLE_INPUT_ID}" value="${DEFAULT_SAMPLES}" min="50" step="100"
               style="width:90px;margin-left:8px;padding:2px 6px;border:1px solid #ccc;border-radius:4px">
      </label>
    </div>
  `;
  return el;
}

function readControls(container) {
  const groupSel  = container.querySelector(`#${GROUP_SEL_ID}`);
  const sampleRaw = container.querySelector(`#${SAMPLE_INPUT_ID}`)?.value ?? DEFAULT_SAMPLES;
  // value='' is the "All groups" option — filter(Boolean) drops it, leaving [] = all groups.
  const selectedGroups = groupSel
    ? [...groupSel.selectedOptions].map(o => o.value).filter(Boolean)
    : [];
  return {
    selectedGroups,
    selectedFile:    container.querySelector(`#${FILE_SEL_ID}`)?.value ?? '',
    samplesPerGroup: Math.max(50, parseInt(sampleRaw, 10) || DEFAULT_SAMPLES),
  };
}

function wireControls(container, onDraw) {
  // Group select: enforce "All groups" mutual-exclusivity before re-fetching.
  const groupSel = container.querySelector(`#${GROUP_SEL_ID}`);
  if (groupSel) {
    groupSel.addEventListener('change', () => {
      const allOpt   = groupSel.options[0];
      const specific = [...groupSel.options].slice(1).filter(o => o.selected);
      if (allOpt.selected && specific.length > 0) {
        allOpt.selected = false;       // specific selected alongside All → drop All
      } else if (!allOpt.selected && specific.length === 0) {
        allOpt.selected = true;        // nothing selected → revert to All
      }
      onDraw();
    });
  }
  for (const id of [FILE_SEL_ID, SAMPLE_INPUT_ID])
    container.querySelector(`#${id}`)?.addEventListener('change', onDraw);

  // Individual-image lines replace the group mean, so "spread around the mean" is meaningless then.
  const indivEl  = container.querySelector(`#${INDIV_ID}`);
  const spreadEl = container.querySelector(`#${SPREAD_ID}`);
  indivEl?.addEventListener('change', () => {
    if (spreadEl) spreadEl.disabled = indivEl.checked;
    onDraw();
  });
}
