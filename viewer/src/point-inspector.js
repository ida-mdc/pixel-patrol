/**
 * Point-inspector drawer: clicking a plot point or mosaic thumbnail (carrying a
 * customdata file_row_number) opens a side drawer for that image - header,
 * acquisition metadata, per-group metrics, and intensity histograms. The
 * metadata and metric lists come from the schema. Data via DuckDB (ctx.query).
 */

import { escapeHtml, niceName, accent } from './plot-utils.js';
import { onPointClick, setSelectedPoint } from './point-selection.js';
import { drawThumbnailRGBA, SPRITE } from './exhibit.js';
import { META_COLS, NON_USER_FACING_NUMERIC_COLS } from './schema.js';
import { DATE_COLS, DATE_FMT } from './constants.js';

// Kept out of the Acquisition block: header fields, long-format infra, and the
// shared non-user-facing numeric columns. Everything else scalar is metadata.
const NON_ACQUISITION_COLS = new Set([
  'name', 'path', 'type', 'obs_level',
  ...NON_USER_FACING_NUMERIC_COLS,
]);
const MUTED = '#898781';

let drawer = null, bodyEl = null;
const DRAWER_W = 432;

// Clicking any registered plot point (see point-selection.js) opens the drawer.
onPointClick(({ fileRowNumber, ctx, metric }) => openInspector(fileRowNumber, ctx, { metric }));

// On a wide screen, shrink the report (padding on #main-app) so the drawer takes
// width and the report re-centers in the remainder instead of being covered.
// On narrow screens the drawer overlays (there isn't room to give away).
function pushMain(open) {
  const app = document.getElementById('main-app');
  if (!app) return;
  const wide = window.innerWidth >= 1100;
  app.style.transition = 'padding-right 0.2s ease';
  app.style.paddingRight = (open && wide) ? `${DRAWER_W}px` : '';
  window.dispatchEvent(new Event('resize'));
  setTimeout(() => window.dispatchEvent(new Event('resize')), 240);
}

function ensureDrawer() {
  if (drawer) return;
  drawer = document.createElement('aside');
  drawer.className = 'point-inspector';
  drawer.style.cssText =
    'position:fixed;top:0;right:0;height:100vh;width:432px;max-width:94vw;z-index:9998;' +
    'background:var(--card-bg,#fff);color:var(--text,#222);box-shadow:-2px 0 16px rgba(0,0,0,0.22);' +
    'transform:translateX(100%);transition:transform 0.2s ease;display:flex;flex-direction:column;' +
    'font-size:13px;overflow:hidden;';

  const header = document.createElement('div');
  header.style.cssText =
    'display:flex;align-items:flex-start;justify-content:space-between;gap:8px;' +
    'padding:14px 18px;border-bottom:1px solid rgba(128,128,128,0.25);flex:0 0 auto;';
  header.innerHTML = '<div id="pi-title"></div>';
  const close = document.createElement('button');
  close.textContent = '✕';
  close.title = 'Close';
  close.style.cssText = 'border:none;background:none;cursor:pointer;font-size:16px;color:inherit;line-height:1;flex:0 0 auto;';
  close.addEventListener('click', closeInspector);
  header.appendChild(close);

  bodyEl = document.createElement('div');
  bodyEl.style.cssText = 'padding:18px;overflow:auto;flex:1 1 auto;display:flex;flex-direction:column;gap:22px;';

  drawer.append(header, bodyEl);
  document.body.appendChild(drawer);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeInspector(); });
}

export function closeInspector() {
  if (drawer) drawer.style.transform = 'translateX(100%)';
  pushMain(false);
  setSelectedPoint(null);
}

// ── formatting ───────────────────────────────────────────────────────────────
function fmt(n) {
  if (n == null || n === '') return '—';
  const x = Number(n);
  if (!isFinite(x)) return String(n);
  if (x !== 0 && (Math.abs(x) >= 1e6 || Math.abs(x) < 1e-3)) return x.toExponential(2);
  if (Number.isInteger(x)) return x.toLocaleString();
  return String(Number(x.toPrecision(4)));
}
function bytesFmt(b) {
  b = Number(b); if (!isFinite(b)) return '—';
  const u = ['B', 'KB', 'MB', 'GB']; let i = 0;
  while (b >= 1024 && i < 3) { b /= 1024; i++; }
  return (i ? b.toFixed(b < 10 ? 1 : 0) : b) + ' ' + u[i];
}

// ── arrow helpers ────────────────────────────────────────────────────────────
function getCol(table, name) {
  const fields = table?.schema?.fields ?? [];
  for (let i = 0; i < fields.length; i++) {
    if (fields[i]?.name === name) return typeof table.getChildAt === 'function' ? table.getChildAt(i) : null;
  }
  return null;
}
const sqlStr = (s) => `'${String(s).replace(/'/g, "''")}'`;

// Row SELECT that STRFTIMEs date columns to strings (like the rest of the report),
// passing everything else through.
function rowSelect(ctx) {
  const dateCols = (ctx.schema?.allCols ?? []).filter(c => DATE_COLS.has(c));
  if (!dateCols.length) return '*';
  const excluded = dateCols.map(c => `"${c}"`).join(', ');
  const formatted = dateCols.map(c => `STRFTIME("${c}", ${DATE_FMT}) AS "${c}"`).join(', ');
  return `* EXCLUDE (${excluded}), ${formatted}`;
}

// ── histogram SVG ────────────────────────────────────────────────────────────
// For integer images, don't draw sub-integer bins: merge the stored 256 bins
// down to unit-width bins (one per integer value) when the value range is < 256.
function binForDisplay(hist, mn, mx, dtype) {
  if (!hist || !hist.length) return hist;
  if (!/int/i.test(String(dtype || ''))) return hist;   // uint8/int16/… ; floats keep 256
  const R = Number(mx) - Number(mn);
  if (!isFinite(R) || R <= 0) return hist;
  const w = R / hist.length;
  if (w >= 1) return hist;                                // stored bins already ≥ 1 wide
  const n = Math.max(1, Math.floor(R) + 1);
  if (n >= hist.length) return hist;
  const out = new Array(n).fill(0);
  for (let k = 0; k < hist.length; k++) out[Math.min(n - 1, Math.floor((k + 0.5) * w))] += hist[k];
  return out;
}

function barRects(hist, W, H) {
  const n = hist.length, max = Math.max(...hist.map(v => Math.log1p(v))) || 1, bw = W / n;
  let s = '';
  for (let i = 0; i < n; i++) {
    const h = Math.log1p(hist[i]) / max * (H - 3);
    if (h <= 0.2) continue;
    s += `<rect x="${(i * bw).toFixed(2)}" y="${(H - h).toFixed(2)}" width="${Math.max(0.6, bw * 0.85).toFixed(2)}" height="${h.toFixed(2)}"/>`;
  }
  return s;
}
function histSVG(hist, mn, mx, col) {
  const W = 396, H = 76;
  return `<svg viewBox="0 0 ${W} ${H + 16}" style="width:100%;display:block;font:10px ui-monospace,monospace">`
    + `<g fill="${col}" fill-opacity="0.9">${barRects(hist, W, H)}</g>`
    + `<text x="0" y="${H + 12}" fill="${MUTED}" text-anchor="start">${escapeHtml(fmt(mn))}</text>`
    + `<text x="${W}" y="${H + 12}" fill="${MUTED}" text-anchor="end">${escapeHtml(fmt(mx))}</text></svg>`;
}
function miniHist(hist, col) {
  const W = 120, H = 30;
  return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;display:block"><g fill="${col}" fill-opacity="0.9">${barRects(hist, W, H)}</g></svg>`;
}

// ── data fetchers ────────────────────────────────────────────────────────────
async function fetchArrayCol(ctx, sql, col) {
  const table = await ctx.query(sql);
  const c = getCol(table, col);
  const out = [];
  for (let i = 0; i < Number(table.numRows ?? 0); i++) out.push(ctx.data.extractBinary(c?.get(i) ?? null));
  return { table, values: out };
}

/** Per-channel intensity histograms for the same image (channel marginal of the tree). */
async function fetchChannelHists(ctx, row) {
  const dimCols = ctx.schema?.dimCols ?? [];
  if (!dimCols.includes('dim_c') || !(Number(row.size_C) > 1) || row.path == null) return [];
  const others = dimCols.filter(c => c !== 'dim_c').map(c => `"${c}" IS NULL`).join(' AND ');
  const where = `path = ${sqlStr(row.path)} AND "dim_c" IS NOT NULL${others ? ' AND ' + others : ''}`;
  const table = await ctx.query(
    `SELECT "dim_c", histogram_counts, histogram_min, histogram_max FROM pp_all WHERE ${where} ORDER BY "dim_c"`);
  const dc = getCol(table, 'dim_c'), hc = getCol(table, 'histogram_counts'),
        mn = getCol(table, 'histogram_min'), mx = getCol(table, 'histogram_max');
  const out = [];
  for (let i = 0; i < Number(table.numRows ?? 0); i++) {
    const h = ctx.data.extractBinary(hc?.get(i) ?? null);
    if (h && h.length) out.push({ idx: Number(dc.get(i)), hist: h, min: mn?.get(i), max: mx?.get(i) });
  }
  return out;
}

// For each metricCols value present on this image: the group min / median / max
// and this image's percentile.
async function fetchRefs(ctx, row) {
  const metricCols = ctx.schema?.metricCols ?? [];
  const present = metricCols.filter(k => row[k] != null && isFinite(Number(row[k])));
  if (!present.length) return {};
  const q = ctx.sql.q;
  const sel = present.map((k, i) => {
    const c = q(k), v = Number(row[k]);
    return `min(${c}) AS mn${i}, approx_quantile(${c},0.5) AS md${i}, max(${c}) AS mx${i}, ` +
           `(COUNT(*) FILTER (WHERE ${c} < ${v}))::DOUBLE / NULLIF(COUNT(${c}) - 1, 0) AS pc${i}`;
  }).join(', ');
  // Compare within the image's own group when a grouping is active, else the whole set.
  let where = 'obs_level = 0';
  if (ctx.state?.groupCol && row[ctx.state.groupCol] != null) {
    where += ` AND ${q(ctx.state.groupCol)} = ${sqlStr(row[ctx.state.groupCol])}`;
  }
  const [r] = await ctx.queryRows(`SELECT ${sel} FROM pp_all WHERE ${where}`);
  const refs = {};
  present.forEach((k, i) => {
    refs[k] = { val: Number(row[k]), min: Number(r['mn' + i]), med: Number(r['md' + i]), max: Number(r['mx' + i]),
                pct: r['pc' + i] == null ? 0.5 : Number(r['pc' + i]) };
  });
  return refs;
}

// ── section builders ─────────────────────────────────────────────────────────
function heading(text) { return `<div style="font:600 10.5px ui-monospace,monospace;letter-spacing:.09em;text-transform:uppercase;color:${MUTED};margin-bottom:9px">${escapeHtml(text)}</div>`; }

// Metadata columns for this image: present scalars that aren't metrics, blobs or
// header/infra. META_COLS lead in order, then extras alphabetically.
export function acquisitionCols(row, ctx) {
  const metrics = new Set(ctx.schema?.metricCols ?? []);
  const blobs   = new Set(ctx.schema?.blobCols ?? []);
  const all     = ctx.schema?.allCols ?? Object.keys(row);
  const has = (c) => !metrics.has(c) && !blobs.has(c) && !NON_ACQUISITION_COLS.has(c)
    && row[c] != null && row[c] !== '';
  const lead   = META_COLS.filter(c => all.includes(c) && has(c));
  const extras = all.filter(c => !META_COLS.includes(c) && has(c)).sort();
  return [...lead, ...extras];
}

// Light per-column formatting, else generic number / string rendering.
export function metaValue(col, v) {
  if (col === 'size_bytes') return bytesFmt(v);
  if (col === 'file_extension') return String(v).toUpperCase();
  const n = Number(v);
  return (typeof v !== 'string' && isFinite(n)) ? fmt(n) : String(v);
}

function acquisitionSection(row, ctx) {
  const cols = acquisitionCols(row, ctx);
  if (!cols.length) return '';
  const dl = cols.map(c =>
    `<dt style="color:${MUTED}">${escapeHtml(niceName(c))}</dt>` +
    `<dd style="margin:0;text-align:right;font-family:ui-monospace,monospace;font-variant-numeric:tabular-nums;word-break:break-word">${escapeHtml(metaValue(c, row[c]))}</dd>`).join('');
  return `<div>${heading('Acquisition')}<dl style="display:grid;grid-template-columns:auto 1fr;gap:7px 16px;margin:0;font-size:12.5px">${dl}</dl></div>`;
}

// A small raw-image thumbnail that sits beside the file name in the header.
function headerThumb(thumbBytes) {
  if (!thumbBytes || thumbBytes.length < SPRITE * SPRITE) return null;
  const cv = document.createElement('canvas'); cv.width = SPRITE; cv.height = SPRITE;
  cv.style.cssText = 'width:60px;height:60px;image-rendering:pixelated;border-radius:8px;border:1px solid rgba(128,128,128,0.25);background:rgba(128,128,128,0.12);display:block';
  drawThumbnailRGBA(cv.getContext('2d'), thumbBytes);
  return cv;
}

// Group metric keys by producing processor, in first-appearance order, with the
// unknown-producer group ('') last. Returns [[producer, keys], ...].
export function groupMetricsByProducer(keys, producerByCol = {}) {
  const groups = new Map();
  for (const k of keys) {
    const g = producerByCol[k] || '';
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(k);
  }
  const ordered = [...groups.entries()].filter(([g]) => g !== '');
  if (groups.has('')) ordered.push(['', groups.get('')]);
  return ordered;
}

function metricGroupLabel(name) {
  const el = document.createElement('div');
  el.textContent = name === '' ? 'Other' : niceName(name.replace(/-/g, ' '));
  el.style.cssText = `font:600 10px ui-monospace,monospace;letter-spacing:.06em;color:${MUTED};margin:2px 0 7px`;
  return el;
}

function metricCard(k, r, clickedMetric) {
  const p = Math.max(0, Math.min(1, r.pct)) * 100, hit = k === clickedMetric;
  const el = document.createElement('div');
  el.style.cssText = 'padding:8px 10px;border-radius:10px;'
    + `border:1px solid ${hit ? accent() : 'rgba(128,128,128,0.22)'};`
    + `background:${hit ? `color-mix(in srgb, ${accent()} 10%, transparent)` : 'transparent'}`;
  el.innerHTML =
    '<div style="display:flex;justify-content:space-between;gap:8px;align-items:baseline">'
    + `<span style="font-size:12px">${escapeHtml(niceName(k))}</span>`
    + `<span style="font:600 13px ui-monospace,monospace;font-variant-numeric:tabular-nums">${escapeHtml(fmt(r.val))}</span></div>`
    + `<div title="${p < 50 ? 'below' : 'above'} the group median" style="position:relative;height:4px;border-radius:2px;background:rgba(128,128,128,0.25);margin-top:8px">`
    + `<span style="position:absolute;top:-2px;left:50%;width:2px;height:8px;background:${MUTED};transform:translateX(-50%)"></span>`
    + `<i style="position:absolute;top:-3px;left:${p.toFixed(0)}%;width:9px;height:9px;border-radius:50%;background:${accent()};transform:translateX(-50%);box-shadow:0 0 0 2px var(--card-bg,#fff)"></i></div>`
    + `<div style="display:flex;justify-content:space-between;font:9px ui-monospace,monospace;color:${MUTED};margin-top:5px;font-variant-numeric:tabular-nums"><span>${escapeHtml(fmt(r.min))}</span><span>${escapeHtml(fmt(r.max))}</span></div>`;
  return el;
}

// Metric cards showing where the image sits in its group, grouped by producing
// processor when that info is available. The clicked metric is outlined.
function metricsElement(refs, clickedMetric, producerByCol = {}) {
  const wrap = document.createElement('div');
  const grouped = groupMetricsByProducer(Object.keys(refs), producerByCol);
  if (!grouped.length) return wrap;
  wrap.insertAdjacentHTML('beforeend', heading('Metrics · where this image sits in its group'));
  const labelled = grouped.length > 1;
  for (const [producer, keys] of grouped) {
    if (labelled) wrap.appendChild(metricGroupLabel(producer));
    const list = document.createElement('div');
    list.style.cssText = `display:grid;grid-template-columns:1fr 1fr;gap:8px${labelled ? ';margin-bottom:12px' : ''}`;
    for (const k of keys) list.appendChild(metricCard(k, refs[k], clickedMetric));
    wrap.appendChild(list);
  }
  return wrap;
}

function histogramsSection(ctx, row, wholeHist, wholeMin, wholeMax, channels) {
  const col = accent();
  let html = `<div>${heading('Intensity — whole image')}`;
  html += wholeHist && wholeHist.length ? histSVG(binForDisplay(wholeHist, wholeMin, wholeMax, row.dtype), wholeMin, wholeMax, col)
    : `<div style="color:${MUTED};font-size:12px">No histogram for this row.</div>`;
  if (channels.length) {
    const ncol = Math.min(channels.length, channels.length > 6 ? 5 : 3);
    const cells = channels.map(c =>
      `<div style="display:flex;flex-direction:column;gap:2px"><div style="font:9.5px ui-monospace,monospace;color:${MUTED}">C${c.idx}</div>${miniHist(binForDisplay(c.hist, c.min, c.max, row.dtype), col)}</div>`).join('');
    html += `<div style="margin-top:15px">${heading('Per channel')}<div style="display:grid;grid-template-columns:repeat(${ncol},1fr);gap:9px 10px">${cells}</div></div>`;
  }
  return html + '</div>';
}

/**
 * Open the drawer for one row.
 * @param {number} fileRowNumber
 * @param {object} ctx  plugin ctx (needs ctx.query / ctx.queryRows).
 */
export async function openInspector(fileRowNumber, ctx, opts = {}) {
  ensureDrawer();
  drawer.style.transform = 'translateX(0)';
  pushMain(true);
  setSelectedPoint(Number(fileRowNumber));
  bodyEl.textContent = 'Loading…';

  let row;
  try {
    [row] = await ctx.queryRows(`SELECT ${rowSelect(ctx)} FROM pp_all WHERE file_row_number = ${Number(fileRowNumber)} LIMIT 1`);
  } catch (err) { bodyEl.textContent = `Could not load point: ${err.message}`; return; }
  if (!row) { bodyEl.textContent = 'Point not found.'; return; }

  // Subtitle stays a plain "what does this row cover" label; the dimensions and
  // dtype live once in the Acquisition block below, not repeated here.
  const kind = row.type === 'sub_file' ? 'sub-image' : 'image';
  const pinned = String(row.dim_order || '').split('')
    .map(a => ({ axis: a.toUpperCase(), o: row[`dim_${a.toLowerCase()}`] })).filter(d => d.o != null);
  const sub = pinned.length ? `${pinned.map(d => `${d.axis}=${d.o}`).join(', ')} slice of ${kind}` : `Whole ${kind}`;
  const titleEl = document.getElementById('pi-title');
  titleEl.style.cssText = 'display:flex;gap:12px;align-items:flex-start;flex:1 1 auto;min-width:0';
  titleEl.innerHTML =
    '<div id="pi-thumb" style="flex:0 0 auto"></div>' +
    '<div style="flex:1 1 auto;min-width:0">' +
      `<div style="font-size:16px;font-weight:640;letter-spacing:-.01em;word-break:break-word">${escapeHtml(String(row.name ?? row.path ?? ''))}</div>` +
      `<div style="color:${MUTED};font:11.5px ui-monospace,monospace;margin-top:3px;letter-spacing:.03em">${escapeHtml(sub)}</div>` +
    '</div>';

  const clickedMetric = opts.metric || null;
  const wthumb = ctx.schema?.blobCols?.includes('thumbnail');
  const blobQ = wthumb
    ? ctx.query(`SELECT "thumbnail" FROM pp_all WHERE file_row_number = ${Number(fileRowNumber)} LIMIT 1`)
    : Promise.resolve(null);
  const wsql = `SELECT histogram_counts, histogram_min, histogram_max FROM pp_all WHERE file_row_number = ${Number(fileRowNumber)} LIMIT 1`;

  let blobTable = null, channels = [], refs = {}, whole = { table: null, values: [null] }, wmin, wmax;
  try {
    [blobTable, channels, refs, whole] = await Promise.all([
      blobQ, fetchChannelHists(ctx, row), fetchRefs(ctx, row), fetchArrayCol(ctx, wsql, 'histogram_counts'),
    ]);
    wmin = getCol(whole.table, 'histogram_min')?.get(0);
    wmax = getCol(whole.table, 'histogram_max')?.get(0);
  } catch (err) { console.warn('[viewer] inspector data load failed:', err); }

  const thumbBytes = blobTable ? ctx.data.extractBinary(getCol(blobTable, 'thumbnail')?.get(0) ?? null) : null;
  const thumb = headerThumb(thumbBytes);
  if (thumb) document.getElementById('pi-thumb').appendChild(thumb);

  bodyEl.textContent = '';
  bodyEl.insertAdjacentHTML('beforeend', acquisitionSection(row, ctx));
  bodyEl.appendChild(metricsElement(refs, clickedMetric, ctx.schema?.producerByCol));
  bodyEl.insertAdjacentHTML('beforeend', histogramsSection(ctx, row, whole.values[0], wmin, wmax, channels));
}
