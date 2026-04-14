import { q, groupExpr } from '../sql.js';
import { groupColor } from '../colors.js';
import { escapeHtml } from '../plot-utils.js';

// Matches thumbnail generation (pixel_patrol_base ThumbnailProcessor): SPRITE_SIZE=64.
// (Older report UI code used 32px for display, but the stored bytes are 64×64×4.)
const SPRITE   = 64;
const BORDER   = 2;
const GAP      = 3; // tiny margin between tiles
const CELL     = SPRITE + BORDER * 2 + GAP;
// Limit thumbnails for performance (client-side sprite + Plotly render).
const MAX_IMGS = 256;

const SORT_COL_ID = 'mosaic-sort-select';
const SORT_DIR_ID = 'mosaic-sort-dir';
const PICK_ID          = 'mosaic-pick-mode';
const PICK_SAMPLE_ALL  = 'sample_all';
const PICK_TOP_ALL     = 'top_all';
const DISPLAY_ID  = 'mosaic-display-mode';
const DISPLAY_NORM = 'normalized';
const DISPLAY_DENORM = 'denormalized';

/**
 * Image mosaic plugin — matches Dash ImageMosaikWidget behaviour:
 *   - Canvas sprite (fast) with hover tooltip
 *   - Coloured group borders
 *   - Sort by metric dropdown
 *   - Up to MAX_IMGS thumbnails
 */
export default {
  id: 'mosaic',
  label: 'Image Mosaic',
  info: [
    'Displays an **image mosaic**, one thumbnail per file.',
    '',
    '- Thumbnails are generated from the central slice in all non-XY dimensions.',
    '- Sorting by a measurement (e.g. mean, min, max) can reveal visual trends.',
    '- Border colors indicate the group of each image.',
    '- **Hover** over an image to see its filename.',
  ].join('\n'),

  requires(schema) {
    const hasMetric = schema.metricCols.some(c => schema.allCols.includes(c));
    return schema.blobCols.includes('thumbnail') && hasMetric;
  },

  async render(container, ctx) {
    // ── Sort control ──────────────────────────────────────────────────────
    const sortableMetrics = ctx.schema.metricCols.filter(
      c => ctx.schema.allCols.includes(c),
    );

    const hasNormCols =
      ctx.schema.allCols.includes('thumbnail_norm_min') &&
      ctx.schema.allCols.includes('thumbnail_norm_max') &&
      ctx.schema.allCols.includes('thumbnail_dtype');

    const defaultSort = sortableMetrics.includes('mean_intensity')
      ? 'mean_intensity'
      : (sortableMetrics[0] ?? '');

    const controlRow = document.createElement('div');
    controlRow.style.cssText = 'display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap';
    controlRow.innerHTML = `
      <label for="${PICK_ID}" style="white-space:nowrap;font-weight:500;font-size:14px">
        Pick:
      </label>
      <select id="${PICK_ID}" class="form-select form-select-sm" style="max-width:220px">
        <option value="${PICK_TOP_ALL}" selected>Top ${MAX_IMGS} by metric</option>
        <option value="${PICK_SAMPLE_ALL}">Random ${MAX_IMGS}</option>
      </select>
      <label for="${SORT_COL_ID}" style="white-space:nowrap;font-weight:500;font-size:14px">
        Sort mosaic by:
      </label>
      <select id="${SORT_COL_ID}" class="form-select form-select-sm" style="max-width:260px">
        ${sortableMetrics.map(c => `<option value="${c}"${c === defaultSort ? ' selected' : ''}>${c}</option>`).join('')}
      </select>
      <select id="${SORT_DIR_ID}" class="form-select form-select-sm" style="max-width:160px">
        <option value="asc">Ascending</option>
        <option value="desc" selected>Descending</option>
      </select>
      ${hasNormCols ? `
        <div style="display:flex;align-items:center;gap:8px;margin-left:8px">
          <span style="white-space:nowrap;font-weight:500;font-size:14px">Display:</span>
          <label style="display:flex;align-items:center;gap:6px;margin:0">
            <input type="radio" name="${DISPLAY_ID}" value="${DISPLAY_NORM}" checked />
            <span style="font-size:14px">Normalised</span>
          </label>
          <label style="display:flex;align-items:center;gap:6px;margin:0">
            <input type="radio" name="${DISPLAY_ID}" value="${DISPLAY_DENORM}" />
            <span style="font-size:14px">Denormalised</span>
          </label>
        </div>
      ` : ``}
    `;
    container.appendChild(controlRow);

    const plotContainer = document.createElement('div');
    container.appendChild(plotContainer);

    // Render immediately, then re-render when sort changes.
    const getDisplayMode = () => {
      if (!hasNormCols) return DISPLAY_NORM;
      const checked = controlRow.querySelector(`input[name="${DISPLAY_ID}"]:checked`);
      return checked?.value || DISPLAY_NORM;
    };

    const getPickMode = () => document.getElementById(PICK_ID)?.value || PICK_TOP_ALL;
    const getSortDir  = () => document.getElementById(SORT_DIR_ID)?.value || 'desc';

    // Show a spinner and load mosaic in the background so other widgets aren't blocked.
    const _loadMosaic = (col, dm, opts) => {
      plotContainer.innerHTML = '<div class="text-center p-4">'
        + '<div class="spinner-border spinner-border-sm text-secondary me-2" role="status"></div>'
        + '<span class="text-muted small">Loading thumbnails…</span></div>';
      return renderMosaic(plotContainer, ctx, col, dm, opts).catch(err => {
        plotContainer.innerHTML = `<div class="text-danger small p-2">${err.message}</div>`;
        console.error('[mosaic] render error:', err);
      });
    };

    // Re-render with current control values. Used by all change handlers.
    const refresh = () => _loadMosaic(
      document.getElementById(SORT_COL_ID).value,
      getDisplayMode(),
      { pickMode: getPickMode(), sortDir: getSortDir() },
    );

    // Non-blocking initial render: render() returns here, other plugins continue.
    _loadMosaic(defaultSort, getDisplayMode(), { pickMode: getPickMode(), sortDir: getSortDir() });

    document.getElementById(SORT_COL_ID).onchange = refresh;
    document.getElementById(SORT_DIR_ID).onchange = refresh;
    document.getElementById(PICK_ID).onchange     = refresh;

    if (hasNormCols) {
      for (const el of controlRow.querySelectorAll(`input[name="${DISPLAY_ID}"]`)) {
        el.onchange = refresh;
      }
    }
  },
};

async function renderMosaic(container, ctx, sortCol, displayMode = DISPLAY_NORM, { pickMode = PICK_TOP_ALL, sortDir = 'desc' } = {}) {
  container.innerHTML = '';

  const t0 = performance.now();
  let pickIdsMs = 0;
  let thumbnailsMs = 0;

  const sortableMetrics = ctx.schema.metricCols.filter(c => ctx.schema.allCols.includes(c));
  const defaultMetric   = sortableMetrics[0] ?? null;
  const metricCol       = (sortCol && ctx.schema.allCols.includes(sortCol))
    ? sortCol
    : defaultMetric;

  if (String(pickMode) === PICK_TOP_ALL && !metricCol) {
    container.innerHTML = '<div class="no-data">No numeric metric column for ordering.</div>';
    return;
  }

  // ── Query setup ───────────────────────────────────────────────────────────
  const hoverCol = ctx.schema.allCols.includes('name') ? 'name'
    : ctx.schema.allCols.includes('imported_path_short') ? 'imported_path_short'
    : null;

  const gcSel    = groupExpr(ctx.state);
  const hoverSel = hoverCol ? `, ${q(hoverCol)} AS __label__` : '';
  const sortSel  = metricCol ? `, ${q(metricCol)} AS __sort__` : '';
  const normSel  =
    ctx.schema.allCols.includes('thumbnail_norm_min') &&
    ctx.schema.allCols.includes('thumbnail_norm_max') &&
    ctx.schema.allCols.includes('thumbnail_dtype')
      ? `, "thumbnail_norm_min" AS __tn_min__, "thumbnail_norm_max" AS __tn_max__, "thumbnail_dtype" AS __tn_dtype__`
      : '';

  const dir  = (String(sortDir).toLowerCase() === 'desc') ? 'DESC' : 'ASC';
  const mode = String(pickMode);
  if (mode !== PICK_TOP_ALL && mode !== PICK_SAMPLE_ALL) {
    container.innerHTML = '<div class="no-data">Invalid mosaic mode.</div>';
    return;
  }

  const needMetricInRows = Boolean(metricCol);

  // Stable row id: export column or DuckDB virtual file_row_number.
  const idCol =
    ctx.schema.rowIdColumn ??
    (ctx.schema.allCols.includes('row_index') ? 'row_index' : null);

  // ── Fetch items ────────────────────────────────────────────────────────────
  let items;

  if (idCol) {
    // ── 2-step path: pick ids, then join pp_data for thumbnails ──────────
    let idRows;
    const tPick0 = performance.now();
    if (mode === PICK_TOP_ALL) {
      idRows = await ctx.queryRows(`
        SELECT ${rowIdSql(idCol)} AS __rid__
        FROM pp_data ${ctx.where}
        ORDER BY ${q(metricCol)} ${dir} NULLS LAST
        LIMIT ${MAX_IMGS}
      `);
    } else {
      idRows = await ctx.queryRows(`
        SELECT ${rowIdSql(idCol)} AS __rid__
        FROM pp_data ${ctx.where}
        USING SAMPLE ${MAX_IMGS} ROWS (reservoir, 42)
      `);
    }
    pickIdsMs = performance.now() - tPick0;

    const rids = (idRows ?? []).map(r => Number(r.__rid__)).filter(Number.isFinite);
    if (!rids.length) {
      container.innerHTML = '<div class="no-data">No thumbnail data.</div>';
      return;
    }

    const idsList = rids.join(', ');
    const sql = `
      WITH ids AS (
        SELECT * FROM UNNEST([${idsList}]) WITH ORDINALITY AS t(rid, ord)
      )
      SELECT
        ids.ord AS __ord__,
        ${gcSel},
        "thumbnail"${hoverSel}${sortSel}${normSel}
      FROM pp_data
      JOIN ids ON ${rowIdSql(idCol)} = ids.rid
      ORDER BY ids.ord
    `;

    const tBlob0 = performance.now();
    const result = await ctx.query(sql);
    thumbnailsMs = performance.now() - tBlob0;

    const numRows = Number(result?.numRows ?? 0);
    if (!numRows) {
      container.innerHTML = '<div class="no-data">No thumbnail data.</div>';
      return;
    }
    items = tableToItems(result, { hoverCol, hasSort: needMetricInRows, hasNorm: Boolean(normSel) });

  } else {
    // ── Fallback: single query (no stable row id available) ───────────────
    let sql;
    if (mode === PICK_TOP_ALL) {
      sql = `
        SELECT ${gcSel}, "thumbnail"${hoverSel}${sortSel}${normSel}
        FROM pp_data ${ctx.where}
        ORDER BY ${q(metricCol)} ${dir} NULLS LAST
        LIMIT ${MAX_IMGS}
      `;
    } else {
      sql = `
        SELECT ${gcSel}, "thumbnail"${hoverSel}${sortSel}${normSel}
        FROM pp_data ${ctx.where}
        USING SAMPLE ${MAX_IMGS} ROWS (reservoir, 42)
      `;
    }

    const tBlob0 = performance.now();
    const result = await ctx.query(sql);
    thumbnailsMs = performance.now() - tBlob0;

    const numRows = Number(result?.numRows ?? 0);
    if (!numRows) {
      container.innerHTML = '<div class="no-data">No thumbnail data.</div>';
      return;
    }
    items = tableToItems(result, { hoverCol, hasSort: needMetricInRows, hasNorm: Boolean(normSel) });
  }

  const t1 = performance.now();

  if (!items?.length) {
    container.innerHTML = '<div class="no-data">No thumbnail data.</div>';
    return;
  }

  // Restore order for random sample + optional metric sort.
  if (mode === PICK_SAMPLE_ALL && metricCol) {
    const dirMul = (String(sortDir).toLowerCase() === 'desc') ? -1 : 1;
    items.sort((a, b) => dirMul * ((a.sort ?? 0) - (b.sort ?? 0)));
  }

  const t2 = performance.now();

  // ── Build sprite on canvas ────────────────────────────────────────────────
  const n      = items.length;
  const perRow = Math.ceil(Math.sqrt(n));
  const nRows  = Math.ceil(n / perRow);
  const W      = perRow * CELL;
  const H      = nRows  * CELL;

  const canvas = makeCanvas(W, H);
  const ctx2d  = canvas.getContext('2d');
  canvas.style.imageRendering = 'pixelated';
  canvas.style.maxWidth = '100%';

  // Keep the mosaic background truly transparent (alpha=0) so padding pixels
  // are actually transparent in Plotly.
  ctx2d.clearRect(0, 0, W, H);

  for (let i = 0; i < items.length; i++) {
    const col    = i % perRow;
    const row    = Math.floor(i / perRow);
    const x      = col * CELL;
    const y      = row * CELL;
    const tile   = CELL - GAP;
    const half   = tile / 2;

    const gColor = groupColor(ctx.colorMap, items[i].group);
    // Border-only coloring: draw a colored ring, keep the interior background dark.
    ctx2d.fillStyle = gColor;
    ctx2d.fillRect(x, y, tile, tile);
    // Make the inner area transparent (not black).
    ctx2d.clearRect(x + BORDER, y + BORDER, SPRITE, SPRITE);

    // Pass the group color so transparent padding shows group color,
    // matching Dash's PIL paste(..., mask=alpha) behaviour.
    if (items[i].thumb?.kind === 'raw') {
      const raw = (displayMode === DISPLAY_DENORM && items[i].tnMin != null && items[i].tnMax != null && items[i].tnDtype)
        ? denormalizeThumbnailRGBA(items[i].thumb.data, items[i].tnMin, items[i].tnMax, items[i].tnDtype)
        : items[i].thumb.data;
      // In Dash, only the border is group-colored; the interior is the plot background.
      drawThumbnailRGBA(ctx2d, raw, x + BORDER, y + BORDER);
    }
  }
  const t3 = performance.now();

  // ── Hover tooltip (fast) ──────────────────────────────────────────────────
  container.style.position = 'relative';
  const tip = document.createElement('div');
  tip.style.cssText = [
    'position:absolute',
    'display:none',
    'pointer-events:none',
    'z-index:10',
    'background:rgba(0,0,0,0.85)',
    'color:white',
    'padding:6px 8px',
    'border-radius:6px',
    'font-size:12px',
    'max-width:360px',
    'white-space:nowrap',
    'overflow:hidden',
    'text-overflow:ellipsis',
  ].join(';');
  container.appendChild(canvas);
  container.appendChild(tip);

  const tile = CELL - GAP;
  canvas.onmousemove = e => {
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    const px = (e.clientX - rect.left) * sx;
    const py = (e.clientY - rect.top) * sy;

    const col = Math.floor(px / CELL);
    const row = Math.floor(py / CELL);
    if (col < 0 || row < 0) { tip.style.display = 'none'; return; }

    const localX = px - col * CELL;
    const localY = py - row * CELL;
    // Ignore gap region.
    if (localX >= tile || localY >= tile) { tip.style.display = 'none'; return; }

    const idx = row * perRow + col;
    if (idx < 0 || idx >= items.length) { tip.style.display = 'none'; return; }

    const text = items[idx].label || items[idx].group;
    tip.textContent = text;
    tip.style.left = `${Math.min(rect.width - 10, (e.clientX - rect.left) + 12)}px`;
    tip.style.top  = `${Math.min(rect.height - 10, (e.clientY - rect.top) + 12)}px`;
    tip.style.display = 'block';
  };
  canvas.onmouseleave = () => { tip.style.display = 'none'; };

  // ── Legend (cheap, DOM) ───────────────────────────────────────────────────
  const presentGroups = [...new Set(items.map(it => it.group))].sort();
  if (presentGroups.length > 1) {
    const legend = document.createElement('div');
    legend.style.cssText = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:8px';
    for (const g of presentGroups) {
      const item = document.createElement('div');
      item.style.cssText = 'display:flex;align-items:center;gap:6px;font-size:13px';
      const c = groupColor(ctx.colorMap, g);
      item.innerHTML = `<span style="display:inline-block;width:10px;height:10px;border-radius:999px;background:${c}"></span>${escapeHtml(g)}`;
      legend.appendChild(item);
    }
    container.appendChild(legend);
  }

  if (n === MAX_IMGS && ctx.filteredCount > MAX_IMGS) {
    const note = document.createElement('p');
    note.className = 'text-muted small mt-1 mb-0';
    note.textContent = `Showing ${MAX_IMGS} of ${ctx.filteredCount.toLocaleString()} images.`;
    container.appendChild(note);
  }

  // Debug timing (visible in devtools console)
  // eslint-disable-next-line no-console
  console.log('[mosaic] timings ms', {
    pickIds: pickIdsMs.toFixed(1),
    thumbnails: thumbnailsMs.toFixed(1),
    query: (pickIdsMs + thumbnailsMs).toFixed(1),
    tableToItems: (t2 - t1).toFixed(1),
    draw: (t3 - t2).toFixed(1),
    total: (t3 - t0).toFixed(1),
    drawn: items.length,
    mode: idCol ? '2step' : '1step',
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** DuckDB `rowid` must stay unquoted; normal columns use q(). */
function rowIdSql(name) {
  if (!name) return '';
  return name === 'rowid' ? 'rowid' : q(name);
}

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

function extractBytes(val) {
  if (!val) return null;
  if (val instanceof Uint8Array) return val;
  if (Array.isArray(val)) return Uint8Array.from(val, v => Number(v) & 0xff);
  if (val.values instanceof Uint8Array) return val.values;
  return null;
}

function drawThumbnailRGBA(ctx2d, pixels, x, y) {
  const rgba = pixels;
  const iData = ctx2d.createImageData(SPRITE, SPRITE);
  const d     = iData.data;

  const isRGBA = rgba.length >= SPRITE * SPRITE * 4;

  for (let i = 0; i < SPRITE * SPRITE; i++) {
    const idx = i * 4;
    if (isRGBA) {
      const a = rgba[i * 4 + 3] ?? 255;
      if (a < 128) {
        // Preserve transparency for padding pixels (alpha=0).
        d[idx]   = 0; d[idx+1] = 0; d[idx+2] = 0; d[idx+3] = 0;
      } else {
        d[idx]   = rgba[i * 4];
        d[idx+1] = rgba[i * 4 + 1];
        d[idx+2] = rgba[i * 4 + 2];
        d[idx+3] = 255;   // content pixels are opaque in our thumbnails
      }
    } else {
      // Grayscale
      const v = pixels[i] ?? 0;
      d[idx] = d[idx+1] = d[idx+2] = v;  d[idx+3] = 255;
    }
  }

  ctx2d.putImageData(iData, x, y);
}

function extractThumbnail(val) {
  const bytes = extractBytes(val);
  if (!bytes) return null;

  // Match current thumbnail generator exactly:
  // pixel_patrol_base.plugins.processors.thumbnail_processor outputs raw RGBA uint8
  // with fixed shape SPRITE×SPRITE×4.
  if (bytes.length !== SPRITE * SPRITE * 4) return null;
  return { kind: 'raw', data: bytes };
}

function colIndex(table, name) {
  const fields = table?.schema?.fields ?? [];
  for (let i = 0; i < fields.length; i++) {
    if (fields[i]?.name === name) return i;
  }
  return -1;
}

function getCol(table, name) {
  const idx = colIndex(table, name);
  if (idx < 0) return null;
  return typeof table.getChildAt === 'function' ? table.getChildAt(idx) : null;
}

function tableToItems(table, { hoverCol, hasSort, hasNorm }) {
  const n = Number(table.numRows ?? 0);
  const gCol   = getCol(table, '__group__');
  const tCol   = getCol(table, 'thumbnail');
  const oCol   = getCol(table, '__ord__');
  const lCol   = hoverCol ? getCol(table, '__label__') : null;
  const sCol   = hasSort ? getCol(table, '__sort__') : null;
  const minCol = hasNorm ? getCol(table, '__tn_min__') : null;
  const maxCol = hasNorm ? getCol(table, '__tn_max__') : null;
  const dtCol  = hasNorm ? getCol(table, '__tn_dtype__') : null;

  const out = [];
  for (let i = 0; i < n; i++) {
    const group = gCol ? String(gCol.get(i)) : 'all';
    const bytes = tCol ? tCol.get(i) : null; // Uint8Array for Binary
    const thumb = extractThumbnail(bytes);
    if (!thumb) continue;
    out.push({
      ord:   oCol ? Number(oCol.get(i)) : null,
      group,
      thumb,
      label: lCol ? String(lCol.get(i) ?? '') : '',
      sort:  sCol ? Number(sCol.get(i) ?? 0) : 0,
      tnMin: minCol ? minCol.get(i) : null,
      tnMax: maxCol ? maxCol.get(i) : null,
      tnDtype: dtCol ? String(dtCol.get(i) ?? '') : null,
    });
  }
  if (out.some(it => it.ord != null && Number.isFinite(it.ord))) {
    out.sort((a, b) => (a.ord ?? 0) - (b.ord ?? 0));
  }
  return out;
}


function denormalizeThumbnailRGBA(rgbaBytes, normMin, normMax, dtypeName) {
  const mm = Number(normMin);
  const mx = Number(normMax);
  if (!Number.isFinite(mm) || !Number.isFinite(mx)) return rgbaBytes;

  const info = dtypeInfo(dtypeName);
  if (!info) return rgbaBytes;

  const { lo, hi } = info;
  const scale = (mx - mm) / 255.0;
  const denom = (hi - lo) || 1;

  // Work on a copy to avoid mutating cached bytes.
  const out = new Uint8Array(rgbaBytes);
  for (let i = 0; i < SPRITE * SPRITE; i++) {
    const o = i * 4;
    // Only adjust visible pixels.
    const a = out[o + 3];
    if (a < 1) continue;

    for (let c = 0; c < 3; c++) {
      const v = out[o + c];
      const unnorm = (v * scale + mm - lo) / denom * 255.0;
      out[o + c] = clamp8(unnorm);
    }
  }
  return out;
}

function clamp8(x) {
  if (x <= 0) return 0;
  if (x >= 255) return 255;
  return x | 0;
}

function dtypeInfo(name) {
  const n = String(name || '').toLowerCase();
  switch (n) {
    case 'uint8':   return { lo: 0, hi: 255 };
    case 'int8':    return { lo: -128, hi: 127 };
    case 'uint16':  return { lo: 0, hi: 65535 };
    case 'int16':   return { lo: -32768, hi: 32767 };
    case 'uint32':  return { lo: 0, hi: 4294967295 };
    case 'int32':   return { lo: -2147483648, hi: 2147483647 };
    default:        return null;
  }
}

