const DTYPE_BYTES_SQL = `CASE "dtype"
  WHEN 'uint8' THEN 1 WHEN 'int8' THEN 1
  WHEN 'uint16' THEN 2 WHEN 'int16' THEN 2 WHEN 'float16' THEN 2
  WHEN 'uint32' THEN 4 WHEN 'int32' THEN 4 WHEN 'float32' THEN 4
  WHEN 'uint64' THEN 8 WHEN 'int64' THEN 8 WHEN 'float64' THEN 8 END`;

const BYTES_PER_SAMPLE = 'bytes_per_sample';
const STORAGE_LABEL = 'storage_label';
const FULL_COVERAGE = 0.999;
const NOT_RECORDED = 'not recorded';
const HEADER_BACKGROUND = '#f1f3f5';
const COLUMNS = ['Format', 'Codec', 'Files', 'Stored', 'Uncompressed', 'Compression', 'Zero-valued',
  'Compression excl. zeros', 'Bytes / sample (mean)', 'File size', 'Lossless'];

const LOSSLESS_CODEC = /\b(none|lossless|deflate|zlib|gzip|lzw|packbits|zstd\d*|lzma|lz4|bz2|bzip2|snappy|brotli|blosc|shuffle|bitshuffle|delta|predictor|crc32c|transpose|sharding\w*|bytes)\b/;
const LOSSY_CODEC = /\b(lossy|ojpeg|zfp\w*|fixedscaleoffset|quantize|bitround)\b|^jpeg$/;
const EITHER_CODEC = /xr|2000|jpegxl|jpeg xl|webp|lerc|jpeg.?ls|jxr|avif|heic|camera|system/;

// ---- rows (pure) ----

const LOSSLESSNESS_ORDER = ['lossy', 'either', 'unknown', 'lossless'];

function stageLosslessness(stage) {
  const label = stage.trim().toLowerCase();
  if (LOSSY_CODEC.test(label)) return 'lossy';
  if (EITHER_CODEC.test(label)) return 'either';
  return LOSSLESS_CODEC.test(label) ? 'lossless' : 'unknown';
}

// A chain of stages (Zarr filters, mixed CZI sub-blocks) is as lossy as its worst stage.
export function losslessness(codec) {
  if (codec == null) return null;
  const kinds = String(codec).split('+').map(stageLosslessness);
  return LOSSLESSNESS_ORDER.find(k => kinds.includes(k));
}

const ICON_GLYPH = {
  check: '<path d="M5 8.5l2.2 2.2L11.2 5.5" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>',
  cross: '<path d="M5.5 5.5l5 5M10.5 5.5l-5 5" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>',
  question: '<text x="8" y="11.6" text-anchor="middle" font-size="10" font-weight="700" fill="#fff" font-family="sans-serif">?</text>',
};

const svgIcon = (glyph, colour) =>
  `<svg width="16" height="16" viewBox="0 0 16 16" style="vertical-align:middle"><circle cx="8" cy="8" r="8" fill="${colour}"/>${ICON_GLYPH[glyph]}</svg>`;

const LOSSLESSNESS_ICON = {
  lossless: [svgIcon('check', '#2f9e44'), 'Lossless: the original values can be recovered exactly.'],
  lossy: [svgIcon('cross', '#e03131'), 'Lossy: the codec discards information, so decoded values differ from the original.'],
  either: [svgIcon('question', '#868e96'), 'Lossy or lossless: this codec supports both modes and the file header does not say which was used.'],
  unknown: [svgIcon('question', '#868e96'), 'This codec is not classified.'],
};

export function compressionRow(r) {
  const disk = Number(r.disk);
  const raw = Number(r.raw);
  const zeroShare = r.zero_share == null ? null : Number(r.zero_share);
  const zeroCovered = r.zero_covered == null ? 0 : Number(r.zero_covered);
  return {
    group: String(r.__group__), format: String(r.ext), codec: r.codec == null ? null : String(r.codec),
    losslessness: losslessness(r.codec),
    files: Number(r.files), disk, raw,
    ratio: raw > 0 ? raw / disk : null,
    bytesPerSample: r.mean_bps == null ? null : Number(r.mean_bps),
    zeroShare,
    contentRatio: raw > 0 && zeroShare !== null && zeroCovered >= FULL_COVERAGE && r.codec !== 'none' ? raw * (1 - zeroShare) / disk : null,
    smallest: Number(r.smallest), largest: Number(r.largest),
  };
}

export function sortRows(rows, groupOrder) {
  const rank = (g) => { const i = groupOrder.indexOf(g); return i < 0 ? groupOrder.length : i; };
  return [...rows].sort((a, b) =>
    rank(a.group) - rank(b.group) || a.group.localeCompare(b.group)
    || a.format.localeCompare(b.format) || (a.codec ?? '').localeCompare(b.codec ?? '') || b.disk - a.disk);
}

export function sections(rows, groupOrder) {
  const out = [];
  for (const row of sortRows(rows, groupOrder)) {
    const last = out.at(-1);
    if (last?.group === row.group) last.rows.push(row);
    else out.push({ group: row.group, rows: [row] });
  }
  return out.map(s => ({
    ...s,
    files: s.rows.reduce((n, r) => n + r.files, 0),
    disk: s.rows.reduce((n, r) => n + r.disk, 0),
  }));
}


// ---- data ----

const hasCol = (ctx, col) => ctx.schema.allCols.includes(col);

const ZERO_WIDTH_SPACE = 8203;

// The x axis shows the format. Each group gets the label padded with a
// different number of zero-width spaces: identical on screen, but a separate
// category per group, so a box is centred on its tick instead of sharing a
// slot with groups that lack that format.
function storageLabelSql(ctx) {
  const format = `COALESCE("file_extension", '(none)')`;
  return `${format} || repeat(chr(${ZERO_WIDTH_SPACE}), (DENSE_RANK() OVER (ORDER BY ${ctx.sql.groupCol()}) - 1)::INT)`;
}

// The share of samples at the dtype minimum is a zero count only for unsigned
// integers, and only where the quality processor measured it.
function zeroColumnsSql(ctx, window) {
  if (!hasCol(ctx, 'dark_clipping_fraction')) return 'NULL AS file_zeros, NULL AS file_zero_basis';
  const unsigned = `"dtype" LIKE 'uint%' AND "dark_clipping_fraction" IS NOT NULL`;
  return `SUM(CASE WHEN ${unsigned} THEN "dark_clipping_fraction" * "num_pixels" END) OVER ${window} AS file_zeros,
      SUM(CASE WHEN ${unsigned} THEN "num_pixels" END) OVER ${window} AS file_zero_basis`;
}

// One row per file: a container file repeats size_bytes on every image row, so
// samples are summed over its images before they are compared with the size.
// A file can be larger than its pixels (metadata, thumbnails, padding), which
// says nothing about the codec, so the stored size is capped at the raw size.
function perFileTable(ctx) {
  const window = hasCol(ctx, 'path') ? '(PARTITION BY "path")' : '(PARTITION BY "file_row_number")';
  const where = ctx.sql.andWhere(ctx.where, `"size_bytes" > 0 AND "num_pixels" > 0`);
  return `(SELECT *, ${storageLabelSql(ctx)} AS ${STORAGE_LABEL},
      SUM("num_pixels") OVER ${window} AS file_samples,
      SUM("num_pixels" * ${DTYPE_BYTES_SQL}) OVER ${window} AS file_raw,
      ${zeroColumnsSql(ctx, window)},
      LEAST("size_bytes", SUM("num_pixels" * ${DTYPE_BYTES_SQL}) OVER ${window})::DOUBLE AS file_stored,
      LEAST("size_bytes", SUM("num_pixels" * ${DTYPE_BYTES_SQL}) OVER ${window})::DOUBLE
        / SUM("num_pixels") OVER ${window} AS ${BYTES_PER_SAMPLE}
    FROM pp_data ${where}
    QUALIFY ROW_NUMBER() OVER ${window} = 1)`;
}

async function fetchCompression(ctx) {
  const codec = hasCol(ctx, 'compression') ? `"compression"` : 'NULL';
  const rows = await ctx.queryRows(`
    SELECT ${ctx.sql.groupExpr()}, COALESCE("file_extension", '(none)') AS ext, ${codec} AS codec, COUNT(*) AS files,
           SUM(file_stored) AS disk, SUM(file_raw) AS raw, AVG(bytes_per_sample) AS mean_bps,
           MIN("size_bytes") AS smallest, MAX("size_bytes") AS largest,
           SUM(file_zeros) / NULLIF(SUM(file_zero_basis), 0) AS zero_share,
           SUM(file_zero_basis) / NULLIF(SUM(file_samples), 0) AS zero_covered
    FROM ${perFileTable(ctx)} GROUP BY 1, 2, 3`);
  return rows.map(compressionRow);
}


// ---- rendering ----

const ratio = (v) => (v ? `${v.toFixed(1)}×` : 'n/a');
const share = (v) => (v === null ? 'n/a' : `${(100 * v).toFixed(1)}%`);

function cell(html, align = 'right') {
  return `<td style="padding:4px 10px;text-align:${align}">${html}</td>`;
}

function losslessCell(r) {
  if (!r.losslessness) return cell('');
  const [icon, hint] = LOSSLESSNESS_ICON[r.losslessness];
  return cell(`<span title="${hint}" aria-label="${hint}">${icon}</span>`);
}

function dataRow(r, ctx) {
  const { formatBytes, escapeHtml } = ctx.plot;
  const size = r.files > 1 ? `${formatBytes(r.smallest)} – ${formatBytes(r.largest)}` : formatBytes(r.largest);
  const codec = r.codec ?? `<span style="color:#868e96">${NOT_RECORDED}</span>`;
  return `<tr>${cell(escapeHtml(r.format), 'left')}${cell(r.codec === null ? codec : escapeHtml(codec), 'left')}
    ${cell(r.files.toLocaleString())}${cell(formatBytes(r.disk))}${cell(formatBytes(r.raw))}${cell(ratio(r.ratio))}
    ${cell(share(r.zeroShare))}${cell(ratio(r.contentRatio))}
    ${cell(r.bytesPerSample ? r.bytesPerSample.toFixed(3) : 'n/a')}${cell(size)}${losslessCell(r)}</tr>`;
}

function groupHeader(section, ctx) {
  const { formatBytes, escapeHtml } = ctx.plot;
  const colour = ctx.color.group(section.group);
  const label = escapeHtml(ctx.groupLabel?.(section.group) ?? section.group);
  return `<tr><td colspan="${COLUMNS.length}" style="padding:8px 10px;font-weight:600;background:${HEADER_BACKGROUND};border-left:4px solid ${colour}">
    ${label} <span style="font-weight:400;color:#495057">· ${section.files.toLocaleString()} files · ${formatBytes(section.disk)} stored</span></td></tr>`;
}

function compressionTable(sectionList, ctx) {
  const body = sectionList.map(s => groupHeader(s, ctx) + s.rows.map(r => dataRow(r, ctx)).join('')).join('');
  const table = document.createElement('table');
  table.className = 'table table-sm';
  table.style.cssText = 'font-size:0.88rem;width:100%';
  const heads = COLUMNS.map((h, i) => `<th style="padding:4px 10px;text-align:${i < 2 ? 'left' : 'right'}">${h}</th>`).join('');
  table.innerHTML = `<thead><tr>${heads}</tr></thead><tbody>${body}</tbody>`;
  return table;
}

function groupSeries(ctx) {
  return {
    sql: ctx.sql.groupExpr(),
    build: (rows) => ({
      groups: ctx.groups.filter(g => rows.some(r => String(r.__group__) === g)),
      colorFn: (g) => ctx.color.group(g),
      labelFn: (g) => ctx.groupLabel?.(g) ?? g,
    }),
  };
}

function compressionDistribution(host, ctx) {
  return ctx.plot.engine.renderDistribution(host, ctx, {
    numCol: BYTES_PER_SAMPLE,
    source: { table: perFileTable(ctx), where: '' },
    catSql: STORAGE_LABEL,
    catLabel: 'format',
    yLabel: 'bytes stored per sample',
    title: 'Bytes stored per sample<br><sup>one point per file</sup>',
    series: groupSeries(ctx),
    layout: { yaxis: { title: 'bytes stored per sample', rangemode: 'tozero' } },
  });
}

function note(text) {
  const el = document.createElement('p');
  el.className = 'text-muted small';
  el.textContent = text;
  return el;
}

function missingCodecNote(rows) {
  if (rows.some(r => r.codec !== null)) return null;
  return note('This report has no compression column. Reprocess it with a current PixelPatrol to record each file\'s codec.');
}

// ---- lossy warning ----

const filesOf = (rows) => rows.reduce((n, r) => n + r.files, 0);

function codecList(rows, escape) {
  return rows.map(r => `<code>${escape(r.format)} · ${escape(r.codec)}</code>`).join(', ');
}

export function losslessWarning(rows, escape = String) {
  const lossy = rows.filter(r => r.losslessness === 'lossy');
  const either = rows.filter(r => r.losslessness === 'either');
  const total = filesOf(rows).toLocaleString();
  if (lossy.length) {
    return { level: 'red', html: `${filesOf(lossy).toLocaleString()} of ${total} files store their pixels with a lossy codec (${codecList(lossy, escape)}), so the pixel values in them are approximations of what was originally recorded.` };
  }
  if (either.length) {
    return { level: 'yellow', html: `${filesOf(either).toLocaleString()} of ${total} files use a codec that can store pixels lossy or lossless (${codecList(either, escape)}). The file header does not say which, so their pixel values may be approximations of what was originally recorded.` };
  }
  return null;
}


// ---- overview tile ----

async function overviewMessage(ctx) {
  try {
    const rows = await fetchCompression(ctx);
    if (!rows.length) return null;
    const disk = rows.reduce((n, r) => n + r.disk, 0);
    const raw = rows.reduce((n, r) => n + r.raw, 0);
    const text = `Stored at <strong>${(raw / disk).toFixed(1)}×</strong> compression overall.`;
    const warning = losslessWarning(rows);
    if (!warning) return text;
    const unsure = warning.level === 'red' ? 'lossy' : 'possibly lossy';
    return { text: `${text} <strong>${unsure}</strong> files present.`, warning: true };
  } catch { return null; }
}

function overviewPlot(container, ctx) {
  return ctx.plot.engine.renderDistribution(container, ctx, {
    numCol: BYTES_PER_SAMPLE,
    source: { table: perFileTable(ctx), where: '' },
    catSql: ctx.sql.groupCol(),
    yLabel: 'bytes / sample',
    series: { isCategory: true },
    categoriesOrder: ctx.groups,
    catLabelFn: ctx.groupLabel,
    mini: true,
    maxRawPoints: 0,
  });
}


const INFO = [
  'Shows **which codec** each file uses and **how well it compresses**, to explain why the same pixels take very different amounts of disk.',
  '',
  '- The **plot** shows the distribution of bytes stored per sample, one point per file (the table shows the mean of these points), split by file type and coloured by group. A wide violin means compressibility depends on the content of each file.',
  '- The **table** is sorted by group, then file type, then codec.',
  '- **Codec** is read from the file header when the report is made (TIFF compression and predictor, Zarr codec chain, CZI sub-block codec). `none` means stored raw.',
  '- **Lossless** (last column, also summarised in a warning above the plot) is derived from the codec: green check: `none`, deflate, LZW, zstd and the like are lossless; red cross: baseline JPEG is lossy; grey question mark: JPEG XR, JPEG 2000 and WebP can be either and the header does not say which. A chain is as lossy as its worst stage. Lossy rows compress by throwing information away, so compare their ratios only with each other.',
  '- **Uncompressed** is samples × bytes per sample; **Compression** is uncompressed ÷ stored.',
  '- **Zero-valued** is the share of samples equal to zero: blank canvas, empty tile slots, black borders. These cost almost nothing on disk (empty Zarr chunks are never written), so they raise the ratio without the codec doing better. It comes from the raster-quality processor and is only available for unsigned integer images.',
  '- **Compression excl. zeros** is the ratio on the remaining samples, which is what the codec itself achieved. Compare it across formats. It is only shown when the zero share was measured for every file in the row, and not for `none`, where zeros are stored like any other sample.',
  '- **Stored** is the file size, capped at the uncompressed size: a file larger than its pixels (metadata, thumbnails, padding) counts as 1×, since that says nothing about the codec. **File size** is always the real size.',
  '- **File size** shows the smallest and largest file; a wide range means compressibility depends on content, as for lossless codecs on noisy or detailed pictures.',
  '- A lossless ratio is bounded by the information in the pixels: flat, low-contrast pictures compress well, noisy ones barely. A predictor or planar layout often adds 20-30% for smooth data.',
].join('\n');

export default {
  id: 'compression',
  required_inputs: ['size_bytes', 'num_pixels'],
  inputs: ['dtype', 'compression', 'file_extension', 'dark_clipping_fraction'],
  group: 'Dataset Stats',
  scope: 'file',
  label: 'Compression',
  shortLabel: 'Compression',
  info: INFO,

  requires(schema) {
    return ['size_bytes', 'num_pixels', 'dtype'].every(c => schema.allCols.includes(c));
  },

  overviewMessage,
  overviewPlot,

  async render(container, ctx) {
    const rows = await fetchCompression(ctx);
    if (!rows.length) { container.innerHTML = '<div class="no-data">No file sizes in the current selection.</div>'; return; }
    const ordered = sections(rows, ctx.groups ?? []);
    const plotHost = document.createElement('div');
    container.append(...[missingCodecNote(rows)].filter(Boolean), plotHost, compressionTable(ordered, ctx));
    const warning = losslessWarning(rows, ctx.plot.escapeHtml);
    if (warning) ctx.plot.prependWarning(container, warning);
    await compressionDistribution(plotHost, ctx);
  },
};
