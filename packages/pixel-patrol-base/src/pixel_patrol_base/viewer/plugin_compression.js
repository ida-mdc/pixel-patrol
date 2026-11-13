const DTYPE_BYTES_SQL = `CASE "dtype"
  WHEN 'uint8' THEN 1 WHEN 'int8' THEN 1
  WHEN 'uint16' THEN 2 WHEN 'int16' THEN 2 WHEN 'float16' THEN 2
  WHEN 'uint32' THEN 4 WHEN 'int32' THEN 4 WHEN 'float32' THEN 4
  WHEN 'uint64' THEN 8 WHEN 'int64' THEN 8 WHEN 'float64' THEN 8 END`;

const COMPRESSION_FACTOR = 'compression_factor';
const STORAGE_LABEL = 'storage_label';
const NOT_RECORDED = 'not recorded';
const HEADER_BACKGROUND = '#f1f3f5';

const COLUMNS = [
  ['Format', 'File extension.'],
  ['Compression algorithm', 'Read from the file header when the report was made (TIFF compression and predictor, Zarr codec chain, CZI sub-block compression). "none" means stored raw.'],
  ['Compression type', 'Derived from the algorithm: uncompressed, lossless (the original values can be recovered exactly) or lossy (information is discarded). Algorithms that support both modes without saying which are shown as "lossy or lossless".'],
  ['Files', 'Number of files.'],
  ['Bytes on disk', 'File size, capped at the uncompressed size to not include metadata into the compression factor.'],
  ['Uncompressed', 'Number of pixels × bytes per pixel of the data type.'],
  ['Compression factor', 'Mean of the files\' compression factors: uncompressed pixel bytes ÷ file size.'],
  ['File size', 'Real size of the smallest and the largest file.'],
];
const LEFT_ALIGNED_COLUMNS = 3;

// ---- rows (pure) ----

const TYPES = {
  'uncompressed': { colour: '#495057', hint: 'The pixels are stored raw.' },
  'lossless': { colour: '#2f9e44', hint: 'The original values can be recovered exactly.' },
  'lossy': { colour: '#e03131', hint: 'Information is discarded, so decoded values differ from the original.' },
  'lossy or lossless': { colour: '#868e96', hint: 'The algorithm supports both modes.' },
  'unknown': { colour: '#868e96', hint: 'This algorithm is not classified.' },
};
const EITHER = 'lossy or lossless';

export function compressionRow(r) {
  const disk = Number(r.disk);
  const raw = Number(r.raw);
  return {
    group: String(r.__group__), format: String(r.ext), algorithm: r.algorithm == null ? null : String(r.algorithm),
    type: r.type == null ? null : String(r.type),
    files: Number(r.files), disk, raw,
    factor: r.factor == null ? null : Number(r.factor),
    smallest: Number(r.smallest), largest: Number(r.largest),
  };
}

export function sortRows(rows, groupOrder) {
  const rank = (g) => { const i = groupOrder.indexOf(g); return i < 0 ? groupOrder.length : i; };
  return [...rows].sort((a, b) =>
    rank(a.group) - rank(b.group) || a.group.localeCompare(b.group)
    || a.format.localeCompare(b.format) || (a.algorithm ?? '').localeCompare(b.algorithm ?? '') || b.disk - a.disk);
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

// One row per file: a container file repeats size_bytes on every image row, so
// its images are summed before they are compared with the size. A file can be
// larger than its pixels (metadata, thumbnails, padding), so the size on disk
// is capped at the uncompressed size.
function perFileTable(ctx) {
  const window = hasCol(ctx, 'path') ? '(PARTITION BY "path")' : '(PARTITION BY "file_row_number")';
  const where = ctx.sql.andWhere(ctx.where, `"size_bytes" > 0 AND "num_pixels" > 0`);
  const raw = `SUM("num_pixels" * ${DTYPE_BYTES_SQL}) OVER ${window}`;
  const stored = `LEAST("size_bytes", ${raw})::DOUBLE`;
  return `(SELECT *, ${storageLabelSql(ctx)} AS ${STORAGE_LABEL},
      ${raw} AS file_raw, ${stored} AS file_stored
    FROM pp_data ${where}
    QUALIFY ROW_NUMBER() OVER ${window} = 1)`;
}

async function fetchCompression(ctx) {
  const algorithm = hasCol(ctx, 'compression') ? `"compression"` : 'NULL';
  const type = hasCol(ctx, 'compression_type') ? `"compression_type"` : 'NULL';
  const rows = await ctx.queryRows(`
    SELECT ${ctx.sql.groupExpr()}, COALESCE("file_extension", '(none)') AS ext, ${algorithm} AS algorithm, ${type} AS type, COUNT(*) AS files,
           SUM(file_stored) AS disk, SUM(file_raw) AS raw, AVG("compression_factor") AS factor, MIN("size_bytes") AS smallest, MAX("size_bytes") AS largest
    FROM ${perFileTable(ctx)} GROUP BY 1, 2, 3, 4`);
  return rows.map(compressionRow);
}


// ---- rendering ----

const factor = (v) => (v ? `${v.toFixed(1)}×` : 'n/a');

function cell(html, align = 'right') {
  return `<td style="padding:4px 10px;text-align:${align}">${html}</td>`;
}

function typeCell(r) {
  if (!r.type) return cell('', 'left');
  const { colour, hint } = TYPES[r.type] ?? TYPES.unknown;
  return cell(`<span title="${hint}" style="color:${colour};font-weight:600">${r.type}</span>`, 'left');
}

function dataRow(r, ctx) {
  const { formatBytes, escapeHtml } = ctx.plot;
  const size = r.files > 1 ? `${formatBytes(r.smallest)} – ${formatBytes(r.largest)}` : formatBytes(r.largest);
  const algorithm = r.algorithm === null ? `<span style="color:#868e96">${NOT_RECORDED}</span>` : escapeHtml(r.algorithm);
  return `<tr>${cell(escapeHtml(r.format), 'left')}${cell(algorithm, 'left')}${typeCell(r)}
    ${cell(r.files.toLocaleString())}${cell(formatBytes(r.disk))}${cell(formatBytes(r.raw))}${cell(factor(r.factor))}${cell(size)}</tr>`;
}

function groupHeader(section, ctx) {
  const { formatBytes, escapeHtml } = ctx.plot;
  const colour = ctx.color.group(section.group);
  const label = escapeHtml(ctx.groupLabel?.(section.group) ?? section.group);
  return `<tr><td colspan="${COLUMNS.length}" style="padding:8px 10px;font-weight:600;background:${HEADER_BACKGROUND};border-left:4px solid ${colour}">
    ${label} <span style="font-weight:400;color:#495057">· ${section.files.toLocaleString()} files · ${formatBytes(section.disk)} on disk</span></td></tr>`;
}

function columnHeader([label, hint], index) {
  const align = index < LEFT_ALIGNED_COLUMNS ? 'left' : 'right';
  return `<th style="padding:4px 10px;text-align:${align}">${label} <span title="${hint}" style="cursor:help;color:#868e96">ⓘ</span></th>`;
}

function compressionTable(sectionList, ctx) {
  const body = sectionList.map(s => groupHeader(s, ctx) + s.rows.map(r => dataRow(r, ctx)).join('')).join('');
  const table = document.createElement('table');
  table.className = 'table table-sm';
  table.style.cssText = 'font-size:0.88rem;width:100%';
  table.innerHTML = `<thead><tr>${COLUMNS.map(columnHeader).join('')}</tr></thead><tbody>${body}</tbody>`;
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
    numCol: COMPRESSION_FACTOR,
    source: { table: perFileTable(ctx), where: '' },
    catSql: STORAGE_LABEL,
    catLabel: 'format',
    yLabel: 'compression factor',
    title: 'Compression factor<br><sup>one point per file</sup>',
    series: groupSeries(ctx),
    layout: { yaxis: { title: 'compression factor', rangemode: 'tozero' } },
  });
}

// ---- lossy warning ----

const filesOf = (rows) => rows.reduce((n, r) => n + r.files, 0);

function algorithmList(rows, escape) {
  return rows.map(r => `<code>${escape(r.format)} · ${escape(r.algorithm)}</code>`).join(', ');
}

export function lossyWarning(rows, escape = String) {
  const lossy = rows.filter(r => r.type === 'lossy');
  const either = rows.filter(r => r.type === EITHER);
  const total = filesOf(rows).toLocaleString();
  if (lossy.length) {
    return { level: 'red', html: `${filesOf(lossy).toLocaleString()} of ${total} files store their pixels with a lossy compression algorithm (${algorithmList(lossy, escape)}), so the pixel values in them are approximations of what was originally recorded.` };
  }
  if (either.length) {
    return { level: 'yellow', html: `${filesOf(either).toLocaleString()} of ${total} files use a compression algorithm that can store pixels lossy or lossless (${algorithmList(either, escape)}). The file header does not say which, so their pixel values may be approximations of what was originally recorded.` };
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
    const warning = lossyWarning(rows);
    if (!warning) return text;
    const unsure = warning.level === 'red' ? 'lossy' : 'possibly lossy';
    return { text: `${text} <strong>${unsure}</strong> files present.`, warning: true };
  } catch { return null; }
}

function overviewPlot(container, ctx) {
  return ctx.plot.engine.renderDistribution(container, ctx, {
    numCol: COMPRESSION_FACTOR,
    source: { table: perFileTable(ctx), where: '' },
    catSql: ctx.sql.groupCol(),
    yLabel: 'compression factor',
    series: { isCategory: true },
    categoriesOrder: ctx.groups,
    catLabelFn: ctx.groupLabel,
    mini: true,
    maxRawPoints: 0,
  });
}


const INFO = [
  'Shows **which compression algorithm** each file uses and **how well it compresses**.',
  '',
  'Hover the ⓘ next to a column name for what it means. The plot shows the compression factor of every file, split by file type and coloured by group.',
].join('\n');

export default {
  id: 'compression',
  required_inputs: ['size_bytes', 'num_pixels', 'compression_factor'],
  inputs: ['dtype', 'compression', 'compression_type', 'file_extension'],
  group: 'Dataset Stats',
  scope: 'file',
  label: 'Compression',
  shortLabel: 'Compression',
  info: INFO,

  requires(schema) {
    return ['size_bytes', 'num_pixels', 'dtype', 'compression_factor'].every(c => schema.allCols.includes(c));
  },

  overviewMessage,
  overviewPlot,

  async render(container, ctx) {
    const rows = await fetchCompression(ctx);
    if (!rows.length) { container.innerHTML = '<div class="no-data">No file sizes in the current selection.</div>'; return; }
    const ordered = sections(rows, ctx.groups ?? []);
    const plotHost = document.createElement('div');
    container.append(plotHost, compressionTable(ordered, ctx));
    const warning = lossyWarning(rows, ctx.plot.escapeHtml);
    if (warning) ctx.plot.prependWarning(container, warning);
    await compressionDistribution(plotHost, ctx);
  },
};
