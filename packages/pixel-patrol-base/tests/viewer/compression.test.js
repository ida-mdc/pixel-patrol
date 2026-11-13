import { describe, expect, it } from 'vitest';
import widget, { compressionRow, lossyWarning, sections, sortRows } from '../../src/pixel_patrol_base/viewer/plugin_compression.js';

const TYPE_OF = { none: 'uncompressed', deflate: 'lossless', lzw: 'lossless', JPEG: 'lossy', 'JPEG XR': 'lossy or lossless',
  'blosc(zstd, bitshuffle) level 3': 'lossless' };

const raw = (group, ext, algorithm, disk, rawBytes, files = 1, type = TYPE_OF[algorithm] ?? null) => ({
  __group__: group, ext, algorithm, type, files, disk, raw: rawBytes, factor: rawBytes / disk, smallest: disk / files, largest: disk / files,
});

describe('lossyWarning', () => {
  const row = (algorithm, files = 1) => compressionRow(raw('g', 'czi', algorithm, 10, 100, files));

  it('is red and lists the lossy algorithms', () => {
    const w = lossyWarning([row('JPEG', 2), row('deflate')]);
    expect(w.level).toBe('red');
    expect(w.html).toContain('2 of 3 files');
    expect(w.html).toContain('czi · JPEG');
  });

  it('is yellow when an algorithm might be lossy', () => {
    expect(lossyWarning([row('JPEG XR'), row('none')]).level).toBe('yellow');
  });

  it('is absent for lossless and uncompressed files', () => {
    expect(lossyWarning([row('deflate'), row('none'), row(null)])).toBeNull();
  });
});

describe('compressionRow', () => {
  it('divides the uncompressed size by the size on disk', () => {
    const r = compressionRow(raw('g', 'ome.zarr', 'blosc(zstd, bitshuffle) level 3', 32.68e9, 76.4e9));
    expect(r.factor).toBeCloseTo(2.34, 2);
    expect(r.algorithm).toBe('blosc(zstd, bitshuffle) level 3');
    expect(r.type).toBe('lossless');
  });

  it('keeps a missing algorithm as null', () => {
    const r = compressionRow(raw('g', 'tif', null, 1e6, 2e6));
    expect(r.algorithm).toBeNull();
    expect(r.type).toBeNull();
  });
});

describe('sorting and grouping', () => {
  const rows = [
    raw('b', 'tif', 'deflate', 5, 10), raw('a', 'tif', 'lzw', 4, 10), raw('a', 'czi', 'JPEG XR', 1, 10),
    raw('a', 'tif', 'deflate', 6, 10), raw('b', 'czi', 'JPEG XR', 2, 10),
  ].map(compressionRow);

  it('orders by group, then file type, then algorithm', () => {
    expect(sortRows(rows, ['a', 'b']).map(r => `${r.group}/${r.format}/${r.algorithm}`)).toEqual([
      'a/czi/JPEG XR', 'a/tif/deflate', 'a/tif/lzw', 'b/czi/JPEG XR', 'b/tif/deflate']);
  });

  it('follows the viewer group order, not the alphabet', () => {
    expect(sortRows(rows, ['b', 'a'])[0].group).toBe('b');
  });

  it('puts unknown groups last and totals each section', () => {
    const out = sections(rows, ['b']);
    expect(out.map(s => s.group)).toEqual(['b', 'a']);
    expect(out[1]).toMatchObject({ files: 3, disk: 11 });
  });
});

describe('widget spec', () => {
  it('is the compression widget and needs sizes, dtype and the compression factor', () => {
    expect(widget.id).toBe('compression');
    expect(widget.requires({ allCols: ['size_bytes', 'num_pixels', 'dtype', 'compression_factor'] })).toBe(true);
    expect(widget.requires({ allCols: ['size_bytes', 'num_pixels', 'dtype'] })).toBe(false);
  });

  it('renders group headers, named columns with hovers and the compression type', async () => {
    const plots = [];
    const rows = [raw('a', 'czi', 'JPEG XR', 1e9, 2.7e10), raw('b', 'tif', null, 2e9, 4e9, 4), raw('b', 'zarr', 'none', 1e9, 1e9)];
    const queries = [];
    const ctx = {
      where: '', groups: ['a', 'b'],
      schema: { allCols: ['size_bytes', 'num_pixels', 'dtype', 'compression', 'compression_type', 'compression_factor'] },
      sql: { groupExpr: () => '"g" AS __group__', groupCol: () => '"g"', andWhere: (w, c) => `WHERE ${c}` },
      queryRows: async (sql) => { queries.push(sql); return rows; },
      groupLabel: (g) => g.toUpperCase(), color: { group: () => '#4c78a8' },
      plot: { escapeHtml: (s) => String(s), prependWarning: (c, w) => c.prepend(Object.assign(document.createElement('div'), { className: `warn-${w.level}`, innerHTML: w.html })), LAYOUT: {}, formatBytes: (b) => `${b} B`, engine: { renderDistribution: async (host, c, spec) => { plots.push(spec); return true; } } },
    };
    const el = document.createElement('div');
    await widget.render(el, ctx);
    const body = [...el.querySelectorAll('tbody tr')].map(tr => tr.textContent.replace(/\s+/g, ' ').trim());
    expect(body[0]).toContain('A');
    expect(body[1]).toContain('JPEG XR');
    expect(body[1]).toContain('lossy or lossless');
    expect(body[2]).toContain('B');
    expect(body[3]).toContain('not recorded');
    expect(body[4]).toContain('uncompressed');
    const heads = [...el.querySelectorAll('thead th')];
    expect(heads.map(th => th.textContent.replace('ⓘ', '').trim())).toEqual([
      'Format', 'Compression algorithm', 'Compression type', 'Files', 'Bytes on disk', 'Uncompressed', 'Compression factor', 'File size']);
    expect(heads.every(th => th.querySelector('[title]').title.length > 0)).toBe(true);
    expect(queries[0]).toContain('"compression_type" AS type');
    const [spec] = plots;
    expect(spec.numCol).toBe('compression_factor');
    expect(spec.source.table).toContain('size_bytes');
    expect(spec.source.table).toContain('"file_extension"');
    expect(spec.series.sql).toBe('"g" AS __group__');
    expect(spec.series.build([{ __group__: 'b' }]).groups).toEqual(['b']);
    expect(spec.catSql).toBe('storage_label');
    expect(spec.source.table).toContain('DENSE_RANK() OVER (ORDER BY "g")');
    expect(spec.layout.violinmode).toBeUndefined();
  });
});

describe('overview tile', () => {
  const rows = [raw('a', 'czi', 'JPEG XR', 1e9, 2.7e10), raw('b', 'tif', null, 2e9, 4e9)];
  const calls = [];
  const ctx = {
    where: '', groups: ['a', 'b'],
    schema: { allCols: ['size_bytes', 'num_pixels', 'dtype'] },
    sql: { groupExpr: () => '"g" AS __group__', groupCol: () => '"g"', andWhere: (w, c) => `WHERE ${c}` },
    queryRows: async () => rows,
    groupLabel: (g) => g,
    plot: { engine: { renderDistribution: async (host, c, spec) => { calls.push(spec); return true; } } },
  };

  it('summarises the overall compression factor', async () => {
    const message = await widget.overviewMessage(ctx);
    expect(message.text).toContain('10.3×');
    expect(message.warning).toBe(true);
  });

  it('draws a compact box summary per group', async () => {
    await widget.overviewPlot(document.createElement('div'), ctx);
    expect(calls[0]).toMatchObject({ mini: true, maxRawPoints: 0, numCol: 'compression_factor', categoriesOrder: ['a', 'b'] });
  });
});
