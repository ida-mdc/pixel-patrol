import { describe, expect, it } from 'vitest';
import widget, { compressionRow, losslessWarning, losslessness, sections, sortRows } from '../../src/pixel_patrol_base/viewer/plugin_compression.js';

const raw = (group, ext, codec, disk, rawBytes, files = 1, zero = 0) => ({
  __group__: group, ext, codec, files, disk, raw: rawBytes, mean_bps: disk / rawBytes, smallest: disk / files, largest: disk / files, zero_share: zero, zero_covered: zero === null ? null : 1,
});

describe('losslessness', () => {
  it.each([
    ['none', 'lossless'], ['deflate', 'lossless'], ['deflate + horizontal predictor', 'lossless'],
    ['blosc(zstd, bitshuffle) level 3', 'lossless'], ['zstd1', 'lossless'], ['gzip level 4 + shuffle', 'lossless'],
    ['JPEG', 'lossy'], ['zfpy', 'lossy'], ['JPEG XR', 'either'], ['jpeg2000', 'either'],
    ['JPEG XR + zstd0', 'either'], ['deflate + JPEG', 'lossy'], ['frobnicate', 'unknown'],
  ])('%s is %s', (codec, expected) => expect(losslessness(codec)).toBe(expected));

  it('is null when the codec was not recorded', () => {
    expect(losslessness(null)).toBeNull();
  });
});

describe('losslessWarning', () => {
  const row = (codec, files = 1) => compressionRow({ ...raw('g', 'czi', codec, 10, 100, files) });

  it('is red and lists the lossy codecs', () => {
    const w = losslessWarning([row('JPEG', 2), row('deflate')]);
    expect(w.level).toBe('red');
    expect(w.html).toContain('2 of 3 files');
    expect(w.html).toContain('czi · JPEG');
  });

  it('is yellow when a codec might be lossy', () => {
    expect(losslessWarning([row('JPEG XR'), row('none')]).level).toBe('yellow');
  });

  it('is absent for lossless codecs', () => {
    expect(losslessWarning([row('deflate'), row(null)])).toBeNull();
  });
});

describe('compressionRow', () => {
  it('separates the ratio from the empty share', () => {
    const r = compressionRow(raw('g', 'ome.zarr', 'blosc(zstd, bitshuffle) level 3', 32.68e9, 76.4e9, 1, 0.192));
    expect(r.ratio).toBeCloseTo(2.34, 2);
    expect(r.contentRatio).toBeCloseTo(1.89, 2);
    expect(r.bytesPerSample).toBeCloseTo(0.428, 3);
    expect(r.codec).toBe('blosc(zstd, bitshuffle) level 3');
  });

  it('only compares without zeros when every file was measured', () => {
    const partial = compressionRow({ ...raw('g', 'tif', null, 1e6, 2e6, 1, 0.2), zero_covered: 0.4 });
    expect(partial.zeroShare).toBeCloseTo(0.2);
    expect(partial.contentRatio).toBeNull();
  });

  it('keeps a missing codec and zero share as null', () => {
    const r = compressionRow({ ...raw('g', 'tif', null, 1e6, 2e6), zero_share: null });
    expect(r.codec).toBeNull();
    expect(r.zeroShare).toBeNull();
    expect(r.contentRatio).toBeNull();
  });
});

describe('sorting and grouping', () => {
  const rows = [
    raw('b', 'tif', 'deflate', 5, 10), raw('a', 'tif', 'lzw', 4, 10), raw('a', 'czi', 'JPEG XR', 1, 10),
    raw('a', 'tif', 'deflate', 6, 10), raw('b', 'czi', 'JPEG XR', 2, 10),
  ].map(compressionRow);

  it('orders by group, then file type, then codec', () => {
    expect(sortRows(rows, ['a', 'b']).map(r => `${r.group}/${r.format}/${r.codec}`)).toEqual([
      'a/czi/JPEG XR', 'a/tif/deflate', 'a/tif/lzw', 'b/czi/JPEG XR', 'b/tif/deflate']);
  });

  it('follows the viewer group order, not the alphabet', () => {
    expect(sortRows(rows, ['b', 'a'])[0].group).toBe('b');
  });

  it('has no content ratio for uncompressed rows, which store zeros like any sample', () => {
    expect(compressionRow(raw('g', 'tif', 'none', 100, 100, 1, 0.2)).contentRatio).toBeNull();
  });

  it('puts unknown groups last and totals each section', () => {
    const out = sections(rows, ['b']);
    expect(out.map(s => s.group)).toEqual(['b', 'a']);
    expect(out[1]).toMatchObject({ files: 3, disk: 11 });
  });
});

describe('widget spec', () => {
  it('is the compression widget and needs sizes and dtype', () => {
    expect(widget.id).toBe('compression');
    expect(widget.requires({ allCols: ['size_bytes', 'num_pixels', 'dtype'] })).toBe(true);
    expect(widget.requires({ allCols: ['size_bytes'] })).toBe(false);
  });

  it('renders group headers with the codec column', async () => {
    const plots = [];
    const rows = [raw('a', 'czi', 'JPEG XR', 1e9, 2.7e10), raw('b', 'tif', null, 2e9, 4e9, 4)];
    const ctx = {
      where: '', groups: ['a', 'b'],
      schema: { allCols: ['size_bytes', 'num_pixels', 'dtype', 'compression'] },
      sql: { groupExpr: () => '"g" AS __group__', groupCol: () => '"g"', andWhere: (w, c) => `WHERE ${c}` },
      queryRows: async () => rows,
      groupLabel: (g) => g.toUpperCase(), color: { group: () => '#4c78a8' },
      plot: { escapeHtml: (s) => String(s), prependWarning: (c, w) => c.prepend(Object.assign(document.createElement('div'), { className: `warn-${w.level}`, innerHTML: w.html })), LAYOUT: {}, formatBytes: (b) => `${b} B`, engine: { renderDistribution: async (host, c, spec) => { plots.push(spec); return true; } } },
    };
    const el = document.createElement('div');
    await widget.render(el, ctx);
    const body = [...el.querySelectorAll('tbody tr')].map(tr => tr.textContent.replace(/\s+/g, ' ').trim());
    expect(body[0]).toContain('A');
    expect(body[1]).toContain('JPEG XR');
    expect(body[2]).toContain('B');
    expect(body[3]).toContain('not recorded');
    expect(el.querySelector('thead').textContent).toContain('Codec');
    const [spec] = plots;
    expect(spec.numCol).toBe('bytes_per_sample');
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

  it('summarises the overall compression ratio', async () => {
    const message = await widget.overviewMessage(ctx);
    expect(message.text).toContain('10.3×');
    expect(message.warning).toBe(true);
  });

  it('draws a compact box summary per group', async () => {
    await widget.overviewPlot(document.createElement('div'), ctx);
    expect(calls[0]).toMatchObject({ mini: true, maxRawPoints: 0, numCol: 'bytes_per_sample', categoriesOrder: ['a', 'b'] });
  });
});
