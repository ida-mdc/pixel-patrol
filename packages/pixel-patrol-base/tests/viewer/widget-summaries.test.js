import { describe, it, expect } from 'vitest';

// Load every shipped widget and index the plugin objects by id so each test can
// grab the exact widget it exercises (array exports are flattened).
const modules = import.meta.glob('../../src/pixel_patrol_base/viewer/plugin_*.js', { eager: true });
const byId = new Map();
for (const mod of Object.values(modules)) {
  const exported = mod.default ?? mod;
  for (const plugin of Array.isArray(exported) ? exported : [exported]) byId.set(plugin.id, plugin);
}
const widget = id => {
  const p = byId.get(id);
  if (!p) throw new Error(`widget "${id}" not found - shipped ids: ${[...byId.keys()].join(', ')}`);
  return p;
};

// Widgets never import viewer internals: at runtime the viewer hands them every
// helper through `ctx`. These are faithful copies of those helpers (viewer/src
// plot-utils.js + sql.js) so the asserted summary strings match production - if
// the viewer's copy drifts, that is a contract change worth surfacing here.
const escapeHtml = s => String(s)
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
const niceName = col => col.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
const humanList = items => {
  const a = (items ?? []).filter(x => x != null && x !== '');
  if (a.length <= 1) return a[0] ?? '';
  return `${a.slice(0, -1).join(', ')} and ${a.at(-1)}`;
};
const formatBytes = v => {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let val = Number(v), u = 0;
  while (val >= 1024 && u < units.length - 1) { val /= 1024; u++; }
  return val >= 10 ? `${Math.round(val)} ${units[u]}` : `${val.toFixed(1)} ${units[u]}`;
};
const q = name => `"${String(name).replaceAll('"', '""')}"`;
const andWhere = (where, condition) =>
  !condition ? where : (where ? `${where} AND ${condition}` : `WHERE ${condition}`);
const groupCol = state => (state.groupCol ? q(state.groupCol) : `'__ALL__'`);
const groupExpr = state => `${groupCol(state)} AS __group__`;

// condensedMessage() only ever reaches for ctx.queryRows plus the pure string
// helpers above, so a thin fake ctx is enough to drive it deterministically.
// `rows` is either a function (sql) => rows[] (a router) or a fixed array.
function makeCtx({ allCols = [], metricCols = [], blobCols = [], where = '', rows = [], state = {} } = {}) {
  const queryRows = typeof rows === 'function' ? async sql => rows(sql) : async () => rows;
  return {
    where,
    state: { groupCol: null, ...state },
    schema: { allCols, metricCols, blobCols, dimensionInfo: {} },
    queryRows,
    sql: { q, andWhere, groupCol: () => groupCol(state), groupExpr: () => groupExpr(state) },
    plot: { escapeHtml, niceName, humanList, formatBytes },
  };
}

describe('summary widget · condensedMessage', () => {
  const summary = widget('summary');

  it('reports a plain file count when there is one image per file', async () => {
    const out = await summary.condensedMessage(makeCtx({
      allCols: ['size_bytes', 'file_extension'],
      rows: [{ file_count: 42 }],
    }));
    expect(out).toContain('<strong>42 files</strong>');
    expect(out).toContain('one image per file');
  });

  it('reports the image total when files contain multiple images', async () => {
    const out = await summary.condensedMessage(makeCtx({
      allCols: ['size_bytes', 'file_extension', 'n_images'],
      rows: [{ file_count: 10, n_images_sum: 30 }],
    }));
    expect(out).toContain('<strong>10 files</strong>');
    expect(out).toContain('<strong>30 images</strong>');
  });

  it('uses the singular for a single file', async () => {
    const out = await summary.condensedMessage(makeCtx({
      allCols: ['size_bytes', 'file_extension'],
      rows: [{ file_count: 1 }],
    }));
    expect(out).toContain('<strong>1 file</strong>');
  });
});

describe('file-stats widget · condensedMessage', () => {
  const fileStats = widget('file-stats');
  // The widget fires two queries in parallel: distinct extensions, then per-group
  // counts. Route by a marker in the SQL so order doesn't matter.
  const router = ({ exts, counts }) => sql =>
    /file_extension/.test(sql)
      ? exts.map(ext => ({ ext }))
      : counts.map(c => ({ c }));

  it('says all files share one extension when nothing varies', async () => {
    const out = await fileStats.condensedMessage(makeCtx({
      allCols: ['file_extension', 'size_bytes'],
      rows: router({ exts: ['tif'], counts: [5] }),
    }));
    expect(out).toBe('All <strong>tif</strong>.');
  });

  it('flags mixed file types as a warning', async () => {
    const out = await fileStats.condensedMessage(makeCtx({
      allCols: ['file_extension', 'size_bytes'],
      rows: router({ exts: ['tif', 'png'], counts: [5] }),
    }));
    expect(out).toMatchObject({ warning: true });
    expect(out.text).toContain('file type');
  });

  it('flags an imbalanced per-group file count', async () => {
    const out = await fileStats.condensedMessage(makeCtx({
      allCols: ['file_extension', 'size_bytes'],
      rows: router({ exts: ['tif'], counts: [10, 3] }),
    }));
    expect(out).toMatchObject({ warning: true });
    expect(out.text).toContain('file count');
  });
});

describe('sunburst widget · condensedMessage', () => {
  const sunburst = widget('sunburst');

  it('describes the folder structure by file count', async () => {
    const out = await sunburst.condensedMessage(makeCtx({ rows: [{ n: 7 }] }));
    expect(out).toBe('Folder structure of <strong>7 files</strong>.');
  });

  it('uses the singular for one file', async () => {
    const out = await sunburst.condensedMessage(makeCtx({ rows: [{ n: 1 }] }));
    expect(out).toContain('<strong>1 file</strong>');
  });
});

describe('image-table widget · condensedMessage', () => {
  it('lists the image count with a sort/search hint', async () => {
    const out = await widget('image-table').condensedMessage(makeCtx({ rows: [{ n: 3 }] }));
    expect(out).toContain('<strong>3 images</strong>');
    expect(out).toContain('sort');
  });
});

describe('histogram widget · condensedMessage', () => {
  const histogram = widget('histogram');

  it('teases the comparison when every image has a histogram', async () => {
    const out = await histogram.condensedMessage(makeCtx({ rows: [{ total: 10, n: 10 }] }));
    expect(out).toBe('Compare pixel intensity distributions across groups.');
  });

  it('warns when only some images have histograms', async () => {
    const out = await histogram.condensedMessage(makeCtx({ rows: [{ total: 10, n: 4 }] }));
    expect(out).toMatchObject({ warning: true });
    expect(out.text).toContain('<strong>4</strong>/10');
  });
});

describe('mosaic widget · condensedMessage', () => {
  it('names the metric the thumbnails are sorted by', async () => {
    const out = await widget('mosaic').condensedMessage(makeCtx({
      blobCols: ['thumbnail'],
      metricCols: ['mean_intensity'],
      allCols: ['mean_intensity'],
      rows: [{ n: 12 }],
    }));
    expect(out).toContain('Thumbnails sorted by');
    expect(out).toContain(niceName('mean_intensity'));
  });
});

describe('metadata widget · condensedMessage', () => {
  const metadata = widget('metadata');
  // distinct-count probe per DIST col, then a MODE() lookup for the headline.
  const router = ({ nDistinct, mode }) => sql =>
    /COUNT\(DISTINCT/.test(sql) ? [{ n: nDistinct }] : [{ dtype: mode }];

  it('reports a single pixel type when dtype is uniform', async () => {
    const out = await metadata.condensedMessage(makeCtx({
      allCols: ['dtype'],
      rows: router({ nDistinct: 1, mode: 'uint8' }),
    }));
    expect(out).toBe('All <strong>uint8</strong> pixels.');
  });

  it('warns when pixel types are mixed', async () => {
    const out = await metadata.condensedMessage(makeCtx({
      allCols: ['dtype'],
      rows: router({ nDistinct: 2, mode: 'uint8' }),
    }));
    expect(out).toMatchObject({ warning: true });
    expect(out.text).toContain('pixel type');
  });
});

describe('dim-size widget · condensedMessage', () => {
  const dimSize = widget('dim-size');
  // COUNT(*) total first, then the X/Y distinctness probe.
  const router = ({ total, ndx, ndy }) => sql =>
    /COUNT\(DISTINCT/.test(sql) ? [{ ndx, ndy }] : [{ total }];

  it('says all images are 2D and the same size', async () => {
    const out = await dimSize.condensedMessage(makeCtx({
      allCols: ['size_X', 'size_Y'],
      rows: router({ total: 5, ndx: 1, ndy: 1 }),
    }));
    expect(out).toBe('All 2D, same size.');
  });

  it('warns when image size varies', async () => {
    const out = await dimSize.condensedMessage(makeCtx({
      allCols: ['size_X', 'size_Y'],
      rows: router({ total: 5, ndx: 3, ndy: 1 }),
    }));
    expect(out).toMatchObject({ warning: true });
    expect(out.text).toContain('image size');
  });
});

describe('custom-plot widget · condensedMessage', () => {
  const customPlot = widget('custom-plot');

  it('invites exploration with no suggestion when no familiar columns exist', async () => {
    const out = await customPlot.condensedMessage(makeCtx({ allCols: [] }));
    expect(out).toContain('pick any two columns');
    expect(out).not.toContain('e.g.');
  });

  it('suggests a sharpness comparison when intensity + laplacian are present', async () => {
    const out = await customPlot.condensedMessage(makeCtx({
      allCols: ['mean_intensity', 'laplacian_variance'],
    }));
    expect(out).toContain('Laplacian Variance');
  });
});

describe('violin widgets · condensedMessage', () => {
  it('basic teases per-image intensity stats when pixel types are uniform', async () => {
    const out = await widget('violin-basic').condensedMessage(makeCtx({ allCols: ['mean_intensity'] }));
    expect(out).toBe('Compare per-image intensity statistics across groups.');
  });

  it('basic warns when pixel types are mixed (intensities not comparable)', async () => {
    const out = await widget('violin-basic').condensedMessage(makeCtx({
      allCols: ['mean_intensity', 'dtype'],
      rows: [{ n_dtypes: 2 }],
    }));
    expect(out).toMatchObject({ warning: true });
    expect(out.text).toContain('Mixed pixel types');
  });

  it('quality returns a static, query-free teaser', async () => {
    const out = await widget('violin-quality').condensedMessage(makeCtx());
    expect(out).toBe('Spot quality differences and outliers across groups.');
  });
});

describe('stats-across-dims widgets · condensedMessage', () => {
  it('names the varying dimensions in upper case, sorted', async () => {
    const ctx = makeCtx();
    ctx.schema.dimensionInfo = { c: [0, 1], t: [0, 1, 2] };
    const out = await widget('stats-across-dims-basic').condensedMessage(ctx);
    expect(out).toBe('Per-slice trend along <strong>C, T</strong>.');
  });

  it('returns null when no dimension varies', async () => {
    const out = await widget('stats-across-dims-quality').condensedMessage(makeCtx());
    expect(out).toBeNull();
  });
});

describe('condensedMessage robustness', () => {
  // A failing query (DB hiccup, unexpected schema) must degrade to null rather
  // than throwing and breaking the whole condensed gallery render.
  const summaries = [...byId.values()].filter(p => typeof p.condensedMessage === 'function');

  it('covers every widget that declares a condensedMessage', () => {
    expect(summaries.length).toBeGreaterThanOrEqual(11);
  });

  it.each(summaries.map(p => [p.id, p]))(
    '%s never throws when the query fails',
    async (_id, plugin) => {
      const ctx = makeCtx({
        allCols: ['dtype', 'mean_intensity'],
        rows: () => { throw new Error('db down'); },
      });
      const out = await plugin.condensedMessage(ctx);
      // Either a graceful null or a still-valid summary (string / {text}).
      const ok = out == null || typeof out === 'string' || typeof out?.text === 'string';
      expect(ok).toBe(true);
    },
  );
});
