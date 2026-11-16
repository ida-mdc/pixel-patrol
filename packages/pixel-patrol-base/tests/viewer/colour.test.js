import { describe, expect, it } from 'vitest';
import widgets, {
  colourAssessment, groupCentre, hsvSaturation, hsvToRgb, hueDegrees, hueName, imageFromRow, imageTable, overviewSummary,
  quantile, wheelPoint, wheelRadius,
} from '../../src/pixel_patrol_base/viewer/plugin_colour.js';

const [widget] = widgets;
const luma = ([r, g, b]) => 0.299 * r + 0.587 * g + 0.114 * b;
const image = (rgb, extra = {}) => ({ group: 'g', key: 'k', rgb, brightness: luma(rgb), contrast: 10, saturation: hsvSaturation(rgb), coherence: null, ...extra });
const images = (rgb, n = 5, extra = {}) => Array.from({ length: n }, () => image(rgb, extra));
const angle = (degrees) => [Math.cos((degrees * Math.PI) / 180), Math.sin((degrees * Math.PI) / 180)];

describe('colour maths', () => {
  it('finds hues, names them, and takes quantiles', () => {
    expect(hueDegrees([200, 100, 50])).toBeCloseTo(19.1, 1);
    expect(hueDegrees([0, 0, 0])).toBeNull();
    expect([0, 60, 120, 180, 240, 300, 359].map(hueName)).toEqual(['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'red']);
    expect(quantile([1, 2, 3, 4, 5], 0.5)).toBe(3);
    expect(quantile([], 0.5)).toBeNull();
  });

  it('reads saturation and converts hues to colours', () => {
    expect(hsvSaturation([200, 100, 50])).toBeCloseTo(0.75);
    expect(hsvSaturation([0, 0, 0])).toBe(0);
    expect([0, 120, 240].map(hue => hsvToRgb(hue, 1, 1))).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    expect(hsvToRgb(60, 0, 0.5)).toEqual([0.5, 0.5, 0.5]);
  });
});

describe('hue/saturation wheel', () => {
  it.each([[[1, 0, 0], 0], [[1, 1, 0], 60], [[0, 1, 0], 120], [[0, 1, 1], 180], [[0, 0, 1], 240], [[1, 0, 1], 300]])(
    'puts %j on the rim at %i degrees', (rgb, degrees) => {
      const { x, y, saturation } = wheelPoint(rgb);
      expect(saturation).toBe(1);
      [x, y].forEach((value, i) => expect(value).toBeCloseTo(angle(degrees)[i], 6));
    });

  it('puts grey in the centre, ignores a missing colour, and grows with the square root of saturation whatever the brightness', () => {
    expect(wheelPoint([120, 120, 120])).toMatchObject({ x: 0, y: 0 });
    expect(wheelPoint([0, 0, 0])).toMatchObject({ x: 0, y: 0 });
    expect(wheelPoint(null)).toBeNull();
    expect([wheelRadius(0.25), wheelRadius(2)]).toEqual([0.5, 1]);
    const [dim, bright] = [wheelPoint([20, 10, 10]), wheelPoint([200, 100, 100])];
    expect(dim.x).toBeCloseTo(bright.x, 6);
    expect(dim.y).toBeCloseTo(bright.y, 6);
  });
});

describe('assessment', () => {
  it.each([
    ['a neutral group', images([100, 100, 100]), 'neutral'],
    ['just under the neutral limit', images([104, 100, 100]), 'neutral'],
    ['a slight cast', images([110, 100, 100]), 'slight red cast'],
    ['a strong cast', images([200, 100, 50]), 'strong orange cast'],
    ['a single-hue tint', images([200, 100, 50], 5, { coherence: 0.95 }), 'strong orange cast'],
    ['colourful pixels whose hues disagree', images([150, 137, 189], 5, { saturation: 0.72, coherence: 0.1 }), 'mixed hues, average leans blue'],
    ['hues that cancel to grey', images([120, 120, 120], 5, { saturation: 0.72, coherence: 0.1 }), 'mixed hues, average is neutral'],
    ['no hue measurement', images([150, 137, 189]), 'strong blue cast'],
  ])('reads the average colour of %s', (_, group, text) => {
    expect(colourAssessment(group).balance.text).toBe(text);
  });

  it('reads the verdict from the point the group is drawn at, further out for a stronger cast', () => {
    const [near, far] = [groupCentre(images([104, 100, 100])), groupCentre(images([112, 100, 100]))];
    expect(Math.hypot(far.x, far.y)).toBeGreaterThan(Math.hypot(near.x, near.y));
    expect(groupCentre(images([110, 100, 100])).saturation).toBeCloseTo(0.0909, 3);
    expect(groupCentre(images([200, 100, 50])).hue).toBeCloseTo(19.1, 1);
    expect(groupCentre([])).toBeNull();
  });

  it.each([[0, 'grey'], [0.05, 'grey'], [0.2, 'muted'], [0.4, 'moderate'], [0.75, 'vivid']])(
    'calls saturation %f %s', (saturation, word) => {
      expect(colourAssessment(images([200, 100, 50], 5, { saturation })).saturation.text).toBe(word);
    });

  describe('a group that mixes grey and vivid images', () => {
    const grey = { saturation: 0.002, coherence: 0.002 };
    const vivid = { saturation: 0.73, coherence: 0.74 };
    const mixed = [
      ...images([100, 100, 100], 4, grey), image([105, 100, 112], { saturation: 0.12, coherence: 0.49 }),
      image([60, 120, 190], vivid), image([110, 60, 40], { saturation: 0.54, coherence: 0.75 }),
    ];

    it('lists the classes instead of a median that hides them', () => {
      const found = colourAssessment(mixed);
      expect(found.saturation).toMatchObject({ varied: true, text: 'varies: 4 grey, 1 muted, 1 moderate, 1 vivid' });
      expect(found.saturation.parts).toEqual([['grey', 4], ['muted', 1], ['moderate', 1], ['vivid', 1]]);
      expect(found.balance).toMatchObject({ varied: true });
      expect(found.balance.parts[0]).toEqual(['neutral', 4]);
      expect(found.balance.text).toContain('strong cast');
      expect(overviewSummary(new Map([['day', mixed]]))).toBe('7 RGB images, average colour <strong>varies between images</strong> in every group.');
    });

    it('says mostly when a large majority agrees, and nothing extra when all do', () => {
      const mostly = [...images([100, 100, 100], 7, grey), image([60, 120, 190], vivid)];
      expect([colourAssessment(mostly).saturation.text, colourAssessment(mostly).balance.text]).toEqual(['mostly grey', 'mostly neutral']);
      expect(colourAssessment(images([100, 100, 100], 5, grey)).saturation.text).toBe('grey');
    });
  });
});

describe('SQL', () => {
  it('turns a slice row into an image', () => {
    const row = { __group__: 'a', __key__: '/d/x.png', __name__: 'x.png', __frn__: '7', r: 200, g: '100', b: 50, brightness: 124.2, contrast: 12, saturation: 0.75, coherence: null };
    expect(imageFromRow(row)).toEqual({ group: 'a', key: '/d/x.png', name: 'x.png', frn: 7, rgb: [200, 100, 50], brightness: 124.2, contrast: 12, saturation: 0.75, coherence: null });
    expect(imageFromRow({ ...row, __name__: undefined, __frn__: null })).toMatchObject({ name: '/d/x.png', frn: null });
  });

  it('builds one row per complete image from the channel means and the processor columns, under the current filter', () => {
    const ctx = {
      schema: { allCols: [] }, state: { dimensions: {} }, where: 'WHERE "day" = 1',
      sql: { dimSubsetWhere: () => ['"obs_level" = 0'], andWhere: (w, c) => (w ? `${w} AND ${c}` : `WHERE ${c}`) },
    };
    const sql = imageTable(ctx);
    ['r', 'g', 'b'].forEach(name => expect(sql).toContain(`"ch_${name}_mean"`));
    expect(sql).toContain('"dim_s" = 2');
    expect(sql).toContain('"obs_level" = 0');
    expect(sql).toContain('GROUP BY "path"');
    expect(sql).toContain('WHERE "day" = 1 AND "ch_r_mean" IS NOT NULL');
    expect(sql).toContain('ROUND("rgb_saturation_mean"');
    expect(sql).not.toMatch(/COALESCE|std_intensity/);
  });
});

const rows = (group, key, r, g, b, extra = {}) => ({ __group__: group, __key__: key, __name__: `${key}.png`, __frn__: key.charCodeAt(0), r, g, b, brightness: luma([r, g, b]), contrast: 10, saturation: hsvSaturation([r, g, b]), coherence: null, ...extra });
const DATA = [rows('warm', 'a', 200, 100, 50), rows('warm', 'b', 190, 100, 55), rows('grey', 'c', 100, 100, 100)];

const makeContext = (data, groups = ['warm', 'grey']) => {
  const calls = { distributions: [], plots: [], minis: [], grids: [], registered: [], queries: [] };
  return {
    calls, groups, schema: { allCols: [] }, state: { dimensions: {}, showSignificance: false }, where: '',
    sql: { groupCol: () => '"g"', groupExpr: () => '"g" AS __group__', dimSubsetWhere: () => ['"obs_level" = 0'], andWhere: (w, c) => (w ? `${w} AND ${c}` : `WHERE ${c}`) },
    queryRows: async (sql) => { calls.queries.push(sql); return data; },
    groupLabel: (g) => g.toUpperCase(), color: { group: () => '#4c78a8' },
    plot: {
      LAYOUT: {}, escapeHtml: String, groupingLabel: () => 'Group',
      append: (host, traces, layout) => { const div = document.createElement('div'); host.appendChild(div); calls.plots.push({ traces, layout, div }); return div; },
      registerPointPlot: (div, c, metric) => calls.registered.push({ div, metric }),
      appendMini: (host, traces, layout) => calls.minis.push({ traces, layout }),
      flexGrid: (parent, perRow) => { const wrap = document.createElement('div'); parent.appendChild(wrap); calls.grids.push(perRow); return { wrap, flexBasisPct: 48 }; },
      engine: { CONSTANTS: { MAX_VIOLIN_POINTS: 5000 }, renderDistribution: async (host, c, spec) => { calls.distributions.push(spec); return true; } },
    },
  };
};

describe('widget', () => {
  it('needs the channel means and the processor columns along S', () => {
    const base = { allCols: ['mean_intensity', 'rgb_luma_std', 'rgb_saturation_mean'], dimCols: ['dim_s'] };
    expect(widget.requires(base)).toBe(true);
    expect(widget.requires({ ...base, dimCols: ['dim_t'] })).toBe(false);
    expect(widget.requires({ ...base, allCols: ['mean_intensity'] })).toBe(false);
  });

  it('puts the wheel without a legend next to the table, and the shared plots below', async () => {
    const ctx = makeContext(DATA);
    const container = document.createElement('div');
    await widget.render(container, ctx);

    const [top] = container.children;
    expect(top.style.display).toBe('flex');
    expect(top.style.alignItems).toBe('center');
    const table = top.querySelector('table');
    expect(table.textContent).toContain('WARM');
    expect(table.textContent).not.toContain('image');
    expect([...table.querySelectorAll('th')].map(th => th.textContent)).toEqual(['Group', 'Average colour', 'Saturation']);
    expect(ctx.calls.plots[0].layout.showlegend).toBe(false);

    const [brightness, contrast, , cast] = ctx.calls.distributions;
    expect(ctx.calls.distributions.map(spec => spec.numCol)).toEqual(['colour_brightness', 'rgb_luma_std', 'colour_saturation', 'colour_cast']);
    expect(brightness).toMatchObject({ series: { isCategory: true }, categoriesOrder: ['warm', 'grey'], yLabel: 'Brightness (luma)' });
    expect(brightness.source.table).toContain('pp_all');
    expect(contrast.title).toContain('Distribution of Contrast<br>');
    expect(cast.title).toContain('Distribution of Colour cast<br>');
    expect(cast.title).toContain('n=3');
    expect(cast.yLabel).toBe('Colour cast (saturation of the average colour)');
    expect(cast.layout.yaxis).toEqual({ title: cast.yLabel, rangemode: 'nonnegative' });
    expect(ctx.calls.grids).toEqual([3]);
  });

  it('gives many groups one plot per row, like the other widgets', async () => {
    const groups = ['a', 'b', 'c', 'd'];
    const ctx = makeContext(groups.map(g => rows(g, g, 100, 100, 100)), groups);
    await widget.render(document.createElement('div'), ctx);
    expect(ctx.calls.grids).toEqual([1]);
  });

  it('makes the wheel dots clickable, named on hover, and keeps the group medians out of the selection', async () => {
    const ctx = makeContext(DATA);
    await widget.render(document.createElement('div'), ctx);
    const [{ traces, div }] = ctx.calls.plots;
    const warm = traces.find(trace => trace.name === 'WARM' && trace.customdata);
    expect(warm.text[0]).toContain('<b>a.png</b>');
    expect(warm.customdata).toEqual(['a'.charCodeAt(0), 'b'.charCodeAt(0)]);
    expect(warm.hovertemplate).toContain('%{text}');
    expect(ctx.calls.registered).toEqual([{ div, metric: 'colour_cast' }]);
    const median = traces.find(trace => trace.marker?.symbol === 'diamond');
    expect(median.customdata).toBeUndefined();
    expect(median.text[0]).toContain('median of the group');
    expect(ctx.calls.queries[0]).toContain('"file_row_number" AS __frn__');

    delete ctx.plot.registerPointPlot;
    await widget.render(document.createElement('div'), ctx);
  });

  it('shows a group that differs as "varies" with the classes listed under it, and says so when there are no RGB images', async () => {
    const mixed = [rows('a', 'x', 100, 100, 100, { saturation: 0.001 }), rows('a', 'y', 100, 100, 100, { saturation: 0.001 }), rows('a', 'z', 60, 120, 190, { saturation: 0.7, coherence: 0.7 })];
    const container = document.createElement('div');
    await widget.render(container, makeContext(mixed, ['a']));
    const cell = container.querySelectorAll('tbody td')[2];
    expect(cell.textContent.startsWith('varies')).toBe(true);
    expect(cell.querySelector('div').textContent).toBe('2 grey · 1 vivid');

    const empty = document.createElement('div');
    await widget.render(empty, makeContext([]));
    expect(empty.textContent).toContain('No RGB images found');
  });
});

describe('overview tile', () => {
  it('summarises whether the groups share an average colour', async () => {
    expect(await widget.overviewMessage(makeContext(DATA))).toBe('3 RGB images, average colour <strong>differs between groups</strong>.');
    const same = new Map([['a', images([100, 100, 100], 1)], ['b', images([110, 110, 110], 1)]]);
    expect(overviewSummary(same)).toBe('2 RGB images, average colour <strong>neutral</strong> in every group.');
    expect(await widget.overviewMessage(makeContext([]))).toBeNull();
  });

  it('draws the wheel without legend or labels, and nothing without colour data', async () => {
    const ctx = makeContext(DATA);
    expect(await widget.overviewPlot(document.createElement('div'), ctx)).toBe(true);
    const [{ traces, layout }] = ctx.calls.minis;
    expect(layout).toMatchObject({ showlegend: false, annotations: [] });
    expect(traces.some(trace => trace.name === 'WARM')).toBe(true);
    expect(await widget.overviewPlot(document.createElement('div'), makeContext([]))).toBe(false);
  });
});
