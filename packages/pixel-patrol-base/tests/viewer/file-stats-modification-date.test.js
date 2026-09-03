import { describe, it, expect } from 'vitest';
import { renderModificationDates } from '../../src/pixel_patrol_base/viewer/plugin_file_stats.js';

// renderModificationDates only ever reaches for ctx.queryRows, ctx.sql and the
// grouped-bar plot helpers, so a thin fake ctx drives it deterministically.
// Routes bucket queries by the STRFTIME format literal so callers don't need
// to know the exact SQL shape, only which granularity they expect.
function makeCtx(byFmt = {}) {
  const appendCalls = [];
  return {
    ctx: {
      where: '',
      groups: [],
      sql: {
        andWhere: (where, cond) => (cond ? `WHERE ${cond}` : where),
        groupCol: () => `'__ALL__'`,
        fileCount: () => 'COUNT(DISTINCT "path")',
      },
      queryRows: async sql => {
        const fmt = sql.match(/STRFTIME\([^,]+,\s*'([^']+)'\)/)?.[1];
        return byFmt[fmt] ?? [];
      },
      plot: {
        groupedBarTraces: (categories, getValue) => ({ categories, getValue }),
        append: (container, traces, layout) => appendCalls.push({ traces, layout }),
        bargap: () => 0.3,
      },
    },
    appendCalls,
  };
}

describe('renderModificationDates', () => {
  it('reports a single shared timestamp as a full-precision invariant', async () => {
    const { ctx } = makeCtx();
    const invariants = [];
    await renderModificationDates({}, ctx, [{
      min_fmt: '2026-07-02 23:00:31', max_fmt: '2026-07-02 23:00:31', span_ms: 0, n_unique: 1,
    }], invariants);
    expect(invariants).toEqual([['Modification Date', '2026-07-02 23:00:31']]);
  });

  it('reports sub-second variance within one second as a range, not a bare date', async () => {
    const { ctx } = makeCtx();
    const invariants = [];
    await renderModificationDates({}, ctx, [{
      min_fmt: '2026-07-02 23:00:31', max_fmt: '2026-07-02 23:00:31', span_ms: 248, n_unique: 40,
    }], invariants);
    expect(invariants).toEqual([['Modification Date', '2026-07-02 23:00:31 (span < 1s)']]);
  });

  it('reports sub-second variance crossing a second boundary with both endpoints', async () => {
    const { ctx } = makeCtx();
    const invariants = [];
    await renderModificationDates({}, ctx, [{
      min_fmt: '2026-07-02 23:00:31', max_fmt: '2026-07-02 23:00:32', span_ms: 500, n_unique: 2,
    }], invariants);
    expect(invariants).toEqual([['Modification Date', '2026-07-02 23:00:31 – 2026-07-02 23:00:32 (span < 1s)']]);
  });

  it('charts by day when the span crosses multiple days', async () => {
    const { ctx, appendCalls } = makeCtx({
      '%Y-%m-%d': [
        { bucket: '2026-07-01', __group__: '__ALL__', count: 2 },
        { bucket: '2026-07-02', __group__: '__ALL__', count: 3 },
      ],
    });
    const invariants = [];
    await renderModificationDates({}, ctx, [{
      min_fmt: 'a', max_fmt: 'b', span_ms: 3 * 86_400_000, n_unique: 5,
    }], invariants);
    expect(invariants).toEqual([]);
    expect(appendCalls).toHaveLength(1);
    expect(appendCalls[0].layout.title.text).toBe('File Count by Modification Date');
    expect(appendCalls[0].layout.xaxis.title).toBe('Date');
  });

  it('charts by hour when the whole span is under a day', async () => {
    const { ctx, appendCalls } = makeCtx({
      '%Y-%m-%d %H:00': [
        { bucket: '2026-07-02 10:00', __group__: '__ALL__', count: 4 },
        { bucket: '2026-07-02 11:00', __group__: '__ALL__', count: 6 },
      ],
    });
    const invariants = [];
    await renderModificationDates({}, ctx, [{
      min_fmt: 'a', max_fmt: 'b', span_ms: 2 * 3_600_000, n_unique: 10,
    }], invariants);
    expect(invariants).toEqual([]);
    expect(appendCalls[0].layout.xaxis.title).toBe('Hour');
  });

  it('rolls up to months once daily buckets exceed the readable cap', async () => {
    const dayRows = Array.from({ length: 25 }, (_, i) => ({
      bucket: `2026-01-${String(i + 1).padStart(2, '0')}`, __group__: '__ALL__', count: 1,
    }));
    const { ctx, appendCalls } = makeCtx({
      '%Y-%m-%d': dayRows,
      '%Y-%m': [{ bucket: '2026-01', __group__: '__ALL__', count: 25 }],
    });
    const invariants = [];
    await renderModificationDates({}, ctx, [{
      min_fmt: 'a', max_fmt: 'b', span_ms: 30 * 86_400_000, n_unique: 25,
    }], invariants);
    expect(appendCalls[0].layout.xaxis.title).toBe('Month');
  });
});
