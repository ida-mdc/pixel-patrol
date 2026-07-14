import { describe, it, expect } from 'vitest';
import { choosePlotKind, sortCats, COUNT_Y, NULL_LABEL, niceDateAxis, setDateCols, DATE_COLS } from '../plot-engine.js';

// choosePlotKind is query-free apart from two injected cardinality probes, so we
// drive it with fakes. `card` maps column → distinct-count for both probes.
function chooser(numericCols, card, overrides = {}) {
  return (xCol, yCol) => choosePlotKind({
    xCol, yCol,
    numericSet: new Set(numericCols),
    catCount: async (c) => card[c] ?? 1,
    distinct: async (c) => card[c] ?? 1,
    ...overrides,
  });
}

describe('choosePlotKind', () => {
  it('numeric × numeric → scatter', async () => {
    const pick = chooser(['a', 'b'], { a: 100, b: 100 });
    expect((await pick('a', 'b')).kind).toBe('scatter');
  });

  it('numeric × near-constant numeric → numNumTable', async () => {
    const pick = chooser(['a', 'b'], { a: 100, b: 1 });
    expect((await pick('a', 'b')).kind).toBe('numNumTable');
  });

  it('categorical × categorical → heatmap', async () => {
    const pick = chooser([], { a: 3, b: 4 });
    expect((await pick('a', 'b')).kind).toBe('heatmap');
  });

  it('too many categories on a heatmap axis → message', async () => {
    const pick = chooser([], { a: 3, b: 999 });
    const r = await pick('a', 'b');
    expect(r.kind).toBe('message');
    expect(r.message).toMatch(/too many/);
  });

  it('categorical × numeric → distribution', async () => {
    const pick = chooser(['n'], { cat: 4, n: 50 });
    const r = await pick('cat', 'n');
    expect(r.kind).toBe('distribution');
    expect(r.catCol).toBe('cat');
    expect(r.numCol).toBe('n');
    expect(r.flipped).toBe(false);
  });

  it('numeric × categorical → distribution, flipped', async () => {
    const pick = chooser(['n'], { cat: 4, n: 50 });
    const r = await pick('n', 'cat');
    expect(r.kind).toBe('distribution');
    expect(r.catCol).toBe('cat');
    expect(r.numCol).toBe('n');
    expect(r.flipped).toBe(true);
  });

  it('single category or near-constant numeric → catNumTable', async () => {
    expect((await chooser(['n'], { cat: 1, n: 50 })('cat', 'n')).kind).toBe('catNumTable');
    expect((await chooser(['n'], { cat: 4, n: 1 })('cat', 'n')).kind).toBe('catNumTable');
  });

  it('too many categories on a categorical axis → message', async () => {
    const r = await chooser(['n'], { cat: 999, n: 50 })('cat', 'n');
    expect(r.kind).toBe('message');
  });

  it('y = (count) → countBar, or countTable for a single category', async () => {
    const pick = chooser([], { a: 5 });
    expect((await pick('a', COUNT_Y)).kind).toBe('countBar');
    expect((await chooser([], { a: 1 })('a', COUNT_Y)).kind).toBe('countTable');
  });

  it('count over too many categories → message', async () => {
    const r = await chooser([], { a: 999 })('a', COUNT_Y);
    expect(r.kind).toBe('message');
  });

  it('same column for X and Y → invalid', async () => {
    expect((await chooser(['a'], { a: 100 })('a', 'a')).kind).toBe('invalid');
  });

  it('respects a custom maxCat', async () => {
    const r = await choosePlotKind({
      xCol: 'a', yCol: COUNT_Y, numericSet: new Set(),
      catCount: async () => 10, distinct: async () => 10, maxCat: 5,
    });
    expect(r.kind).toBe('message');
  });
});

describe('sortCats', () => {
  it('sorts alphabetically with the "(missing)" bucket last', () => {
    expect(sortCats(['b', NULL_LABEL, 'a'])).toEqual(['a', 'b', NULL_LABEL]);
  });
  it('leaves a list without a missing bucket sorted', () => {
    expect(sortCats(['c', 'a', 'b'])).toEqual(['a', 'b', 'c']);
  });
});

describe('niceDateAxis', () => {
  it('pins a floor dtick and pads the range for a near-zero-variance span', () => {
    const t = Date.parse('2026-01-01T00:00:00Z');
    const axis = niceDateAxis([t, t + 1], 'Modified');
    expect(axis.type).toBe('date');
    expect(axis.dtick).toBe(1000);
    expect(axis.range[0]).toBeLessThan(t);
    expect(axis.range[1]).toBeGreaterThan(t + 1);
  });

  it('still floors the dtick to 1s for a small-but-nonzero span', () => {
    const min = Date.parse('2026-01-01T00:00:00Z');
    const max = min + 2000; // 2 seconds apart - Plotly would go sub-second here
    const axis = niceDateAxis([min, max], 'Modified');
    expect(axis.dtick).toBe(1000);
    expect(axis.range).toBeUndefined(); // span exceeds one tick, no padding needed
  });

  it('defers to Plotly (no dtick) once the span clears the sub-second floor', () => {
    const min = Date.parse('2026-01-01T00:00:00Z');
    const max = min + 3_600_000; // 1 hour apart
    const axis = niceDateAxis([min, max], 'Modified');
    expect(axis.dtick).toBeUndefined();
    expect(axis.range).toBeUndefined();
    expect(axis.type).toBe('date');
  });

  it('floors just below the span threshold and defers at it', () => {
    const t = Date.parse('2026-01-01T00:00:00Z');
    expect(niceDateAxis([t, t + 3999], 'Modified').dtick).toBe(1000);
    expect(niceDateAxis([t, t + 4000], 'Modified').dtick).toBeUndefined();
  });

  it('handles no valid values without throwing', () => {
    const axis = niceDateAxis([], 'Modified');
    expect(axis.type).toBe('date');
    expect(axis.dtick).toBeUndefined();
  });

  it('scales tick labels by dtick instead of forcing one fixed format', () => {
    const min = Date.parse('2026-01-01T00:00:00Z');
    const max = min + 365 * 86_400_000; // a year apart - ticks should read as bare dates
    const axis = niceDateAxis([min, max], 'Modified');
    expect(axis.tickformat).toBeUndefined();
    expect(axis.tickformatstops).toBeInstanceOf(Array);
    expect(axis.tickformatstops.at(-1)).toMatchObject({ dtickrange: [86_400_000, null], value: '%Y-%m-%d' });
    // hover always shows full precision - it's a single value, not a whole axis of them
    expect(axis.hoverformat).toBe('%Y-%m-%d %H:%M:%S');
  });
});

describe('setDateCols', () => {
  it('replaces the DATE_COLS set contents', () => {
    setDateCols(['acquisition_date']);
    expect(DATE_COLS.has('acquisition_date')).toBe(true);
    expect(DATE_COLS.has('modification_date')).toBe(false);

    setDateCols(['modification_date']);
    expect(DATE_COLS.has('modification_date')).toBe(true);
    expect(DATE_COLS.has('acquisition_date')).toBe(false);
  });
});
