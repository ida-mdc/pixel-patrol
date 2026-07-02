import { describe, it, expect } from 'vitest';
import { choosePlotKind, sortCats, COUNT_Y, NULL_LABEL } from '../plot-engine.js';

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
