import { describe, it, expect } from 'vitest';
import {
  dtypeKind,
  remapToAxis,
  accumulateGroupHistograms,
  classifyDtypes,
  DTYPE_NORM,
  populatedExtent,
} from '../../src/pixel_patrol_base/viewer/plugin_histogram.js';

const NBINS = 256;

// A length-NBINS array with `mass` at bin `at`, zero elsewhere.
function spike(at, mass = 1) {
  const a = new Array(NBINS).fill(0);
  a[at] = mass;
  return a;
}

describe('dtypeKind', () => {
  it('buckets by prefix', () => {
    expect(dtypeKind('uint8')).toBe('uint');
    expect(dtypeKind('uint64')).toBe('uint');
    expect(dtypeKind('int16')).toBe('int');
    expect(dtypeKind('float32')).toBe('float');
  });

  it('returns null for unrecognized dtypes', () => {
    expect(dtypeKind('bool')).toBeNull();
    expect(dtypeKind(undefined)).toBeNull();
  });
});

describe('DTYPE_NORM', () => {
  it('has a normalization divisor for every integer bit width up to 64', () => {
    for (const bits of [8, 16, 32, 64]) {
      expect(DTYPE_NORM[`uint${bits}`]).toBe(2 ** bits);
      expect(DTYPE_NORM[`int${bits}`]).toBe(2 ** (bits - 1));
    }
  });
});

describe('classifyDtypes', () => {
  const row = dtype => ({ dtype });

  it('assumes a single uint panel when the dataset has no dtype column at all', () => {
    expect(classifyDtypes([], false)).toEqual({ presentKinds: ['uint'], multiKind: false, unsupportedDtypes: [] });
  });

  it('reports a single, non-multi kind for a uniform dtype', () => {
    const out = classifyDtypes([row('uint16')], true);
    expect(out).toEqual({ presentKinds: ['uint'], multiKind: false, unsupportedDtypes: [] });
  });

  it('flags multiKind when more than one kind is present, in a stable uint/int/float order', () => {
    const out = classifyDtypes([row('float32'), row('uint8')], true);
    expect(out.presentKinds).toEqual(['uint', 'float']);
    expect(out.multiKind).toBe(true);
  });

  it('surfaces dtypes it cannot classify instead of silently dropping them', () => {
    const out = classifyDtypes([row('uint8'), row('bool')], true);
    expect(out.presentKinds).toEqual(['uint']);
    expect(out.unsupportedDtypes).toEqual(['bool']);
  });

  it('reports zero present kinds (not a crash) when every dtype is unsupported', () => {
    const out = classifyDtypes([row('bool')], true);
    expect(out.presentKinds).toEqual([]);
    expect(out.multiKind).toBe(false);
    expect(out.unsupportedDtypes).toEqual(['bool']);
  });
});

describe('remapToAxis', () => {
  it('is a no-op when source and target axes already match', () => {
    const values = spike(10, 3);
    const out = remapToAxis(values, 0, 256, 0, 256);
    expect(out).toBe(values); // same reference: no allocation on the fast path
  });

  it('redistributes a narrow source range onto a wider shared target', () => {
    // Source: all mass in bin 0 of a [100, 356) image (one bin = one intensity level).
    const values = spike(0, 1);
    const out = remapToAxis(values, 100, 356, 100, 1356); // target is 5x wider
    const total = out.reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1);
    // Bin-centre of source value ~100.5 lands near the very start of the wider target.
    expect(out[0]).toBeCloseTo(1);
  });

  it('places a constant-value source at its correct target bin', () => {
    const values = spike(0, 1); // irrelevant which bin holds the mass when srcLo===srcHi
    const out = remapToAxis(values, 1200, 1200, 0, 2400);
    const total = out.reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(1);
    // 1200 is exactly the midpoint of [0, 2400) -> bin 128.
    const nonZero = out.findIndex(v => v > 0);
    expect(nonZero).toBe(128);
  });

  it('collapses everything into bin 0 when the target itself is a single point', () => {
    const values = spike(5, 2).map((v, i) => (i === 50 ? 3 : v)); // two nonzero bins, mass 2 + 3
    const out = remapToAxis(values, 0, 256, 7, 7);
    expect(out[0]).toBe(5);
    expect(out.reduce((a, b) => a + b, 0)).toBe(5);
  });
});

describe('populatedExtent', () => {
  it('returns null when every bin is zero', () => {
    expect(populatedExtent(0, 256, new Array(NBINS).fill(0))).toBeNull();
  });

  it('trims leading and trailing zeros to the real data span', () => {
    // Narrow integer data widened to a 256-wide window: only the first 10 levels are populated.
    const values = new Array(NBINS).fill(0);
    for (let i = 0; i < 10; i++) values[i] = 1;
    expect(populatedExtent(100, 356, values)).toEqual([100, 110]);
  });

  it('preserves a real internal gap between two populated clusters', () => {
    const values = spike(5, 1);
    values[250] = 1;
    const [lo, hi] = populatedExtent(0, 256, values);
    expect(lo).toBe(5);
    expect(hi).toBe(251);
  });

  it('matches binXs bin placement for a constant-value (zero-width) window', () => {
    // hi === lo: same degenerate case binXs falls back to step = 1/NBINS for.
    const values = spike(0, 1);
    const [lo, hi] = populatedExtent(100, 100, values);
    expect(lo).toBeCloseTo(100);
    expect(hi).toBeCloseTo(100 + 1 / NBINS);
  });
});

describe('accumulateGroupHistograms - the core cross-image bug fix', () => {
  function record({ kind = 'uint', group = 'g', total = 100, aMin, aMax, nMin = aMin, nMax = aMax, spikeAt = 0, nanCount = 0 }) {
    return { kind, group, counts: spike(spikeAt, total), total, aMin, aMax, nMin, nMax, nanCount };
  }

  it('does not collapse images with different ranges into the same bin (the reported bug)', () => {
    // Three images, each a narrow constant-ish cluster, all landing in bin 0 of
    // their OWN per-image range - exactly the "100 vs 1200 vs 30000" scenario.
    const records = [
      record({ aMin: 100, aMax: 356 }),
      record({ aMin: 1200, aMax: 1456 }),
      record({ aMin: 30000, aMax: 30256 }),
    ];
    const { uint: { g: gd } } = accumulateGroupHistograms(records);

    expect(gd.aMin).toBe(100);
    expect(gd.aMax).toBe(30256);

    // Naive index-wise summing would put all three at bin 0. The fix must spread
    // them across three distinct target bins reflecting their real values.
    const nonZeroBins = gd.sums.reduce((acc, v, i) => (v > 0 ? [...acc, i] : acc), []);
    expect(nonZeroBins.length).toBe(3);
    expect(new Set(nonZeroBins).size).toBe(3);

    // Each image contributes a full 1.0 of normalized mass, averaged over 3 images.
    const total = gd.sums.reduce((a, b) => a + b, 0);
    expect(total).toBeCloseTo(3);
  });

  it('sums images sharing the same range without distorting values', () => {
    const records = [
      record({ aMin: 200, aMax: 300, spikeAt: 5 }),
      record({ aMin: 200, aMax: 300, spikeAt: 5 }),
      record({ aMin: 200, aMax: 300, spikeAt: 5 }),
    ];
    const { uint: { g: gd } } = accumulateGroupHistograms(records);
    expect(gd.sums[5]).toBeCloseTo(3);
    expect(gd.sums.filter(v => v > 0).length).toBe(1);
  });

  it('keeps different kinds in separate buckets', () => {
    const records = [
      record({ kind: 'uint', aMin: 0, aMax: 256 }),
      record({ kind: 'float', aMin: 0, aMax: 256 }),
    ];
    const result = accumulateGroupHistograms(records);
    expect(Object.keys(result).sort()).toEqual(['float', 'uint']);
    expect(result.uint.g.count).toBe(1);
    expect(result.float.g.count).toBe(1);
  });

  it('remaps the dtype-normalized axis independently of the actual-value axis', () => {
    // Same two images as the mixed-dtype-within-kind scenario: uint8 (0-71 -> normalized
    // window [0, 256)) and uint16 (5000-5300, normalized window far from zero).
    const records = [
      record({ aMin: 0, aMax: 256, nMin: 0, nMax: 1, spikeAt: 0 }),
      record({ aMin: 5000, aMax: 5256, nMin: 0.076, nMax: 0.084, spikeAt: 0 }),
    ];
    const { uint: { g: gd } } = accumulateGroupHistograms(records);

    // Actual-value axis: union spans the full 0-5256 range.
    expect(gd.aMin).toBe(0);
    expect(gd.aMax).toBe(5256);
    // Normalized axis: independently unioned from nMin/nMax, not derived from aMin/aMax.
    // The uint8 image's own normalized window already spans the full [0, 1), so it
    // dominates the union regardless of where the uint16 image's narrower window sits.
    expect(gd.nMin).toBeCloseTo(0);
    expect(gd.nMax).toBeCloseTo(1);

    const actualBins = gd.sums.reduce((acc, v, i) => (v > 0 ? [...acc, i] : acc), []);
    const normBins   = gd.sumsNorm.reduce((acc, v, i) => (v > 0 ? [...acc, i] : acc), []);
    expect(new Set(actualBins).size).toBe(2);
    expect(new Set(normBins).size).toBe(2);
    // The two axes place the second image at different relative positions.
    expect(actualBins).not.toEqual(normBins);
  });

  it('skips normalized accumulation for float and when computeNormalized is false', () => {
    const floatOnly = accumulateGroupHistograms([record({ kind: 'float', aMin: 0, aMax: 256 })]);
    expect(floatOnly.float.g.sumsNorm).toBe(floatOnly.float.g.sums);

    const noNorm = accumulateGroupHistograms(
      [record({ kind: 'uint', aMin: 0, aMax: 256 })],
      { computeNormalized: false },
    );
    expect(noNorm.uint.g.sumsNorm).toBe(noNorm.uint.g.sums);
  });

  it('remaps sumSq (the ±1 std input) onto the exact same bins as sums, not just the same count', () => {
    const records = [
      record({ aMin: 100, aMax: 356, spikeAt: 0 }),
      record({ aMin: 1200, aMax: 1456, spikeAt: 0 }),
    ];
    const { uint: { g: gd } } = accumulateGroupHistograms(records);
    const nonZeroBins = arr => arr.reduce((acc, v, i) => (v > 0 ? [...acc, i] : acc), []);
    const sumsBins  = nonZeroBins(gd.sums);
    const sumSqBins = nonZeroBins(gd.sumSq);
    expect(sumsBins.length).toBe(2); // sanity: the two images really do land on different bins
    expect(sumSqBins).toEqual(sumsBins);
  });

  it('defaults nanCount to 0, matching prior no-NaN behavior', () => {
    const { uint: { g: gd } } = accumulateGroupHistograms([record({ aMin: 0, aMax: 256 })]);
    expect(gd.nanSum).toBe(0);
  });

  it('folds NaN pixels into the per-image denominator, shrinking its bin mass accordingly', () => {
    // 100 finite pixels + 100 NaN pixels -> each bin's mass is halved, nanFrac is 0.5.
    const records = [record({ aMin: 0, aMax: 256, total: 100, nanCount: 100 })];
    const { uint: { g: gd } } = accumulateGroupHistograms(records);
    expect(gd.nanSum).toBeCloseTo(0.5);
    expect(gd.sums.reduce((a, b) => a + b, 0)).toBeCloseTo(0.5);
  });
});
