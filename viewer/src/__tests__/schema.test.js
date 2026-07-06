import { describe, it, expect } from 'vitest';
import { detectSchema, pickDefaultGroupCol } from '../schema.js';

// Helper: build a column descriptor array from {name: type} pairs.
// detectSchema requires an `obs_level` column, so every fixture below spells it
// out (see the gate test for the rejection path).
function cols(obj) {
  return Object.entries(obj).map(([name, type]) => ({ name, type }));
}

describe('detectSchema', () => {
  it('rejects schemas without the required obs_level column', () => {
    expect(() => detectSchema([{ name: 'mean_intensity', type: 'Float64' }]))
      .toThrow(/obs_level/);
  });


  it('classifies numeric columns as metricCols', () => {
    const { metricCols } = detectSchema(cols({
      obs_level: 'Int64',
      mean_intensity: 'Float64',
      max_intensity: 'Float32',
      size_bytes: 'Int64',
    }));
    expect(metricCols).toContain('mean_intensity');
    expect(metricCols).toContain('max_intensity');
  });

  it('excludes SKIP_METRIC_COLS from metricCols', () => {
    const { metricCols } = detectSchema(cols({
      obs_level: 'Int64',
      row_index: 'Int32',
      file_row_number: 'Int32',
      ndim: 'Int32',
      mean_intensity: 'Float64',
    }));
    expect(metricCols).not.toContain('row_index');
    expect(metricCols).not.toContain('file_row_number');
    expect(metricCols).not.toContain('ndim');
    expect(metricCols).toContain('mean_intensity');
  });

  it('routes dim_* columns into dimCols, out of metric/allCols', () => {
    const { dimCols, metricCols, allCols } = detectSchema(cols({
      obs_level: 'Int64',
      mean_intensity: 'Float64',
      dim_t: 'Int32',
      dim_c: 'Int32',
    }));
    expect(dimCols).toEqual(expect.arrayContaining(['dim_t', 'dim_c']));
    expect(metricCols).toContain('mean_intensity');
    expect(metricCols).not.toContain('dim_t');
    expect(allCols).not.toContain('dim_t');
  });

  it('excludes blob cols from metricCols and puts them in blobCols', () => {
    const { metricCols, blobCols } = detectSchema(cols({
      obs_level: 'Int64',
      thumbnail: 'LargeBinary',
      histogram_counts: 'LargeBinary',
      mean_intensity: 'Float64',
    }));
    expect(blobCols).toContain('thumbnail');
    expect(blobCols).toContain('histogram_counts');
    expect(metricCols).not.toContain('thumbnail');
  });

  it('classifies known string group columns into groupCols', () => {
    const { groupCols } = detectSchema(cols({
      obs_level: 'Int64',
      imported_path_short: 'Utf8',
      folder_top: 'Utf8',
      dtype: 'Utf8',
      dim_order: 'Utf8',
      channel: 'Utf8',
    }));
    expect(groupCols).toContain('imported_path_short');
    expect(groupCols).toContain('folder_top');
    expect(groupCols).toContain('dtype');
    expect(groupCols).toContain('dim_order');
  });

  it('includes novel string cols not in KNOWN_GROUP_COLS via the string heuristic', () => {
    // These names are not in KNOWN_GROUP_COLS, so they are admitted via the
    // generic string-column heuristic (any string col is a group candidate).
    const { groupCols } = detectSchema(cols({
      obs_level: 'Int64',
      tissue_type: 'Utf8',
      experiment_arm: 'Utf8',
    }));
    expect(groupCols).toContain('tissue_type');
    expect(groupCols).toContain('experiment_arm');
  });

  it('admits string columns as group candidates (cardinality filtering is deferred to runtime)', () => {
    // detectSchema is a pure structural classifier: every string column becomes
    // a group candidate here, and high-cardinality ones (path, filename, …) are
    // pruned later by a runtime DB cardinality query, not at this stage.
    const { groupCols } = detectSchema(cols({
      obs_level: 'Int64',
      path: 'Utf8',
      filename: 'Utf8',
      dtype: 'Utf8',
    }));
    expect(groupCols).toContain('path');
    expect(groupCols).toContain('filename');
    expect(groupCols).toContain('dtype');
  });

  it('leaves dimensionInfo empty (it is populated later from a runtime DB query)', () => {
    const { dimensionInfo } = detectSchema(cols({
      obs_level: 'Int64',
      mean_intensity: 'Float64',
      dim_t: 'Int32',
    }));
    expect(dimensionInfo).toEqual({});
  });

  it('always includes canonical group cols even with non-standard Arrow types', () => {
    // Dictionary type would bypass the Utf8 heuristic but should still be promoted
    const { groupCols } = detectSchema(cols({
      obs_level: 'Int64',
      imported_path_short: 'Dictionary',
      folder_top: 'Dictionary',
      common_base: 'Dictionary',
    }));
    expect(groupCols).toContain('imported_path_short');
    expect(groupCols).toContain('folder_top');
    expect(groupCols).toContain('common_base');
  });

  it('allCols contains every column including blobs', () => {
    const { allCols } = detectSchema(cols({
      obs_level: 'Int64',
      thumbnail: 'LargeBinary',
      mean_intensity: 'Float64',
      dtype: 'Utf8',
    }));
    expect(allCols).toContain('thumbnail');
    expect(allCols).toContain('mean_intensity');
    expect(allCols).toContain('dtype');
  });
});

describe('pickDefaultGroupCol', () => {
  it('prefers imported_path_short when present', () => {
    const allCols = ['imported_path_short', 'common_base', 'dtype'];
    const groupCols = ['imported_path_short', 'common_base', 'dtype'];
    expect(pickDefaultGroupCol(allCols, groupCols)).toBe('imported_path_short');
  });

  it('returns null when only common_base present (no sub-path grouping)', () => {
    const allCols = ['common_base', 'dtype'];
    const groupCols = ['common_base', 'dtype'];
    expect(pickDefaultGroupCol(allCols, groupCols)).toBeNull();
  });

  it('groups by name for small common_base reports so each file is its own group', () => {
    const allCols = ['common_base', 'name', 'dtype'];
    const groupCols = ['common_base', 'dtype'];
    expect(pickDefaultGroupCol(allCols, groupCols, 4)).toBe('name');
  });

  it('falls back to first groupCol when no priority col present', () => {
    const allCols = ['modality', 'dtype'];
    const groupCols = ['modality', 'dtype'];
    expect(pickDefaultGroupCol(allCols, groupCols)).toBe('modality');
  });

  it('returns null when groupCols is empty', () => {
    expect(pickDefaultGroupCol([], [])).toBeNull();
  });
});
