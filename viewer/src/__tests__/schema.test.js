import { describe, it, expect } from 'vitest';
import { detectSchema, pickDefaultGroupCol } from '../schema.js';

// Helper: build a column descriptor array from {name: type} pairs
function cols(obj) {
  return Object.entries(obj).map(([name, type]) => ({ name, type }));
}

describe('detectSchema', () => {
  it('excludes DuckDB list/array types from metricCols (AVG/STDDEV invalid)', () => {
    const { metricCols } = detectSchema(cols({
      obs_level: 'BIGINT',
      dim_c: 'BIGINT',
      mean_intensity: 'DOUBLE',
      shape: 'BIGINT[]',
      vec: 'DOUBLE[]',
    }));
    expect(metricCols).toContain('mean_intensity');
    expect(metricCols).not.toContain('shape');
    expect(metricCols).not.toContain('vec');
  });

  it('classifies numeric columns as metricCols', () => {
    const { metricCols } = detectSchema(cols({
      mean_intensity: 'Float64',
      max_intensity: 'Float32',
      size_bytes: 'Int64',
    }));
    expect(metricCols).toContain('mean_intensity');
    expect(metricCols).toContain('max_intensity');
  });

  it('excludes SKIP_METRIC_COLS from metricCols', () => {
    const { metricCols } = detectSchema(cols({
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

  it('puts dimension-suffix numeric columns into dimMetricCols', () => {
    const { metricCols, dimMetricCols } = detectSchema(cols({
      mean_intensity: 'Float64',
      mean_intensity_t0: 'Float64',
      mean_intensity_c1: 'Float32',
    }));
    expect(metricCols).toContain('mean_intensity');
    expect(metricCols).not.toContain('mean_intensity_t0');
    expect(dimMetricCols).toContain('mean_intensity_t0');
    expect(dimMetricCols).toContain('mean_intensity_c1');
  });

  it('excludes blob cols from metricCols and puts them in blobCols', () => {
    const { metricCols, blobCols } = detectSchema(cols({
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
    // These names are not in KNOWN_GROUP_COLS and don't match the high-cardinality regex,
    // so they should be admitted via the generic string-column heuristic.
    const { groupCols } = detectSchema(cols({
      tissue_type: 'Utf8',
      experiment_arm: 'Utf8',
    }));
    expect(groupCols).toContain('tissue_type');
    expect(groupCols).toContain('experiment_arm');
  });

  it('excludes high-cardinality string columns from groupCols', () => {
    const { groupCols } = detectSchema(cols({
      path: 'Utf8',
      filename: 'Utf8',
      description: 'Utf8',
      id: 'Utf8',
      uuid: 'Utf8',
      hash: 'Utf8',
    }));
    expect(groupCols).not.toContain('path');
    expect(groupCols).not.toContain('filename');
    expect(groupCols).not.toContain('description');
  });

  it('builds dimensionInfo from dim-suffix columns', () => {
    const { dimensionInfo } = detectSchema(cols({
      mean_intensity_t0: 'Float64',
      mean_intensity_t1: 'Float64',
      mean_intensity_c0: 'Float64',  // only one c index → not in dimensionInfo
    }));
    expect(dimensionInfo).toHaveProperty('t');
    expect(dimensionInfo.t).toEqual(['0', '1']);
    // c has only one index — should not be promoted
    expect(dimensionInfo).not.toHaveProperty('c');
  });

  it('sorts dimension indices numerically', () => {
    const { dimensionInfo } = detectSchema(cols({
      metric_t0: 'Float64',
      metric_t2: 'Float64',
      metric_t10: 'Float64',
    }));
    expect(dimensionInfo.t).toEqual(['0', '2', '10']);
  });

  it('always includes canonical group cols even with non-standard Arrow types', () => {
    // Dictionary type would bypass the Utf8 heuristic but should still be promoted
    const { groupCols } = detectSchema(cols({
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
    const allCols = ['imported_path_short', 'folder_top', 'dtype'];
    const groupCols = ['imported_path_short', 'folder_top', 'dtype'];
    expect(pickDefaultGroupCol(allCols, groupCols)).toBe('imported_path_short');
  });

  it('returns null when only common_base present (no sub-path grouping)', () => {
    const allCols = ['common_base', 'dtype'];
    const groupCols = ['common_base', 'dtype'];
    expect(pickDefaultGroupCol(allCols, groupCols)).toBeNull();
  });

  it('falls back to folder_top when imported_path_short is absent', () => {
    const allCols = ['folder_top', 'dtype'];
    const groupCols = ['folder_top', 'dtype'];
    expect(pickDefaultGroupCol(allCols, groupCols)).toBe('folder_top');
  });

  it('falls back to report_group', () => {
    const allCols = ['report_group', 'dtype'];
    const groupCols = ['report_group', 'dtype'];
    expect(pickDefaultGroupCol(allCols, groupCols)).toBe('report_group');
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