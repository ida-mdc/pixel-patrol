import { describe, it, expect } from 'vitest';
import { acquisitionCols, metaValue, groupMetricsByProducer } from '../point-inspector.js';

// A schema shaped like the sample dive-log extension: canonical metadata +
// its own metadata column (depth_zone) + a built-in and an extension metric.
const ctx = {
  schema: {
    allCols: [
      'name', 'path', 'type', 'file_row_number', 'obs_level',
      'dim_order', 'dtype', 'size_X', 'size_Y', 'size_Z', 'size_C',
      'ndim', 'pixel_size_X', 'pixel_size_Y', 'num_pixels', 'size_bytes',
      'file_extension', 'imported_path_short', 'depth_zone',
      'mean_intensity', 'std_intensity', 'laplacian_variance', 'glow_count',
      'thumbnail', 'histogram_counts', 'histogram_min', 'histogram_max',
    ],
    metricCols: ['mean_intensity', 'std_intensity', 'laplacian_variance', 'glow_count'],
    blobCols: ['thumbnail', 'histogram_counts'],
  },
};

const row = {
  name: 'img.tif', path: '/data/img.tif', type: 'file', file_row_number: 3, obs_level: 0,
  dim_order: 'XYC', dtype: 'uint16', size_X: 512, size_Y: 512, size_Z: null, size_C: 3,
  ndim: 3, pixel_size_X: 0.65, pixel_size_Y: 0.65, num_pixels: 786432, size_bytes: 1536,
  file_extension: 'tif', imported_path_short: 'siteA', depth_zone: 'abyss',
  mean_intensity: 123.456, std_intensity: 10, laplacian_variance: 42, glow_count: 7,
  thumbnail: new Uint8Array(1), histogram_counts: new Uint8Array(1),
  histogram_min: 0, histogram_max: 4095,
};

describe('acquisitionCols', () => {
  const cols = acquisitionCols(row, ctx);

  it('leads with canonical META_COLS in their documented order, then extras alphabetically', () => {
    expect(cols).toEqual([
      'dim_order', 'dtype', 'size_Y', 'size_X', 'size_C', 'ndim', 'pixel_size_X', 'pixel_size_Y',
      'depth_zone', 'file_extension', 'imported_path_short', 'num_pixels', 'size_bytes',
    ]);
  });

  it('surfaces extension metadata (depth_zone) rather than a fixed set', () => {
    expect(cols).toContain('depth_zone');
  });

  it('excludes metrics — they belong to the Metrics section', () => {
    for (const m of ['mean_intensity', 'std_intensity', 'laplacian_variance', 'glow_count']) {
      expect(cols).not.toContain(m);
    }
  });

  it('excludes blobs, internal histogram range scalars, and header/infra columns', () => {
    for (const c of ['thumbnail', 'histogram_counts', 'histogram_min', 'histogram_max',
                     'name', 'path', 'type', 'file_row_number', 'obs_level']) {
      expect(cols).not.toContain(c);
    }
  });

  it('drops columns with no value on this image (size_Z is null)', () => {
    expect(cols).not.toContain('size_Z');
  });
});

describe('groupMetricsByProducer', () => {
  const producerByCol = {
    mean_intensity: 'raster-basic', std_intensity: 'raster-basic',
    laplacian_variance: 'raster-quality', glow_count: 'glow-by-depth',
  };

  it('groups by producer in first-appearance order, unknown producer last', () => {
    const keys = ['mean_intensity', 'laplacian_variance', 'std_intensity', 'unlabeled', 'glow_count'];
    expect(groupMetricsByProducer(keys, producerByCol)).toEqual([
      ['raster-basic', ['mean_intensity', 'std_intensity']],
      ['raster-quality', ['laplacian_variance']],
      ['glow-by-depth', ['glow_count']],
      ['', ['unlabeled']],
    ]);
  });

  it('falls back to a single unlabeled group when no producer info is available', () => {
    expect(groupMetricsByProducer(['a', 'b'], {})).toEqual([['', ['a', 'b']]]);
  });
});

describe('metaValue', () => {
  it('humanizes byte sizes and upper-cases the file extension', () => {
    expect(metaValue('size_bytes', 1536)).toBe('1.5 KB');
    expect(metaValue('file_extension', 'tif')).toBe('TIF');
  });

  it('formats numbers and passes strings through', () => {
    expect(metaValue('mean_intensity', 123.456)).toBe('123.5');
    expect(metaValue('dtype', 'uint16')).toBe('uint16');
    expect(metaValue('dim_order', 'XYC')).toBe('XYC');
  });
});
