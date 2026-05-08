// Ported from report/src/components/schema.js — no Observable dependencies.
import { FILE_ROW_NUMBER } from './constants.js';

// Columns that contain binary data. Must correspond to actual Arrow binary types!!
// TODO: maybe 'histogram_min', 'histogram_max', should be excluded in another way - they are numeric but not metrics.
export const BLOB_COLS = new Set([
  'thumbnail', 'histogram_counts', 'histogram_min', 'histogram_max',
  'coloc_pearson_r', 'coloc_ssim',
  'coloc_ssim_luminance', 'coloc_ssim_contrast', 'coloc_ssim_structure',
]);

export const SKIP_METRIC_COLS = new Set([
  'row_index', FILE_ROW_NUMBER, 'depth', 'modification_month', 'n_images', 'ndim', 'num_pixels',
  'Y_size', 'X_size', 'Z_size', 'T_size', 'C_size', 'S_size',
  'size_bytes',
  // Co-localisation scalar — a count, not a plottable metric.
  'coloc_n_channels',
]);

export const META_COLS = [
  'dim_order', 'dtype', 'Y_size', 'X_size', 'Z_size', 'T_size', 'C_size',
  'ndim', 'pixel_size_X', 'pixel_size_Y', 'pixel_size_Z',
];

export const KNOWN_GROUP_COLS = new Set([
  'imported_path_short', 'folder_top', 'report_group', 'group', 'file_extension',
  'dtype', 'dim_order', 'channel', 'modality', 'condition', 'batch',
  'class', 'label', 'category', 'cohort', 'experiment', 'sample_type',
  // Dash "NO_GROUPING_COL" equivalent sometimes present in exported tables.
  'common_base',
]);

// dim_t, dim_c, dim_z … columns in the long format
const DIM_COL_RE = /^dim_([a-z])$/;

// Columns that are infrastructure for the long format, not metrics
const LONG_FORMAT_COLS = new Set(['obs_level']);

/**
 * Detect schema from an array of {name, type} column descriptors.
 * `type` is the string from Arrow's field type (e.g. "Float64", "Utf8", "Int32").
 *
 * Long-format files have dim_t/dim_c/... nullable int columns + obs_level.
 *
 * @returns {{
 *   metricCols: string[],
 *   dimCols: string[],
 *   groupCols: string[],
 *   dimensionInfo: Object.<string, number[]>,
 *   defaultGroupCol: string|null,
 *   allCols: string[],
 *   blobCols: string[],
 *   isLongFormat: boolean,
 *   allTable: string,
 * }}
 */
export function detectSchema(columns) {
  const metricCols    = [];
  const dimCols       = [];   // long-format: ['dim_t', 'dim_c', …]
  const groupCols     = [];
  const blobCols      = [];
  const allCols       = [];

  const isLongFormat  = columns.some(c => c.name === 'obs_level');
  if (!isLongFormat) {
    throw new Error('Unsupported schema: obs_level column is required (long-format only).');
  }

  for (const { name, type } of columns) {
    if (BLOB_COLS.has(name)) {
      blobCols.push(name);
      allCols.push(name);
      continue;
    }

    // Skip long-format infrastructure columns from metric/group/allCols
    if (LONG_FORMAT_COLS.has(name)) continue;

    const isNumeric = /^(int|uint|float|double|decimal|bigint|smallint|tinyint|real|int8|int16|int32|int64|uint8|uint16|uint32|uint64|float32|float64)/i.test(type);
    const isString  = /^(utf8|string|large_utf8|bool|date|time|timestamp|varchar|char|text)/i.test(type);
    const isDimCol  = DIM_COL_RE.test(name); // long format: dim_t, dim_c, …

    // dim_* columns in long format: track but don't add to allCols/metricCols
    if (isDimCol) {
      dimCols.push(name);
      continue;
    }

    allCols.push(name);

    if (isNumeric && !SKIP_METRIC_COLS.has(name)) {
      metricCols.push(name);
    }

    if (!SKIP_METRIC_COLS.has(name)) {
      if (KNOWN_GROUP_COLS.has(name)) {
        groupCols.push(name);
      } else if (isString) {
        groupCols.push(name);
      }
    }
  }

  const dimensionInfo = {};
  // Long-format dimensionInfo is populated later in finishLoad() via a DB query.

  for (const mustHave of ['imported_path_short', 'folder_top', 'report_group', 'common_base']) {
    if (allCols.includes(mustHave) && !groupCols.includes(mustHave)) {
      groupCols.push(mustHave);
    }
  }

  return {
    metricCols, dimCols,
    groupCols, dimensionInfo, allCols, blobCols,
    isLongFormat,
    allTable: 'pp_all',
  };
}

/**
 * Pick the default group column after cardinality filtering.
 * Prefers well-known columns in priority order, falls back to the first available.
 *
 * If only `common_base` is present (no `imported_path_short`), the data was not
 * grouped during processing — default to no grouping (null).
 */
export function pickDefaultGroupCol(allCols, groupCols) {
  if (allCols.includes('imported_path_short')) return 'imported_path_short';
  // No sub-paths were specified during processing — don't impose a default grouping.
  if (!allCols.includes('imported_path_short') && allCols.includes('common_base')) return null;
  if (groupCols.includes('folder_top'))        return 'folder_top';
  if (groupCols.includes('report_group'))      return 'report_group';
  return groupCols[0] ?? null;
}
