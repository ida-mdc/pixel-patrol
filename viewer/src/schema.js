// Ported from report/src/components/schema.js - no Observable dependencies.
import { FILE_ROW_NUMBER } from './constants.js';

// Columns that contain binary data. Must correspond to actual Arrow binary types!!
export const BLOB_COLS = new Set([
  'thumbnail', 'histogram_counts',
]);

// Numeric columns never shown to users: row ids, aggregation bookkeeping, and
// internal histogram / thumbnail bounds. Shared with the point inspector.
export const NON_USER_FACING_NUMERIC_COLS = new Set([
  'row_index', FILE_ROW_NUMBER, 'depth', 'modification_month', 'n_images',
  'histogram_min', 'histogram_max',
  'thumbnail_norm_min', 'thumbnail_norm_max',
]);

// Not plottable metrics: the non-user-facing columns, plus derived descriptors.
export const SKIP_METRIC_COLS = new Set([
  ...NON_USER_FACING_NUMERIC_COLS,
  'ndim', 'num_pixels', 'size_bytes',
]);

// Per-axis pixel-extent columns (size_X, size_Y, size_Z, …). Matched by shape
// rather than a fixed axis list so any dimension a dataset carries is excluded
// from metrics, not just the canonical X/Y/Z/T/C/S set.
const AXIS_SIZE_RE = /^size_[A-Za-z]$/;

export const META_COLS = [
  'dim_order', 'dtype', 'size_Y', 'size_X', 'size_Z', 'size_T', 'size_C',
  'ndim', 'pixel_size_X', 'pixel_size_Y', 'pixel_size_Z',
];

export const KNOWN_GROUP_COLS = new Set([
  'imported_path_short', 'folder_top', 'report_group', 'group', 'file_extension',
  'dtype', 'dim_order', 'channel', 'modality', 'condition', 'batch',
  'class', 'label', 'category', 'cohort', 'experiment', 'sample_type',
  // Dash "NO_GROUPING_COL" equivalent sometimes present in exported tables.
  'common_base',
]);

// dim_t, dim_c, dim_z … per-dimension index columns
const DIM_COL_RE = /^dim_([a-z])$/;

// Infrastructure columns, not metrics
const INFRA_COLS = new Set(['obs_level']);

/**
 * Detect schema from an array of {name, type} column descriptors.
 * `type` is the string from Arrow's field type (e.g. "Float64", "Utf8", "Int32").
 *
 * Files carry per-dimension dim_t/dim_c/... nullable int columns plus obs_level.
 *
 * @returns {{
 *   metricCols: string[],
 *   dimCols: string[],
 *   groupCols: string[],
 *   dimensionInfo: Object.<string, number[]>,
 *   allCols: string[],
 *   blobCols: string[],
 *   allTable: string,
 * }}
 */
export function detectSchema(columns) {
  const metricCols    = [];
  const dimCols       = [];   // per-dimension index columns: ['dim_t', 'dim_c', …]
  const groupCols     = [];
  const blobCols      = [];
  const allCols       = [];

  if (!columns.some(c => c.name === 'obs_level')) {
    throw new Error('Unsupported schema: required obs_level column is missing.');
  }

  for (const { name, type } of columns) {
    if (BLOB_COLS.has(name)) {
      blobCols.push(name);
      allCols.push(name);
      continue;
    }

    // Skip infrastructure columns from metric/group/allCols
    if (INFRA_COLS.has(name)) continue;

    const isNumeric = /^(int|uint|float|double|decimal|bigint|smallint|tinyint|real|int8|int16|int32|int64|uint8|uint16|uint32|uint64|float32|float64)/i.test(type);
    const isString  = /^(utf8|string|large_utf8|bool|date|time|timestamp|varchar|char|text)/i.test(type);
    const isDimCol  = DIM_COL_RE.test(name); // dim_t, dim_c, …

    // dim_* columns: track but don't add to allCols/metricCols
    if (isDimCol) {
      dimCols.push(name);
      continue;
    }

    allCols.push(name);

    const isSkipped = SKIP_METRIC_COLS.has(name) || AXIS_SIZE_RE.test(name);

    if (isNumeric && !isSkipped) {
      metricCols.push(name);
    }

    if (!isSkipped) {
      if (KNOWN_GROUP_COLS.has(name)) {
        groupCols.push(name);
      } else if (isString) {
        groupCols.push(name);
      }
    }
  }

  const dimensionInfo = {};
  // dimensionInfo is populated later in finishLoad() via a DB query.

  for (const mustHave of ['imported_path_short', 'common_base']) {
    if (allCols.includes(mustHave) && !groupCols.includes(mustHave)) {
      groupCols.push(mustHave);
    }
  }

  return {
    metricCols, dimCols,
    groupCols, dimensionInfo, allCols, blobCols,
    allTable: 'pp_all',
  };
}

/**
 * Pick the default group column after cardinality filtering.
 *
 * `imported_path_short` means the data was grouped during processing, so it is
 * the natural default. If only `common_base` is present the data was not grouped
 * - default to no grouping (null), unless the report is small (≤4 rows) in which
 * case `name` is used so each file gets its own group. Otherwise fall back to the
 * first available group column.
 */
export function pickDefaultGroupCol(allCols, groupCols, totalRows = Infinity) {
  if (allCols.includes('imported_path_short')) return 'imported_path_short';
  if (allCols.includes('common_base')) {
    if (totalRows <= 4 && allCols.includes('name')) return 'name';
    return null;
  }
  return groupCols[0] ?? null;
}
