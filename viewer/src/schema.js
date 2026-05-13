// Ported from report/src/components/schema.js — no Observable dependencies.
import { FILE_ROW_NUMBER } from './constants.js';

// Columns that contain binary data. Must correspond to actual Arrow binary types!!
// TODO: maybe 'histogram_min', 'histogram_max', should be excluded in another way - they are numeric but not metrics.
export const BLOB_COLS = new Set([
  'thumbnail', 'histogram_counts', 'histogram_min', 'histogram_max',
]);

export const SKIP_METRIC_COLS = new Set([
  'row_index', FILE_ROW_NUMBER, 'depth', 'modification_month', 'n_images', 'ndim', 'num_pixels',
  'Y_size', 'X_size', 'Z_size', 'T_size', 'C_size', 'S_size',
  'size_bytes',
]);

export const META_COLS = [
  'dim_order', 'dtype', 'Y_size', 'X_size', 'Z_size', 'T_size', 'C_size',
  'ndim', 'pixel_size_X', 'pixel_size_Y', 'pixel_size_Z',
];

export const DIM_PATTERN = /(_[a-z]\d+)+$/;

export const KNOWN_GROUP_COLS = new Set([
  'imported_path_short', 'folder_top', 'report_group', 'group', 'file_extension',
  'dtype', 'dim_order', 'channel', 'modality', 'condition', 'batch',
  'class', 'label', 'category', 'cohort', 'experiment', 'sample_type',
  // Dash "NO_GROUPING_COL" equivalent sometimes present in exported tables.
  'common_base',
]);

const HIGH_CARD_RE = /^(path|name|filename|filepath|full_path|imported_path|uuid|id|hash|md5|sha\d*|url|uri|description|comment|notes?)$/i;

/**
 * Detect schema from an array of {name, type} column descriptors.
 * `type` is the string from Arrow's field type (e.g. "Float64", "Utf8", "Int32").
 *
 * @returns {{
 *   metricCols: string[],
 *   dimMetricCols: string[],
 *   groupCols: string[],
 *   dimensionInfo: Object.<string, string[]>,
 *   defaultGroupCol: string|null,
 *   allCols: string[],
 *   blobCols: string[],
 * }}
 */
export function detectSchema(columns) {
  const metricCols = [];
  const dimMetricCols = [];
  const groupCols = [];
  const dims = {};
  const blobCols = [];
  const allCols = [];

  for (const { name, type } of columns) {
    allCols.push(name);

    if (BLOB_COLS.has(name)) {
      blobCols.push(name);
      continue;
    }

    const isNumeric = /^(int|uint|float|double|decimal|bigint|smallint|tinyint|real|int8|int16|int32|int64|uint8|uint16|uint32|uint64|float32|float64)/i.test(type);
    const isString  = /^(utf8|string|large_utf8|bool|date|time|timestamp)/i.test(type);
    const isDim     = DIM_PATTERN.test(name);

    if (isNumeric && !SKIP_METRIC_COLS.has(name)) {
      if (!isDim) {
        metricCols.push(name);
      } else {
        // Numeric columns with dimension suffixes (e.g. mean_intensity_t0, laplacian_variance_c1)
        dimMetricCols.push(name);
      }
    }

    if (!isDim && !SKIP_METRIC_COLS.has(name)) {
      if (KNOWN_GROUP_COLS.has(name)) {
        groupCols.push(name);
      } else if (isString && !HIGH_CARD_RE.test(name)) {
        groupCols.push(name);
      }
    }

    // Collect dimension letter → indices
    const re = /_([a-z])(\d+)/g;
    let m;
    while ((m = re.exec(name)) !== null) {
      const d = m[1], idx = m[2];
      if (!dims[d]) dims[d] = new Set();
      dims[d].add(idx);
    }
  }

  const dimensionInfo = {};
  for (const [d, vals] of Object.entries(dims)) {
    if (vals.size > 1) {
      dimensionInfo[d] = [...vals].sort((a, b) => parseInt(a) - parseInt(b));
    }
  }

  // Ensure canonical grouping columns are always offered when present.
  // Some exports may store these with non-primitive Arrow types (e.g. Dictionary),
  // which can bypass the heuristics above.
  for (const mustHave of ['imported_path_short', 'folder_top', 'report_group', 'common_base']) {
    if (allCols.includes(mustHave) && !groupCols.includes(mustHave)) {
      groupCols.push(mustHave);
    }
  }

  return { metricCols, dimMetricCols, groupCols, dimensionInfo, allCols, blobCols };
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
