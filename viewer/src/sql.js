import { GROUP_ALL, GROUP_COL_ALIAS } from './constants.js';

/**
 * SQL helpers.
 *
 * All user-supplied values are escaped via parameterised-style escaping
 * (single-quote doubling). DuckDB WASM doesn't expose prepared statements
 * through its async API in a convenient way, so we build strings carefully.
 */

function esc(v) {
  return String(v).replace(/'/g, "''");
}

/**
 * Build a WHERE clause fragment from the current filter state.
 * Returns an empty string when no filter is active.
 */
export function buildWhere(filter) {
  const { col, op, val } = filter;
  if (!col || !op || val === '') return '';

  const c = q(col);

  switch (op) {
    case 'contains':
      return `WHERE ${c}::VARCHAR LIKE '%${esc(val)}%'`;
    case 'not_contains':
      return `WHERE ${c}::VARCHAR NOT LIKE '%${esc(val)}%'`;
    case 'eq':
      return `WHERE ${c}::VARCHAR = '${esc(val)}'`;
    case 'gt':
      return `WHERE TRY_CAST(${c} AS DOUBLE) > ${parseFloat(val)}`;
    case 'ge':
      return `WHERE TRY_CAST(${c} AS DOUBLE) >= ${parseFloat(val)}`;
    case 'lt':
      return `WHERE TRY_CAST(${c} AS DOUBLE) < ${parseFloat(val)}`;
    case 'le':
      return `WHERE TRY_CAST(${c} AS DOUBLE) <= ${parseFloat(val)}`;
    case 'in': {
      const list = val.split(',').map(v => `'${esc(v.trim())}'`).join(', ');
      return `WHERE ${c}::VARCHAR IN (${list})`;
    }
    default:
      return '';
  }
}

/**
 * Extend an existing WHERE fragment with an extra AND condition.
 * Use this instead of appending a raw `WHERE …` when ctx.where may already
 * contain a WHERE clause (which would produce invalid SQL).
 *
 *   andWhere('',                   '"ext" IS NOT NULL') → 'WHERE "ext" IS NOT NULL'
 *   andWhere('WHERE "x" = \'1\'', '"ext" IS NOT NULL') → 'WHERE "x" = \'1\' AND "ext" IS NOT NULL'
 */
export function andWhere(where, condition) {
  if (!condition) return where;
  return where ? `${where} AND ${condition}` : `WHERE ${condition}`;
}

/**
 * Return a safely double-quoted SQL identifier.
 * Embedded double-quotes are escaped by doubling them (standard SQL).
 */
export function q(name) {
  return `"${String(name).replaceAll('"', '""')}"`;
}

/**
 * USING SAMPLE clause for approximate sampling.
 * DuckDB 1.x supports USING SAMPLE n ROWS (reservoir).
 */
export function sample(n) {
  return `USING SAMPLE ${n} ROWS (reservoir, 42)`;
}

/**
 * WHERE-part strings that pin a single long-format aggregation subset.
 *
 * Long-format parquet carries one row per fixed-dim subset (the power set of the
 * dimensions): a row's `obs_level` is the size of its fixed subset and every
 * other `dim_*` column is NULL. To isolate exactly the subset (fixed ∪ split)
 * and avoid pooling values from different aggregated row classes, constrain:
 *   - fixed dims  → `= idx`              (pinned to a coordinate)
 *   - split dims  → `IS NOT NULL`        (kept un-aggregated; one point each)
 *   - every other → `IS NULL`
 *
 * obs_level defaults to |fixed| + |split|. Pass `obsLevel: null` to omit it -
 * spatial strips (varying x or y) span datasets that mix dim_orders (YX vs CYX
 * vs TCYX in one file), so obs_level is not reliable per row class there; the
 * IS NULL / IS NOT NULL constraints alone select the right rows.
 *
 * Returns the parts; callers AND-join them into a WHERE clause.
 *
 * @param {object} o
 * @param {(s:string)=>string} o.q                 identifier quoter
 * @param {string[]} o.dimCols                      all dim_* columns (schema.dimCols)
 * @param {string} [o.baseWhere='']                 extra predicate, no leading WHERE
 * @param {Record<string,number>} [o.fixed={}]      {letter: index} dims pinned to a value
 * @param {Set<string>} [o.split=new Set()]         letters kept un-aggregated
 * @param {number|null} [o.obsLevel]                obs_level to require (default auto)
 * @returns {string[]}
 */
export function dimSubsetWhere({ q: quote = q, dimCols = [], baseWhere = '', fixed = {}, split = new Set(), obsLevel } = {}) {
  const parts = [];
  if (baseWhere) parts.push(baseWhere);
  const level = obsLevel === undefined ? Object.keys(fixed).length + split.size : obsLevel;
  if (level != null) parts.push(`obs_level = ${level}`);
  for (const [letter, idx] of Object.entries(fixed)) parts.push(`${quote(`dim_${letter}`)} = ${idx}`);
  for (const col of dimCols) {
    const letter = col.slice(4); // strip "dim_"
    if (letter in fixed) continue; // already pinned above
    parts.push(split.has(letter) ? `${quote(col)} IS NOT NULL` : `${quote(col)} IS NULL`);
  }
  return parts;
}

/** Strip a leading `WHERE ` so a fragment can be AND-joined into a larger clause. */
export function stripWhere(where) {
  return where ? where.replace(/^\s*WHERE\s+/i, '') : '';
}

/**
 * Bare SQL expression for the active group column.
 * Returns `"col"` when grouping is active, or `'all'` when not.
 * Use this when you write ` AS __group__` yourself in the query.
 */
export function groupCol(state) {
  return state.groupCol ? q(state.groupCol) : `'${GROUP_ALL}'`;
}

/**
 * Full SELECT expression for the group column including alias.
 * Returns `"col" AS __group__` or `'all' AS __group__`.
 * Use this in SELECT lists directly.
 */
export function groupExpr(state) {
  return `${groupCol(state)} AS ${GROUP_COL_ALIAS}`;
}

/**
 * Count files, not rows: a container file has one row per sub-image.
 * Falls back to `COUNT(*)` when there is no `path` column.
 */
export function fileCount(allCols = []) {
  return allCols.includes('path') ? `COUNT(DISTINCT ${q('path')})` : 'COUNT(*)';
}

/**
 * One row per file, to FROM instead of `pp_data` when summing a file-level
 * measure like `size_bytes` - a container's size sits on each sub-image row.
 * Counting doesn't need it; see fileCount.
 * Filters before collapsing, so a file survives if any of its images pass.
 */
export function perFile(where = '', allCols = []) {
  if (!allCols.includes('path')) {
    return `(SELECT *, 1 AS __n_images__ FROM pp_data ${where}) AS pp_file`;
  }
  return `(SELECT *, COUNT(*) OVER (PARTITION BY ${q('path')}) AS __n_images__
           FROM pp_data ${where}
           QUALIFY ROW_NUMBER() OVER (PARTITION BY ${q('path')}) = 1) AS pp_file`;
}
