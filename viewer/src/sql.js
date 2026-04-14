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

  const c = `"${col}"`;

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
 * Bare SQL expression for the active group column.
 * Returns `"col"` when grouping is active, or `'all'` when not.
 * Use this when you write ` AS __group__` yourself in the query.
 */
export function groupCol(state) {
  return state.groupCol ? q(state.groupCol) : `'all'`;
}

/**
 * Full SELECT expression for the group column including alias.
 * Returns `"col" AS __group__` or `'all' AS __group__`.
 * Use this in SELECT lists directly.
 */
export function groupExpr(state) {
  return `${groupCol(state)} AS __group__`;
}
