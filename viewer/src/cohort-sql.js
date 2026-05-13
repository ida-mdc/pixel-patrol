/**
 * SQL fragments for cohort filtering (user filter + dimension slice).
 * Shared by the renderer and offline snapshot export.
 */

import { buildWhere, andWhere, q } from './sql.js';
import { FILE_ROW_NUMBER } from './constants.js';

export function resolveCohortJoinColumn(schema) {
  if (!schema?.isLongFormat) return null;
  const joinCol = (schema.rowIdColumn && schema.rowIdColumn !== FILE_ROW_NUMBER)
    ? schema.rowIdColumn
    : (schema.allCols.includes('path') ? 'path' : null);
  return joinCol;
}

export function buildDimCohortCondition(schema, dimensions) {
  if (!schema?.isLongFormat) return '';
  const selected = Object.entries(dimensions ?? {})
    .map(([letter, idxRaw]) => [letter, Number(idxRaw)])
    .filter(([, idx]) => Number.isFinite(idx));
  if (!selected.length) return '';

  const joinCol = resolveCohortJoinColumn(schema);
  if (!joinCol) return '';

  const jq = q(joinCol);
  const predicates = selected.map(([letter, idx]) => `s.${q(`dim_${letter}`)} = ${idx}`);
  if (!predicates.length) return '';

  return `pp_data.${jq} IN (
    SELECT DISTINCT s.${jq}
    FROM pp_all s
    WHERE ${predicates.join(' AND ')}
  )`;
}

/**
 * WHERE clause for queries against `pp_data` that mirror the interactive viewer.
 */
export function buildScopedWhere(schema, state) {
  const userWhere = buildWhere(state.filter);
  return andWhere(userWhere, buildDimCohortCondition(schema, state.dimensions ?? {}));
}
