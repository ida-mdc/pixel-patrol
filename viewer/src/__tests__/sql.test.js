import { describe, it, expect } from 'vitest';
import { buildWhere, andWhere, q, groupCol, groupExpr, dimSubsetWhere, stripWhere } from '../sql.js';

describe('buildWhere', () => {
  it('returns empty string when col is missing', () => {
    expect(buildWhere({ col: '', op: 'eq', val: 'x' })).toBe('');
  });

  it('returns empty string when op is missing', () => {
    expect(buildWhere({ col: 'name', op: '', val: 'x' })).toBe('');
  });

  it('returns empty string when val is empty string', () => {
    expect(buildWhere({ col: 'name', op: 'eq', val: '' })).toBe('');
  });

  it('builds eq clause', () => {
    expect(buildWhere({ col: 'dtype', op: 'eq', val: 'uint8' }))
      .toBe(`WHERE "dtype"::VARCHAR = 'uint8'`);
  });

  it('escapes single quotes in val for eq', () => {
    expect(buildWhere({ col: 'name', op: 'eq', val: "O'Brien" }))
      .toBe(`WHERE "name"::VARCHAR = 'O''Brien'`);
  });

  it('builds contains clause', () => {
    expect(buildWhere({ col: 'path', op: 'contains', val: 'runA' }))
      .toBe(`WHERE "path"::VARCHAR LIKE '%runA%'`);
  });

  it('builds not_contains clause', () => {
    expect(buildWhere({ col: 'path', op: 'not_contains', val: 'runA' }))
      .toBe(`WHERE "path"::VARCHAR NOT LIKE '%runA%'`);
  });

  it('builds gt clause with numeric val', () => {
    expect(buildWhere({ col: 'mean_intensity', op: 'gt', val: '50.5' }))
      .toBe(`WHERE TRY_CAST("mean_intensity" AS DOUBLE) > 50.5`);
  });

  it('builds ge clause', () => {
    expect(buildWhere({ col: 'size', op: 'ge', val: '100' }))
      .toBe(`WHERE TRY_CAST("size" AS DOUBLE) >= 100`);
  });

  it('builds lt clause', () => {
    expect(buildWhere({ col: 'score', op: 'lt', val: '0.5' }))
      .toBe(`WHERE TRY_CAST("score" AS DOUBLE) < 0.5`);
  });

  it('builds le clause', () => {
    expect(buildWhere({ col: 'score', op: 'le', val: '1' }))
      .toBe(`WHERE TRY_CAST("score" AS DOUBLE) <= 1`);
  });

  it('builds in clause with multiple values', () => {
    expect(buildWhere({ col: 'ext', op: 'in', val: 'tif, png, jpg' }))
      .toBe(`WHERE "ext"::VARCHAR IN ('tif', 'png', 'jpg')`);
  });

  it('escapes single quotes in in-list values', () => {
    const result = buildWhere({ col: 'name', op: 'in', val: "a'b, c" });
    expect(result).toContain("'a''b'");
  });

  it('returns empty string for unknown op', () => {
    expect(buildWhere({ col: 'x', op: 'unknown_op', val: 'v' })).toBe('');
  });
});

describe('andWhere', () => {
  it('returns a WHERE clause when no existing where', () => {
    expect(andWhere('', '"ext" IS NOT NULL')).toBe('WHERE "ext" IS NOT NULL');
  });

  it('appends AND when existing WHERE is present', () => {
    expect(andWhere(`WHERE "x" = '1'`, '"ext" IS NOT NULL'))
      .toBe(`WHERE "x" = '1' AND "ext" IS NOT NULL`);
  });

  it('returns existing where unchanged when condition is empty', () => {
    expect(andWhere(`WHERE "x" = '1'`, '')).toBe(`WHERE "x" = '1'`);
  });

  it('returns empty string when both are empty', () => {
    expect(andWhere('', '')).toBe('');
  });
});

describe('q', () => {
  it('wraps identifier in double quotes', () => {
    expect(q('my_col')).toBe('"my_col"');
  });

  it('escapes embedded double quotes', () => {
    expect(q('col"name')).toBe('"col""name"');
  });
});

describe('stripWhere', () => {
  it('strips a leading WHERE (case-insensitive, any whitespace)', () => {
    expect(stripWhere(`WHERE "x" = 1`)).toBe(`"x" = 1`);
    expect(stripWhere(`  where "x" = 1`)).toBe(`"x" = 1`);
  });
  it('returns empty string for empty input', () => {
    expect(stripWhere('')).toBe('');
    expect(stripWhere(undefined)).toBe('');
  });
});

describe('dimSubsetWhere', () => {
  const dimCols = ['dim_t', 'dim_c', 'dim_z', 'dim_y', 'dim_x'];

  it('whole-image aggregate: obs_level 0, every dim NULL', () => {
    expect(dimSubsetWhere({ dimCols }))
      .toEqual([
        'obs_level = 0',
        '"dim_t" IS NULL', '"dim_c" IS NULL', '"dim_z" IS NULL',
        '"dim_y" IS NULL', '"dim_x" IS NULL',
      ]);
  });

  it('one split dim: that dim IS NOT NULL, others NULL, obs_level 1', () => {
    expect(dimSubsetWhere({ dimCols, split: new Set(['c']) }))
      .toEqual([
        'obs_level = 1',
        '"dim_t" IS NULL', '"dim_c" IS NOT NULL', '"dim_z" IS NULL',
        '"dim_y" IS NULL', '"dim_x" IS NULL',
      ]);
  });

  it('fixed dim is pinned and excluded from the NULL loop; obs_level counts it', () => {
    expect(dimSubsetWhere({ dimCols, fixed: { y: 3 }, split: new Set(['c']) }))
      .toEqual([
        'obs_level = 2',
        '"dim_y" = 3',
        '"dim_t" IS NULL', '"dim_c" IS NOT NULL', '"dim_z" IS NULL',
        '"dim_x" IS NULL',
      ]);
  });

  it('baseWhere is prepended', () => {
    expect(dimSubsetWhere({ dimCols, baseWhere: `"dtype" = 'uint8'`, split: new Set(['t']) }))
      .toEqual([
        `"dtype" = 'uint8'`,
        'obs_level = 1',
        '"dim_t" IS NOT NULL', '"dim_c" IS NULL', '"dim_z" IS NULL',
        '"dim_y" IS NULL', '"dim_x" IS NULL',
      ]);
  });

  it('obsLevel: null omits the obs_level clause (spatial strips)', () => {
    expect(dimSubsetWhere({ dimCols, split: new Set(['x']), obsLevel: null }))
      .toEqual([
        '"dim_t" IS NULL', '"dim_c" IS NULL', '"dim_z" IS NULL',
        '"dim_y" IS NULL', '"dim_x" IS NOT NULL',
      ]);
  });

  it('explicit obsLevel overrides the auto count', () => {
    expect(dimSubsetWhere({ dimCols: ['dim_c'], split: new Set(['c']), obsLevel: 5 }))
      .toEqual(['obs_level = 5', '"dim_c" IS NOT NULL']);
  });
});

describe('groupCol / groupExpr', () => {
  it('groupCol returns quoted column when groupCol is set', () => {
    expect(groupCol({ groupCol: 'dtype' })).toBe('"dtype"');
  });

  it("groupCol returns 'all' literal when groupCol is null", () => {
    expect(groupCol({ groupCol: null })).toBe("'all'");
  });

  it('groupExpr includes AS __group__ alias', () => {
    expect(groupExpr({ groupCol: 'dtype' })).toBe('"dtype" AS __group__');
  });
});