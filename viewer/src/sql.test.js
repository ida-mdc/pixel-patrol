import { describe, it, expect } from 'vitest';
import { fileCount, perFile, groupCol } from './sql.js';

// A container file contributes one pp_data row per sub-image, all carrying the
// same file-level facts. These two helpers keep widgets at file grain.
const ALL = ['path', 'size_bytes', 'file_extension', 'dtype', 'mean_intensity'];
const gc = groupCol({ groupCol: null });

describe('fileCount', () => {
  it('counts distinct paths, so a container counts once', () => {
    expect(fileCount(ALL)).toBe('COUNT(DISTINCT "path")');
  });

  it('falls back to counting rows when there is no path column', () => {
    expect(fileCount(['size_bytes'])).toBe('COUNT(*)');
  });
});

describe('perFile', () => {
  it('collapses to one row per file and reports the sub-image count', () => {
    const sql = perFile({ groupCol: gc, allCols: ALL });
    expect(sql).toContain('COUNT(*) AS __n_images__');
    expect(sql).toContain('GROUP BY ALL');
  });

  it('carries file-level columns through, and no image-level ones', () => {
    const sql = perFile({ groupCol: gc, allCols: ALL });
    expect(sql).toContain('"size_bytes"');
    expect(sql).toContain('"file_extension"');
    expect(sql).not.toContain('dtype');
    expect(sql).not.toContain('mean_intensity');
  });

  it('skips file-level columns the schema does not have', () => {
    expect(perFile({ groupCol: gc, allCols: ['path', 'size_bytes'] }))
      .not.toContain('modification_date');
  });

  it('keys by group as well as by file, so an image-level grouping can split one file', () => {
    expect(perFile({ groupCol: gc, allCols: ALL })).toContain('__group__');
  });

  it('omits the group key for dataset-wide totals', () => {
    expect(perFile({ grouped: false, groupCol: gc, allCols: ALL })).not.toContain('__group__');
  });

  // Filter-then-collapse: a file survives when any of its images pass.
  it('applies the caller WHERE inside the subquery, before collapsing', () => {
    const sql = perFile({ where: `WHERE "dtype" = 'uint8'`, groupCol: gc, allCols: ALL });
    expect(sql).toMatch(/WHERE "dtype" = 'uint8'\s+GROUP BY ALL/);
  });

  it('passes rows through when there is no path to collapse on', () => {
    const sql = perFile({ groupCol: gc, allCols: ['size_bytes'] });
    expect(sql).toContain('1 AS __n_images__');
    expect(sql).not.toContain('GROUP BY');
  });

  it('is usable directly after FROM', () => {
    expect(perFile({ groupCol: gc, allCols: ALL })).toMatch(/\) AS pp_file$/);
  });
});
