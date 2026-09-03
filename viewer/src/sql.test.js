import { describe, it, expect } from 'vitest';
import { fileCount, perFile } from './sql.js';

const ALL = ['path', 'size_bytes', 'file_extension', 'dtype'];

describe('fileCount', () => {
  it('counts distinct paths, so a container counts once', () => {
    expect(fileCount(ALL)).toBe('COUNT(DISTINCT "path")');
  });

  it('falls back to counting rows when there is no path column', () => {
    expect(fileCount(['size_bytes'])).toBe('COUNT(*)');
  });
});

describe('perFile', () => {
  it('keeps one row per path and reports the sub-image count', () => {
    const sql = perFile('', ALL);
    expect(sql).toContain('QUALIFY ROW_NUMBER() OVER (PARTITION BY "path") = 1');
    expect(sql).toContain('COUNT(*) OVER (PARTITION BY "path") AS __n_images__');
  });

  // Filter-then-collapse: a file survives when any of its images pass.
  it('applies the caller WHERE before collapsing', () => {
    expect(perFile(`WHERE "dtype" = 'uint8'`, ALL))
      .toMatch(/WHERE "dtype" = 'uint8'\s+QUALIFY/);
  });

  it('passes rows through when there is no path to collapse on', () => {
    const sql = perFile('', ['size_bytes']);
    expect(sql).toContain('1 AS __n_images__');
    expect(sql).not.toContain('QUALIFY');
  });

  it('is usable directly after FROM', () => {
    expect(perFile('', ALL)).toMatch(/\) AS pp_file$/);
  });
});
