import { describe, it, expect } from 'vitest';
import { buildGroupLabels } from '../group-labels.js';

describe('buildGroupLabels', () => {
  it('returns identity map for a single group', () => {
    const result = buildGroupLabels(['only_one']);
    expect(result).toEqual({ only_one: 'only_one' });
  });

  it('returns identity map when all values are short (≤20 chars)', () => {
    const groups = ['alpha', 'beta', 'gamma'];
    expect(buildGroupLabels(groups)).toEqual({ alpha: 'alpha', beta: 'beta', gamma: 'gamma' });
  });

  it('returns identity map for two short groups', () => {
    expect(buildGroupLabels(['a', 'b'])).toEqual({ a: 'a', b: 'b' });
  });

  it('uses path-based shortening for path-like groups', () => {
    const groups = [
      '/very/long/path/to/experiment/condition_A',
      '/very/long/path/to/experiment/condition_B',
      '/very/long/path/to/experiment/condition_C',
    ];
    const result = buildGroupLabels(groups);
    // Each label should be shorter than the original and still unique
    const labels = Object.values(result);
    expect(new Set(labels).size).toBe(3);
    for (const [orig, label] of Object.entries(result)) {
      expect(label.length).toBeLessThanOrEqual(orig.length);
    }
    // Should use the trailing path component to disambiguate
    expect(result[groups[0]]).toContain('condition_A');
  });

  it('path shortening uses backslash separator when no forward slashes', () => {
    const groups = [
      'C:\\very\\long\\path\\to\\experiment\\condition_A',
      'C:\\very\\long\\path\\to\\experiment\\condition_B',
    ];
    const result = buildGroupLabels(groups);
    const labels = Object.values(result);
    expect(new Set(labels).size).toBe(2);
    expect(result[groups[0]]).toContain('condition_A');
    expect(result[groups[1]]).toContain('condition_B');
  });

  it('path shortening uses minimum tail segments needed for uniqueness', () => {
    // These share the final path segment — need 2 components to distinguish
    const groups = [
      '/data/runA/images/sample',
      '/data/runB/images/sample',
    ];
    const result = buildGroupLabels(groups);
    expect(Object.values(result)).toHaveLength(2);
    expect(new Set(Object.values(result)).size).toBe(2);
  });

  it('uses prefix/suffix shortening for long non-path groups with shared prefix', () => {
    const groups = [
      'experiment_batch_2024_condition_A_replicate_01',
      'experiment_batch_2024_condition_B_replicate_01',
      'experiment_batch_2024_condition_C_replicate_01',
    ];
    const result = buildGroupLabels(groups);
    const labels = Object.values(result);
    // Labels should be unique
    expect(new Set(labels).size).toBe(3);
    // Labels should be shorter (prefix+suffix stripped)
    for (const label of labels) {
      expect(label.length).toBeLessThan(groups[0].length);
    }
  });

  it('falls back to identity when prefix is too short to strip (<10 chars)', () => {
    // Groups differ early on, so no meaningful shared prefix
    const groups = [
      'alpha_extra_padding_to_exceed_twenty_chars',
      'beta_extra_padding_to_exceed_twenty_chars',
    ];
    const result = buildGroupLabels(groups);
    // No meaningful prefix/suffix shortening possible — identity
    expect(result[groups[0]]).toBe(groups[0]);
    expect(result[groups[1]]).toBe(groups[1]);
  });

  it('falls back to identity when shortened labels would not be unique', () => {
    // All groups have the same suffix content after stripping prefix
    const groups = [
      'aaaaaaaaaa_X_bbbbbbbbbb',
      'aaaaaaaaaa_X_bbbbbbbbbb',  // duplicate — identity required
    ];
    const result = buildGroupLabels(groups);
    expect(result[groups[0]]).toBe(groups[0]);
  });
});