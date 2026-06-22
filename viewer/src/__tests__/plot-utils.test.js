import { describe, it, expect } from 'vitest';
import { humanList, formatBytes } from '../plot-utils.js';

describe('humanList', () => {
  it('joins 0, 1, 2 and 3+ items the way English does', () => {
    expect(humanList([])).toBe('');
    expect(humanList(['a'])).toBe('a');
    expect(humanList(['a', 'b'])).toBe('a and b');
    expect(humanList(['a', 'b', 'c'])).toBe('a, b and c');
  });

  it('drops null/empty entries before joining', () => {
    expect(humanList(['a', null, '', 'b'])).toBe('a and b');
    expect(humanList(undefined)).toBe('');
  });
});

describe('formatBytes', () => {
  it('scales units and rounds by magnitude', () => {
    expect(formatBytes(512)).toBe('512 B');
    expect(formatBytes(5)).toBe('5.0 B');
    expect(formatBytes(1536)).toBe('1.5 KB');
    expect(formatBytes(12_000_000)).toBe('11 MB');
  });
});
