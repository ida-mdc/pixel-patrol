import { describe, it, expect, beforeEach } from 'vitest';
import { readUrlParams, writeUrlParams } from '../url-params.js';

// Reset the URL to bare '/' before each test so tests don't bleed into each other.
beforeEach(() => {
  history.pushState({}, '', '/');
});

describe('readUrlParams', () => {
  it('returns empty object when no params', () => {
    expect(readUrlParams()).toEqual({});
  });

  it('reads palette param', () => {
    history.pushState({}, '', '?palette=viridis');
    expect(readUrlParams().palette).toBe('viridis');
  });

  it('reads group param', () => {
    history.pushState({}, '', '?group=dtype');
    expect(readUrlParams().groupCol).toBe('dtype');
  });

  it('reads empty group param as null', () => {
    history.pushState({}, '', '?group=');
    expect(readUrlParams().groupCol).toBeNull();
  });

  it('reads filter params fc/fo/fv', () => {
    history.pushState({}, '', '?fc=size&fo=gt&fv=100');
    const { filter } = readUrlParams();
    expect(filter).toEqual({ col: 'size', op: 'gt', val: '100' });
  });

  it('does not produce filter when fc/fo/fv incomplete', () => {
    history.pushState({}, '', '?fc=size&fo=gt');  // fv missing
    expect(readUrlParams().filter).toBeUndefined();
  });

  it('reads dims param', () => {
    history.pushState({}, '', '?dims=c0.t1');
    const { dimensions } = readUrlParams();
    expect(dimensions).toEqual({ c: '0', t: '1' });
  });

  it('ignores malformed dim segments', () => {
    history.pushState({}, '', '?dims=c0.INVALID.t2');
    const { dimensions } = readUrlParams();
    expect(dimensions).toEqual({ c: '0', t: '2' });
  });

  it('reads sig=1 as showSignificance true', () => {
    history.pushState({}, '', '?sig=1');
    expect(readUrlParams().showSignificance).toBe(true);
  });

  it('reads sig=0 as showSignificance false', () => {
    history.pushState({}, '', '?sig=0');
    expect(readUrlParams().showSignificance).toBe(false);
  });

  it('reads hidden widgets', () => {
    history.pushState({}, '', '?hidden=widget-a.widget-b');
    const { hiddenWidgets } = readUrlParams();
    expect(hiddenWidgets).toBeInstanceOf(Set);
    expect(hiddenWidgets.has('widget-a')).toBe(true);
    expect(hiddenWidgets.has('widget-b')).toBe(true);
  });
});

describe('writeUrlParams', () => {
  it('writes palette when different from default (tab10)', () => {
    writeUrlParams({ palette: 'viridis' });
    expect(window.location.search).toContain('palette=viridis');
  });

  it('omits palette when it equals the default', () => {
    writeUrlParams({ palette: 'tab10' });
    expect(window.location.search).not.toContain('palette');
  });

  it('writes group param', () => {
    writeUrlParams({ groupCol: 'dtype' });
    expect(window.location.search).toContain('group=dtype');
  });

  it('omits group when null/falsy', () => {
    writeUrlParams({ groupCol: null });
    expect(window.location.search).not.toContain('group');
  });

  it('writes filter params when all three fields present', () => {
    writeUrlParams({ filter: { col: 'size', op: 'gt', val: '50' } });
    const search = window.location.search;
    expect(search).toContain('fc=size');
    expect(search).toContain('fo=gt');
    expect(search).toContain('fv=50');
  });

  it('omits filter params when filter is empty/missing', () => {
    writeUrlParams({ filter: { col: '', op: '', val: '' } });
    const search = window.location.search;
    expect(search).not.toContain('fc=');
    expect(search).not.toContain('fo=');
    expect(search).not.toContain('fv=');
  });

  it('writes sig=1 when showSignificance is true', () => {
    writeUrlParams({ showSignificance: true });
    expect(window.location.search).toContain('sig=1');
  });

  it('omits sig when showSignificance is false', () => {
    writeUrlParams({ showSignificance: false });
    expect(window.location.search).not.toContain('sig');
  });

  it('writes hidden widgets sorted', () => {
    writeUrlParams({ hiddenWidgets: new Set(['widget-b', 'widget-a']) });
    const search = window.location.search;
    expect(search).toContain('hidden=widget-a.widget-b');
  });

  it('omits hidden when set is empty', () => {
    writeUrlParams({ hiddenWidgets: new Set() });
    expect(window.location.search).not.toContain('hidden');
  });

  it('round-trips through writeUrlParams → readUrlParams', () => {
    const state = {
      palette: 'plasma',
      groupCol: 'dtype',
      filter: { col: 'size', op: 'gt', val: '100' },
      dimensions: { c: '0', t: '1' },
      showSignificance: true,
      hiddenWidgets: new Set(['w1', 'w2']),
    };
    writeUrlParams(state);
    const read = readUrlParams();

    expect(read.palette).toBe('plasma');
    expect(read.groupCol).toBe('dtype');
    expect(read.filter).toEqual({ col: 'size', op: 'gt', val: '100' });
    expect(read.dimensions).toEqual({ c: '0', t: '1' });
    expect(read.showSignificance).toBe(true);
    expect(read.hiddenWidgets).toEqual(new Set(['w1', 'w2']));
  });
});