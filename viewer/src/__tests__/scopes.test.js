import { describe, it, expect } from 'vitest';
import { scopeBadgeHtml, setScopeBadge, syncPinnedScopeBadge } from '../scopes.js';

describe('scopeBadgeHtml', () => {
  it('renders a badge for a known scope', () => {
    const html = scopeBadgeHtml('image');
    expect(html).toContain('widget-scope-badge');
    expect(html).toContain('per image');
  });

  it('returns empty string for an unset/unknown scope', () => {
    expect(scopeBadgeHtml(undefined)).toBe('');
    expect(scopeBadgeHtml('bogus')).toBe('');
  });
});

describe('setScopeBadge', () => {
  it('updates text/title/color on the given element', () => {
    const el = document.createElement('span');
    setScopeBadge(el, 'slice');
    expect(el.textContent).toContain('per slice');
    expect(el.title.length).toBeGreaterThan(0);
  });

  it('is a no-op when el is missing or scope is unknown', () => {
    expect(() => setScopeBadge(null, 'image')).not.toThrow();
    const el = document.createElement('span');
    setScopeBadge(el, 'bogus');
    expect(el.textContent).toBe('');
  });
});

// Widget card DOM shape produced by renderer.js's createCard(): the badge is
// a sibling of the render() container, found via .closest('.widget-card').
function makeCard() {
  const card = document.createElement('div');
  card.className = 'widget-card';
  const badge = document.createElement('span');
  badge.className = 'widget-scope-badge';
  card.appendChild(badge);
  const body = document.createElement('div');
  card.appendChild(body);
  return { card, body, badge };
}

describe('syncPinnedScopeBadge', () => {
  it('sets "image" when no dim is pinned and no runtime slice', () => {
    const { body, badge } = makeCard();
    syncPinnedScopeBadge(body, { state: { dimensions: {} } });
    expect(badge.textContent).toContain('per image');
  });

  it('sets "slice" when a sidebar dimension is pinned', () => {
    const { body, badge } = makeCard();
    syncPinnedScopeBadge(body, { state: { dimensions: { c: '1' } } });
    expect(badge.textContent).toContain('per slice');
  });

  it('ignores non-finite pinned values (e.g. "" for "All")', () => {
    const { body, badge } = makeCard();
    syncPinnedScopeBadge(body, { state: { dimensions: { c: '' } } });
    expect(badge.textContent).toContain('per image');
  });

  it('sets "slice" via runtimeSlice even with no pinned dims', () => {
    const { body, badge } = makeCard();
    syncPinnedScopeBadge(body, { state: { dimensions: {} } }, true);
    expect(badge.textContent).toContain('per slice');
  });

  it('does nothing when the container has no ancestor .widget-card', () => {
    const orphan = document.createElement('div');
    expect(() => syncPinnedScopeBadge(orphan, { state: { dimensions: {} } })).not.toThrow();
  });
});

describe('pinnedDims', () => {
  it('treats empty, blank, null and non-numeric values as not pinned', async () => {
    const { pinnedDims } = await import('../sql.js');
    expect(pinnedDims({ c: '1', t: '', z: ' ', y: null, x: 'abc', a: '0' })).toEqual({ c: 1, a: 0 });
  });
});
