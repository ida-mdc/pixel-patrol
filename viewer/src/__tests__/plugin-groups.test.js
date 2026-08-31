import { describe, it, expect } from 'vitest';
import { pluginGroup, orderedGroupNames, groupPlugins } from '../plugin-groups.js';

describe('pluginGroup', () => {
  it('canonicalizes a known group regardless of case', () => {
    expect(pluginGroup({ group: 'dataset stats' })).toBe('Dataset Stats');
  });

  it('falls back to "Other Widgets" for an unset group', () => {
    expect(pluginGroup({})).toBe('Other Widgets');
  });

  it('passes through an unknown group name as-is', () => {
    expect(pluginGroup({ group: 'Third Party' })).toBe('Third Party');
  });
});

describe('orderedGroupNames', () => {
  it('orders known groups per GROUP_ORDER and unknown groups alphabetically after', () => {
    const plugins = [
      { group: 'Visualization' }, { group: 'Zeta' }, { group: 'Summary' }, { group: 'Alpha' },
    ];
    expect(orderedGroupNames(plugins)).toEqual(['Summary', 'Visualization', 'Alpha', 'Zeta']);
  });
});

describe('groupPlugins', () => {
  it('sorts Dataset Stats plugins per the explicit WIDGET_ORDER regardless of input order', () => {
    const plugins = [
      { id: 'stats-across-dims-basic', group: 'Dataset Stats' },
      { id: 'stats-across-dims-quality', group: 'Dataset Stats' },
      { id: 'violin-basic', group: 'Dataset Stats' },
      { id: 'violin-quality', group: 'Dataset Stats' },
    ];
    const grouped = groupPlugins(plugins);
    expect(grouped.get('Dataset Stats').map(p => p.id)).toEqual([
      'violin-basic', 'violin-quality', 'stats-across-dims-basic', 'stats-across-dims-quality',
    ]);
  });

  it('appends plugins not in WIDGET_ORDER after the listed ones, keeping their relative order', () => {
    const plugins = [
      { id: 'stats-across-dims-basic', group: 'Dataset Stats' },
      { id: 'third-party-b', group: 'Dataset Stats' },
      { id: 'violin-basic', group: 'Dataset Stats' },
      { id: 'third-party-a', group: 'Dataset Stats' },
    ];
    const grouped = groupPlugins(plugins);
    expect(grouped.get('Dataset Stats').map(p => p.id)).toEqual([
      'violin-basic', 'stats-across-dims-basic', 'third-party-b', 'third-party-a',
    ]);
  });

  it('leaves registration order untouched for groups with no explicit WIDGET_ORDER', () => {
    const plugins = [
      { id: 'z', group: 'Summary' },
      { id: 'a', group: 'Summary' },
    ];
    const grouped = groupPlugins(plugins);
    expect(grouped.get('Summary').map(p => p.id)).toEqual(['z', 'a']);
  });

  it('does not mutate the input array', () => {
    const plugins = [
      { id: 'stats-across-dims-basic', group: 'Dataset Stats' },
      { id: 'violin-basic', group: 'Dataset Stats' },
    ];
    const snapshot = [...plugins];
    groupPlugins(plugins);
    expect(plugins).toEqual(snapshot);
  });
});
