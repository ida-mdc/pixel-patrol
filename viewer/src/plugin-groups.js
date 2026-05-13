const GROUP_ORDER = ['Summary', 'File Stats', 'Metadata', 'Dataset Stats', 'Visualization', 'Other Widgets'];

const CANON = new Map(GROUP_ORDER.map(g => [g.toLowerCase(), g]));

export function pluginGroup(plugin) {
  const raw = String(plugin?.group ?? '').trim();
  if (!raw) return 'Other Widgets';
  return CANON.get(raw.toLowerCase()) ?? raw;
}

export function orderedGroupNames(plugins) {
  const names = [...new Set(plugins.map(pluginGroup))];
  return [
    ...GROUP_ORDER.filter(g => names.includes(g)),
    ...names.filter(g => !GROUP_ORDER.includes(g)).sort(),
  ];
}

