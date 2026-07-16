const GROUP_ORDER = ['Summary', 'File Stats', 'Metadata', 'Dataset Stats', 'Visualization', 'Other Widgets'];

// Explicit widget order within a group, by plugin id. Without this, order
// within a group just falls out of plugin *registration* order, which for
// auto_detect extensions is an alphabetical glob of plugin_*.js filenames -
// an accident of naming, not a real ordering decision (e.g. it put "stats
// across dims" before "violin" purely because "s" < "v"). Anything not
// listed here keeps its registration order, appended after the listed ones.
const WIDGET_ORDER = {
  'Dataset Stats': ['violin-basic', 'violin-quality', 'stats-across-dims-basic', 'stats-across-dims-quality'],
};

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

/**
 * Group plugins by their canonical group (Map<groupName, plugin[]>, insertion
 * order = first-seen group order). Within each group, plugins are sorted per
 * WIDGET_ORDER when that group has an explicit order; otherwise they keep
 * registration order. The sort is stable, so unlisted plugins in an ordered
 * group keep their relative registration order too, appended after the
 * listed ones.
 */
export function groupPlugins(plugins) {
  const grouped = new Map();
  for (const p of plugins) {
    const g = pluginGroup(p);
    if (!grouped.has(g)) grouped.set(g, []);
    grouped.get(g).push(p);
  }
  for (const [g, list] of grouped) {
    const order = WIDGET_ORDER[g];
    if (!order) continue;
    const rank = id => { const i = order.indexOf(id); return i === -1 ? order.length : i; };
    list.sort((a, b) => rank(a.id) - rank(b.id));
  }
  return grouped;
}

