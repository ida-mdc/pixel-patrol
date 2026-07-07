import { describe, it, expect } from 'vitest';

// Built-in widgets are plain ES modules shipped in this package and loaded into
// the viewer at runtime. Import every one of them so the widget contract is
// checked against the real, shipped files - a newly added widget is picked up
// automatically.
const modules = import.meta.glob('../../src/pixel_patrol_base/viewer/plugin_*.js', { eager: true });

// A module's default export is either a single plugin object or an array of them
// (e.g. the violin / across-dimensions widgets ship a basic + quality variant),
// so flatten into one list of { plugin, file } records.
const widgets = Object.entries(modules).flatMap(([file, mod]) => {
  const exported = mod.default ?? mod;
  return (Array.isArray(exported) ? exported : [exported]).map(plugin => ({ plugin, file }));
});

// Group-ordering mirrors viewer/src/plugin-groups.js. Kept local so this package
// has no import dependency on the viewer - widgets only ever see `group` strings.
const GROUP_ORDER = ['Summary', 'File Stats', 'Metadata', 'Dataset Stats', 'Visualization', 'Other Widgets'];
const CANON = new Map(GROUP_ORDER.map(g => [g.toLowerCase(), g]));
const pluginGroup = plugin => {
  const raw = String(plugin?.group ?? '').trim();
  if (!raw) return 'Other Widgets';
  return CANON.get(raw.toLowerCase()) ?? raw;
};

// A schema with all array fields present so requires() never trips over an
// undefined field; merge in just the keys a given widget cares about.
function schema(over = {}) {
  return {
    metricCols: [], dimCols: [], groupCols: [], dimensionInfo: {},
    allCols: [], blobCols: [],
    ...over,
  };
}

describe('widget set', () => {
  it('discovers the shipped built-in widgets', () => {
    // Guards against the glob silently matching nothing (wrong path, moved files).
    expect(widgets.length).toBeGreaterThanOrEqual(11);
  });

  it('every widget has a unique id', () => {
    const ids = widgets.map(w => w.plugin.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe('widget contract', () => {
  it.each(widgets.map(w => [w.plugin.id || w.file, w.plugin]))(
    '%s satisfies the plugin contract',
    (_name, plugin) => {
      expect(typeof plugin.id).toBe('string');
      expect(plugin.id.length).toBeGreaterThan(0);
      expect(typeof plugin.label).toBe('string');
      expect(plugin.label.length).toBeGreaterThan(0);
      expect(typeof plugin.requires).toBe('function');
      expect(typeof plugin.render).toBe('function');
      // condensedMessage is optional, but when present must be callable.
      if ('condensedMessage' in plugin) {
        expect(typeof plugin.condensedMessage).toBe('function');
      }
      // Every widget resolves to a (string) group name.
      expect(typeof pluginGroup(plugin)).toBe('string');
    },
  );

  it('every widget hides itself when given an empty schema', () => {
    // requires() must never throw and must return a real boolean; with nothing
    // in the schema no widget should claim it can render.
    for (const { plugin, file } of widgets) {
      const ok = plugin.requires(schema());
      expect(typeof ok, `${file} requires() should return a boolean`).toBe('boolean');
      expect(ok, `${plugin.id} should be hidden on an empty schema`).toBe(false);
    }
  });
});

// Minimal schema fragment that should make each widget's requires() return true.
// Keyed by widget id so a renamed/removed widget surfaces as a clear failure.
const ENABLING_SCHEMA = {
  'summary':                   { allCols: ['size_bytes', 'file_extension'] },
  'file-stats':                { allCols: ['file_extension', 'size_bytes'] },
  'histogram':                 { blobCols: ['histogram_counts'] },
  'mosaic':                    { blobCols: ['thumbnail'], metricCols: ['mean_intensity'], allCols: ['mean_intensity'] },
  'sunburst':                  { allCols: ['path', 'size_bytes'] },
  'image-table':               { allCols: ['size_bytes'] },
  'custom-plot':               { allCols: ['size_bytes'] },
  'dim-size':                  { allCols: ['size_X', 'size_Y'] },
  'metadata':                  { allCols: ['dtype'] },
  'violin-basic':              { metricCols: ['mean_intensity'] },
  'violin-quality':            { metricCols: ['laplacian_variance'] },
  'stats-across-dims-basic':   { metricCols: ['mean_intensity'], hasDimSlices: true },
  'stats-across-dims-quality': { metricCols: ['laplacian_variance'], hasDimSlices: true },
};

describe('widget requires()', () => {
  it('every shipped widget has an enabling-schema case', () => {
    // Forces this table to be kept in sync with the widgets that actually ship.
    const ids = new Set(widgets.map(w => w.plugin.id));
    expect(new Set(Object.keys(ENABLING_SCHEMA))).toEqual(ids);
  });

  it.each(widgets.map(w => [w.plugin.id, w.plugin]))(
    '%s shows itself once its required columns are present',
    (id, plugin) => {
      const over = ENABLING_SCHEMA[id];
      expect(over, `no enabling schema declared for ${id}`).toBeTruthy();
      expect(plugin.requires(schema(over))).toBe(true);
    },
  );

  it('metadata also shows itself when only dimension info is present', () => {
    const metadata = widgets.find(w => w.plugin.id === 'metadata').plugin;
    expect(metadata.requires(schema({ dimensionInfo: { t: [0, 1] } }))).toBe(true);
  });

  it('violin-quality stays hidden when only basic metrics exist', () => {
    const quality = widgets.find(w => w.plugin.id === 'violin-quality').plugin;
    expect(quality.requires(schema({ metricCols: ['mean_intensity'] }))).toBe(false);
  });
});
