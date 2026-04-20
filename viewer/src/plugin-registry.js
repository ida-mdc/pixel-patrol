/**
 * Runtime plugin registry.
 *
 * Built-in plugins are bundled at build time. External plugins are loaded via
 * an extension manifest — a JSON file listing the plugin JS URLs:
 *
 *   Remote:  ?extension=https://example.com/my-extension/extension.json
 *   Local:   window.__PP_EXTENSION_URLS = ['/extension/extension.json']
 *            (injected automatically when serve_viewer(extension=...) is used)
 *
 * Manifest format:
 *   { "plugins": ["./plugin_a.js", "./plugin_b.js"] }
 *   Relative paths are resolved against the manifest URL.
 *
 * Plugin contract:
 *   {
 *     id:       string   — unique identifier
 *     label:    string   — card header title
 *     requires(schema) → bool        — return false to hide when columns absent
 *     async render(container, ctx) → void  — draw into the provided DOM element
 *   }
 *
 * ctx fields available to plugins:
 *   ctx.query(sql)           → raw Arrow Table  (use for binary / blob columns)
 *   ctx.queryRows(sql)       → plain JS objects (binary cols as Uint8Array)
 *   ctx.querySample(cols, n) → sampled scalar query shorthand
 *   ctx.schema               → { metricCols, groupCols, dimensionInfo, allCols, blobCols }
 *   ctx.state                → { palette, groupCol, filter, dimensions }
 *   ctx.colorMap             → { groupValue: hexColor }
 *   ctx.where                → SQL WHERE fragment (or '')
 *   ctx.groups               → distinct group values
 *   ctx.filteredCount        → rows matching current filter
 *   ctx.totalRows            → total rows in file
 */

import summary          from './plugins/summary.js';
import fileStats        from './plugins/file-stats.js';
import sunburst         from './plugins/sunburst.js';
import metadata         from './plugins/metadata.js';
import dimSize          from './plugins/dim-size.js';
import violinPlugins        from './plugins/violin.js';
import histogram            from './plugins/histogram.js';
import statsAcrossDimsPlugins from './plugins/stats-across-dims.js';
import mosaic           from './plugins/mosaic.js';

const BUILTIN_PLUGINS = [
  summary,
  fileStats,
  sunburst,
  metadata,
  dimSize,
  ...violinPlugins,
  ...statsAcrossDimsPlugins,
  histogram,
  mosaic,
];

// ── Registry ──────────────────────────────────────────────────────────────────

const _plugins   = [...BUILTIN_PLUGINS];
const _listeners = new Set();

export const registry = {
  /** Current list of registered plugins (built-ins + runtime additions). */
  get plugins() { return _plugins; },

  /**
   * Register a plugin object directly.
   * If a plugin with the same id already exists it is replaced in-place.
   * Triggers all onAdd listeners after registration.
   */
  register(plugin) {
    if (!plugin || typeof plugin.id !== 'string' || typeof plugin.render !== 'function') {
      console.warn('[PixelPatrol] registerPlugin: invalid plugin — needs id (string) and render (function):', plugin);
      return false;
    }
    const idx = _plugins.findIndex(p => p.id === plugin.id);
    if (idx >= 0) {
      _plugins[idx] = plugin;
    } else {
      _plugins.push(plugin);
    }
    _listeners.forEach(fn => fn(plugin));
    return true;
  },

  /**
   * Dynamically import a plugin from url and register it.
   * The module must export the plugin object as its default export.
   * Returns the registered plugin object.
   */
  async loadFromUrl(url) {
    // @vite-ignore tells Vite not to analyse this dynamic import statically.
    const mod    = await import(/* @vite-ignore */ url);
    const plugin = mod.default ?? mod;
    this.register(plugin);
    return plugin;
  },

  /**
   * Subscribe to plugin additions / replacements.
   * fn is called with the newly registered plugin object.
   * Returns an unsubscribe function.
   */
  onAdd(fn) {
    _listeners.add(fn);
    return () => _listeners.delete(fn);
  },
};

