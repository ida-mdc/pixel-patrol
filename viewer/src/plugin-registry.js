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

// Built-in plugins are now loaded at runtime from installed Python packages
// (pixel_patrol.viewer_extensions entry-point group) rather than bundled here.
// This means plugin JS files can be edited without rebuilding the viewer.
const BUILTIN_PLUGINS = [];

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
   * Dynamically import one or more plugins from url and register them.
   * The module's default export may be a single plugin object or an array.
   * Returns the array of registered plugin objects.
   */
  async loadFromUrl(url) {
    // @vite-ignore tells Vite not to analyse this dynamic import statically.
    const mod      = await import(/* @vite-ignore */ url);
    const exported = mod.default ?? mod;
    const plugins  = Array.isArray(exported) ? exported : [exported];
    plugins.forEach(p => this.register(p));
    return plugins;
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

