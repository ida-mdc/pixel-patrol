/**
 * Plugin index — re-exports the live plugin list from the runtime registry.
 *
 * The canonical plugin list and registration API lives in ../plugin-registry.js.
 * Built-in plugins are listed and ordered there.
 *
 * To add a built-in plugin at build time:
 *   1. Create src/plugins/my-plugin.js exporting a default object.
 *   2. Import it in ../plugin-registry.js and add it to BUILTIN_PLUGINS.
 *
 * To add a plugin at runtime (no rebuild):
 *   - window.PixelPatrol.registerPlugin({ id, label, requires, render })
 *   - window.PixelPatrol.loadPlugin('./my-plugin.js')
 *   - window.__PP_PLUGINS = [...]          (before viewer boots)
 *   - window.__PP_PLUGIN_URLS = [...]      (before viewer boots)
 */

export { registry } from '../plugin-registry.js';
