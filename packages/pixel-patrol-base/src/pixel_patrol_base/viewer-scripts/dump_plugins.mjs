// Dump static metadata of the viewer widget plugins as JSON to stdout.
//
// Usage:  node dump_plugins.mjs <extension-dir> [<extension-dir> ...]
//
// Each <extension-dir> contains an `extension.json` listing plugin modules.
// We import each module's default export (an object or an array of objects) and
// emit the descriptive, statically-known fields. Functions like `requires` are
// skipped; `inputs` (declarative column dependencies) is included when present.
//
// Used by pixel_patrol_base.core.schema_catalog to add the `widgets` section to
// the schema catalog. Best-effort: errors for a single plugin are reported on
// stderr and that plugin is skipped, so one bad module can't break the export.

import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const FIELDS = ['id', 'group', 'scope', 'label', 'shortLabel', 'info', 'inputs'];

function pick(plugin) {
  const out = {};
  for (const key of FIELDS) {
    if (plugin && plugin[key] !== undefined && typeof plugin[key] !== 'function') {
      out[key] = plugin[key];
    }
  }
  return out;
}

async function loadExtension(dir) {
  const manifestPath = path.join(dir, 'extension.json');
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const widgets = [];
  for (const rel of manifest.plugins ?? []) {
    const modPath = path.resolve(dir, rel);
    try {
      const mod = await import(pathToFileURL(modPath).href);
      const exported = mod.default;
      const list = Array.isArray(exported) ? exported : [exported];
      for (const plugin of list) {
        if (plugin && plugin.id) widgets.push(pick(plugin));
      }
    } catch (err) {
      process.stderr.write(`dump_plugins: skipped '${rel}': ${err.message}\n`);
    }
  }
  return widgets;
}

async function main() {
  const dirs = process.argv.slice(2);
  const all = [];
  for (const dir of dirs) {
    try {
      all.push(...await loadExtension(dir));
    } catch (err) {
      process.stderr.write(`dump_plugins: skipped dir '${dir}': ${err.message}\n`);
    }
  }
  all.sort((a, b) => String(a.id).localeCompare(String(b.id)));
  process.stdout.write(JSON.stringify(all, null, 2));
}

main();
