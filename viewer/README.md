# Pixel Patrol Viewer

A single-page web app that visualises Pixel Patrol `.parquet` files.  
Built with [Vite](https://vitejs.dev/), [DuckDB WASM](https://duckdb.org/docs/api/wasm/overview), [Plotly](https://plotly.com/javascript/), and vanilla JS — no framework.

---

## Building

```sh
cd viewer
npm install
npm run build
```

`npm run build` compiles to `dist/` **and** automatically copies the output to
`packages/pixel-patrol-base/src/pixel_patrol_base/viewer_dist/`, so the Python
CLI (`pixel-patrol view`) always picks up the latest build without a manual copy step.

For a hot-reloading development server:

```sh
npm run dev
```

The dev server adds `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers
so DuckDB can use `SharedArrayBuffer` (multi-threaded mode). The same headers are sent by
the Python viewer server; on GitHub Pages DuckDB falls back to the single-threaded bundle
automatically.

---

## Usage

### Local file (browser only)

Open `dist/index.html` (or the dev server URL) in a browser and drag & drop a `.parquet`
file onto the welcome screen, or use the file picker.

### Served by the Python CLI

```sh
pixel-patrol view path/to/data.parquet
```

This starts a local HTTP server and opens the viewer in the default browser. In this mode
all SQL is executed server-side by a native DuckDB connection instead of WASM, which is
significantly faster for large files.

### URL parameter

Append `?file=<url>` to load a remote parquet file directly:

```
http://localhost:8052/?file=https://example.com/data.parquet
```

---

## Extending with plugins

A plugin is a plain JS object with four fields:

```js
{
  id:       'my-widget',           // unique string
  label:    'My Widget',           // card header
  requires(schema) { ... },        // return false to hide when columns are absent
  async render(container, ctx) {   // draw into container (a <div>)
    const rows = await ctx.queryRows('SELECT col FROM pp_data LIMIT 10');
    container.textContent = JSON.stringify(rows);
  },
}
```

`ctx` fields available inside `render`:

| Field | Type | Description |
|-------|------|-------------|
| `ctx.query(sql)` | `async → Arrow Table` | Raw Arrow result (use for binary/blob columns) |
| `ctx.queryRows(sql)` | `async → object[]` | Plain JS objects (blobs as `Uint8Array`) |
| `ctx.querySample(cols, n)` | `async → object[]` | Sampled scalar query shorthand |
| `ctx.schema` | object | `{ metricCols, groupCols, dimensionInfo, allCols, blobCols }` |
| `ctx.state` | object | `{ palette, groupCol, filter, dimensions }` |
| `ctx.colorMap` | object | `{ groupValue: hexColor }` |
| `ctx.where` | string | SQL `WHERE` fragment reflecting the current filter (or `''`) |
| `ctx.groups` | string[] | Distinct values of the active group column |
| `ctx.filteredCount` | number | Rows matching the current filter |
| `ctx.totalRows` | number | Total rows in the file |

The main DuckDB table is always called `pp_data`.

### Registration options

**At page load — inline object:**
```html
<script>
window.__PP_PLUGINS = [{ id: 'my-widget', label: '...', requires: () => true, render: async (el, ctx) => { ... } }];
</script>
```

**At page load — external ES module:**
```html
<script>
window.__PP_PLUGIN_URLS = ['./my-plugin.js'];
</script>
```
The module must export the plugin object as its `default` export.

**At runtime:**
```js
window.PixelPatrol.registerPlugin(plugin);           // object
await window.PixelPatrol.loadPlugin('./my-plugin.js'); // URL → default export
```

If a plugin with the same `id` is registered twice the second call replaces the first,
and the dashboard re-renders immediately.
