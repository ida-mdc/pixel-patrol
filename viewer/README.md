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

### Drop a file in the browser

Open the deployed viewer or `dist/index.html` and drag & drop a `.parquet` file, or use the file picker.

### Python CLI

```sh
pixel-patrol view path/to/data.parquet
```

Starts a local HTTP server and opens the viewer. SQL runs server-side via native DuckDB — significantly faster than WASM for large files.

### URL parameters

| Parameter | Description |
|---|---|
| `?data=<url>` | Load a remote parquet file on startup |
| `?extension=<url>` | Load an extension manifest (repeatable) |

Example:

```
https://ida-mdc.github.io/pixel-patrol/?data=https://example.com/data.parquet&extension=https://example.com/my-extension/extension.json
```

---

## Extensions

An extension is a folder containing an `extension.json` manifest and one or more plugin JS files.

### Manifest format

```json
{
  "name": "My Extension",
  "plugins": ["./plugin_a.js", "./plugin_b.js"]
}
```

Relative paths in `plugins` are resolved against the manifest URL, so the JS files can live alongside it.

### Loading an extension

**Remotely** — pass the manifest URL as a query parameter:

```
?extension=https://your-host/my-extension/extension.json
```

Multiple extensions can be chained: `&extension=url1&extension=url2`

**Locally** — pass the extension directory to `serve_viewer`:

```python
from pixel_patrol_base.viewer_server import serve_viewer
serve_viewer('data.parquet', extension='path/to/my-extension/')
```

The server serves the directory at `/extension/` and injects the manifest URL automatically.

### Writing a plugin

A plugin is an ES module that exports a single object:

```js
export default {
  id:    'my-widget',   // unique across all loaded plugins
  label: 'My Widget',   // shown in the sidebar widget list

  requires(schema) {
    // return false to hide the widget when expected columns are absent
    return schema.allCols.includes('my_column');
  },

  async render(container, ctx) {
    const rows = await ctx.queryRows(`
      SELECT my_column, COUNT(*) AS cnt
      FROM pp_data
      ${ctx.where}
      GROUP BY 1 ORDER BY 2 DESC
    `);
    // write into container using plain DOM, Plotly (window.Plotly), or any CDN library
  },
};
```

`Plotly` is exposed as `window.Plotly` so plugins can use it without a separate import.

The DuckDB table is always named `pp_data`. Use `${ctx.where}` to respect the active filter —
note that `ctx.where` is already a full `WHERE …` clause (or empty string), so if your query
has its own `WHERE`, merge with `AND`:

```js
WHERE my_col IS NOT NULL
  ${ctx.where ? 'AND ' + ctx.where.replace(/^WHERE\s+/i, '') : ''}
```

### `ctx` reference

| Field | Type | Description |
|---|---|---|
| `ctx.queryRows(sql)` | `async → object[]` | Query returning plain JS objects |
| `ctx.query(sql)` | `async → Arrow Table` | Raw Arrow result (for binary/blob columns) |
| `ctx.querySample(cols, n)` | `async → object[]` | Sampled scalar shorthand |
| `ctx.schema` | object | `{ metricCols, groupCols, dimensionInfo, allCols, blobCols }` |
| `ctx.state` | object | `{ palette, groupCol, filter, dimensions }` |
| `ctx.colorMap` | object | `{ groupValue: hexColor }` |
| `ctx.where` | string | SQL `WHERE` clause for the current filter (or `''`) |
| `ctx.groups` | string[] | Distinct values of the active group column |
| `ctx.filteredCount` | number | Rows matching the current filter |
| `ctx.totalRows` | number | Total rows in the file |
