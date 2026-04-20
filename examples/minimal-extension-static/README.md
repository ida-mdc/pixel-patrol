# Minimal Extension (Static Viewer)

A complete, self-contained Pixel Patrol extension example.  It covers the full
extension surface: a custom **loader**, a custom **processor**, and two custom
**viewer plugins** that visualise the resulting data.

```
minimal-extension-static/
├── data/                        example Markdown diary entries
│   ├── 2024/
│   └── 2025/
├── src/
│   ├── my_loader.py             custom loader  — reads .md files
│   ├── my_processor.py          custom processor — computes positivity score
│   └── plugin_registry.py       registers loader + processor as PP plugins
├── viewer/
│   ├── extension.json           manifest listing the viewer plugins
│   ├── plugin_mood_trend.js     mood-count bar chart + positivity trend
│   └── plugin_word_frequency.js top-40 word frequency chart
├── create_and_show_report.py    processes data and opens the viewer
└── pyproject.toml
```

---

## The loader

`src/my_loader.py` implements `MarkdownDiaryLoader` (name: `markdown-diary`).
It reads `.md` files with a YAML front-matter block:

```markdown
---
date: 2024-01-05
moods: tired, calm
---
Slow start to the year. ...
```

Each file becomes a record with three metadata fields:

| Column | Type | Description |
|---|---|---|
| `entry_date` | `str` | ISO date from front-matter |
| `moods` | `list[str]` | mood tags from front-matter |
| `free_text` | `str` | body text after the front-matter block |

---

## The processor

`src/my_processor.py` implements `MarkdownMoodProcessor` (name:
`markdown-mood`).  It runs on every `text/markdown` record and adds:

| Column | Type | Description |
|---|---|---|
| `positivity_factor` | `float` | `(positive_moods − negative_moods) / total_moods`, range `[−1, 1]` |

---

## The viewer plugins

Both plugins are ES modules that export a plugin object.  They are listed in
`viewer/extension.json` and loaded automatically by the viewer.

**`plugin_mood_trend.js`** — requires `moods`, `entry_date`, `positivity_factor`, `imported_path_short`
- Stacked bar chart: mood occurrence counts per folder
- Line chart: mean positivity across the year, one line per folder/year

**`plugin_word_frequency.js`** — requires `free_text`
- Bar chart of the 40 most frequent non-stopword words in the diary entries

---

## Running locally

Install and run:

```sh
uv run python create_and_show_report.py
```

This processes the example diary files, saves `out/report.parquet`, and serves
the viewer at `http://127.0.0.1:8052` with the `viewer/` extension loaded.

---

## Deploying to GitHub Pages

A GitHub Actions workflow is already included at
`.github/workflows/deploy-pages.yml`.  To activate it:

1. Copy this folder as a new repository.
2. Go to Settings → Pages → Source and select **GitHub Actions**.
3. Push to `main` — the workflow deploys the `viewer/` folder automatically.

Your extension manifest will be live at:

```
https://<your-org>.github.io/<your-repo>/extension.json
```

---

## Linking to the viewer

Open the deployed Pixel Patrol viewer and pass your extension URLs:

```
https://ida-mdc.github.io/pixel-patrol/
  &extension=https://<your-org>.github.io/<your-repo>/extension.json
```

Multiple extensions can be chained by repeating `&extension=`.

---

## Writing your own plugin

Each plugin is an ES module with four fields:

```js
export default {
  id:    'my-widget',
  label: 'My Widget',

  requires(schema) {
    return schema.allCols.includes('my_column');
  },

  async render(container, ctx) {
    const rows = await ctx.queryRows(`
      SELECT my_column, COUNT(*) AS cnt
      FROM pp_data ${ctx.where}
      GROUP BY 1 ORDER BY 2 DESC
    `);
    // write into container using plain DOM, Plotly, or any CDN library
  },
};
```

Register it in `viewer/extension.json`:

```json
{
  "name": "My Extension",
  "plugins": ["./plugin_my_widget.js"]
}
```

### `ctx` reference

| Field | Type | Description |
|---|---|---|
| `ctx.queryRows(sql)` | `async → object[]` | Query returning plain JS objects |
| `ctx.query(sql)` | `async → Arrow Table` | Raw Arrow result (for binary columns) |
| `ctx.querySample(cols, n)` | `async → object[]` | Sampled scalar shorthand |
| `ctx.schema` | object | `{ metricCols, groupCols, dimensionInfo, allCols, blobCols }` |
| `ctx.state` | object | `{ palette, groupCol, filter, dimensions }` |
| `ctx.colorMap` | object | `{ groupValue: hexColor }` |
| `ctx.where` | string | SQL `WHERE` clause for the current filter (or `''`) |
| `ctx.groups` | string[] | Distinct values of the active group column |
| `ctx.filteredCount` | number | Rows matching the current filter |
| `ctx.totalRows` | number | Total rows in the file |

The DuckDB table is always named `pp_data`.
