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
│   └── markdown_diary_tracker_static/
│       ├── __init__.py
│       ├── my_loader.py             custom loader  — reads .md files
│       ├── my_processor.py          custom processor — computes positivity score
│       ├── plugin_registry.py       registers loader, processor + viewer extension
│       └── viewer/
│           ├── extension.json       manifest listing the viewer plugins
│           ├── plugin_mood_trend.js     mood-count bar chart + positivity trend
│           └── plugin_word_frequency.js top-40 word frequency chart
├── create_and_show_report.py    processes data and opens the viewer
└── pyproject.toml
```

---

## The loader

`src/markdown_diary_tracker_static/my_loader.py` implements `MarkdownDiaryLoader`
(name: `markdown-diary`).
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

`src/markdown_diary_tracker_static/my_processor.py` implements
`MarkdownMoodProcessor` (name: `markdown-mood`).  It runs on every
`text/markdown` record and adds:

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
the viewer at `http://127.0.0.1:8052`.  The viewer extensions are loaded
automatically because `serve_viewer` discovers the `pixel_patrol.viewer_extensions`
entry-point declared in `pyproject.toml` — no explicit path needed.

---

## Sharing the report

### Install the pip package

Because the JS viewer plugins are **bundled inside the pip package**
(`src/markdown_diary_tracker_static/viewer/`), the recipient only needs to
install the package:

```sh
pip install markdown-diary-tracker-static   # or uv add / pip install -e .
```

Then open any report generated with this extension — the viewer plugins load
automatically, no extra arguments required:

```python
from pixel_patrol_base.viewer_server import serve_viewer
serve_viewer("path/to/report.parquet")
```

### Host on GitHub Pages (no Python required)

A GitHub Actions workflow is included at `.github/workflows/deploy-pages.yml`.
It deploys the `viewer/` folder so the extension can be referenced by URL.

To activate it:

1. Copy this folder as a new repository.
2. Go to Settings → Pages → Source and select **GitHub Actions**.
3. Push to `main` — the workflow deploys automatically.

Your extension manifest will be live at:

```
https://<your-org>.github.io/<your-repo>/extension.json
```

Open the deployed Pixel Patrol viewer and pass your extension URL:

```
https://ida-mdc.github.io/pixel-patrol/
  ?data=<parquet-url>&extension=https://<your-org>.github.io/<your-repo>/extension.json
```

Multiple extensions can be chained by repeating `&extension=`.

---

## Writing your own plugin

See the [viewer README](../../viewer/README.md) for the full plugin writing guide,
`ctx` reference, and extension format documentation.
