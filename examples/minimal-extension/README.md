# Pixel HAI Watch — Minimal Extension

<img src="assets/shark.png" alt="A softly glowing cartoon shark" width="260" align="right">

A complete, self-contained Pixel Patrol extension — and, more importantly, a
guide to building your own. It walks through the **entire** extension
surface: a custom **loader**, a custom **processor**, and two custom
**viewer widgets** (one that visualises loader metadata, one that visualises
a processor's output).

**You almost certainly don't need all four.** A loader, a processor, and a
viewer widget are independent pieces — Pixel Patrol discovers and combines
whatever you ship, and nothing requires you to write more than one. This
example bundles all four side by side purely so you can see how each is
*shaped* and how they fit together; in the wild you're far more likely to
publish, say, just a processor that computes a metric on anyone's images, or
just a loader for one niche file format, or just a widget that charts columns
other people's extensions already produce. Skip straight to the piece you
need — "[Defining the package](#defining-the-package)" shows how each is
declared on its own, and you can delete the rest.

```
minimal-extension/
├── data/                                tiny generated dataset (8 patches, 2 folders)
│   ├── azores_log/
│   └── kermadec_log/
├── make_dataset.py                      (re)generates data/ — run once, or it's auto-run
├── src/
│   └── pixel_patrol_hai_watch/
│       ├── __init__.py
│       ├── my_loader.py                 custom loader     — reads .parquet "dive patches"
│       ├── my_processor.py              custom processor  — counts the bioluminescent glow in each patch
│       ├── plugin_registry.py           registers loader, processor + viewer extension
│       └── viewer/
│           ├── extension.json                  manifest listing the viewer plugins
│           ├── plugin_dives_logged.js          metadata widget    — dives logged, by depth zone & site
│           └── plugin_glow_by_depth.js         processor-output widget — glow spotted, by depth zone
├── create_and_show_report.py            generates data (if needed), processes it, opens the viewer
└── pyproject.toml
```

---

## The dataset

`make_dataset.py` writes eight tiny `.parquet` files into `data/<folder>/`.
Each one represents a 16x16 grayscale snapshot from a deep-sea camera:

- **Pixel data** — one `uint8` column per pixel column (`px_00` … `px_15`),
  one row per pixel row. Stack the columns side by side and you get the YX
  pixel grid directly — the table *is* the image.
- **Fake image metadata** — `depth_zone` (`sunlit`/`twilight`/`midnight`/
  `abyss`) is stored as a key/value pair in the parquet schema's metadata
  (`table.schema.metadata`), exactly the slot real formats (OME-XML in TIFFs,
  EXIF in JPEGs, ...) use to carry instrument/acquisition info.

There's no real microscopy or biology here — it's all synthetic, generated
by a few lines of numpy. That's the point: a `.parquet` table of numbers can
stand in for "an image" just as well as a TIFF can, as long as something
hands it to Pixel Patrol with a shape, a dtype, and a `kind`. Open
`make_dataset.py` if you're curious how the patches are painted, or just
treat `data/` as "some files my loader needs to read."

The two folders (`azores_log` / `kermadec_log`) play the same role as the
year folders in a typical Pixel Patrol dataset: they become the `path` /
`imported_path_short` grouping used throughout the report.

---

## The loader

`src/pixel_patrol_hai_watch/my_loader.py` implements `SharkCamLoader`
(name: `shark-cam`).

A loader is any class that satisfies the `PixelPatrolLoader` protocol from
`pixel_patrol_base.core.contracts` — a [`typing.Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol),
not a base class, so `SharkCamLoader` needs no import or inheritance from
`pixel_patrol_base`; matching the shape below is enough. It needs:

| Member | Type | Purpose |
|---|---|---|
| `NAME` | `str` | unique identifier passed to `create_project(..., loader=...)` |
| `SUPPORTED_EXTENSIONS` | `set[str]` | file extensions this loader can read (lower-case, no dot) |
| `FOLDER_EXTENSIONS` | `set[str]` | "extensions" that mark a *folder* as a single loadable unit (e.g. OME-Zarr stores); usually empty |
| `CONTAINER_EXTENSIONS` | `set[str]` | extensions that may contain more than one image (multi-series OME-TIFF, LMDB, ...); usually empty |
| `OUTPUT_SCHEMA` | `dict[str, type]` | extra metadata columns this loader adds to the report, with their types |
| `OUTPUT_SCHEMA_PATTERNS` | `list[tuple[str, type]]` | regex/type pairs for dynamically-named metadata columns (e.g. `pixel_size_X`); usually empty |
| `is_folder_supported(path)` | `(Path) -> bool` | whether a *folder* (not a file) should be treated as one image |
| `read_header(path)` | `(Path) -> FileInfo` | cheap shape/dtype/dim-order probe, **no pixel data loaded** |
| `load(path)` | `(Path) -> Record` | loads one image and returns a `Record` |
| `load_range(path, start, stop)` | `(Path, int, int) -> Iterator[(str, Record)]` | yields sub-images for container formats; raise `NotImplementedError` otherwise |

`SharkCamLoader` reads each table with `pyarrow.parquet`, stacks its columns
into a 2-D `uint8` array, decodes `depth_zone` out of the schema metadata,
and wraps everything with `record_from(...)`. It declares `kind="intensity"`
and `dim_order="YX"` — which is what makes the *built-in* processors (basic
metrics, histogram, thumbnail) pick the patches up automatically, right
alongside our custom one. That's the point of the exercise: to a Pixel
Patrol pipeline, a "dive patch" parquet table behaves just like any other 2-D
image.

**On its own:** a loader needs nothing but a name, a couple of small
declarations, and these five methods — no processor, no widget, nothing else
to register. If your only problem is "Pixel Patrol can't read my file
format," this is the entire surface you need to implement.

---

## The processor

`src/pixel_patrol_hai_watch/my_processor.py` implements `GlowSpotterProcessor`
(name: `glow-spotter`).

A processor is any class that satisfies the `PixelPatrolProcessor` protocol:

| Member | Type | Purpose |
|---|---|---|
| `NAME` | `str` | unique identifier (shown in pipeline logs) |
| `CHUNK_KIND` | `ChunkKind` | `LEAF` (user-configured tiles/slices), `MEMORY` (whole record, memory-safe chunking), or `FULL_RECORD` |
| `INPUT` | `RecordSpec` | which records this processor runs on (`kinds`, `axes`, `capabilities`, ...) |
| `OUTPUT` | `"features" \| "record"` | whether `run_chunk` returns columns to merge, or a brand-new `Record` |
| `OUTPUT_SCHEMA` | `dict[str, type]` | the columns this processor adds, with their types |
| `run_chunk(record)` | `(Record) -> dict` | does the actual computation on one chunk |
| `get_aggregation(name)` | `(str) -> Callable \| None` | how to combine multiple chunks' values for column `name` into the per-image value |

`GlowSpotterProcessor` runs on every `intensity` record with `X`/`Y` axes
(`RecordSpec(axes={"X", "Y"}, kinds={"intensity"})`), uses `CHUNK_KIND.MEMORY`
because these patches are small enough to process whole, and adds:

| Column | Type | Description |
|---|---|---|
| `glow_count` | `int` | number of pixels that stand out clearly brighter than the patch's median (`> median + 60`) — the patch's "glow". Sunlit patches have almost none, deep ones light up with plenty, by construction. |

Because each patch is processed in a single chunk, `get_aggregation` simply
returns the lone chunk's value (`rows[0][col]`) — see `RasterProcessor` in
`pixel_patrol_base` for processors that genuinely need to combine many chunks.

**On its own — and this is the important bit:** notice that
`GlowSpotterProcessor` never imports, names, or in any way refers to
`SharkCamLoader`. It only declares the *kind* of record it needs (an
`intensity` image with `X`/`Y` axes) and works on whatever satisfies that —
these synthetic dive patches, a microscopy TIFF stack, a photo, anything. A
processor is a pure function from "image-shaped data" to "a few new columns";
write it once, and it runs against every loader — yours, ours, or a third
party's — that produces a matching record. If your only goal is "compute this
metric for everyone's images," a processor like this *is* your entire
extension.

---

## The viewer widgets

This extension ships **two** widgets on purpose — one for each kind of data a
loader can surface:

- **`plugin_dives_logged.js`** ("Dives Logged") visualises the *fake image
  metadata* on its own: a stacked bar chart of how many dive snapshots were
  logged in each `depth_zone`, split by dive site (`imported_path_short`) —
  `depth_zone` is read straight out of the parquet schema metadata by
  `SharkCamLoader`, while the site grouping comes for free with every Pixel
  Patrol report.
- **`plugin_glow_by_depth.js`** ("Glow Sightings by Depth") visualises a
  *metric derived from the actual pixel data*: a jittered scatter of
  `glow_count` (computed by `GlowSpotterProcessor`) against `depth_zone`,
  colored by site just like the built-in widgets. A scatter is used rather
  than a box/violin plot because each category only holds a handful of
  points — exactly the kind of small sample where distributional summaries
  would mislead rather than inform.

Both are listed in `viewer/extension.json` and loaded automatically by the
viewer, and both declare `group: 'Pixel HAI Watch'` so they get their own
named sidebar section instead of being lumped under the generic "Other
Widgets".

A plugin is a JS module that exports a single default object:

```js
export default {
  id:    'my-widget',          // unique across all loaded plugins
  label: 'My Widget',          // shown in the sidebar widget list
  group: 'My Extension Name',  // optional — gives the widget its own sidebar section

  requires(schema) {
    // return false to hide the widget when its columns are absent
    return schema.allCols.includes('my_column');
  },

  async render(container, ctx) {
    const rows = await ctx.queryRows(`
      SELECT my_column, COUNT(*) AS cnt
      FROM pp_data
      ${ctx.where}
      GROUP BY 1 ORDER BY 2 DESC
    `);
    // write into `container` using plain DOM, Plotly (window.Plotly), or any CDN library
  },
};
```

The DuckDB table is always named `pp_data`; `Plotly` is exposed globally as
`window.Plotly`. The extension manifest just lists the plugin files:

```json
{
  "name": "Pixel HAI Watch Extension",
  "plugins": ["./plugin_dives_logged.js", "./plugin_glow_by_depth.js"]
}
```

### `ctx` reference

| Field | Type | Description |
|---|---|---|
| `ctx.queryRows(sql)` | `async → object[]` | query returning plain JS objects |
| `ctx.query(sql)` | `async → Arrow Table` | raw Arrow result (for binary/blob columns) |
| `ctx.querySample(cols, n)` | `async → object[]` | sampled scalar shorthand |
| `ctx.schema` | `object` | `{ metricCols, groupCols, dimensionInfo, allCols, blobCols }` |
| `ctx.state` | `object` | `{ palette, groupCol, filter, dimensions }` |
| `ctx.colorMap` | `object` | `{ groupValue: hexColor }` |
| `ctx.where` | `string` | SQL `WHERE` clause for the active filter (or `''`) — merge with `AND` if your query has its own `WHERE` |
| `ctx.groups` | `string[]` | distinct values of the active group column |
| `ctx.filteredCount` / `ctx.totalRows` | `number` | row counts |

See the [viewer README](../../viewer/README.md) for the full guide and the
extension-manifest format in detail.

**On its own:** notice neither widget imports anything from the Python side
or refers to `SharkCamLoader`/`GlowSpotterProcessor` by name — they just
query for the columns they need (`depth_zone`, `glow_count`) and politely
hide themselves via `requires()` when those columns aren't there. A widget is
just a query plus a chart; you can publish one that lights up for *any*
report containing the right columns, including ones produced entirely by
someone else's loader and processor.

---

## Defining the package

Any extension is a regular, installable Python package. The pieces that make
Pixel Patrol find it:

1. **`pyproject.toml` entry points** — three optional groups, each pointing at
   a function in your `plugin_registry` module:

   ```toml
   [project.entry-points."pixel_patrol.loader_plugins"]
   my_extension_loaders = "my_package.plugin_registry:register_loader_plugins"

   [project.entry-points."pixel_patrol.processor_plugins"]
   my_extension_processors = "my_package.plugin_registry:register_processor_plugins"

   [project.entry-points."pixel_patrol.viewer_extensions"]
   my_extension_viewer = "my_package.plugin_registry:get_viewer_extension_dir"
   ```

2. **`plugin_registry.py`** — `register_loader_plugins` / `register_processor_plugins`
   each return a *list of classes* (not instances); `get_viewer_extension_dir`
   returns the `Path` to the folder containing `extension.json`:

   ```python
   # my_package/plugin_registry.py
   from pathlib import Path
   from my_package.my_loader import MyLoader
   from my_package.my_processor import MyProcessor

   def register_loader_plugins():
       return [MyLoader]

   def register_processor_plugins():
       return [MyProcessor]

   def get_viewer_extension_dir():
       return Path(__file__).parent / "viewer"
   ```

**You only need to declare the entry-point groups your extension actually
uses** — this is the crux of "pick and choose." Shipping only a processor?
Drop the `loader_plugins` and `viewer_extensions` groups (and the
`my_loader.py`/`viewer/` files) entirely; `register_processor_plugins` is the
only function `plugin_registry.py` needs. A viewer-only extension can drop
both Python groups and ship `plugin_registry.py` with nothing but
`get_viewer_extension_dir`. Nothing about the discovery mechanism assumes
you've implemented all three.

Once the package is installed in the same environment as `pixel_patrol_base`,
everything is discovered automatically at runtime — no explicit registration
or path needed when calling `create_project(..., loader="shark-cam")` or
`serve_viewer(report_path)`.

---

## Running locally

Install and run:

```sh
uv run python create_and_show_report.py
```

This (re)generates the tiny dataset if `data/` is missing, processes it with
the custom loader and processor, saves `out/report.parquet`, and serves the
viewer at `http://127.0.0.1:8052`. The viewer extension is loaded
automatically because `serve_viewer` discovers the `pixel_patrol.viewer_extensions`
entry-point declared in `pyproject.toml` — no explicit path needed.

Open the viewer and you'll see a short blurb sitting right below the project
title — that's `process_files(..., description=...)` in
`create_and_show_report.py`: a free-form, project-level caption that's stored
in the report's own metadata and rendered in the header, independent of
anything a loader extracts from individual files (compare `depth_zone` above,
which *is* per-file, loader-extracted metadata).

To regenerate the dataset on its own (e.g. after tweaking `make_dataset.py`):

```sh
uv run python make_dataset.py
```

---

## Sharing the report

### Install the pip package

Because the JS viewer plugins are **bundled inside the pip package**
(`src/pixel_patrol_hai_watch/viewer/`), the recipient only needs to install
the package:

```sh
pip install pixel-patrol-hai-watch   # or uv add / pip install -e .
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

## Writing your own extension

Ready to build? Copy this folder, then work through it one piece at a time:

1. Decide which piece(s) you actually need — see the table at the top, and
   remember you can stop after just one.
2. Update `pyproject.toml`'s `[project]` metadata and keep only the
   entry-point groups you need.
3. Replace `my_loader.py` / `my_processor.py` / `viewer/*.js` with your own
   — or delete the files (and the corresponding registry function) for the
   pieces you're skipping.
4. Run `create_and_show_report.py` (or your own pipeline script) as you go.
   The protocols above tell you exactly what's still missing, and nothing
   stops you from running an unfinished extension while you build it out.

See the [viewer README](../../viewer/README.md) for the full plugin writing
guide, `ctx` reference, and extension format documentation, and
[`docs/extensions.md`](../../docs/extensions.md) for the broader extension
overview.
