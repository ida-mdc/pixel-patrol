# Pixel Sky Watch — Minimal Extension

A complete, self-contained Pixel Patrol extension example. It covers the full
extension surface: a custom **loader**, a custom **processor**, and two custom
**viewer plugins** that visualise the resulting data.

The twist: there are no real images here. `.parquet` *tables* are read as if
they were photos of a patch of sky — each table's numeric columns become the
pixel grid of a tiny snapshot, and a couple of playful key/value pairs tucked
into the parquet schema metadata stand in for the kind of instrument metadata
real loaders extract (channel names, pixel sizes, acquisition stamps, ...).

```
minimal-extension/
├── data/                            tiny generated dataset (8 sky patches, 2 folders)
│   ├── rooftop_log/
│   └── campsite_log/
├── make_dataset.py                  (re)generates data/ — run once, or it's auto-run
├── src/
│   └── pixel_sky_watch/
│       ├── __init__.py
│       ├── my_loader.py             custom loader     — reads .parquet "sky patches"
│       ├── my_processor.py          custom processor  — counts the stars in each patch
│       ├── plugin_registry.py       registers loader, processor + viewer extension
│       └── viewer/
│           ├── extension.json                manifest listing the viewer plugins
│           ├── plugin_sky_patches_logged.js  metadata widget    — patches logged, by time & cloud cover
│           └── plugin_stars_by_time.js       image-data widget  — stars spotted, by time of day
├── create_and_show_report.py        generates data (if needed), processes it, opens the viewer
└── pyproject.toml
```

---

## The dataset

`make_dataset.py` writes eight tiny `.parquet` files into `data/<folder>/`.
Each one represents a 16x16 grayscale snapshot of a patch of sky:

- **Pixel data** — one `uint8` column per pixel column (`px_00` … `px_15`),
  one row per pixel row. Stack the columns side by side and you get the YX
  pixel grid directly — the table *is* the image.
- **Fake image metadata** — `time_of_day` (dawn/day/dusk/night) and
  `cloud_cover` (clear/cloudy) are stored as key/value pairs in the parquet
  schema's metadata (`table.schema.metadata`), exactly the slot real formats
  (OME-XML in TIFFs, EXIF in JPEGs, ...) use to carry instrument/acquisition
  info.

Each patch's brightness follows its `time_of_day` (bright at midday, dark at
night), painted with a soft vertical "sky" gradient and a sprinkle of bright
"stars" on top — generously at night, rarely during the day — so the loader
and processor below have something intuitive to read and count: of course you
spot more stars at night, just like in real life.

The two folders (`rooftop_log` / `campsite_log`) play the same role as the
year folders in a typical Pixel Patrol dataset: they become the `path` /
`imported_path_short` grouping used throughout the report.

---

## The loader

`src/pixel_sky_watch/my_loader.py` implements `SkyPatchLoader`
(name: `sky-patch`).

A loader is any class that satisfies the `PixelPatrolLoader` protocol from
`pixel_patrol_base.core.contracts` — a [`typing.Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol),
not a base class, so `SkyPatchLoader` needs no import or inheritance from
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

`SkyPatchLoader` reads each table with `pyarrow.parquet`, stacks its columns
into a 2-D `uint8` array, decodes the schema metadata into `time_of_day` /
`cloud_cover`, and wraps everything with `record_from(...)`. It declares
`kind="intensity"` and `dim_order="YX"` — which is what makes the *built-in*
processors (basic metrics, histogram, thumbnail) pick the patches up
automatically, right alongside our custom one. That's the point of the
exercise: to a Pixel Patrol pipeline, a "sky patch" parquet table behaves just
like any other 2-D image.

---

## The processor

`src/pixel_sky_watch/my_processor.py` implements `StarSpotterProcessor`
(name: `star-spotter`).

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

`StarSpotterProcessor` runs on every `intensity` record with `X`/`Y` axes
(`RecordSpec(axes={"X", "Y"}, kinds={"intensity"})`), uses `CHUNK_KIND.MEMORY`
because these patches are small enough to process whole, and adds:

| Column | Type | Description |
|---|---|---|
| `star_count` | `int` | number of pixels that stand out clearly brighter than the patch's median (`> median + 60`) — the patch's "stars". Night patches light up with many, daytime patches with almost none, by construction. |

Because each patch is processed in a single chunk, `get_aggregation` simply
returns the lone chunk's value (`rows[0][col]`) — see `RasterProcessor` in
`pixel_patrol_base` for processors that genuinely need to combine many chunks.

---

## The viewer plugins

This extension ships **two** widgets on purpose — one for each kind of data a
loader can surface:

- **`plugin_sky_patches_logged.js`** ("Sky Patches Logged") visualises the
  *fake image metadata* on its own: a stacked bar chart of how many patches
  were logged at each `time_of_day`, split by `cloud_cover` — both fields read
  straight out of the parquet schema metadata by `SkyPatchLoader`.
- **`plugin_stars_by_time.js`** ("Stars Spotted by Time of Day") visualises a
  *metric derived from the actual pixel data*: a jittered scatter of
  `star_count` (computed by `StarSpotterProcessor`) against `time_of_day`,
  colored by folder just like the built-in widgets. A scatter is used rather
  than a box/violin plot because each category only holds a handful of points
  — exactly the kind of small sample where distributional summaries would
  mislead rather than inform.

Both are listed in `viewer/extension.json` and loaded automatically by the
viewer, and both declare `group: 'Pixel Sky Watch'` so they get their own
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
  "name": "Pixel Sky Watch Extension",
  "plugins": ["./plugin_sky_patches_logged.js", "./plugin_stars_by_time.js"]
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

You only need to declare the entry-point groups your extension actually uses —
a viewer-only extension, say, can omit the loader/processor groups entirely.

Once the package is installed in the same environment as `pixel_patrol_base`,
everything is discovered automatically at runtime — no explicit registration
or path needed when calling `create_project(..., loader="sky-patch")` or
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

To regenerate the dataset on its own (e.g. after tweaking `make_dataset.py`):

```sh
uv run python make_dataset.py
```

---

## Sharing the report

### Install the pip package

Because the JS viewer plugins are **bundled inside the pip package**
(`src/pixel_sky_watch/viewer/`), the recipient only needs to install the
package:

```sh
pip install pixel-sky-watch   # or uv add / pip install -e .
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

See the [viewer README](../../viewer/README.md) for the full plugin writing
guide, `ctx` reference, and extension format documentation, and
[`docs/extensions.md`](../../docs/extensions.md) for the broader extension
overview.
