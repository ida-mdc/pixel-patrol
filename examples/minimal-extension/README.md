# Minimal Extension Example - HAI Watch

<img src="assets/shark.png" alt="A softly glowing cartoon shark" width="260" align="right">

A guide to building your own Pixel Patrol extension - shown here as a
complete, self-contained example. It walks through the **entire** extension
surface: a custom **loader**, a custom **processor**, and two custom
**viewer widgets** (one that visualises loader metadata, one that visualises
a processor's output).

**You almost certainly don't need all four.** A loader, a processor, and a
viewer widget are independent - ship just the one you need and skip the rest.
"[Defining the package](#defining-the-package)" shows how each is declared
on its own.

```
minimal-extension/
├── src/
│   └── pixel_patrol_hai_watch/
│       ├── my_loader.py          custom loader
│       ├── my_processor.py       custom processor
│       ├── plugin_registry.py    registers loader, processor & viewer extension
│       └── viewer/               custom viewer widgets
└── pyproject.toml
```

(There are a few more files in the folder - a script that generates a tiny
toy dataset and one that runs the pipeline end to end - but those are just
helpers, not part of the extension itself.)

---

## The loader

**If Pixel Patrol can't read your file format - or doesn't read it (and its
metadata) the way you want - write a loader extension.**

`src/pixel_patrol_hai_watch/my_loader.py` implements `SharkCamLoader`
(name: `shark-cam`).

A loader is any class that satisfies the `PixelPatrolLoader` protocol from
`pixel_patrol_base.core.contracts` - a [`typing.Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol),
not a base class, so `SharkCamLoader` needs no import or inheritance from
`pixel_patrol_base`; matching the shape below is enough. It needs:

| Member | Type | Required? | Purpose |
|---|---|---|---|
| `NAME` | `str` | yes | unique identifier passed to `create_project(..., loader=...)` |
| `SUPPORTED_EXTENSIONS` | `set[str]` | yes | file extensions this loader can read (lower-case, no dot) |
| `OUTPUT_SCHEMA` | `dict[str, type]` | yes | extra metadata columns this loader adds to the report, with their types |
| `read_header(path)` | `(Path) -> FileInfo` | yes | cheap shape/dtype/dim-order probe, **no pixel data loaded** |
| `load(path)` | `(Path) -> Record` | yes | loads one image and returns a `Record` |
| `load_range(path, start, stop)` | `(Path, int, int) -> Iterator[(str, Record)]` | yes | yields sub-images for container formats; raise `NotImplementedError` otherwise |
| `FOLDER_EXTENSIONS` | `set[str]` | no - defaults to empty | "extensions" that mark a *folder* as a single loadable unit (e.g. OME-Zarr stores) |
| `CONTAINER_EXTENSIONS` | `set[str]` | no - defaults to empty | extensions that may contain more than one image (multi-series OME-TIFF, LMDB, ...) |
| `OUTPUT_SCHEMA_PATTERNS` | `list[tuple[str, type]]` | no - defaults to empty | regex/type pairs for dynamically-named metadata columns (e.g. `pixel_size_X`) |
| `is_folder_supported(path)` | `(Path) -> bool` | no - only if `FOLDER_EXTENSIONS` is non-empty | whether a *folder* (not a file) should be treated as one image |

In our toy example here, `SharkCamLoader` reads a table with `pyarrow.parquet`,
stacks its columns into a 2-D `uint8` array, reads out the metadata field our
toy dataset carries (`depth_zone`), and wraps it all with
`record_from(..., kind="intensity")` - the `kind` is what lets the *built-in*
processors (basic metrics, histogram, thumbnail) pick the patches up
automatically too, right alongside our custom one.

---

## The processor

**If you want to compute a metric on images - any images, regardless of who
loaded them - write a processor extension.**

`src/pixel_patrol_hai_watch/my_processor.py` implements `GlowSpotterProcessor`
(name: `glow-spotter`).

A processor is any class that satisfies the `PixelPatrolProcessor` protocol -
every member below is required:

| Member | Type | Purpose |
|---|---|---|
| `NAME` | `str` | unique identifier (shown in pipeline logs) |
| `CHUNK_KIND` | `ChunkKind` | `LEAF` (user-configured tiles/slices) or `MEMORY` (whole record, memory-safe chunking) |
| `INPUT` | `RecordSpec` | which records this processor runs on (`kinds`, `axes`, `capabilities`, ...) |
| `OUTPUT` | `"features" \| "record"` | whether `run_chunk` returns columns to merge, or a brand-new `Record` |
| `OUTPUT_SCHEMA` | `dict[str, type]` | the columns this processor adds, with their types |
| `run_chunk(record)` | `(Record) -> dict` | does the actual computation on one chunk |
| `get_aggregation(name)` | `(str) -> Callable \| None` | how to combine multiple chunks' values for column `name` into the per-image value |

In our toy example here, `GlowSpotterProcessor` runs on every `intensity`
record with `X`/`Y` axes (`RecordSpec(axes={"X", "Y"}, kinds={"intensity"})`),
uses `CHUNK_KIND.LEAF`, and adds a single column, `glow_count` (`int`): the
number of pixels that stand out clearly brighter than the patch's median
(`> median + 60`) - the patch's "glow". Sunlit patches have almost none,
deep ones light up with plenty, by construction. Because glows are
independent per pixel, `get_aggregation` simply sums the counts from every
chunk into the per-image total.

Notice `GlowSpotterProcessor` never imports or refers to `SharkCamLoader` by
name - it only declares the *kind* of record it needs (an `intensity` image
with `X`/`Y` axes) and runs against anything that produces one: these
synthetic dive patches, a microscopy stack, a photo, any loader's output.

---

## The viewer widgets

**If you want to visualise report data in the browser - your own extension's
columns or anyone else's - write a viewer widget.**

`src/pixel_patrol_hai_watch/viewer/` ships **two** widgets, one for each kind
of data a loader can surface:

- **`plugin_dives_logged.js`** ("Dives Logged") visualises the *fake image
  metadata* on its own: a stacked bar chart of how many dive snapshots were
  logged in each `depth_zone`, split by dive site. `depth_zone` comes straight
  from `SharkCamLoader`; the site (`imported_path_short`, here `azores_log` /
  `kermadec_log`) is derived automatically from each file's folder by Pixel
  Patrol itself, so every report has it - no loader involvement needed.
- **`plugin_glow_by_depth.js`** ("Glow Sightings by Depth") visualises a
  *metric derived from the pixel data*: a jittered scatter of `glow_count`
  (computed by `GlowSpotterProcessor`) against `depth_zone`, colored by site.
  A scatter rather than a box/violin plot, because each category here only
  holds a handful of points - too few for a distributional summary to be
  meaningful.

Both are listed in `viewer/extension.json` and loaded automatically by the
viewer, and both declare `group: 'Pixel HAI Watch'` so they get their own
named sidebar section instead of being lumped under the generic "Other
Widgets".

A plugin is a JS module that exports a single default object:

```js
export default {
  id:    'my-widget',          // unique across all loaded plugins
  label: 'My Widget',          // shown in the sidebar widget list
  group: 'My Extension Name',  // optional - gives the widget its own sidebar section
  scope: 'image',              // optional - 'file' | 'image' | 'slice', shown as a badge
                               // describing what one datapoint in this widget represents

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
| `ctx.color.getColors(palette, n)` | `(string, number) → string[]` | `n` colors from the named palette - for ad-hoc groupings not covered by `colorMap` |
| `ctx.color.getPaletteNames()` | `() → string[]` | palette names accepted by `ctx.color.getColors` |
| `ctx.where` | `string` | SQL `WHERE` clause for the active filter (or `''`) - merge with `AND` if your query has its own `WHERE` |
| `ctx.groups` | `string[]` | distinct values of the active group column |
| `ctx.filteredCount` / `ctx.totalRows` | `number` | row counts |

See the [viewer README](../../viewer/README.md) for the full guide and the
extension-manifest format in detail.

Notice neither widget imports anything from the Python side or refers to
`SharkCamLoader`/`GlowSpotterProcessor` by name - they just query for the
columns they need (`depth_zone`, `glow_count`) and hide themselves via
`requires()` when those columns are missing. A widget is just a query plus a
chart that lights up for any report with the right columns, including ones
produced entirely by someone else's loader and processor.

---

## Defining the package

Any extension is a regular, installable Python package. The pieces that make
Pixel Patrol find it:

1. **`pyproject.toml` entry points** - three optional groups, each pointing at
   a function in your `plugin_registry` module:

   ```toml
   [project.entry-points."pixel_patrol.loader_plugins"]
   my_extension_loaders = "my_package.plugin_registry:register_loader_plugins"

   [project.entry-points."pixel_patrol.processor_plugins"]
   my_extension_processors = "my_package.plugin_registry:register_processor_plugins"

   [project.entry-points."pixel_patrol.viewer_extensions"]
   my_extension_viewer = "my_package.plugin_registry:get_viewer_extension_dir"
   ```

2. **`plugin_registry.py`** - `register_loader_plugins` / `register_processor_plugins`
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
uses** - this is the crux of "pick and choose." Shipping only a processor?
Drop the `loader_plugins` and `viewer_extensions` groups (and the
`my_loader.py`/`viewer/` files) entirely; `register_processor_plugins` is the
only function `plugin_registry.py` needs. A viewer-only extension can drop
both Python groups and ship `plugin_registry.py` with nothing but
`get_viewer_extension_dir`. Nothing about the discovery mechanism assumes
you've implemented all three.

Once the package is installed in the same environment as `pixel_patrol_base`,
everything is discovered automatically at runtime - no explicit registration
or path needed when calling `create_project(..., loader="shark-cam")` or
`serve_viewer(report_path)`.

---

## Running locally

Pixel Patrol discovers loaders, processors, and viewer extensions through
Python entry points - which only works if your package is installed in the
*same environment* as `pixel_patrol_base`. So first, make sure you're in that
environment, then install this package into it:

```sh
uv pip install -e .
```

If you want to test our minimal extension with the toy data to generate a
report, run the pipeline:

```sh
uv run python create_and_show_report.py
```

This (re)generates the tiny dataset if `data/` is missing, processes it with
the custom loader and processor, saves `out/report.parquet`, and serves the
viewer at `http://127.0.0.1:8052`. The viewer extension is loaded
automatically because `serve_viewer` discovers the `pixel_patrol.viewer_extensions`
entry-point declared in `pyproject.toml` - no explicit path needed.

---

## Writing your own extension

Ready to build? Copy this folder, then work through it one piece at a time:

1. Decide which piece(s) you actually need - loader/processor/widget (see the
   table at the top).
2. Update `pyproject.toml`'s `[project]` metadata and keep only the
   entry-point groups you need.
3. Replace `my_loader.py` / `my_processor.py` / `viewer/*.js` with your own
   - or delete the files (and the corresponding registry function) for the
   pieces you're skipping.
4. You probably want to create your own toy dataset to test your new
   extension, and run the full pipeline on it.

Check out our [docs](https://ida-mdc.github.io/pixel-patrol/docs/) and
[tutorials](https://ida-mdc.github.io/pixel-patrol/docs/tutorials/) for more
info on how to create your own Pixel Patrol extension.
