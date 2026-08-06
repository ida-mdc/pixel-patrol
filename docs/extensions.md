# Extensions

PixelPatrol is designed to be extended. You can add custom loaders, processors, and viewer widgets as standalone Python packages - no fork required.

[`pixel-patrol-example-extension`](https://github.com/ida-mdc/pixel-patrol-example-extension) is a complete, working template - "Pixel HAI Watch", which reads `.parquet` tables as if they were tiny snapshots from a deep-sea shark camera. It implements:

- A custom **loader** (reads `.parquet` tables as pixel grids, with fake image metadata - which ocean depth zone the snapshot was taken in)
- A custom **processor** (counts the bioluminescent "glows" in each patch)
- Two JavaScript **viewer widgets** (one for the loader's metadata, one for the processor's metric)

A loader, a processor, and a viewer widget are independent pieces - ship just the one you need and skip the rest. Use the example as a starting point: copy it, update the `pyproject.toml` metadata, and replace the example identifiers with your own.

Loader, processor, and viewer-widget contracts are defined as [`typing.Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)s in `pixel_patrol_base.core.contracts`, not base classes - your classes just need to match the expected shape (the right `NAME`, methods, attributes, ...), with no import or inheritance from `pixel_patrol_base` required. That's what keeps extensions standalone, decoupled packages.

---

## How extensions are discovered

PixelPatrol uses Python entry points. When your package is installed in the same environment, its loaders, processors, and viewer plugins are discovered automatically at runtime.

Register them in your `pyproject.toml`:

```toml
[project.entry-points."pixel_patrol.loader_plugins"]
my_extension_loaders = "my_package.plugin_registry:register_loader_plugins"

[project.entry-points."pixel_patrol.processor_plugins"]
my_extension_processors = "my_package.plugin_registry:register_processor_plugins"

[project.entry-points."pixel_patrol.viewer_extensions"]
my_extension_viewer = "my_package.plugin_registry:get_viewer_extension_dir"
```

You only need to declare the groups your extension actually uses. Each register function returns a list of classes (for loaders/processors) or a `Path` (for the viewer extension directory):

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

---

## Loader

If PixelPatrol can't read your file format - or doesn't read it (and its metadata) the way you want - write a loader extension. A loader turns a file into a `Record` (pixel data plus metadata) by implementing the `PixelPatrolLoader` protocol: `NAME`, `SUPPORTED_EXTENSIONS`, `OUTPUT_SCHEMA`, `read_header`, `load`, and `load_range` are required; `FOLDER_EXTENSIONS`, `CONTAINER_EXTENSIONS`, `OUTPUT_SCHEMA_PATTERNS`, and `is_folder_supported` are optional and default to "none of that".

`dim_order` must be set independently in both `read_header` (via `FileInfo`) and in the metadata passed to `record_from()` inside `load` — the pipeline never carries one into the other. For spatial images, `dim_order` must include at least `X` or `Y` (usually both); omitting them causes the pipeline to fall back to generic dim names and process every pixel as a separate leaf block. If your format uses different axis names, map them to the canonical labels: `H` or `height` → `Y`, `W` or `width` → `X`, rows → `Y`, columns → `X`.

If your format stores certain dims in a way that makes spatial splitting inefficient (e.g. packed colour channels that are always decoded together), set `deferred_dims` on the returned `FileInfo` to a string of those dim letters. The pipeline will then avoid splitting those dims in memory chunks until all other dims have been exhausted. This is optional — omit it when all dims can be split equally efficiently.

`dim_order` is the only metadata field that must be correct. `shape` and `dtype` are always derived from the pixel array, not from `meta`. If `dim_order`/`dim_names` doesn't match the array's `ndim`, it's silently dropped and replaced with generic `A`/`B`/`C`… labels - check the length matches.

The pipeline overwrites these after loading, so don't bother setting them: `shape`, `ndim`, `num_pixels`, `size_<axis>`, `dim_<axis>`, `dtype`, and the filesystem columns (`path`, `name`, `type`, `parent`, `depth`, `size_bytes`, `file_extension`, `modification_date`, `imported_path`, `common_base`, `child_id`).

Any other field in `meta` is passed through to the table as-is, whether or not it's declared in `OUTPUT_SCHEMA` - a loader can dump every field it read from a file's metadata without knowing what each one means. `OUTPUT_SCHEMA` is only for the schema catalog/docs (see [schema doc](schema.md)): undeclared columns still work, they just won't show up there. Exception: loaders using the shared `RASTER_IMAGE_LOADER_SCHEMA` (BioIO/tifffile/Zarr-style), where `dim_order` and `dtype` are marked required in that schema's "Required columns".

See [`pixel-patrol-example-extension`](https://github.com/ida-mdc/pixel-patrol-example-extension) for a full working example, including the complete protocol table.

---

## Processor

If you want to compute a metric on images - any images, regardless of who loaded them - write a processor extension. A processor receives loaded records and returns derived values that get merged into the table as new columns, by implementing the `PixelPatrolProcessor` protocol: `NAME`, `CHUNK_KIND`, `INPUT`, `OUTPUT`, `OUTPUT_SCHEMA`, `run_chunk`, and `get_aggregation` - every member is required.

See [`pixel-patrol-example-extension`](https://github.com/ida-mdc/pixel-patrol-example-extension) for a full working example, including the complete protocol table.

---

## Viewer plugin

If you want to visualize interactive report data in the browser - your own extension's columns or anyone else's - write a viewer widget. A viewer plugin is a small JavaScript module that renders a custom widget in the viewer's sidebar, with full access to the interactive report's data through an in-browser DuckDB instance (the table is always called `pp_data`). Plugins are declared in an `extension.json` manifest, and the `pixel_patrol.viewer_extensions` entry point in `pyproject.toml` points to the directory containing it.

**Option A — explicit list** (recommended for published extensions): list every plugin file by name. Clear and predictable.

```json
{
  "name": "My Extension",
  "plugins": ["./plugin_foo.js", "./plugin_bar.js"]
}
```

**Option B — auto-detect** (convenient during development): set `"auto_detect": true` and every file matching `plugin_*.js` in the directory is loaded automatically, sorted alphabetically. No list to maintain.

```json
{
  "name": "My Extension",
  "auto_detect": true
}
```

Each plugin's `render(container, ctx)` method receives a `ctx` object - the render context - which gives it everything it needs to query data and respond to the viewer's current state:

- `ctx.queryRows(sql)` - run a DuckDB query, returns plain JS objects
- `ctx.where` - SQL WHERE clause for the active filter (or `''`) - append with `AND` if you have your own conditions
- `ctx.schema` - available columns, grouped by type (`metricCols`, `groupCols`, `allCols`, ...)
- `ctx.state` - current viewer state: active group column, filter, selected dimensions
- `ctx.colorMap` - maps group values to hex colors from the active palette
- `ctx.groups` - distinct values of the active group column
- `ctx.totalRows` / `ctx.filteredCount` - row counts

To show your widget correctly in Overview mode (the default tile-gallery view), also define `overviewMessage(ctx)` and `overviewPlot(container, ctx)` - a short summary sentence and a small preview plot shown on the widget's tile. Both are optional.

See the [`pixel-patrol-example-extension` README](https://github.com/ida-mdc/pixel-patrol-example-extension/blob/main/README.md) for the full `ctx` reference and worked examples.

---

## Licensing

For easy redistribution, use a permissive open-source license for your extension - options like MIT, BSD-2-Clause, BSD-3-Clause, and Apache-2.0 all work well. Add a `LICENSE` file to your package and declare it in your `pyproject.toml`:

```toml
[project]
license = "MIT"
license-files = ["LICENSE"]
```

Avoid copyleft licenses (GPL, AGPL): they would require anyone who combines your extension with pixel-patrol to release their own code under the same terms, which is usually not what extension authors or users want.

The direct dependencies of `pixel-patrol-base` are all permissively licensed (MIT, BSD, Apache-2.0), so depending on it is safe. If your extension adds new dependencies of its own, check that their licenses are permissive too - `pip-licenses` (installable via pip) lists the licenses of everything in your environment.

---

!!! tip
    See the [`pixel-patrol-example-extension` README](https://github.com/ida-mdc/pixel-patrol-example-extension/blob/main/README.md) for step-by-step instructions, the full protocol tables, and the complete viewer plugin / `ctx` API.
