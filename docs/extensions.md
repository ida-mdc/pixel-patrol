# Extensions

Pixel Patrol is designed to be extended. You can add custom loaders, processors, and viewer widgets as standalone Python packages - no fork required.

The [`examples/minimal-extension/`](https://github.com/ida-mdc/pixel-patrol/tree/main/examples/minimal-extension) directory in the repository is a complete, working template - "Pixel HAI Watch", which reads `.parquet` tables as if they were tiny snapshots from a deep-sea shark camera. It implements:

- A custom **loader** (reads `.parquet` tables as pixel grids, with fake image metadata - which ocean depth zone the snapshot was taken in)
- A custom **processor** (counts the bioluminescent "glows" in each patch)
- Two JavaScript **viewer widgets** (one for the loader's metadata, one for the processor's metric)

A loader, a processor, and a viewer widget are independent pieces - ship just the one you need and skip the rest. Use the example as a starting point: copy it, update the `pyproject.toml` metadata, and replace the example identifiers with your own.

Loader, processor, and viewer-widget contracts are defined as [`typing.Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)s in `pixel_patrol_base.core.contracts`, not base classes - your classes just need to match the expected shape (the right `NAME`, methods, attributes, ...), with no import or inheritance from `pixel_patrol_base` required. That's what keeps extensions standalone, decoupled packages.

---

## How extensions are discovered

Pixel Patrol uses Python entry points. When your package is installed in the same environment, its loaders, processors, and viewer plugins are discovered automatically at runtime.

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

If Pixel Patrol can't read your file format - or doesn't read it (and its metadata) the way you want - write a loader extension. A loader turns a file into a `Record` (pixel data plus metadata) by implementing the `PixelPatrolLoader` protocol: `NAME`, `SUPPORTED_EXTENSIONS`, `OUTPUT_SCHEMA`, `read_header`, `load`, and `load_range` are required; `FOLDER_EXTENSIONS`, `CONTAINER_EXTENSIONS`, `OUTPUT_SCHEMA_PATTERNS`, and `is_folder_supported` are optional and default to "none of that".

See [`examples/minimal-extension/`](https://github.com/ida-mdc/pixel-patrol/tree/main/examples/minimal-extension) for a full working example, including the complete protocol table.

---

## Processor

If you want to compute a metric on images - any images, regardless of who loaded them - write a processor extension. A processor receives loaded records and returns derived values that get merged into the report as new columns, by implementing the `PixelPatrolProcessor` protocol: `NAME`, `CHUNK_KIND`, `INPUT`, `OUTPUT`, `OUTPUT_SCHEMA`, `run_chunk`, and `get_aggregation` - every member is required.

See [`examples/minimal-extension/`](https://github.com/ida-mdc/pixel-patrol/tree/main/examples/minimal-extension) for a full working example, including the complete protocol table.

---

## Viewer plugin

If you want to visualize report data in the browser - your own extension's columns or anyone else's - write a viewer widget. A viewer plugin is a small JavaScript module that renders a custom widget in the report viewer's sidebar, with full access to the report's data through an in-browser DuckDB instance (the table is always called `pp_data`). Plugins are declared in an `extension.json` manifest:

```json
{
  "plugins": ["my_widget.js"]
}
```

The `pixel_patrol.viewer_extensions` entry point in `pyproject.toml` points to the directory containing `extension.json`.

---

!!! tip
    See [`examples/minimal-extension/README.md`](https://github.com/ida-mdc/pixel-patrol/blob/main/examples/minimal-extension/README.md) in the repository for step-by-step instructions, the full protocol tables, and the complete viewer plugin / `ctx` API.
