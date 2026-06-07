# Extensions

Pixel Patrol is designed to be extended. You can add custom loaders, processors, and viewer widgets as standalone Python packages - no fork required.

The [`examples/minimal-extension/`](https://github.com/ida-mdc/pixel-patrol/tree/main/examples/minimal-extension) directory in the repository is a complete, working template — the playful "Pixel Sky Watch", which reads `.parquet` tables as if they were tiny snapshots of the night sky. It implements:

- A custom **loader** (reads `.parquet` tables as pixel grids, with fake image metadata like time of day and cloud cover)
- A custom **processor** (counts the "stars" in each patch)
- Two JavaScript **viewer plugins** (one for the fake image metadata, one for the star-count metric derived from the pixel data)

Use it as a starting point: update the `pyproject.toml` metadata and replace the example identifiers with your own.

Loader, processor, and viewer-widget contracts are defined as [`typing.Protocol`](https://docs.python.org/3/library/typing.html#typing.Protocol)s in `pixel_patrol_base.core.contracts`, not base classes — your classes just need to match the expected shape (the right `NAME`, methods, attributes, ...), with no import or inheritance from `pixel_patrol_base` required. That's what keeps extensions standalone, decoupled packages.

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

Each register function returns a list of classes (for loaders/processors) or a `Path` (for the viewer extension directory):

```python
# my_package/plugin_registry.py
from my_package.loader import MyLoader
from my_package.processor import MyProcessor
from pathlib import Path

def register_loader_plugins():
    return [MyLoader]

def register_processor_plugins():
    return [MyProcessor]

def get_viewer_extension_dir():
    return Path(__file__).parent / "viewer"
```

---

## Loader

A loader reads image files and returns array data and metadata. Implement the `PixelPatrolLoader` contract from `pixel_patrol_base.core.contracts`.

See [`examples/minimal-extension/`](https://github.com/ida-mdc/pixel-patrol/tree/main/examples/minimal-extension) for a full working example.

---

## Processor

A processor receives image data from the loader and computes derived metrics - statistics, quality scores, etc. - added as columns to the report.

---

## Viewer plugin

Viewer plugins are JavaScript modules declared in an `extension.json` manifest. They receive the DuckDB-backed dataset and render custom widgets in the viewer sidebar.

```json
{
  "plugins": ["my_widget.js"]
}
```

The entry point in `pyproject.toml` points to the directory containing `extension.json`.

---

!!! tip
    See [`examples/minimal-extension/README.md`](https://github.com/ida-mdc/pixel-patrol/blob/main/examples/minimal-extension/README.md) in the repository for step-by-step instructions and the full plugin API.
