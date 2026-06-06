# Extensions

Pixel Patrol is designed to be extended. You can add custom loaders, processors, and viewer widgets as standalone Python packages - no fork required.

The [`examples/minimal-extension/`](https://github.com/ida-mdc/pixel-patrol/tree/main/examples/minimal-extension) directory in the repository is a complete, working template. It implements:

- A custom **loader** (reads Markdown diary files)
- A custom **processor** (mood sentiment score)
- Two JavaScript **viewer plugins** (word frequency chart, mood trend chart)

Use it as a starting point: update the `pyproject.toml` metadata and replace the example identifiers with your own.

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
