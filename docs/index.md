# Quickstart

## Install

We recommend [uv](https://docs.astral.sh/uv/) for fast, clean installs:

```bash
uv venv --python 3.12 pixel-patrol-env
source pixel-patrol-env/bin/activate   # Windows: pixel-patrol-env\Scripts\Activate.ps1
uv pip install pixel-patrol
```

Or with pip:

```bash
pip install pixel-patrol
```

!!! warning
    Installing without a virtual environment is not recommended - it can conflict with other packages. If in doubt, use the uv method above.

Requires Python 3.11+. See [Installation](installation.md) for more options.

---

## The two-step workflow

```bash
pixel-patrol process path/to/images/ -o report.parquet
pixel-patrol view report.parquet
```

The first command reads every supported file in `path/to/images/` - including subdirectories - in parallel and writes a `report.parquet` file. The second opens the interactive viewer in your browser.

---

## Step 1: Process your dataset

For many image formats (TIFF, Zarr, CZI, ND2, PNG, ...) you can choose the `bioio` loader:

```bash
pixel-patrol process path/to/images/ -o report.parquet --loader bioio
```

If you have subfolders representing experimental conditions, or you only want to process images in specific subdirectories, use `-p`:

```bash
pixel-patrol process path/to/images/ -o report.parquet --loader bioio \
  -p condition_a -p condition_b
```

Pixel Patrol will process each path separately and label them in the report, letting you compare conditions in the viewer. You can also change the groupings interactively in the report.

To restrict to specific file types:

```bash
pixel-patrol process path/to/images/ -o report.parquet --loader bioio -e tif -e nd2
```

## Step 2: Open the viewer

```bash
pixel-patrol view report.parquet
```

This starts a local server and opens the viewer in your browser. The viewer shows you plots for exploring your data and lets you filter, group, and compare conditions interactively.

### Sharing a report

The easiest way to share a report is to send the `.parquet` file and open it in the [Pixel Patrol viewer](https://ida-mdc.github.io/pixel-patrol/viewer/) - no installation needed on the recipient's side.

!!! warning
    The browser-based viewer may not be able to load very large parquet files (e.g. 5 GB+). For large reports use `pixel-patrol view` instead, which is backed by a local Python server with native DuckDB.

---

## Using the processing dashboard

If you prefer a visual interface, `pixel-patrol launch` opens a web UI:

```bash
pixel-patrol launch
```

This opens a browser tab where you can set your project, process your data, and finally view the report.

---

## Python API

The same workflow is available as a Python API:

```python
from pixel_patrol_base import api

project = api.create_project(
    "my-project",
    base_dir="path/to/images/",
    loader="bioio",
    output_path="report.parquet",
)

# Optional: define conditions as subdirectory paths
api.add_paths(project, ["condition_a", "condition_b"])

api.process_files(project)
api.view(project)
```

### Loading a saved report

```python
records_df, metadata = api.load("report.parquet")
print(f"{metadata.project_name}: {len(records_df)} records")
```

### Viewing with filters

```python
api.view(
    "report.parquet",
    group_col="path",
    filter_by={"file_extension": {"op": "in", "value": "tif,nd2"}},
    dimensions={"z": "0", "t": "0"},
)
```
