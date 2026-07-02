# Quickstart

## Install

We recommend [uv](https://docs.astral.sh/uv/) for fast, clean installs:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
uv pip install pixel-patrol
```

Or with pip:

!!! warning
    Installing without a virtual environment is not recommended - it can conflict with other packages. If in doubt, use the uv method above.

```bash
pip install pixel-patrol
```

Requires Python 3.11+. See [Installation](installation.md) for more options.

---

## The two-step workflow

```bash
pixel-patrol process path/to/images/ -o results.parquet
pixel-patrol view results.parquet
```

The first command reads every supported file in `path/to/images/` - including subdirectories - in parallel and writes a `results.parquet` table. The second opens the interactive report in your browser.

---

## Step 1: Process your dataset

Pixel Patrol supports several image loaders, each suited to different formats - see [Loaders](processing.md#loaders) for the full list. For many image formats (TIFF, Zarr, CZI, ND2, PNG, ...) the `bioio` loader covers everything:

```bash
pixel-patrol process path/to/images/ -o results.parquet --loader bioio
```

If you have subfolders representing experimental conditions, or you only want to process images in specific subdirectories, use `-p`:

```bash
pixel-patrol process path/to/images/ -o results.parquet --loader bioio \
  -p condition_a -p condition_b
```

Each path becomes a group in the interactive report via the `imported_path` column. You can change the grouping column interactively.

To restrict to specific file extensions:

```bash
pixel-patrol process path/to/images/ -o results.parquet --loader bioio -e tif -e nd2
```

## Step 2: Open the viewer

```bash
pixel-patrol view results.parquet
```

This starts a local server and opens the viewer in your browser. The viewer shows you plots for exploring your data and lets you filter, group, and compare conditions interactively.

### Sharing a table

The easiest way to share a table is to send the `.parquet` file and open it in the [Pixel Patrol viewer](https://pixelpatrol.app/viewer/) - no installation needed on the recipient's side.

!!! warning
    The browser-based viewer may not be able to load very large parquet files (e.g. 5 GB+). For large tables use `pixel-patrol view` instead, which is backed by a local Python server with native DuckDB.

---

## Using the processing dashboard

If you prefer a visual interface, `pixel-patrol launch` opens a web UI:

```bash
pixel-patrol launch
```

This opens a browser tab where you can set your project, process your data, and open the interactive report. An "Open Existing Table" button lets you jump straight to the viewer for a `.parquet` file from a previous run, without reprocessing.

---

## Python API

The same workflow is available as a Python API:

```python
from pixel_patrol_base import api

project = api.create_project(
    "my-project",
    base_dir="path/to/images/",
    loader="bioio",
    output_path="results.parquet",
)

# Optional: define conditions as subdirectory paths
api.add_paths(project, ["condition_a", "condition_b"])

api.process_files(project)
api.view(project)
```

### Loading a saved table

```python
records_df, metadata = api.load("results.parquet")
print(f"{metadata.project_name}: {len(records_df)} records")
```

### Viewing with filters

```python
api.view(
    "results.parquet",
    group_col="path",
    filter_by={"file_extension": {"op": "in", "value": "tif,nd2"}},
    dimensions={"z": "0", "t": "0"},
)
```
