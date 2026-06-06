# Python API

All public functions are in `pixel_patrol_base.api`.

```python
from pixel_patrol_base import api
```

---

## `create_project`

```python
api.create_project(name, base_dir, loader=None, output_path=None) -> Project
```

Creates a new project.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Project name, embedded in the report. Required - cannot be empty or whitespace-only. |
| `base_dir` | `str \| Path` | Root directory containing your dataset. |
| `loader` | `str \| None` | Loader plugin ID, e.g. `"bioio"`. `None` = basic file info only. |
| `output_path` | `str \| Path \| None` | Where to save the `.parquet` file. Defaults to `<base_dir>/<name>.parquet`. |

```python
project = api.create_project(
    "my-experiment",
    base_dir="data/",
    loader="bioio",
    output_path="reports/my-experiment.parquet",
)
```

---

## `add_paths`

```python
api.add_paths(project, paths) -> Project
```

Adds subdirectory paths to the project. If specified, only files within those paths are processed and they become the default grouping when the report opens. Paths are relative to `base_dir` or absolute.

If not called, all supported files under `base_dir` are processed together.

```python
api.add_paths(project, ["control", "treated"])
api.add_paths(project, "/abs/path/to/condition")  # absolute path also works
```

---

## `process_files`

```python
api.process_files(project, **kwargs) -> Project
```

Processes all files in the project paths and writes the `.parquet` report. Most parameters correspond directly to the [CLI options](cli.md#pixel-patrol-process). API-specific notes:

- `slice_size` - dict mapping dimension name to block size, e.g. `{"Z": 1, "Y": 512}`. See [slice-size](processing.md#slice-size).
- `processors_included` / `processors_excluded` - sets of processor IDs, e.g. `{"raster-basic", "thumbnail"}`. See [Available processors](processing.md#available-processors).
- `selected_file_extensions` - set of extensions, e.g. `{"tif", "nd2"}`, or `"all"`.
- `progress_callback` - `Callable[[int, int], None]` called with `(done, total)` after each completed record. `total` is `-1` until the full count is known.

```python
api.process_files(
    project,
    selected_file_extensions={"tif", "nd2"},
    max_workers=8,
    mb_per_task=256,
    description="Batch 3 - fluorescence dataset",
)
```

### External Dask cluster

```python
from dask.distributed import Client

with Client("tcp://host:8786"):
    api.process_files(project)
```

---

## `view`

```python
api.view(source, port=8052, open_browser=True, **kwargs) -> None
```

Opens a report in the interactive viewer, backed by a local DuckDB server. `source` can be a processed `Project` object or a path to an existing `.parquet` file. Parameters correspond to the [CLI options](cli.md#pixel-patrol-view). API-specific notes:

- `filter_by` - dict of the form `{"col": {"op": "eq"|"in"|"gt"|..., "value": "val"}}`.
- `dimensions` - dict of dimension letter to slice index string, e.g. `{"z": "0", "t": "0"}`.
- `widgets_excluded` - set of widget IDs, e.g. `{"histogram", "mosaic"}`. See [Available widgets](viewer.md#available-widgets).

```python
api.view(
    "report.parquet",
    group_col="path",
    filter_by={"file_extension": {"op": "in", "value": "tif,nd2"}},
    dimensions={"z": "0"},
    widgets_excluded={"histogram"},
)
```

---

## `load`

```python
api.load(src) -> tuple[DataFrame, ProjectMetadata]
```

Loads a saved `.parquet` report. Returns a Polars DataFrame and a metadata object.

```python
records_df, metadata = api.load("report.parquet")
print(f"{metadata.project_name}: {len(records_df)} records")
print(records_df.head())
```

---

## `build_viewer`

```python
api.build_viewer(output) -> Path
```

Builds a static viewer for sharing or hosting.

- If `output` ends in `.html` / `.htm`: writes a single self-contained HTML file.
- Otherwise: writes a GitHub Pages-style site folder.

```python
api.build_viewer("viewer.html")     # single file
api.build_viewer("gh-pages-out/")  # site folder
```

!!! warning
    The static viewer runs entirely in the browser and may not be able to load very large parquet files (e.g. 5 GB+). For large reports use `api.view()` instead, which is backed by a local Python server with native DuckDB.
