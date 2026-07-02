# CLI Reference

```bash
pixel-patrol --help
```

---

## `pixel-patrol process`

Scans a directory tree, processes the images, and writes a `.parquet` table with embedded project metadata.

```bash
pixel-patrol process BASE_DIRECTORY -o OUTPUT.parquet [OPTIONS]
```

**Arguments**

| Argument | Description |
|---|---|
| `BASE_DIRECTORY` | Root folder containing your dataset. |

**Options**

| Option | Default | Description |
|---|---|---|
| `-o, --output PATH` | **required** | Path for the output `.parquet` file. |
| `--name TEXT` | folder name | Project name embedded in the parquet metadata. |
| `-p, --paths PATH` | | Subdirectory to process. Repeatable. If specified, only files within those paths are processed and they become the default grouping when the interactive report opens. Paths are relative to `BASE_DIRECTORY`. |
| `-l, --loader TEXT` | *(none)* | Loader plugin, e.g. `bioio`, `zarr`, `tifffile`. Without a loader only basic file info is collected. |
| `-e, --file-extensions EXT` | *(all supported)* | File extension to include, e.g. `tif`. Repeatable. |
| `--flavor TEXT` | | Label shown next to the title in the viewer. |
| `--description TEXT` | | Free-form description shown below the title in the viewer and embedded in the parquet metadata. |
| `--processors-include NAME` | | Run only these processors by ID. Repeatable. Takes precedence over `--processors-exclude`. See [Available processors](processing.md#available-processors). |
| `--processors-exclude NAME` | | Skip these processors by ID. Repeatable. See [Available processors](processing.md#available-processors). |
| `--max-workers N` | auto | Number of parallel Dask workers. Auto-detected from available CPUs and RAM. Use `1` to disable parallelism. |
| `--scheduler URL` | | Connect to an existing Dask scheduler instead of spawning a local one, e.g. `tcp://host:8786`. |
| `--mb-per-task N` | 512 | Work budget per Dask task in MB. Controls batch sizes for small files, spatial splitting for large files, and sub-image batching for container files. See [Task sizing](processing.md#task-sizing). |
| `--max-images-per-task N` | 200 | Max files per batch task or sub-images per container task. |
| `--slice-size DIM=SIZE` | | Per-dimension granularity of statistics in the output table. `Z=1` produces one set of statistics per Z slice; `Z=5` groups every 5 slices. By default X and Y are full extent and all other dims (Z, T, C, S) step by 1. See [per-dimension observations](processing.md#row-structure). Use `-1` for full extent. Repeatable, e.g. `--slice-size Z=1 --slice-size C=1`. |
| `--rows-per-part N` | 10000 | Flush intermediate results to disk every N rows. |
| `--parquet-row-group-size N` | 2048 | Rows per row group in the final parquet. Smaller values speed up thumbnail sampling in the viewer. |
| `--log-file` | off | Write a debug log file alongside the output parquet. |
| `--omit-base-dir` | off | Do not store the base directory in the parquet metadata. Useful when sharing tables without revealing local filesystem paths. |

**Examples**

```bash
# Minimal - basic file info only:
pixel-patrol process my-data/ -o results.parquet

# BioIO loader, two conditions, TIFF and ND2 only:
pixel-patrol process my-data/ -o results.parquet --loader bioio \
  -p control -p treated -e tif -e nd2

# Large dataset on a cluster:
pixel-patrol process my-data/ -o results.parquet --loader bioio \
  --scheduler tcp://host:8786 --mb-per-task 128 --log-file
```

---

## `pixel-patrol view`

Opens a `.parquet` table in the viewer as an interactive report. Starts a local server backed by DuckDB and opens the browser automatically.

```bash
pixel-patrol view PARQUET_FILE [OPTIONS]
```

**Arguments**

| Argument | Description |
|---|---|
| `PARQUET_FILE` | Path to a `.parquet` file (Pixel Patrol table) produced by `process`. |

**Options**

| Option | Default | Description |
|---|---|---|
| `--port N` | 8052 | Port for the local server. |
| `--no-browser` | off | Start the server without opening the browser. |
| `--group-by COL` | `path` | Column to group by on first load. Each distinct value becomes a group with its own color. |
| `--filter-col COL` | | Column to filter on. |
| `--filter-op OP` | | Filter operation: `eq`, `in`, `gt`, `lt`, `ge`, `le`, `contains`, `not_contains`. |
| `--filter-val VAL` | | Filter value. |
| `--dim KEY=VALUE` | | Pre-select a dimension slice shown across all widgets, e.g. `z=0`. Repeatable. |
| `--widgets-exclude NAME` | | Hide a widget by its ID. Repeatable. See [Available widgets](viewer.md#available-widgets). |
| `--significance` | off | Show pairwise statistical significance brackets on violin plots (Mann-Whitney U, Bonferroni corrected). |
| `--palette NAME` | tab10 | Color palette for group colors. |

**Examples**

```bash
pixel-patrol view results.parquet
pixel-patrol view results.parquet --group-by path --dim z=0 --dim t=0
pixel-patrol view results.parquet --filter-col file_extension --filter-op in --filter-val tif,nd2
pixel-patrol view results.parquet --widgets-exclude histogram --no-browser --port 9000
```

---

## `pixel-patrol build-viewer-html`

Packages the viewer as a self-contained static file for sharing or hosting without a running server.

```bash
pixel-patrol build-viewer-html -o OUTPUT [OPTIONS]
```

If `OUTPUT` ends in `.html` or `.htm`, writes a single self-contained HTML file with all JS, CSS, and extensions inlined. Otherwise writes a GitHub Pages-style site folder.

**Examples**

```bash
# Single file - share alongside your .parquet:
pixel-patrol build-viewer-html -o viewer.html

# Site folder - deploy to any static host:
pixel-patrol build-viewer-html -o gh-pages/

# Load a remote parquet via URL:
# Open: gh-pages/index.html?data=https://yourserver.com/results.parquet
```

!!! warning
    The static viewer runs entirely in the browser and may not be able to load very large parquet files (e.g. 5 GB+). For large tables use `pixel-patrol view` instead, which is backed by a local Python server with native DuckDB.

---

## `pixel-patrol schema`

Exports the table schema and plugin catalog as JSON.

```bash
pixel-patrol schema [OPTIONS]
```

Two formats are available:

- **`catalog`** (default) — full plugin catalog: every column with description, type, and producer; loader/processor/widget metadata; and the connection graph between them.
- **`row-schema`** — a standard [JSON Schema](https://json-schema.org/) document describing one parquet table row. Fixed columns are in `properties`; per-axis families (`size_*`, `dim_*`, `pixel_size_*`) are in `patternProperties`.

**Options**

| Option | Default | Description |
|---|---|---|
| `-o, --output PATH` | `schema.json` | Path to write the JSON output. |
| `--print` | off | Print to stdout instead of writing a file. |
| `--no-widgets` | off | Skip widget metadata (requires Node). |
| `--format [catalog\|row-schema]` | `catalog` | Output format. |

**Examples**

```bash
# Full plugin catalog to a file:
pixel-patrol schema -o catalog.json

# Standard JSON Schema for integration with validators or editors:
pixel-patrol schema --format row-schema -o row-schema.json

# Print catalog to stdout:
pixel-patrol schema --print
```

---

## `pixel-patrol launch`

Opens the web-based processing dashboard for configuring and monitoring processing interactively.

```bash
pixel-patrol launch [--port N] [--no-browser]
```

**Options**

| Option | Default | Description |
|---|---|---|
| `--port N` | 8051 | Port for the dashboard server. |
| `--no-browser` | off | Do not open the browser automatically. |
