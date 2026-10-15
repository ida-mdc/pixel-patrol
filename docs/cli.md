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

<!-- process-args-table -->

**Examples**

```bash
# Minimal - basic file info only:
pixel-patrol process my-data/ -o results.parquet

# BioIO loader, two conditions, TIFF and ND2 only:
pixel-patrol process my-data/ -o results.parquet --loader bioio \
  -p control -p treated -e tif -e nd2

# Large dataset on a cluster:
pixel-patrol process my-data/ -o results.parquet --loader bioio \
  --scheduler tcp://host:8786 --log-file

# Process and open the viewer in one step:
pixel-patrol process my-data/ -o results.parquet --loader bioio --view
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
| `PARQUET_FILE` | Path to a `.parquet` file (PixelPatrol table) produced by `process`. |

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
    The static viewer runs entirely in the browser and may not be able to load parquet files larger than a few GB. For large tables use `pixel-patrol view` instead, which is backed by a local Python server with native DuckDB.

---

## `pixel-patrol schema`

Exports the table schema and plugin catalog as JSON.

```bash
pixel-patrol schema [OPTIONS]
```

Two formats are available:

- **`catalog`** (default) - full plugin catalog: every column with description, type, and producer; loader/processor/widget metadata; and the connection graph between them.
- **`row-schema`** - a standard [JSON Schema](https://json-schema.org/) document describing one parquet table row. Fixed columns are in `properties`; per-axis families (`size_*`, `dim_*`, `pixel_size_*`) are in `patternProperties`.

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

## `pixel-patrol list`

Lists available loaders and processors from installed plugins.

```bash
pixel-patrol list [--loaders] [--processors]
```

With no flags, lists both. Use `--loaders` or `--processors` to narrow down.

**Example output**

```
Loaders:    nifti, dicom, bioio, h5, tifffile, zarr, geospatial
Processors: raster-basic, raster-histogram, thumbnail, raster-quality, nodata-statistics
```

---

## `pixel-patrol launch`

Opens the web-based **report manager**: it lists your existing reports, lets you
add new ones (processing runs in the background), import reports stored elsewhere,
and open any report in the built-in viewer.

```bash
pixel-patrol launch [--port N] [--no-browser]
```

Reports are stored in `PIXEL_PATROL_REPORTS_DIR` (default: `~/pixel-patrol-reports`).
Imported reports living outside that folder are remembered across restarts.

**Options**

| Option | Default | Description |
|---|---|---|
| `--port N` | 8051 | Port for the report manager server. |
| `--no-browser` | off | Do not open the browser automatically. |
