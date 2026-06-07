# Report Viewer

The Pixel Patrol viewer is a browser-based interactive dashboard for exploring `.parquet` reports. It is built with [DuckDB](https://duckdb.org/) and [Plotly](https://plotly.com/javascript/) and runs in two modes:

- **Static** - the viewer runs entirely using DuckDB WASM with no server. Used when opening the [hosted viewer](https://ida-mdc.github.io/pixel-patrol/viewer/) or a static HTML file built with `pixel-patrol build-viewer-html`.
- **Python-served** - `pixel-patrol view` starts a local HTTP server backed by native DuckDB. SQL queries run server-side, making it significantly faster for large files.

---

## Opening a report

**From the command line:**

```bash
pixel-patrol view report.parquet
```

Starts a local HTTP server backed by native DuckDB and opens the viewer in your browser. Recommended for large files - SQL queries run server-side rather than in the browser.

**From the hosted viewer:**

Open [ida-mdc.github.io/pixel-patrol/viewer](https://ida-mdc.github.io/pixel-patrol/viewer/) and drag and drop your `.parquet` file, or use the file picker.

**From a built static viewer:**

Build a self-contained viewer file and open it alongside your parquet:

```bash
pixel-patrol build-viewer-html -o viewer.html
```

Open `viewer.html` in any browser and load your parquet from there. To share with someone, send them both files.

**Hosted on a static server:**

Deploy a viewer site folder and load a remote parquet via URL parameter:

```bash
pixel-patrol build-viewer-html -o my-site/
# deploy my-site/ to any static host, then open:
# https://your-host.com/my-site/?data=https://your-host.com/report.parquet
```

**Via URL parameter (hosted viewer):**

```
https://ida-mdc.github.io/pixel-patrol/viewer/?data=https://your-server.com/report.parquet
```

!!! warning
    The static viewer may not be able to load very large parquet files (e.g. 5 GB+). Use `pixel-patrol view` for large reports.

---

## The interface

The report shows the project name and description at the top (if provided when processing). The sidebar on the left contains all controls and widgets:

- **Group by** - choose any column to split the data into groups. Each distinct value becomes a group with its own color across all plots. Defaults to the `path` column (the conditions from `-p`).
- **Filter** - restrict the data to rows matching a column/operator/value combination.
- **Dimension selectors** - for multi-dimensional data (Z, T, C, S), select which slice to display across widgets.
- **Show significance** - toggle statistical significance brackets on violin plots (Mann-Whitney U test, Bonferroni corrected).
- **Save** - export the current data as a `.parquet` or `.csv` file.
- **Widget list** - all available widgets. Click a widget to expand it. Widgets that require columns not present in the report hide themselves automatically.

---

## Available widgets

| ID | Widget | Description |
|---|---|---|
| `summary` | File Data Summary | Per-group summary of file count, total size, and file types present. Quick overview of dataset composition. |
| `file-stats` | File Statistics | File count and total size by extension, file size distribution, and modification timeline. Properties with no variance across files are shown as a summary table instead of a chart. |
| `sunburst` | File Structure Sunburst | Interactive sunburst chart of the file and folder hierarchy, sized by file count or total file size. Click to zoom in; click the center to zoom out. |
| `metadata` | Metadata | Distribution of pixel data types (`dtype`) and dimension orderings (`dim_order`) per group. Also lists properties shared by all files and available dimension ranges. Requires loader metadata. |
| `dim-size` | Dimension Size Distribution | Distributions of image dimension sizes (X, Y, Z, T, C, ...) across the dataset. X/Y scatter plot plus per-dimension strip plots. Useful for spotting size mismatches between groups. |
| `histogram` | Pixel Value Histograms | Mean pixel intensity histogram per group, computed per image and normalised to sum to 1. Supports fixed 0-255 bins or native pixel range. Reveals bit-depth issues, clipping, or exposure differences. |
| `mosaic` | Image Mosaic | Thumbnail grid, one image per file. Sortable by any metric (e.g. `mean_intensity`, `laplacian_variance`) to surface visual outliers. Border colors indicate group membership. |
| `violin-basic` | Pixel Value Statistics | Violin and box plots comparing per-image pixel statistics (`mean_intensity`, `std_intensity`, `min_intensity`, `max_intensity`) across groups. Each point is one image. |
| `violin-quality` | Image Quality Metrics | Violin and box plots comparing image quality metrics across groups. Requires `pixel-patrol-image`. Metrics: **Michelson contrast** (global contrast ratio; higher = greater dynamic range), **MSCN variance** (Mean Subtracted Contrast Normalized variance; sensitive to noise and blur), **Texture heterogeneity** (coefficient of variation of local standard deviations; captures spatial non-uniformity of texture), **Laplacian variance** (variance of discrete Laplacian; higher = sharper image; scale-dependent), **Blocking index** (strength of blocky compression artifacts), **Ringing index** (edge oscillation artifacts from compression). |
| `stats-across-dims-basic` | Basic Statistics Across Dimensions | How pixel statistics (mean, std, min, max) change across Z, T, C, or S slices. Useful for detecting drift or unexpected variation within a dimension. |
| `stats-across-dims-quality` | Quality Metrics Across Dimensions | How image quality metrics change across dimension slices. Useful for detecting focus drift over time (T), channel-specific artifacts (C), or depth-dependent quality changes (Z). Requires `pixel-patrol-image`. |

---

## How it works

In **static mode**, the viewer loads the parquet file into **DuckDB WASM** - a full SQL engine running in a browser Web Worker. All queries run directly against the parquet data in the browser; no data is sent to any server.

In **Python-served mode** (`pixel-patrol view`), the viewer connects to a local Python HTTP server that runs **native DuckDB** server-side. This is significantly faster for large files since native DuckDB can handle queries that would exhaust browser memory.

Widgets subscribe to the current filter, group, and dimension state and re-query DuckDB whenever any of those change. The DuckDB table is always named `pp_data`.

---

## Extensibility

The viewer is designed to be extended. Widgets are JavaScript ES modules loaded from `extension.json` manifests at runtime - no viewer rebuild required to add a custom widget.

See [Extensions](extensions.md) for how to write and load your own plugins.
