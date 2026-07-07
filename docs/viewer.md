# Viewer

The PixelPatrol viewer is a browser-based interactive dashboard for exploring `.parquet` tables. It is built with [DuckDB](https://duckdb.org/) and [Plotly](https://plotly.com/javascript/) and runs in two modes:

- **Static** - the viewer runs entirely using DuckDB WASM with no server. Used when opening the [hosted viewer](https://pixelpatrol.app/viewer/) or a static HTML file built with `pixel-patrol build-viewer-html`.
- **Python-served** - `pixel-patrol view` starts a local HTTP server backed by native DuckDB. SQL queries run server-side, making it significantly faster for large files.

---

## Opening a table

**From the command line:**

```bash
pixel-patrol view results.parquet
```

Starts a local HTTP server backed by native DuckDB and opens the viewer in your browser. Recommended for large files - SQL queries run server-side rather than in the browser.

**From the hosted viewer:**

Open [pixelpatrol.app/viewer](https://pixelpatrol.app/viewer/) and drag and drop your `.parquet` file, or use the file picker.

**From a built static viewer:**

Build a viewer file and open it alongside your parquet:

```bash
pixel-patrol build-viewer-html -o viewer.html
```

By default this writes a **light** file (~7 MB): it inlines the viewer's own
JS/CSS but loads the large DuckDB WASM engine from a CDN (jsdelivr), so it works
for any installed version and needs an internet connection only for DuckDB.

For a fully **self-contained** file that works with no network access, pass
`--offline` (this inlines all JS/CSS/WASM too, producing a large ~50 MB+ file):

```bash
pixel-patrol build-viewer-html -o viewer.html --offline
```

Open `viewer.html` in any browser and load your parquet from there. To share with someone, send them both files. See the [privacy policy](privacy.md) for what's included in the parquet file before sharing it.

**Hosted on a static server:**

Deploy a viewer site folder and load a remote parquet via URL parameter:

```bash
pixel-patrol build-viewer-html -o my-site/
# deploy my-site/ to any static host, then open:
# https://your-host.com/my-site/?data=https://your-host.com/results.parquet
```

**Via URL parameter (hosted viewer):**

```
https://pixelpatrol.app/viewer/?data=https://your-server.com/results.parquet
```

!!! warning
    The static viewer may not be able to load very large parquet files (e.g. 5 GB+). Use `pixel-patrol view` for large tables.

---

## The interface

### View modes

The topbar has an **Overview / Full** toggle.

- **Overview** (default) - widgets appear as a grid of tiles. Each tile shows a short summary and a preview plot. Click a tile to expand it in place; click again (or use **Collapse all**) to close it.
- **Full** - the classic layout: all widgets shown full-width, one after another.

### Sidebar

- **Group by** - choose any column to split the data into groups. Each distinct value becomes a group with its own color across all plots. Defaults to `imported_path_short` when `-p` paths were used, or `name` for small single-directory tables (≤4 files).
- **Filter** - restrict the data to rows matching a column/operator/value combination.
- **Dimension selectors** - for multi-dimensional data (Z, T, C, S), select which slice to display across widgets.
- **Show significance** - toggle statistical significance brackets on violin plots (Mann-Whitney U test, Bonferroni corrected).
- **Save** - export the current data as a `.parquet` or `.csv` file.
- **Widget list** - show or hide individual widgets.

---

## Available widgets

| ID | Widget | Description |
|---|---|---|
| `summary` | File Data Summary | Dataset-wide KPIs (file/image counts, total size, file extensions, common base path) plus a per-group breakdown table when grouped. Quick overview of dataset composition. |
| `image-table` | Image Table | Sortable, searchable table of full-image statistics - one row per image file, no per-slice or per-channel rows. |
| `file-stats` | File Statistics | File count and total size by extension, file size distribution, and modification timeline. Properties with no variance across files are shown as a summary table instead of a chart. |
| `sunburst` | File Structure Sunburst | Interactive sunburst chart of the file and folder hierarchy, sized by file count or total file size. Click to zoom in; click the center to zoom out. |
| `metadata` | Metadata | Distribution of pixel data types (`dtype`) and dimension orderings (`dim_order`) per group. Also lists properties shared by all files and available dimension ranges. Requires loader metadata. |
| `dim-size` | Dimension Size Distribution | Distributions of image dimension sizes (X, Y, Z, T, C, ...) across the dataset. X/Y scatter plot plus per-dimension strip plots. Useful for spotting size mismatches between groups. |
| `histogram` | Pixel Value Histograms | Per-group mean pixel intensity histogram (256 bins per image, area-normalized). Integer and float images are shown in separate plots if the dataset contains both. Integer plots can be toggled between actual pixel values and a dtype-normalized range (0 to 1 for uint, -1 to 1 for int) for comparison across bit depths. Reveals clipping, exposure differences, and bit-depth artifacts. |
| `mosaic` | Image Mosaic | Thumbnail grid, one image per file. Sortable by any metric (e.g. `mean_intensity`, `laplacian_variance`) to surface visual outliers. Border colors indicate group membership. |
| `violin-basic` | Pixel Value Statistics | Violin and box plots comparing pixel statistics (`mean_intensity`, `std_intensity`, `min_intensity`, `max_intensity`) across groups. By default each point is one image; use the **Slice by** toggles to switch to one point per (image × dimension slice) instead. Plots over 5,000 points show a SQL-aggregate box summary instead of a full violin. |
| `violin-quality` | Image Quality Metrics | Violin and box plots comparing image quality metrics across groups. Requires `pixel-patrol-image`. Supports the same **Slice by** toggles and box-summary fallback as Pixel Value Statistics. Metrics: **Michelson contrast** (global contrast ratio; higher = greater dynamic range), **MSCN variance** (Mean Subtracted Contrast Normalized variance; sensitive to noise and blur), **Texture heterogeneity** (coefficient of variation of local standard deviations; captures spatial non-uniformity of texture), **Laplacian variance** (variance of discrete Laplacian; higher = sharper image; scale-dependent), **Blocking index** (strength of blocky compression artifacts), **Ringing index** (edge oscillation artifacts from compression). |
| `stats-across-dims-basic` | Basic Statistics Across Dimensions | How pixel statistics (mean, std, min, max) change across Z, T, C, or S slices. Useful for detecting drift or unexpected variation within a dimension. A dashed line shows the percentage of images that still have a slice at each position, for spotting datasets where images don't all have the same number of slices. |
| `stats-across-dims-quality` | Quality Metrics Across Dimensions | How image quality metrics change across dimension slices. Useful for detecting focus drift over time (T), channel-specific artifacts (C), or depth-dependent quality changes (Z). Requires `pixel-patrol-image`. A dashed line shows the percentage of images that still have a slice at each position, for spotting datasets where images don't all have the same number of slices. |
| `custom-plot` | Custom Plot | Build your own scatter, violin, bar, count, or heatmap plot from any columns in the table, with grouping, coloring, and palette controls. Each plot has its own **Slice by** toggles and per-image/per-slice badge. Add multiple plots, and export any of them as a standalone viewer plugin file. |

---

## Widget data resolution

Each widget card shows a small badge indicating what one datapoint in that widget represents:

- 📄 **per file** - each datapoint is a file (e.g. File Structure Sunburst).
- 🖼️ **per image** - each datapoint is an image. A file containing multiple images (a stack or container) contributes one datapoint per image (e.g. Image Table, Image Mosaic, Pixel Value Statistics).
- 🧩 **per slice** - each datapoint is a slice within an image, such as a channel, Z-plane, or timepoint (e.g. Statistics Across Dimensions).

Pixel Value Statistics, Image Quality Metrics, and Custom Plot can switch between **per image** and **per slice** via their **Slice by** toggles - their badge updates accordingly. File Data Summary spans multiple resolutions (files, images, slices) and shows no badge.

This helps you reason about what's actually being aggregated or plotted, especially for multi-dimensional or multi-image files.

---

## How it works

In **static mode**, the viewer loads the parquet file into **DuckDB WASM** - a full SQL engine running in a browser Web Worker. All queries run directly against the parquet data in the browser; no data is sent to any server.

In **Python-served mode** (`pixel-patrol view`), the viewer connects to a local Python HTTP server that runs **native DuckDB** server-side. This is significantly faster for large files since native DuckDB can handle queries that would exhaust browser memory.

Widgets subscribe to the current filter, group, and dimension state and re-query DuckDB whenever any of those change. The DuckDB table is always named `pp_data`.

---

## Extensibility

The viewer is designed to be extended. Widgets are JavaScript ES modules loaded from `extension.json` manifests at runtime - no viewer rebuild required to add a custom widget.

See [Extensions](extensions.md) for how to write and load your own plugins.
