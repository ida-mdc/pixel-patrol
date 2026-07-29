# Processing

Processing is the step where PixelPatrol scans your images and produces a `.parquet` table. The [Quickstart](index.md) covers the basic command; this page goes deeper into project configuration, loaders, conditions, and performance tuning.

---

## Project configuration

A project requires a **base directory** and an **output path**. A **loader** is technically optional but essential for extracting anything beyond basic file info. Optionally you can also set **paths** (conditions), a **name**, a **description**, and a **flavor** label.

### Base directory and paths

The base directory is the root folder that contains your images. By default, all supported files within it - including those in subdirectories - are scanned and processed together.

If you specify paths with `-p`, only those directories are processed. Paths can be anywhere within the base directory - immediate subdirectories or deeper - and each one becomes an initial grouping condition in the interactive report. You can always regroup interactively in the viewer.

```
my-dataset/
├── control/
│   └── batch_1/
│       └── img_001.tif
└── treated/
    └── img_001.tif
```

```bash
pixel-patrol process my-dataset/ -o results.parquet --loader bioio \
  -p control -p treated
```

Paths are relative to the base directory, or can be absolute.

---

## Loaders

Loaders are responsible for opening image files and reading their metadata - they determine which file formats are supported. Each piece of image metadata a loader extracts becomes a column in the parquet (e.g. `dtype`, pixel size, dimension sizes, acquisition metadata, ...). If multiple loaders support the same format, the metadata they extract may differ, so it is worth choosing the one best suited to your format.

Without a loader, only file-system metadata is collected (`name`, `file_extension`, `size_bytes`, `path`, ...).

Some loaders support **container files** - single files that hold multiple images. Each sub-image in a container generates its own rows in the parquet, identified by a `child_id` column.

| Loader ID | What it reads | Package required |
|---|---|---|
| `bioio` | TIFF, CZI, ND2, LIF, PNG, JPG, and more via BioIO | `pixel-patrol-loader-bio` |
| `zarr` | Zarr datasets | `pixel-patrol-loader-bio` |
| `tifffile` | TIFF only, lightweight | `pixel-patrol-loader-bio` |
| *(none)* | Basic file info only (name, size, extension) | `pixel-patrol-base` |

!!! note
    `pixel-patrol-loader-bio` is included in the `pixel-patrol` bundle. If you installed with `pip install pixel-patrol` or `uv pip install pixel-patrol`, all loaders are already available.

Additional loaders are available in the [GitHub repository](https://github.com/ida-mdc/pixel-patrol), and you can build your own - see [Extensions](extensions.md).

### File extensions

By default, the loader processes all file extensions it supports. Restrict to specific types with `-e`:

```bash
pixel-patrol process my-data/ -o results.parquet --loader bioio -e tif -e nd2 -e czi
```

---

## Processors

Processors compute information from the loaded image data and add one or more columns to the parquet - pixel statistics, histograms, quality scores, thumbnails, and more. All installed processors run by default.

To run only specific processors:

```bash
pixel-patrol process my-data/ -o results.parquet --processors-include raster-basic --processors-include thumbnail
```

To skip specific processors:

```bash
pixel-patrol process my-data/ -o results.parquet --processors-exclude raster-quality
```

!!! note
    `--processors-include` takes precedence; if set, `--processors-exclude` is ignored.

### Available processors

| Processor ID | Package | Columns added to parquet |
|---|---|---|
| `raster-basic` | `pixel-patrol-base` | `min_intensity`, `max_intensity`, `mean_intensity`, `std_intensity`, `finite_pixel_count` |
| `raster-histogram` | `pixel-patrol-base` | `histogram_min`, `histogram_max`, `histogram_nan_count`, `histogram_counts` |
| `thumbnail` | `pixel-patrol-base` | `thumbnail`, `thumbnail_norm_min`, `thumbnail_norm_max`, `thumbnail_dtype` |
| `raster-quality` | `pixel-patrol-base` | `laplacian_variance`, `sobel_gradient_sharpness`, `estimated_noise_std`, `local_range_contrast_variability`, `local_texture_uniformity`, `saturated_pixel_fraction`, `underexposed_pixel_fraction`, `compression_blocking_score` |

Additional processors are available in the [GitHub repository](https://github.com/ida-mdc/pixel-patrol). You can also build your own - see [Extensions](extensions.md).

### How processors see the image

Processors fall into two categories that determine what data they receive:

- **Leaf processors** (`raster-basic`, `raster-histogram`, `raster-quality`) run on individual **leaf blocks** - the smallest spatial unit, by default one 2D plane at a time (one Z slice, one channel, etc.). Their results are aggregated up into the full-image summary (`obs_level=0`). The leaf block shape is controlled by `--slice-size`.

- **Memory processors** (`thumbnail`) run once per **memory chunk**. For images that fit within the `mb_per_task` budget, the memory chunk is the full image. For larger images that are split into spatial sub-regions, each sub-region is one memory chunk and the results are assembled before writing. Memory processors always produce a result at `obs_level=0`.

### `--slice-size`

Controls the per-dimension granularity of statistics in the output table. `Z=1` produces one set of statistics per Z slice; `Z=5` groups every 5 slices into one. This is independent of memory budget or file size - it affects what is saved to the table, not how the data is loaded.

By default X and Y are full-extent (one complete 2D plane) and all other dimensions (Z, T, C, S) step by 1. Override to produce coarser statistics:

```bash
pixel-patrol process my-data/ -o results.parquet --slice-size Z=5 --slice-size Y=512
```

Use `-1` to keep a dimension at full extent.

---

## The parquet file

The output `.parquet` file is the PixelPatrol table. Its columns come from three sources: the file system (always), the loader (if specified), and each processor.

### Row structure

A file does not always map to a single row. The parquet uses an `obs_level` column to represent a hierarchy of observations:

- **`obs_level = 0`** - full-image aggregate: one row per image (or per sub-image in a container), with statistics aggregated over all dimensions.
- **`obs_level = 1`** - per single dimension: one row per Z slice (aggregated over all other dims), one row per C slice (aggregated over all other dims), etc. The `dim_z`, `dim_t`, `dim_c`, `dim_s` columns identify which dimension value the row belongs to; the others are `null`.
- **`obs_level = 2` (and higher)** - combinations of dimensions: one row per unique combination of active dimension values (e.g. each individual Z×C pair).

A simple 2D image produces a single row at obs_level 0. Container files produce rows for each sub-image, linked by a `child_id` column.

**Example: a TIFF with size_Z=2, size_C=2** would produce the following rows (among many other columns):

| `path` | `obs_level` | `dim_z` | `dim_c` | `size_Z` | `size_C` | `mean_intensity` |
|---|---|---|---|---|---|---|
| `condition_a/img01.tif` | `0` | `null` | `null` | `2` | `2` | `285.1` |
| `condition_a/img01.tif` | `1` | `0` | `null` | `1` | `2` | `271.3` |
| `condition_a/img01.tif` | `1` | `1` | `null` | `1` | `2` | `289.4` |
| `condition_a/img01.tif` | `1` | `null` | `0` | `2` | `1` | `280.2` |
| `condition_a/img01.tif` | `1` | `null` | `1` | `2` | `1` | `290.0` |
| `condition_a/img01.tif` | `2` | `0` | `0` | `1` | `1` | `265.1` |
| `condition_a/img01.tif` | `2` | `0` | `1` | `1` | `1` | `277.5` |
| `condition_a/img01.tif` | `2` | `1` | `0` | `1` | `1` | `288.2` |
| `condition_a/img01.tif` | `2` | `1` | `1` | `1` | `1` | `291.4` |

### Columns

Key columns that are always present:

| Column | Description |
|---|---|
| `path` | Path to the source file, relative to the base directory |
| `obs_level` | Observation level (0 = full-image aggregate, 1 = per single dim, 2+ = dim combinations) |
| `name` | Filename |
| `file_extension` | File extension |
| `size_bytes` | File size in bytes |
| `imported_path` | The `-p` path (or `.`) under which this file was found, relative to the base directory |
| `imported_path_short` | Short label for the import base, used as the default grouping column when multiple `-p` paths are given |
| `dim_z`, `dim_t`, `dim_c`, `dim_s` | Slice index for this row (null for level-0 aggregate rows) |
| `child_id` | Sub-image identifier for container files |

**Image metadata columns** (when a loader is used) are extracted by the loader and include dimension sizes (`size_X`, `size_Y`, `size_Z`, `size_T`, `size_C`), `dtype`, pixel size, `dim_order`, and any embedded acquisition metadata. Some of these are shown in the viewer header.

**`size_*` columns at higher obs levels:** For `obs_level=0`, all `size_*` values reflect the full image (e.g. `size_Z=2`). For per-dimension rows, the `size_*` shows its slice size instead.

**Processor columns** are listed in the [Processors](#processors) section above.

### Paths in the table

All path columns (`path`, `parent`, `imported_path`) are stored **relative to the base directory**, so the table does not contain absolute filesystem paths. A file at `/data/images/condition_a/img01.tif` processed with base directory `/data/images/` is stored as `condition_a/img01.tif`.

### Project metadata

Project metadata (name, description, version, processing stats, base directory, paths) is embedded in the parquet file's own metadata fields, not as data columns. It is accessible via `api.load()` and shown in the viewer footer. See the [privacy policy](privacy.md) for what a table contains before sharing it.

To omit the base directory from the parquet metadata (for example when sharing a table without revealing local filesystem paths):

```bash
pixel-patrol process my-data/ -o results.parquet --omit-base-dir
```

```python
api.process_files(project, omit_base_dir=True)
```

---

## Parallelism

Processing runs in parallel using Dask workers. On a local machine (laptop or workstation), PixelPatrol auto-detects an appropriate number of worker processes based on available CPUs and RAM. On an HPC cluster you connect to an existing Dask scheduler instead and workers are whatever the cluster provides.

```bash
# Override the default worker count:
pixel-patrol process my-data/ -o results.parquet --max-workers 8

# Disable parallelism (single process, useful for debugging):
pixel-patrol process my-data/ -o results.parquet --max-workers 1
```

### Connecting to an external Dask cluster

```bash
pixel-patrol process my-data/ -o results.parquet --scheduler tcp://hostname:8786
```

From the Python API:

```python
from dask.distributed import Client
from pixel_patrol_base import api

project = api.create_project("my-project", base_dir="my-data/", loader="bioio")
with Client("tcp://hostname:8786"):
    api.process_files(project)
```

### SLURM clusters (`pixel-patrol-slurm`)

The `pixel-patrol-slurm` package provides a single command that launches a Dask `SLURMCluster`, waits for workers to come online, and then runs `pixel-patrol process` - no manual cluster setup needed.

```bash
pixel-patrol-slurm \
  --jobs 8 --cores 4 --memory 16GB \
  --partition gpu --walltime 02:00:00 \
  -- my-data/ -o results.parquet --loader bioio
```

Everything before `--` controls the SLURM cluster (number of jobs, cores per job, memory, partition, walltime). Everything after `--` is forwarded verbatim to `pixel-patrol process`; the `--scheduler` argument is injected automatically.

---

## Task sizing

PixelPatrol groups work into Dask tasks. Three kinds of task exist, each with its own sizing logic:

- **Batch tasks** - many small files are grouped into one task to reduce scheduling overhead.
- **Memory chunk tasks** - a single large file (whose uncompressed size exceeds `mb_per_task`) is split into spatial sub-regions, each processed as a separate task. Results are assembled before writing.
- **Container tasks** - sub-images from a container file are batched into tasks, again bounded by `mb_per_task`.

### `--mb-per-task`

The memory/work budget per task in MB (default: 512). It controls batch sizes for all three task types:

- **Many small files** - increase to reduce overhead: `--mb-per-task 2048`
- **Large 3D volumes or container files with large images** - decrease to keep individual tasks short: `--mb-per-task 128`

### `--max-images-per-task`

Maximum number of files (or sub-images) per task (default: 200).

```bash
pixel-patrol process my-data/ -o results.parquet --mb-per-task 256 --max-images-per-task 50
```


---

## Output options

### `--rows-per-part`

Number of rows buffered in memory before being flushed to a temporary file on disk. The default (10000) is suitable for most datasets.

### `--parquet-row-group-size`

Controls how records are grouped in the final parquet file (default: 2048). Smaller values speed up thumbnail loading in the viewer:

```bash
pixel-patrol process my-data/ -o results.parquet --parquet-row-group-size 512
```

### `--log-file`

Write a debug log alongside the output parquet for troubleshooting:

```bash
pixel-patrol process my-data/ -o results.parquet --log-file
```

---

## Parquet metadata

Embed project metadata into the parquet file - some fields are shown in the viewer header, all are accessible via `api.load()`:

```bash
pixel-patrol process my-data/ -o results.parquet \
  --name "Experiment 42" \
  --description "Treated vs control, 3 replicates" \
  --flavor "fluorescence"
```

From the API:

```python
api.process_files(
    project,
    flavor="fluorescence",
    description="Treated vs control, 3 replicates",
)
```
