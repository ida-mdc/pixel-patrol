# Processing Strategy: `raster-processor2` branch

## Goal

Process arbitrarily large image datasets — including images that do not fit in RAM — while keeping all workers busy from the moment scanning starts, and without requiring a full file list before work begins.

---

## File discovery and task batching

File discovery and task submission happen simultaneously via a streaming generator. As each file is encountered on disk, it is immediately evaluated and either added to the current batch or turned into one or more slice tasks. Workers start receiving tasks before the scan is complete.

**Small files** accumulate into a shared batch until either the cumulative on-disk size reaches the memory budget (default 1 GB) or the batch has 50 files — whichever comes first. One task covers many files, so a dataset with thousands of small images results in far fewer tasks than files.

**Large files** (whose uncompressed array size exceeds the memory budget) are handled differently. The header is opened lazily to determine the true array dimensions, and the file is split into one or more spatial **slices** (see below), each of which becomes its own task. The header check is only performed when the on-disk file size hints that the array might be large (> 50% of the budget), avoiding unnecessary I/O for small files.

---

## Slicing large files

A large file is split into memory-sized chunks using `raster_slicing_plan`. The plan divides the array dimensions in a fixed priority order:

1. **Channel dims (C, S) — never split.** Every task always receives the full channel stack. This is required so colocalization metrics can always compare all channel pairs within a single computation.
2. **X — tiled first.** As many `tile_size`-wide columns as fit in the memory budget are included per task.
3. **Y — tiled next** with the remaining budget.
4. **Z, T, and any other non-spatial dims — divided last** with whatever budget remains after XY allocation.

The budget is computed as: how many `tile_size × tile_size` atomic tiles (multiplied by all channel pixels) fit within the target memory limit. The result is a list of array slices covering the full image in a Cartesian product of the above ranges.

---

## Tile-based metric computation (raster processor)

Within each chunk (whether a batch file or a large-file slice), `NumpyRasterBackend` computes metrics at tile granularity.

**Folding into a tile grid.** The 2D spatial plane is reshaped from `(..., H, W)` into a tile grid `(n_planes, n_tiles_y, n_tiles_x, tile_size, tile_size)` using `fold_to_tiles`. If H or W is not a multiple of `tile_size`, the array is zero-padded and a boolean mask tracks which pixels are real.

**Batch processing over planes.** All leading (non-spatial) axes — channels, Z, T — are flattened into a plane index. Planes are processed in batches sized to keep intermediate memory (especially the 3×3 neighbourhood stats arrays) within a configurable budget (default 8 GB). Within each plane batch, all metrics are computed as vectorised NumPy operations over the full tile grid at once — there is no Python loop over individual tiles.

**Metrics computed per tile:**

- *Basic* (on by default): min, max, mean, std intensity; finite pixel count
- *Quality* (on by default):
  - **Michelson contrast** — mean 3×3 local range divided by tile std
  - **MSCN variance** — variance of mean-subtracted, locally-normalised (MSCN) coefficients over 3×3 windows; a standard no-reference sharpness proxy
  - **Local std ratio** — mean 3×3 local std divided by tile std; another sharpness indicator
- *Histogram* (on by default): per-tile brightness histogram using an offset-binning trick that counts all tiles in a single `np.bincount` call; histogram bounds are set from the per-plane global min/max
- *Compression artifacts* (off by default): blocking index (average brightness jump at 8-pixel JPEG-block boundaries); ringing index (variance of a simple high-pass residual)

The 3×3 neighbourhood stats needed for MSCN and local-std-ratio are computed once per plane batch and cached, so they are not recomputed for the second metric.

Each valid tile produces one row containing all metric values and its coordinate (`dim_z`, `dim_c`, `dim_y`, `dim_x`, …).

---

## Power-set rollup

After all tile rows for a file are collected, `accumulate_power_set` rolls them up into a complete aggregation tree. Every useful combination of dimensions is computed: per-channel global, per-Z global, per-channel-per-Z, per-tile (if enabled), and fully global — each labelled with an `obs_level` (0 = global summary, higher = more granular). Degenerate spatial axes (images with only one tile in Y or X) are collapsed so the tree does not produce redundant entries.

Each metric has its own rollup rule:
- **max/min** → `nanmax` / `nanmin`
- **mean intensity** → pixel-count-weighted mean (so partial edge tiles contribute proportionally to their valid area)
- **std, contrast, MSCN, sharpness** → pixel-count-weighted mean across tiles
- **finite pixel count** → sum
- **histograms** → merged by remapping each tile's bin centers onto a shared global value range; a fast path short-circuits when all tiles share the same range

For large files split into slices, tile rows from all slice tasks are gathered and fed into `accumulate_slice_rows` as a single combined list before rollup, producing the same result as if the whole image had been loaded at once.

---

## Processors currently hooked up

### `raster-image` — intensity and quality metrics

Implements the tile-based computation and power-set rollup described above. Marked `SLICE_SAFE`: can process any spatial chunk independently; results from all slices are accumulated after all tasks complete.

### `channel-colocalization` — pairwise channel correlation

Runs on images with a C dimension (≥ 2 channels). For each tile it computes **Pearson r** and **SSIM** (plus its luminance, contrast, and structure components) between every channel pair. Channel pairs are derived from all-channel combinations: (0,1), (0,2), …, (C-2, C-1). Each channel is folded into tiles once; all pairs share that fold. Results follow the same power-set rollup structure as the raster processor. `SLICE_SAFE` is guaranteed because C is never split — every slice always has all channels.

### `thumbnail` — spatial mosaic preview

Generates one fixed-size RGBA thumbnail per image. In the normal (small-file) path, the whole image is reduced to a single preview stored in the global row (`obs_level=0`). In the large-file path, each slice task produces a small downsampled patch (≤ 64×64 px) covering its spatial region; `accumulate_slice_rows` places all patches onto a shared canvas at positions proportional to their location in the full image. For multiple Z/T slices at the same XY position, the spatially central patch is used.

---

## Execution and output

Processing always runs through **dask.distributed**. A temporary `LocalCluster` is created automatically when no external scheduler is connected; on HPC the same code path submits to the cluster that is already attached.

Results are accumulated into a Polars DataFrame and, for large datasets, flushed to intermediate Parquet chunks during processing to avoid accumulating everything in memory. After all tasks complete, chunks are combined and dtype-optimised (integers shrunk, float64 downcast to float32) before writing the final output.

---

## Data flow summary

```
filesystem scan (streaming)
        │
        ▼  one file at a time
task planner
  ├─ small files  →  accumulate into batch
  │                  flush when ≥ target_mb or ≥ 50 files  →  batch task
  │
  └─ large file (uncompressed > target_mb)
       open header, run raster_slicing_plan
       ├─ full_file task  (non-slice-safe processors)
       └─ N slice tasks   (slice-safe processors, one per memory-sized chunk)
                                       │
                          dask.distributed workers (local or HPC)
                                       │
                          ┌────────────────────────────────┐
                          │  per chunk / per file:         │
                          │  fold into tile grid           │
                          │  compute metrics (vectorised)  │  ← raster-image
                          │  compute channel pairs         │  ← channel-coloc
                          │  compute thumbnail patch       │  ← thumbnail
                          └────────────────────────────────┘
                                       │
                          accumulate_slice_rows
                          (gather slice tile rows → power-set rollup
                           stitch thumbnail patches)
                                       │
                          parquet flush / final combine
```
