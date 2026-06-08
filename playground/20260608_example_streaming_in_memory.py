"""
Example: process images "as they arrive" — one in-memory dask array at a
time — straight through Pixel Patrol's per-image processing step, and end up
with a normal report parquet: same processors, same columns, same viewer as a
regular `api.process_files` run — without ever touching a file or a `Project`.

This is the pattern for live-acquisition setups (a microscope, a camera, a
processing queue, …): each frame exists only briefly in memory as it's
produced. Pixel Patrol's public API is file-based (`create_project` ->
`add_paths` -> `process_files`, with a loader reading `Record`s from disk),
so faking a watch folder is one option — but if your frames are already plain
(dask) arrays, you can skip the file system entirely and call the very step
`process_files` runs internally on each loaded record: `_process_memory_chunk`
runs the processors on it, `_rollup` turns the result into report rows. That's
the whole pipeline minus "find files on disk and load them".

We also pretend each incoming array already carries Pixel-Patrol-compatible
metadata — `dim_order` for the processors, plus the identity/grouping fields
a loader would normally attach (`name`, `path`, `imported_path_short`, …) —
exactly what a real acquisition adapter would hand over alongside the pixels.
We don't derive any of it ourselves: `_extract_image_meta` reads it straight
off the record and `_rollup` carries it into every report row, so it ends up
in the parquet for free, right alongside the computed metrics.

The frames here come from two simulated sources with different brightness/
noise profiles (e.g. two cameras, or a setup before/after recalibration).
Stamping each one's `imported_path_short` with its source name is what lets
Pixel Patrol compare them as two groups — both mid-run and at the end.
"""
from pathlib import Path

import dask.array as da
import numpy as np
import polars as pl

from pixel_patrol_base import api
from pixel_patrol_base.core.processing import (
    _extract_image_meta, _post_process, _process_memory_chunk, _rollup,
)
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.project_metadata import ProjectMetadata
from pixel_patrol_base.core.record import record_from
from pixel_patrol_base.io.parquet_io import save_parquet
from pixel_patrol_base.plugin_registry import discover_processor_plugins

import logging
logging.basicConfig(level=logging.INFO)

OUTPUT_PATH = Path("out/streaming_report.parquet")

# The exact processors `process_files` would discover and run - basic
# intensity stats, histogram, thumbnail - applied directly to each array.
PROCESSORS = discover_processor_plugins()
CONFIG = ProcessingConfig()

# Two live sources feeding the same pipeline, each with its own
# brightness/noise profile - e.g. two cameras, or a setup before/after
# recalibration. `imported_path_short` (below) is what the viewer groups by.
SOURCES = {
    "camera_a": {"level":  90, "noise":  6},
    "camera_b": {"level": 140, "noise": 14},
}

# Refresh the report after every this-many new frames, so there's something
# to look at *during* acquisition - not just a single result once it's over.
CHECKPOINT_EVERY = 10


def incoming_records(n_per_source: int = 15, seed: int = 0):
    """Stand-in for a live source: yields one in-memory `Record` at a time, as
    the two cameras would interleave their frames. Each one wraps a plain
    dask array - it never touches disk - and already carries the metadata a
    real acquisition adapter would attach: `dim_order` (so Pixel Patrol's
    processors know how to read it) plus `name` / `path` /
    `imported_path_short` (so the report can identify and group it) - exactly
    the kind of `Record` a loader would normally hand to the pipeline."""
    rng = np.random.default_rng(seed)
    counters = {source: 0 for source in SOURCES}
    for i in range(n_per_source * len(SOURCES)):
        source = list(SOURCES)[i % len(SOURCES)]
        params = SOURCES[source]
        frame = rng.normal(loc=params["level"], scale=params["noise"], size=(256, 256))
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        image_id = f"{source}_{counters[source]:04d}"
        counters[source] += 1
        meta = {
            "dim_order":           "YX",
            "name":                f"{image_id}.tif",
            "path":                f"{source}/{image_id}.tif",
            "imported_path_short": source,
            "type":                "file",
            "file_extension":      "tif",
        }
        yield record_from(da.from_array(frame, chunks=frame.shape), meta, kind="intensity")


def process_record(record, file_index: int) -> pl.DataFrame:
    """Run the very step `process_files` runs per loaded image - no loader, no
    file path, just the record itself - and roll the result up into report
    rows. `image_meta` is read straight off the record's (pretend-attached)
    metadata, so the identity/grouping fields ride along into every row."""
    image_meta = _extract_image_meta(record)
    result = _process_memory_chunk(
        record, file_index=file_index, child_id=None,
        processors=PROCESSORS, config=CONFIG,
        file_path=record.meta["path"], image_meta=image_meta,
    )
    return pl.from_dicts(_rollup([result], PROCESSORS, CONFIG.slice_size))


def refresh_report(rows: list, metadata: ProjectMetadata, label: str) -> pl.DataFrame:
    """Combine every row produced so far into one report parquet - the same
    shape `process_files` writes, via the same `_post_process` cleanup and
    `save_parquet` writer - and print a quick per-group snapshot using the
    `imported_path_short` grouping the final viewer uses too."""
    records_df = _post_process(pl.concat(rows, how="diagonal_relaxed"))
    save_parquet(records_df, OUTPUT_PATH, metadata)
    print(f"\n── {label}: {records_df.height} images processed so far ──")
    print(
        records_df.group_by("imported_path_short").agg(
            pl.len().alias("n_images"),
            pl.col("mean_intensity").mean().round(1).alias("avg_mean_intensity"),
            pl.col("std_intensity").mean().round(1).alias("avg_std_intensity"),
        ).sort("imported_path_short")
    )
    return records_df


def main():
    metadata = ProjectMetadata(project_name="Incoming Frames")
    rows: list = []

    print("── images arriving one at a time, as plain in-memory dask arrays ──")
    for i, record in enumerate(incoming_records(), start=1):
        # `record.data` is a dask array that exists only in memory right now -
        # process it the moment it "arrives", just like acquisition software
        # would hand each new frame straight to a processing queue.
        rows.append(process_record(record, file_index=i))
        print(f"  received {record.meta['name']}  (from {record.meta['imported_path_short']})  "
              f"shape={record.data.shape}  dtype={record.data.dtype}")

        if i % CHECKPOINT_EVERY == 0:
            refresh_report(rows, metadata, label=f"checkpoint after {i} frames")

    print("\n── run finished — final report over everything that arrived ───────")
    records_df = refresh_report(rows, metadata, label="final report")
    print(records_df.select("imported_path_short", "name", "mean_intensity", "std_intensity"))

    # --- Show it: the regular Pixel Patrol viewer, comparing the two groups ---
    print("\n── show it ─────────────────────────────────────────────────────────")
    api.view(OUTPUT_PATH, group_col="imported_path_short")


if __name__ == "__main__":
    main()
