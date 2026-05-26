"""Pixel Patrol processing pipeline — Dask-distributed.

Each file yields one record (one logical image → one set of obs rows).
Container files (LMDB, multi-series OME-TIFF) yield one record per sub-image.
Large files are split into spatial memory_chunks processed in parallel.

MEMORY processors receive the full memory_chunk; their output goes to obs_level=0.
LEAF processors run on leaf blocks within each memory_chunk; _rollup aggregates
their rows into obs_levels 0..n (power-set over non-degenerate dims).

CALL GRAPH
──────────
  build_records_df
    ├─ _get_or_create_client         get or spin up a Dask LocalCluster
    ├─ _discover_files               generator: (file_path, file_metadata)
    ├─ _plan_tasks                   generator: FileInfo → Tasks
    └─ _coordinate_pipeline          submit + gather loop
          ├─ _execute_batch_task     [worker] small files
          ├─ _execute_memory_chunk_task [worker] one spatial sub-region
          ├─ _execute_sub_image_task [worker] sub-image batch
          ├─ _RecordAssembler        collects spatial chunks per record
          ├─ _rollup                 MemoryChunkResults → obs_row dicts
          ├─ _join_file_metadata     merge file metadata into obs rows
          └─ _ResultsWriter          buffer → parquet parts → finalize
"""

from __future__ import annotations

import itertools
import math
import logging
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, Iterator, List,
    NamedTuple, Optional, Tuple, Union,
)

import numpy as np
import polars as pl
from dask.distributed import Client, LocalCluster, as_completed, get_client

from pixel_patrol_base.core.contracts import ChunkKind, FileInfo, PixelPatrolLoader, PixelPatrolProcessor
from pixel_patrol_base.core.file_system import _discover_files
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.record import Record, record_from
from pixel_patrol_base.core.specs import is_record_matching_processor
from pixel_patrol_base.io.parquet_io import write_chunk

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _IndexedPath(NamedTuple):
    file_index: int
    file_path: str


@dataclass(frozen=True)
class MemoryChunkSpec:
    """Describes one spatial sub-region of a large file."""
    slices:      Tuple[slice, ...]  # applied as arr[spec.slices]
    origin:      Tuple[int, ...]    # global start coordinate of this sub-region
    dim_order:   Tuple[str, ...]
    image_shape: Tuple[int, ...]    # shape of the full source image


@dataclass(frozen=True)
class BatchTask:
    """One or more small file_paths processed together in a single worker call."""
    files: Tuple[_IndexedPath, ...]


@dataclass(frozen=True)
class MemoryChunkTask:
    """One spatial sub-region (memory_chunk) of a large file."""
    file_index:      int
    file_path:      str
    spec:           MemoryChunkSpec
    n_memory_chunks: int


@dataclass(frozen=True)
class SubImageTask:
    """A budget-sized batch of sub-images from a container file (LMDB, multi-series OME-TIFF, …)."""
    file_index:   int
    file_path:   str
    image_slice: Tuple[int, int]   # (start, stop) half-open


Task = Union[BatchTask, MemoryChunkTask, SubImageTask]


@dataclass
class MemoryChunkResult:
    """Output of processing one memory_chunk."""
    file_index:  int
    child_id:   Optional[str]
    chunk_rows: Dict[str, dict]   # proc.NAME → raw run_chunk output (MEMORY procs)
    leaf_rows:  List[dict]        # one merged dict per leaf block (LEAF procs)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plan tasks  — task construction, consumes _discover_files
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _resolve_leaf_block_shape(
    dim_order: Tuple[str, ...],
    user_spec: Optional[Dict[str, int]],
) -> Dict[str, int]:
    """Return the effective per-dim block size for every dim in dim_order.

    Default: X and Y → -1 (never split); all other dims → 1.
    Values from user_spec override the defaults for any named dim.
    -1 means full extent; positive N means split into blocks of N.
    """
    _FULL_EXTENT_BY_DEFAULT = {"X", "Y"}
    return {
        dim: (
            user_spec[dim]
            if (user_spec and dim in user_spec)
            else (-1 if dim in _FULL_EXTENT_BY_DEFAULT else 1)
        )
        for dim in dim_order
    }


def _split_into_ranges(
    total: int,
    step:  int,
) -> List[Tuple[Optional[int], Optional[int]]]:
    """Split [0, total) into chunks of size step; return (start, stop) pairs.

    Returns [(None, None)] when step >= total — a sentinel that becomes slice(None).
    """
    if step >= total:
        return [(None, None)]
    ranges: List[Tuple[Optional[int], Optional[int]]] = []
    pos = 0
    while pos < total:
        ranges.append((pos, min(pos + step, total)))
        pos += step
    return ranges


def _compute_memory_chunk_specs(
    file_path:        Path,
    info:             FileInfo,
    mb_per_task:      float,
    leaf_block_shape: Optional[Dict[str, int]],
) -> Optional[List[MemoryChunkSpec]]:
    """Return one MemoryChunkSpec per memory chunk for a file described by info.

    Returns None when uncompressed size ≤ budget, all dims are pinned to -1,
    or the resulting n_chunks would be ≤ 1.
    """
    dim_order    = info.dim_order
    shape        = info.shape
    dtype_bytes  = np.dtype(info.dtype).itemsize
    budget_bytes = int(mb_per_task * 1024 * 1024)

    total_bytes = int(np.prod(shape)) * dtype_bytes
    if total_bytes <= budget_bytes:
        return None

    leaf_block_sizes = _resolve_leaf_block_shape(dim_order, leaf_block_shape)
    dim_idx          = {d: i for i, d in enumerate(dim_order)}

    splittable = [d for d in dim_order if leaf_block_sizes[d] != -1]
    if not splittable:
        logger.warning(
            "_compute_memory_chunk_specs: %s (%.1f MB) exceeds budget but all dims are "
            "pinned (-1); file will be processed as a single batch item.",
            file_path, total_bytes / (1024 * 1024),
        )
        return None

    divisible_dims = sorted(
        [d for d in splittable if shape[dim_idx[d]] % leaf_block_sizes[d] == 0],
        key=lambda d: shape[dim_idx[d]],
        reverse=True,
    )

    chunk_dim_sizes: Dict[str, int] = {d: shape[dim_idx[d]] for d in dim_order}

    for dim in divisible_dims:
        leaf_block_dim_size = leaf_block_sizes[dim]
        full_dim_size       = shape[dim_idx[dim]]
        other_dims_elements = int(np.prod([chunk_dim_sizes[d] for d in dim_order if d != dim]))
        min_chunk_bytes     = other_dims_elements * leaf_block_dim_size * dtype_bytes
        if min_chunk_bytes >= budget_bytes:
            chunk_dim_sizes[dim] = leaf_block_dim_size
        else:
            n_blocks = budget_bytes // min_chunk_bytes
            chunk_dim_sizes[dim] = min(full_dim_size, n_blocks * leaf_block_dim_size)
            break

    per_dim_ranges = [_split_into_ranges(shape[dim_idx[d]], chunk_dim_sizes[d]) for d in dim_order]
    n_chunks = math.prod(len(r) for r in per_dim_ranges)

    if n_chunks <= 1:
        logger.warning(
            "_compute_memory_chunk_specs: %s (%.1f MB) exceeds budget but could not be split "
            "— leaf_block_shape=%s, image_shape=%s; processed as a single batch item.",
            file_path, total_bytes / (1024 * 1024),
            dict(leaf_block_sizes),
            dict(zip(dim_order, shape)),
        )
        return None

    specs: List[MemoryChunkSpec] = []
    for combo in itertools.product(*per_dim_ranges):
        slc    = tuple(slice(None) if s is None else slice(s, e) for s, e in combo)
        origin = tuple(0 if s is None else s for s, _ in combo)
        specs.append(MemoryChunkSpec(slices=slc, origin=origin, dim_order=dim_order, image_shape=shape))

    return specs


def _plan_sub_image_tasks(
    file_index:    int,
    file_path:    str,
    info:         FileInfo,
    budget_bytes: int,
) -> List[SubImageTask]:
    """Partition info.n_images into budget-sized batches; return one SubImageTask per batch."""
    per_image_bytes = max(1, int(np.prod(info.shape)) * np.dtype(info.dtype).itemsize)
    images_per_task = max(1, budget_bytes // per_image_bytes)
    image_slices: List[Tuple[int, int]] = []
    pos = 0
    n = info.n_images
    while pos < n:
        image_slices.append((pos, min(pos + images_per_task, n)))
        pos += images_per_task
    return [
        SubImageTask(file_index=file_index, file_path=file_path, image_slice=slc)
        for slc in image_slices
    ]


def _plan_tasks(
    file_stream: Iterator[Tuple[Path, dict]],
    config:      ProcessingConfig,
    loader:      Any,
    files_meta:  List[dict],
) -> Iterator[Task]:
    """Yield Tasks from a streaming file_stream, populating files_meta in-place.

    Routing (evaluated in order):
      n_images > 1  → flush pending batch; yield SubImageTasks.
      uncompressed size > budget → flush pending batch; try spatial chunking;
                                   if unsplittable, fall through to batch.
      otherwise     → accumulate in current batch; flush when budget fills.
    """
    budget_bytes: int = int(config.mb_per_task * 1024 * 1024)
    batch_files: List[_IndexedPath] = []
    batch_bytes: int = 0

    def _flush_batch() -> Optional[BatchTask]:
        nonlocal batch_files, batch_bytes
        if not batch_files:
            return None
        task = BatchTask(files=tuple(batch_files))
        batch_files = []
        batch_bytes = 0
        return task

    for file_path, file_meta in file_stream:
        try:
            info: FileInfo = loader.read_header(file_path)
        except Exception:
            logger.warning("_plan_tasks: read_header failed for %s; skipping", file_path)
            continue

        file_index = len(files_meta)
        files_meta.append(file_meta)

        if info.n_images > 1:
            if pending := _flush_batch():
                yield pending
            yield from _plan_sub_image_tasks(file_index, str(file_path), info, budget_bytes)
            continue

        uncompressed = int(np.prod(info.shape)) * np.dtype(info.dtype).itemsize

        if uncompressed > budget_bytes:
            if pending := _flush_batch():
                yield pending
            specs = _compute_memory_chunk_specs(file_path, info, config.mb_per_task, config.leaf_block_shape)
            if specs:
                n = len(specs)
                for spec in specs:
                    yield MemoryChunkTask(file_index=file_index, file_path=str(file_path), spec=spec, n_memory_chunks=n)
                continue

        batch_files.append(_IndexedPath(file_index=file_index, file_path=str(file_path)))
        batch_bytes += file_meta["size_bytes"]
        if batch_bytes >= budget_bytes:
            if pending := _flush_batch():
                yield pending

    if pending := _flush_batch():
        yield pending


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Execute task  — run inside Dask workers; stateless and picklable
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _stamp_coordinates_to_row(
    row:       dict,
    dim_order: Tuple[str, ...],
    origin:    Tuple[int, ...],
    shape:     Tuple[int, ...],
) -> dict:
    """Stamp global position, extent, and pixel count onto a metric row in-place."""
    row.update({f"dim_{d.lower()}": origin[i] for i, d in enumerate(dim_order)})
    row.update({
        "num_pixels": int(np.prod(shape)),
        **{f"{d.upper()}_size": shape[i] for i, d in enumerate(dim_order)},
    })
    return row


def _iter_leaf_blocks(
    arr:              np.ndarray,
    dim_order:        Tuple[str, ...],
    leaf_block_shape: Optional[Dict[str, int]],
    mem_origin:       Tuple[int, ...],
) -> Iterator[Tuple[np.ndarray, Tuple[int, ...]]]:
    """Yield (leaf_arr, leaf_global_origin) for every leaf block in arr."""
    block = _resolve_leaf_block_shape(dim_order, leaf_block_shape)
    steps = [block[d] if block[d] != -1 else arr.shape[i] for i, d in enumerate(dim_order)]
    ranges = [range(0, arr.shape[i], step) for i, step in enumerate(steps)]
    for starts in itertools.product(*ranges):
        slc = tuple(
            slice(s, min(s + step, arr.shape[i]))
            for i, (s, step) in enumerate(zip(starts, steps))
        )
        yield arr[slc], tuple(mem_origin[i] + s for i, s in enumerate(starts))


def _build_record(record: Record, data: Any, origin: Tuple[int, ...]) -> Record:
    """Build a new Record from a source record template, raw data, and a global origin."""
    arr = data.compute() if hasattr(data, "compute") else np.asarray(data)
    meta = {
        **record.meta,
        "shape": list(arr.shape),
        "ndim":  arr.ndim,
        **{f"dim_{d.lower()}": origin[i] for i, d in enumerate(record.dim_order)},
    }
    return record_from(arr, meta, kind=record.kind)


def _process_memory_chunk(
    mem_record:  Record,
    file_index:  int,
    child_id:    Optional[str],
    processors:  List[Any],
    config:      ProcessingConfig,
    file_path:   str,
) -> MemoryChunkResult:
    """Run all processors on one materialised memory chunk; return a MemoryChunkResult."""
    arr        = mem_record.data
    dim_order  = mem_record.dim_order
    mem_origin = tuple(mem_record.meta.get(f"dim_{d.lower()}", 0) for d in dim_order)

    # MEMORY pass
    chunk_rows: Dict[str, dict] = {}
    for proc in processors:
        if getattr(proc, "CHUNK_KIND", None) != ChunkKind.MEMORY:
            continue
        if not is_record_matching_processor(mem_record, proc.INPUT):
            continue
        try:
            row = proc.run_chunk(mem_record)
        except Exception as exc:
            logger.warning("worker: %s raised on memory chunk of %s: %s", proc.NAME, file_path, exc)
            continue
        if row:
            _stamp_coordinates_to_row(row, dim_order, mem_origin, arr.shape)
            chunk_rows[proc.NAME] = row

    # LEAF pass
    leaf_procs = [
        p for p in processors
        if getattr(p, "CHUNK_KIND", None) == ChunkKind.LEAF
        and is_record_matching_processor(mem_record, p.INPUT)
    ]
    leaf_rows: List[dict] = []

    for leaf_arr, leaf_global_origin in _iter_leaf_blocks(arr, dim_order, config.leaf_block_shape, mem_origin):
        leaf_record = _build_record(mem_record, leaf_arr, leaf_global_origin)
        leaf_row: dict = {}
        for proc in leaf_procs:
            try:
                row = proc.run_chunk(leaf_record)
            except Exception as exc:
                logger.warning("worker: %s raised on leaf of %s: %s", proc.NAME, file_path, exc)
                continue
            if row:
                leaf_row.update(row)
        if leaf_row:
            _stamp_coordinates_to_row(leaf_row, dim_order, leaf_global_origin, leaf_arr.shape)
            leaf_rows.append(leaf_row)

    return MemoryChunkResult(file_index=file_index, child_id=child_id, chunk_rows=chunk_rows, leaf_rows=leaf_rows)


def _execute_batch_task(
    task:       BatchTask,
    loader:     Any,
    processors: List[Any],
    config:     ProcessingConfig,
) -> List[MemoryChunkResult]:
    """Process every file in the task as one complete memory chunk each."""
    results: List[MemoryChunkResult] = []
    for idxed_path in task.files:
        record     = loader.load(Path(idxed_path.file_path))
        mem_record = _build_record(record, record.data, tuple(0 for _ in record.dim_order))
        results.append(
            _process_memory_chunk(mem_record, idxed_path.file_index, None, processors, config, idxed_path.file_path)
        )
    return results


def _execute_memory_chunk_task(
    task:       MemoryChunkTask,
    loader:     Any,
    processors: List[Any],
    config:     ProcessingConfig,
) -> List[MemoryChunkResult]:
    """Load the spatial sub-region defined by task.spec and process it."""
    record     = loader.load(Path(task.file_path))
    mem_record = _build_record(record, record.data[task.spec.slices], task.spec.origin)
    return [_process_memory_chunk(mem_record, task.file_index, None, processors, config, task.file_path)]


def _execute_sub_image_task(
    task:       SubImageTask,
    loader:     Any,
    processors: List[Any],
    config:     ProcessingConfig,
) -> List[MemoryChunkResult]:
    """Load and process a batch of sub-images from a container file."""
    results: List[MemoryChunkResult] = []
    start, stop = task.image_slice
    for child_id, record in loader.load_range(Path(task.file_path), start, stop):
        mem_record = _build_record(record, record.data, tuple(0 for _ in record.dim_order))
        results.append(
            _process_memory_chunk(mem_record, task.file_index, child_id, processors, config, task.file_path)
        )
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Coordinator helpers  — all run in the main (coordinator) process
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _rollup(
    chunk_results: List[MemoryChunkResult],
    processors:    List[Any],
) -> List[dict]:
    """Aggregate a complete set of MemoryChunkResults for one record into obs rows.

    Produces:
      obs_level=0  one global summary row; contains globally aggregated
                   leaf metrics AND chunk processor columns (e.g. thumbnail).
      obs_level=1+ per-dim rows (per-channel, per-Z, per-T, …) from leaf
                   aggregation only.
    Power-set over non-spatial dims; degenerate axes (size=1) skipped.
    """
    all_leaf_rows = [row for cr in chunk_results for row in cr.leaf_rows]

    col_to_proc: Dict[str, Any] = {}
    for proc in processors:
        if getattr(proc, "CHUNK_KIND", None) == ChunkKind.LEAF:
            for col in getattr(proc, "OUTPUT_SCHEMA", {}):
                col_to_proc[col] = proc

    leaf_metric_cols = sorted({
        col for row in all_leaf_rows for col in row
        if col in col_to_proc
    })

    all_dim_keys = sorted({
        col for row in all_leaf_rows for col in row if col.startswith("dim_")
    })
    active_dims = [
        d for d in all_dim_keys
        if len({row.get(d) for row in all_leaf_rows}) > 1
    ]
    n = len(active_dims)

    obs_rows: List[dict] = []

    def _make_aggregate_row(group_rows: List[dict], g_dim_dict: Dict[str, Any]) -> dict:
        obs: dict = {d: None for d in active_dims}
        obs.update(g_dim_dict)
        obs["num_pixels"] = sum(row.get("num_pixels", 0) for row in group_rows)
        obs["obs_level"] = len(g_dim_dict)
        for col in leaf_metric_cols:
            obs[col] = col_to_proc[col].get_aggregation(col)(group_rows, g_dim_dict)
        return obs

    global_obs = _make_aggregate_row(all_leaf_rows, {})
    mem_cols_added = 0
    for proc in processors:
        if getattr(proc, "CHUNK_KIND", None) != ChunkKind.MEMORY:
            continue
        proc_chunk_rows = [
            cr.chunk_rows[proc.NAME]
            for cr in chunk_results
            if proc.NAME in cr.chunk_rows
        ]
        if not proc_chunk_rows:
            continue
        for col in getattr(proc, "OUTPUT_SCHEMA", {}):
            global_obs[col] = proc.get_aggregation(col)(proc_chunk_rows, {})
            mem_cols_added += 1
    if all_leaf_rows or mem_cols_added:
        obs_rows.append(global_obs)

    for r in range(1, n):
        for g_dim_combo in itertools.combinations(active_dims, r):
            groups: Dict[tuple, List[dict]] = {}
            for row in all_leaf_rows:
                key = tuple(row.get(d) for d in g_dim_combo)
                groups.setdefault(key, []).append(row)
            for key_vals, group_rows in groups.items():
                obs_rows.append(_make_aggregate_row(group_rows, dict(zip(g_dim_combo, key_vals))))

    if n > 0:
        for row in all_leaf_rows:
            obs_rows.append({**row, "obs_level": n})

    return obs_rows


def _join_file_metadata(
    obs_rows:  List[dict],
    file_meta: dict,
    child_id:  Optional[str],
) -> pl.DataFrame:
    """Merge file_meta into every obs row and return a Polars DataFrame."""
    rows = [{**row, **file_meta} for row in obs_rows]
    if child_id is not None:
        for row in rows:
            row["child_id"] = child_id
    return pl.from_dicts(rows)


def _is_blob_dtype(dtype) -> bool:
    return dtype == pl.Binary or isinstance(dtype, (pl.List, pl.Array))


_NO_SHRINK = {"size_bytes"}  # protect from int shrink — sums across many files would overflow


def _post_process(df: pl.DataFrame) -> pl.DataFrame:
    """Final cleanup applied once to the fully combined DataFrame.

    1. Mark sub-image rows: where child_id is present and non-null, set type="sub_file".
    2. Drop all-null columns.
    3. Shrink integer dtypes (except size_bytes); downcast float64 → float32.
    4. Reorder: obs_level → dim_* → scalar columns → blob columns.
    """
    # sub-image type annotation
    if "child_id" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("child_id").is_not_null())
            .then(pl.lit("sub_file"))
            .otherwise(pl.col("type"))
            .alias("type")
        )

    # Drop all-null columns
    df = df.select([c for c in df.columns if df[c].null_count() < len(df)])

    # Dtype shrink / downcast
    casts = []
    for col in df.columns:
        dtype = df.schema[col]
        if dtype.is_integer() and col not in _NO_SHRINK:
            new_dtype = df[col].shrink_dtype().dtype
            if new_dtype != dtype:
                casts.append(pl.col(col).cast(new_dtype))
        elif dtype == pl.Float64:
            casts.append(pl.col(col).cast(pl.Float32))
    if casts:
        df = df.with_columns(casts)

    # Column reorder
    dim_cols    = sorted(c for c in df.columns if c.startswith("dim_"))
    blob_cols   = [c for c in df.columns if _is_blob_dtype(df.schema[c])]
    skip        = {"obs_level"} | set(dim_cols) | set(blob_cols)
    scalar_cols = [c for c in df.columns if c not in skip]
    ordered     = ["obs_level"] + dim_cols + scalar_cols + blob_cols
    return df.select([c for c in ordered if c in df.columns])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _RecordAssembler  — collects partial results until a record is complete
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _RecordAssembler:
    """Accumulates MemoryChunkResults from MemoryChunkTasks until all spatial chunks arrive."""

    def __init__(self) -> None:
        self._results:  Dict[Tuple[int, Optional[str]], List[MemoryChunkResult]] = {}
        self._expected: Dict[Tuple[int, Optional[str]], int] = {}

    def add(self, result: MemoryChunkResult, n_memory_chunks: int) -> bool:
        """Register one completed MemoryChunkResult.

        Returns True when all n_memory_chunks spatial chunks for
        (result.file_index, result.child_id) have arrived.
        """
        key = (result.file_index, result.child_id)
        if key not in self._results:
            self._results[key] = []
            self._expected[key] = n_memory_chunks
        self._results[key].append(result)
        return len(self._results[key]) == self._expected[key]

    def pop(self, record_key: Tuple[int, Optional[str]]) -> List[MemoryChunkResult]:
        """Remove and return all accumulated MemoryChunkResults for a record."""
        self._expected.pop(record_key)
        return self._results.pop(record_key)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _ResultsWriter  — buffers complete record rows and writes part files
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _ResultsWriter:
    """Buffers post-rollup obs rows and writes intermediate part files."""

    def __init__(
        self,
        rows_per_part:  int,
        parts_dir:      Optional[Path],
        row_group_size: Optional[int] = None,
    ) -> None:
        self._rows_per_part  = rows_per_part
        self._parts_dir      = parts_dir
        self._row_group_size = row_group_size
        self._buffer:        List[pl.DataFrame] = []
        self._buffer_rows    = 0
        self._part_paths:    List[Path] = []
        self._part_counter   = 0

    def add(self, record_df: pl.DataFrame) -> None:
        """Append one complete record's rows to the buffer."""
        self._buffer.append(record_df)
        self._buffer_rows += len(record_df)
        if self._buffer_rows >= self._rows_per_part:
            self.flush()

    def flush(self) -> None:
        """Write the current buffer to a new part file and clear it."""
        if not self._buffer or self._parts_dir is None:
            return
        self._parts_dir.mkdir(parents=True, exist_ok=True)
        part_path = self._parts_dir / f"part_{self._part_counter:04d}.parquet"
        write_kwargs: Dict[str, Any] = {"compression": "zstd"}
        if self._row_group_size is not None:
            write_kwargs["row_group_size"] = self._row_group_size
        pl.concat(self._buffer, how="diagonal_relaxed").write_parquet(part_path, **write_kwargs)
        self._part_paths.append(part_path)
        self._buffer      = []
        self._buffer_rows = 0
        self._part_counter += 1

    def finalize(self) -> pl.DataFrame:
        """Flush any remaining buffer, combine all parts, and post-process."""
        self.flush()
        if not self._part_paths and not self._buffer:
            return pl.DataFrame()
        if self._part_paths:
            combined = pl.concat([pl.scan_parquet(p) for p in self._part_paths], how="diagonal_relaxed").collect()
        else:
            combined = pl.concat(self._buffer, how="diagonal_relaxed")
        return _post_process(combined)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Coordinate pipeline  — submit + gather loop; runs in the main process
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _coordinate_pipeline(
    client,
    task_stream:  Iterator[Task],
    files_meta:   List[dict],
    loader:       Any,
    processors:   List[Any],
    config:       ProcessingConfig,
    parts_dir:    Optional[Path],
    on_progress:  Optional[Callable[[int, int], None]],
) -> Optional[pl.DataFrame]:
    """Drive the full submit → gather → rollup → join → accumulate cycle.

    Maintains a MAX_PENDING_TASKS sliding window of Dask futures so workers
    stay busy while _plan_tasks is still generating tasks from the file stream.
    """
    max_pending = 2 * len(client.nthreads())
    writer      = _ResultsWriter(config.rows_per_part, parts_dir, config.parquet_row_group_size)
    assembler   = _RecordAssembler()
    future_to_task: Dict[Any, Task] = {}
    completed_records = 0

    _executor_for: Dict[type, Any] = {
        BatchTask:       _execute_batch_task,
        MemoryChunkTask: _execute_memory_chunk_task,
        SubImageTask:    _execute_sub_image_task,
    }

    def _submit(t: Task) -> Any:
        f = client.submit(_executor_for[type(t)], t, loader, processors, config, pure=False)
        future_to_task[f] = t
        return f

    def _handle_record(
        chunk_results: List[MemoryChunkResult],
        file_index:    int,
        child_id:      Optional[str],
    ) -> None:
        nonlocal completed_records
        obs_rows = _rollup(chunk_results, processors)
        if obs_rows:
            writer.add(_join_file_metadata(obs_rows, files_meta[file_index], child_id))
        completed_records += 1
        if on_progress:
            on_progress(completed_records, -1)

    task_iter       = iter(task_stream)
    initial_futures = [_submit(t) for t in itertools.islice(task_iter, max_pending)]
    if not initial_futures:
        return None

    ac = as_completed(initial_futures)
    for future in ac:
        task = future_to_task.pop(future)

        next_task = next(task_iter, None)
        if next_task is not None:
            ac.add(_submit(next_task))

        try:
            results: List[MemoryChunkResult] = future.result()
        except Exception as exc:
            logger.warning("_coordinate_pipeline: worker failed for %s: %s", task, exc)
            continue

        if isinstance(task, MemoryChunkTask):
            result = results[0]
            if assembler.add(result, task.n_memory_chunks):
                key = (result.file_index, result.child_id)
                _handle_record(assembler.pop(key), result.file_index, result.child_id)
        else:
            for result in results:
                _handle_record([result], result.file_index, result.child_id)

    result_df = writer.finalize()
    return result_df if len(result_df) > 0 else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cluster lifecycle
# TODO: move to pixel_patrol_base.dask_utils once Dask pipeline is fully
#       integrated — this is the only place that manages cluster lifecycle.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@contextmanager
def _get_or_create_client(config: ProcessingConfig) -> Generator:
    """Yield an active Dask distributed client.

    Reuses an existing client if one is connected; otherwise spins up a temporary
    LocalCluster with config.max_workers workers and shuts it down on context exit.
    """
    try:
        client = get_client()
        logger.debug("_get_or_create_client: reusing existing client %s", client.scheduler_info()["address"])
        yield client
    except ValueError:
        logger.debug("_get_or_create_client: starting LocalCluster n_workers=%s", config.max_workers)
        cluster = LocalCluster(n_workers=config.max_workers, threads_per_worker=1, processes=True)
        client = Client(cluster)
        try:
            yield client
        finally:
            client.close()
            cluster.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_records_df(
    bases:        List[Path],
    loader:       Any,
    processors:   List[Any],
    config:       Optional[ProcessingConfig] = None,
    parts_dir:    Optional[Path] = None,
    on_progress:  Optional[Callable[[int, int], None]] = None,
) -> Optional[pl.DataFrame]:
    """Scan files, distribute processing across Dask workers, return records_df.

    _discover_files, _plan_tasks, and the submit side of _coordinate_pipeline
    run concurrently via a streaming generator pipeline — workers receive their
    first tasks before the filesystem scan is complete.

    Args:
        bases:       Base directories to scan.
        loader:      Loader instance; must implement read_header, load, load_range.
        processors:  Processor instances to run on each record. Caller is responsible
                     for applying processors_included / processors_excluded filtering.
        config:      Runtime options. Defaults used if None.
        parts_dir:   Directory for intermediate part files. None → in-memory only.
        on_progress: Optional callback(done: int, total: int) called per completed record.
                     total is -1 until the full record count is known.

    Returns records_df, or None if no files are found or processed.

    # TODO: processors_included / processors_excluded are handled by the caller
    #       (project.py); build_records_df receives the already-filtered list.
    """
    cfg = config or ProcessingConfig()

    with _get_or_create_client(cfg) as client:
        files_meta: List[dict] = []
        folder_exts = getattr(loader, "FOLDER_EXTENSIONS", None) if loader is not None else None
        task_stream = _plan_tasks(
            _discover_files(bases, cfg.selected_file_extensions, folder_exts),
            config=cfg,
            loader=loader,
            files_meta=files_meta,
        )
        return _coordinate_pipeline(
            client=client,
            task_stream=task_stream,
            files_meta=files_meta,
            loader=loader,
            processors=processors,
            config=cfg,
            parts_dir=parts_dir,
            on_progress=on_progress,
        )


def cleanup_chunks_dir(parts_dir: Optional[Path]) -> None:
    """Remove the parts directory after a successful save.

    Should be called by the caller (e.g. project.py) after save_parquet succeeds.
    """
    if not parts_dir or not parts_dir.exists():
        return
    try:
        shutil.rmtree(parts_dir)
        logger.info("Processing Core: removed intermediate parts directory '%s'.", parts_dir)
    except OSError as exc:
        logger.warning("Processing Core: could not remove '%s': %s", parts_dir, exc)
