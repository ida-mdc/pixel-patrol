"""Pixel Patrol processing pipeline — Dask-distributed.

Each file yields one record (one logical image → one set of obs rows).
Container files (LMDB, multi-series OME-TIFF) yield one record per sub-image.
Large files are split into memory chunks (sub-regions) processed in parallel.

MEMORY processors receive one memory chunk at a time; when a file is split into
multiple chunks they all run first, then results are assembled before obs_level=0 is written.
LEAF processors run on every spatial leaf block within a chunk; _rollup aggregates
their outputs into one row per unique combination of active dimensions (obs_level 0 = global,
1 = per-Z, per-C, … n = per leaf block).

CALL GRAPH
──────────
  build_records_df
    ├─ _get_or_create_client         get or spin up a Dask LocalCluster
    ├─ _discover_files               generator: (file_path, file_metadata)
    ├─ _plan_tasks                   generator: FileInfo → Tasks
    └─ _coordinate_pipeline          submit + gather loop
          ├─ _execute_batch_task     [worker] small files
          ├─ _execute_memory_chunk_task [worker] one memory chunk (sub-region)
          ├─ _execute_container_task [worker] container file batch
          ├─ _RecordAssembler        collects memory chunks per record
          ├─ _rollup                 MemoryChunkResults → obs_row dicts
          ├─ _join_file_metadata     merge file metadata into obs rows
          └─ _ResultsWriter          buffer → parquet parts → finalize
"""

from __future__ import annotations

import itertools
import math
import logging
import shutil
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, Iterator, List,
    NamedTuple, Optional, Tuple, Union, Set
)

import numpy as np
import polars as pl
import psutil
import pyarrow.parquet as pq
from dask.distributed import Client, LocalCluster, as_completed, get_client
from tqdm.auto import tqdm

from pixel_patrol_base.config import HISTOGRAM_BINS
from pixel_patrol_base.core.contracts import ChunkKind, FileInfo, PixelPatrolLoader, PixelPatrolProcessor
from pixel_patrol_base.core.file_system import _discover_files
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.record import Record, record_from
from pixel_patrol_base.core.specs import is_record_matching_processor

# Dask sets distributed.* loggers to INFO at import time — suppress after importing.
for _name in ("distributed", "distributed.worker", "distributed.scheduler",
              "distributed.nanny", "asyncio", "tornado", "numexpr", "numexpr.utils"):
    logging.getLogger(_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _silence_numcodecs_warning():
    """Same showwarning patch as core/__init__.py, applied to each Dask worker process."""
    import warnings
    _sw = warnings.showwarning
    warnings.showwarning = lambda m, c, f, l, *a, **kw: None if c is DeprecationWarning and "numcodecs" in str(f) else _sw(m, c, f, l, *a, **kw)


# Fields that are PP-internal per-chunk values, not image-level metadata.
# Excluded from image_meta so they don't overwrite PP-computed per-row fields.
_IMAGE_META_SKIP = frozenset({"shape", "num_pixels"})

# Columns exempt from integer dtype shrinkage in _post_process.
# size_bytes is a cumulative sum across many files and can exceed int32 range.
_INT_SHRINK_EXEMPT = frozenset({"size_bytes"})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _IndexedPath(NamedTuple):
    file_index: int
    file_path: str


@dataclass(frozen=True)
class MemoryChunkSpec:
    """Describes one sub-region (memory chunk) of a large file."""
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
    """One sub-region (memory chunk) of a large file."""
    file_index:      int
    file_path:      str
    spec:           MemoryChunkSpec
    n_memory_chunks: int


@dataclass(frozen=True)
class ContainerTask:
    """A budget-sized batch of sub-images from a container file (LMDB, multi-series OME-TIFF, …)."""
    file_index:   int
    file_path:   str
    image_slice: Tuple[int, int]   # (start, stop) half-open


Task = Union[BatchTask, MemoryChunkTask, ContainerTask]


@dataclass
class MemoryChunkResult:
    """Output of processing one memory_chunk."""
    file_index:  int
    child_id:    Optional[str]
    chunk_rows:  Dict[str, dict]   # proc.NAME → raw run_chunk output (MEMORY procs)
    leaf_rows:   List[dict]        # one merged dict per leaf block (LEAF procs)
    image_meta:  Dict[str, Any]    # loader-provided image metadata, merged into all obs rows
    timing:      Dict[str, float]  = field(default_factory=dict)  # "load", "proc_<NAME>" → CPU-seconds


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Plan tasks  — task construction, consumes _discover_files
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _resolve_leaf_block_shape(
    dim_order: Tuple[str, ...],
    user_spec: Optional[Dict[str, int]],
) -> Dict[str, int]:
    """Return the effective per-dim block size for every dim in dim_order.

    Default: X and Y → -1 (never split); all other dims → 1.
    These defaults reflect the conventional 2-D image plane — override any
    dim via user_spec (e.g. leaf_block_shape={'X': 256, 'Z': 1}).
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


def _plan_container_tasks(
    file_index:         int,
    file_path:         str,
    info:              FileInfo,
    budget_bytes:      int,
    max_images_per_task: int,
) -> List[ContainerTask]:
    """Partition info.n_images into budget-sized batches; return one ContainerTask per batch."""
    per_image_bytes = max(1, int(np.prod(info.shape)) * np.dtype(info.dtype).itemsize)
    images_per_task = max(1, min(budget_bytes // per_image_bytes, max_images_per_task))
    image_slices: List[Tuple[int, int]] = []
    pos = 0
    n = info.n_images
    while pos < n:
        image_slices.append((pos, min(pos + images_per_task, n)))
        pos += images_per_task
    return [
        ContainerTask(file_index=file_index, file_path=file_path, image_slice=slc)
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
      n_images > 1  → flush pending batch; yield ContainerTasks.
      uncompressed size > budget → flush pending batch; try spatial chunking;
                                   if unsplittable, fall through to batch.
      otherwise     → accumulate in current batch; flush when budget fills.
    """
    budget_bytes: int = int(config.mb_per_task * 1024 * 1024)
    _MAX_IMAGES_PER_TASK = config.max_images_per_task
    _container_exts: frozenset = frozenset(getattr(loader, "CONTAINER_EXTENSIONS", ()))
    # Folder-based formats (zarr, ome.zarr) report compressed on-disk size which
    # can be orders of magnitude smaller than the uncompressed array — never skip
    # read_header for them or they'll be mis-classified as small files.
    _folder_exts: frozenset    = frozenset(getattr(loader, "FOLDER_EXTENSIONS", ()))
    _small_file_threshold: int = budget_bytes // 8
    _container_hint_done = False  # emit at most once per run
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

    def _should_flush() -> bool:
        return batch_bytes >= budget_bytes or len(batch_files) >= _MAX_IMAGES_PER_TASK

    for file_path, file_meta in file_stream:
        size_bytes: int = file_meta.get("size_bytes", 0)
        ext: str = file_meta.get("file_extension", "")

        if 0 < size_bytes < _small_file_threshold and ext not in _container_exts and ext not in _folder_exts:
            file_index = len(files_meta)
            files_meta.append(file_meta)
            batch_files.append(_IndexedPath(file_index=file_index, file_path=str(file_path)))
            batch_bytes += size_bytes
            if _should_flush():
                if pending := _flush_batch():
                    yield pending
            continue

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
            if not _container_hint_done:
                _container_hint_done = True
                image_bytes = int(np.prod(info.shape)) * np.dtype(info.dtype).itemsize
                if image_bytes > 0:
                    images_per_task = min(budget_bytes // image_bytes, info.n_images, _MAX_IMAGES_PER_TASK)
                    if images_per_task > 20:
                        logger.warning(
                            "Container file with n_images=%d, ~%.1f MB/image; "
                            "mb_per_task=%.0f gives ~%d images/task — tasks may take many minutes. "
                            "Consider --mb-per-task 50 or lower.",
                            info.n_images, image_bytes / 1024 / 1024,
                            config.mb_per_task, images_per_task,
                        )
            yield from _plan_container_tasks(file_index, str(file_path), info, budget_bytes, _MAX_IMAGES_PER_TASK)
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
        batch_bytes += file_meta.get("size_bytes", 0)
        if _should_flush():
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
    # If data is lazy (e.g. a zarr array), read it immediately in this worker process
    # rather than letting Dask schedule it — otherwise Dask submits one sub-task per
    # internal chunk and floods the cluster with thousands of tiny jobs.
    arr = data.compute(scheduler='synchronous') if hasattr(data, "compute") else np.asarray(data)
    meta = {
        **record.meta,
        "shape": list(arr.shape),
        "ndim":  arr.ndim,
        **{f"dim_{d.lower()}": origin[i] for i, d in enumerate(record.dim_order)},
    }
    return record_from(arr, meta, kind=record.kind)


def _extract_image_meta(record: Record) -> Dict[str, Any]:
    """Collect image-level metadata from a fully-loaded (un-chunked) record.

    Starts with all loader-provided fields in record.meta, drops PP-internal
    per-chunk fields (shape, num_pixels) and dim_* coordinate keys, then
    adds/overrides the canonical image-level fields:
    dim_order, dtype, ndim, *_size (full image extent per dim (e.g. S_size=3, Y_size=512))
    Any additional loader-provided fields in record.meta are included as-is. 
    Image metadata is available at all obs_levels.

    For container files (LMDB, multi-series OME-TIFF), call this once per sub-image.
    """
    meta: Dict[str, Any] = {
        k: v for k, v in record.meta.items()
        if not k.startswith("dim_") and k not in _IMAGE_META_SKIP
    }
    meta["dim_order"] = record.dim_order                  # already a clean string
    meta["dtype"]     = str(np.dtype(record.data.dtype))  # canonical form, e.g. "uint16"
    meta["ndim"]      = len(record.dim_order)
    for i, d in enumerate(record.dim_order):
        meta[f"{d}_size"] = int(record.data.shape[i])
    return meta


def _process_memory_chunk(
    mem_record:  Record,
    file_index:  int,
    child_id:    Optional[str],
    processors:  List[Any],
    config:      ProcessingConfig,
    file_path:   str,
    image_meta:  Dict[str, Any],
) -> MemoryChunkResult:
    """Run all processors on one materialised memory chunk; return a MemoryChunkResult."""
    arr        = mem_record.data
    dim_order  = mem_record.dim_order
    mem_origin = tuple(mem_record.meta.get(f"dim_{d.lower()}", 0) for d in dim_order)

    # MEMORY pass
    chunk_rows: Dict[str, dict] = {}
    timing: Dict[str, float] = {}
    for proc in processors:
        if getattr(proc, "CHUNK_KIND", None) != ChunkKind.MEMORY:
            continue
        if not is_record_matching_processor(mem_record, proc.INPUT):
            continue
        try:
            _t = time.perf_counter()
            row = proc.run_chunk(mem_record)
            timing[f"proc_{proc.NAME}"] = timing.get(f"proc_{proc.NAME}", 0.0) + (time.perf_counter() - _t)
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
                _t = time.perf_counter()
                row = proc.run_chunk(leaf_record)
                timing[f"proc_{proc.NAME}"] = timing.get(f"proc_{proc.NAME}", 0.0) + (time.perf_counter() - _t)
            except Exception as exc:
                logger.warning("worker: %s raised on leaf of %s: %s", proc.NAME, file_path, exc)
                continue
            if row:
                leaf_row.update(row)
        if leaf_row:
            _stamp_coordinates_to_row(leaf_row, dim_order, leaf_global_origin, leaf_arr.shape)
            leaf_rows.append(leaf_row)

    return MemoryChunkResult(
        file_index=file_index, child_id=child_id,
        chunk_rows=chunk_rows, leaf_rows=leaf_rows,
        image_meta=image_meta, timing=timing,
    )


def _execute_batch_task(
    task:       BatchTask,
    loader:     Any,
    processors: List[Any],
    config:     ProcessingConfig,
) -> List[MemoryChunkResult]:
    """Process every file in the task as one complete memory chunk each."""
    results: List[MemoryChunkResult] = []
    for idxed_path in task.files:
        _t = time.perf_counter()
        record = loader.load(Path(idxed_path.file_path))
        load_s = time.perf_counter() - _t
        if record is None:
            logger.warning("worker: loader returned None for %s; skipping", idxed_path.file_path)
            continue
        image_meta = _extract_image_meta(record)
        mem_record = _build_record(record, record.data, tuple(0 for _ in record.dim_order))
        result = _process_memory_chunk(mem_record, idxed_path.file_index, None, processors, config,
                                       idxed_path.file_path, image_meta=image_meta)
        result.timing["load"] = result.timing.get("load", 0.0) + load_s
        results.append(result)
    return results


def _execute_memory_chunk_task(
    task:       MemoryChunkTask,
    loader:     Any,
    processors: List[Any],
    config:     ProcessingConfig,
) -> List[MemoryChunkResult]:
    """Load the sub-region defined by task.spec and process it."""
    _t = time.perf_counter()
    record = loader.load(Path(task.file_path))
    load_s = time.perf_counter() - _t
    if record is None:
        logger.warning("worker: loader returned None for %s; skipping", task.file_path)
        return []
    # Extract image_meta from the full record BEFORE slicing — record.data.shape
    # here is the full image shape; task.spec.image_shape is the same value.
    image_meta = _extract_image_meta(record)
    mem_record = _build_record(record, record.data[task.spec.slices], task.spec.origin)
    result = _process_memory_chunk(mem_record, task.file_index, None, processors, config,
                                   task.file_path, image_meta=image_meta)
    result.timing["load"] = result.timing.get("load", 0.0) + load_s
    return [result]


def _execute_container_task(
    task:       ContainerTask,
    loader:     Any,
    processors: List[Any],
    config:     ProcessingConfig,
) -> List[MemoryChunkResult]:
    """Load and process a batch of sub-images from a container file."""
    results: List[MemoryChunkResult] = []
    start, stop = task.image_slice
    for child_id, record in loader.load_range(Path(task.file_path), start, stop):
        if record is None:
            logger.warning("worker: loader returned None for sub-image %s in %s; skipping",
                           child_id, task.file_path)
            continue
        # Each sub-image record carries its own metadata (shape, dtype, pixel sizes, …).
        image_meta = _extract_image_meta(record)
        mem_record = _build_record(record, record.data, tuple(0 for _ in record.dim_order))
        result = _process_memory_chunk(mem_record, task.file_index, child_id, processors, config,
                                       task.file_path, image_meta=image_meta)
        results.append(result)
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
    Power-set over non-degenerate (varying) dims.

    Every returned row is augmented with image_meta so it is
    available at all obs_levels.  image_meta values take priority over
    leaf-block-stamped *_size (which reflect the leaf block size, not the full
    image), so callers always see the full-image extents.
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
    active_dim_set = set(active_dims)

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
            # Strip degenerate (non-varying) dim coordinates from leaf rows.
            # Dims not in active_dims have a single constant value (e.g. dim_x=0
            # when X/Y are full-extent leaves).  Leaving them as 0 instead of
            # null breaks the viewer's obs_level-based per-dim queries, which
            # use `dim_x IS NULL` to identify pre-aggregated rows, and creates
            # spurious single-slice entries in the dim-filter dropdowns.
            filtered = {
                k: v for k, v in row.items()
                if not k.startswith("dim_") or k in active_dim_set
            }
            obs_rows.append({**filtered, "obs_level": n})

    # Merge image-level metadata into every obs row.
    # {**row, **image_meta} lets image_meta override leaf-stamped *_size values
    # (leaf blocks stamp S_size=1 per block; image_meta carries the full count).
    # image_meta never contains dim_* coords, num_pixels, or obs_level, so
    # all PP-computed per-row fields survive unchanged.
    image_meta: Dict[str, Any] = chunk_results[0].image_meta if chunk_results else {}
    return [{**row, **image_meta} for row in obs_rows]


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
    # Aggregate rows (obs_level < n) have None for dim_* columns that only become
    # non-null in later per-dim or leaf rows.  Scan all rows so polars resolves
    # the common supertype (nullable Int64) rather than locking in Null too early.
    return pl.from_dicts(rows, infer_schema_length=len(rows))


def _is_blob_dtype(dtype) -> bool:
    return dtype == pl.Binary or isinstance(dtype, (pl.List, pl.Array))


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

    # Dtype shrink / downcast; fixed-size list promotion for known fixed-width columns.
    casts = []
    for col in df.columns:
        dtype = df.schema[col]
        if dtype.is_integer() and col not in _INT_SHRINK_EXEMPT:
            new_dtype = df[col].shrink_dtype().dtype
            if new_dtype != dtype:
                casts.append(pl.col(col).cast(new_dtype))
        elif dtype == pl.Float64:
            casts.append(pl.col(col).cast(pl.Float32))
        elif col == "histogram_counts" and isinstance(dtype, pl.List):
            casts.append(pl.col(col).list.to_array(HISTOGRAM_BINS))
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
        self._total_rows     = 0
        self._t_start        = time.monotonic()

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
        rows_this_part = self._buffer_rows
        pl.concat(self._buffer, how="diagonal_relaxed").write_parquet(part_path, **write_kwargs)
        self._part_paths.append(part_path)
        self._total_rows  += rows_this_part
        self._buffer      = []
        self._buffer_rows = 0
        elapsed = time.monotonic() - self._t_start
        rate    = self._total_rows / elapsed if elapsed > 0 else 0
        logger.info(
            "Part %04d written: %d rows | cumulative %d rows | elapsed %.1fs | avg %.0f rows/s",
            self._part_counter, rows_this_part, self._total_rows, elapsed, rate,
        )
        self._part_counter += 1

    def finalize(self) -> Optional[pl.DataFrame]:
        """Flush remaining buffer and signal what to do next.

        Returns:
          - None          if parts were written to disk (caller must call
                          save_parquet_from_parts to stream-merge them).
          - pl.DataFrame  if everything is in-memory (parts_dir=None path),
                          with _post_process already applied.
          - empty DF      if nothing was processed at all.

        The previous collect()-all-parts path is intentionally removed: for
        datasets large enough to spill to parts, loading all parts into a single
        DataFrame causes OOM.  See save_parquet_from_parts for the streaming path.
        """
        self.flush()
        if not self._part_paths and not self._buffer:
            logger.warning("Finalize: no parts and no buffer — returning empty DataFrame")
            return pl.DataFrame()
        if self._part_paths:
            # Data is on disk — do not collect; caller streams from parts_dir.
            return None
        # In-memory path (parts_dir=None, small dataset).
        t0 = time.monotonic()
        combined = pl.concat(self._buffer, how="diagonal_relaxed")
        logger.info("Finalize: concat done in %.1fs — %d rows, post-processing ...", time.monotonic() - t0, len(combined))
        result = _post_process(combined)
        logger.info("Finalize: complete — %d rows total, elapsed %.1fs", len(result), time.monotonic() - self._t_start)
        return result


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
    n_workers:    int = 1,
    is_distributed: bool = False,
) -> Tuple[Optional[pl.DataFrame], Dict[str, Any]]:
    """Drive the full submit → gather → rollup → join → accumulate cycle.

    Maintains a MAX_PENDING_TASKS sliding window of Dask futures so workers
    stay busy while _plan_tasks is still generating tasks from the file stream.
    Returns (result_df, stats_dict) where stats_dict carries timing and throughput data.
    """
    max_pending = 2 * sum(client.nthreads().values())
    writer      = _ResultsWriter(config.rows_per_part, parts_dir, config.parquet_row_group_size)
    assembler   = _RecordAssembler()
    future_to_task: Dict[Any, Task] = {}
    completed_records = 0
    n_tasks_submitted = 0
    task_type_counts:  Dict[str, int]   = {}
    all_timing:        Dict[str, float] = {}
    error_records     = 0
    t_pipeline_start  = time.monotonic()
    _LOG_EVERY        = 200

    # Count memory-pressure pause events from Dask's distributed logger.
    class _PauseCounter(logging.Handler):
        def __init__(self) -> None:
            super().__init__(logging.WARNING)
            self.count = 0
        def emit(self, record: logging.LogRecord) -> None:
            if "Pausing worker" in record.getMessage():
                self.count += 1
    _pause_handler = _PauseCounter()
    logging.getLogger("distributed.worker.memory").addHandler(_pause_handler)

    # Scatter the loader and processors to all workers once so they are not
    # re-serialised with every task submission.  broadcast=True sends the data
    # to every worker process upfront; Dask resolves each Future to the actual
    # object on the worker side when the task runs.
    # Note: client.scatter(list) scatters each element individually, so we wrap
    # processors in a single-element list and unpack with [0] to scatter the
    # whole list as one object.
    worker_info = client.scheduler_info().get("workers", {})
    logger.info(
        "Pipeline started: %d workers, max_pending=%d, rows_per_part=%d",
        len(worker_info), max_pending, config.rows_per_part,
    )
    _worker_log = logger.info if is_distributed else logger.debug
    for addr, w in worker_info.items():
        _worker_log("  worker %s  memory_limit=%.2f GiB  nthreads=%d",
                    addr, (w.get("memory_limit") or 0) / 2**30, w.get("nthreads", 1))

    loader_ref     = client.scatter(loader,        broadcast=True)
    processors_ref = client.scatter([processors],  broadcast=True)[0]

    _executor_for: Dict[type, Any] = {
        BatchTask:       _execute_batch_task,
        MemoryChunkTask: _execute_memory_chunk_task,
        ContainerTask:    _execute_container_task,
    }

    def _submit(t: Task) -> Any:
        nonlocal n_tasks_submitted
        f = client.submit(_executor_for[type(t)], t, loader_ref, processors_ref, config, pure=False)
        future_to_task[f] = t
        n_tasks_submitted += 1
        key = type(t).__name__
        task_type_counts[key] = task_type_counts.get(key, 0) + 1
        return f

    pbar = tqdm(
        unit=" images",
        unit_scale=False,
        desc="Processing",
        disable=on_progress is not None,
        dynamic_ncols=True,
        bar_format="{desc}: {n_fmt}{unit} [{elapsed}, {rate_fmt}]",
    )

    def _update_pbar_postfix() -> None:
        vm = psutil.virtual_memory()
        pbar.set_postfix({"RAM": f"{vm.used / 1024**3:.1f}/{vm.total / 1024**3:.1f} GB"}, refresh=False)

    def _handle_record(
        chunk_results: List[MemoryChunkResult],
        file_index:    int,
        child_id:      Optional[str],
    ) -> None:
        nonlocal completed_records
        for cr in chunk_results:
            for k, v in cr.timing.items():
                all_timing[k] = all_timing.get(k, 0.0) + v
        obs_rows = _rollup(chunk_results, processors)
        if obs_rows:
            writer.add(_join_file_metadata(obs_rows, files_meta[file_index], child_id))
        completed_records += 1
        pbar.update(1)
        _update_pbar_postfix()
        if on_progress:
            on_progress(completed_records, -1)

    t_wall_start = time.perf_counter()
    task_iter       = iter(task_stream)
    initial_futures = [_submit(t) for t in itertools.islice(task_iter, max_pending)]
    if not initial_futures:
        pbar.close()
        logger.warning("Pipeline: no tasks were generated — nothing to process")
        return None, {}
    logger.info("Pipeline: %d initial futures submitted", len(initial_futures))

    ac = as_completed(initial_futures)
    all_submitted = False
    for future in ac:
        task = future_to_task.pop(future)

        next_task = next(task_iter, None)
        if next_task is not None:
            ac.add(_submit(next_task))
        elif not all_submitted:
            all_submitted = True
            pbar.total = len(files_meta)
            pbar.refresh()

        try:
            results: List[MemoryChunkResult] = future.result()
        except Exception as exc:
            # Emit one user-visible warning per affected file path.
            if isinstance(task, BatchTask):
                for ip in task.files:
                    logger.warning("Skipping '%s' — worker error: %s", ip.file_path, exc)
                    error_records += 1
            else:
                logger.warning("Skipping '%s' — worker error: %s", task.file_path, exc)
                error_records += 1
            continue

        before = completed_records
        if isinstance(task, MemoryChunkTask):
            result = results[0]
            if assembler.add(result, task.n_memory_chunks):
                key = (result.file_index, result.child_id)
                _handle_record(assembler.pop(key), result.file_index, result.child_id)
        else:
            for result in results:
                _handle_record([result], result.file_index, result.child_id)

        if completed_records > before and completed_records % _LOG_EVERY == 0:
            elapsed = time.monotonic() - t_pipeline_start
            rate    = completed_records / elapsed if elapsed > 0 else 0
            logger.info(
                "Progress: %d images done | %.1f img/s | %d errors | %d futures pending | %d parts written",
                completed_records, rate, error_records, len(future_to_task), writer._part_counter,
            )

    pbar.close()
    wall_s = time.perf_counter() - t_wall_start
    elapsed_total = time.monotonic() - t_pipeline_start
    logger.info(
        "Pipeline done: %d images, %d errors, %d parts, %.1fs total (%.1f img/s avg)",
        completed_records, error_records, writer._part_counter,
        elapsed_total, completed_records / elapsed_total if elapsed_total > 0 else 0,
    )

    logging.getLogger("distributed.worker.memory").removeHandler(_pause_handler)

    try:
        worker_info = client.scheduler_info().get("workers", {})
        n_workers_actual = len(worker_info)
        worker_nodes = sorted({addr.split("//")[-1].split(":")[0] for addr in worker_info})
    except Exception:
        n_workers_actual = n_workers
        worker_nodes = []

    try:
        rss_by_worker = client.run(lambda: psutil.Process().memory_info().rss)
        peak_worker_rss_mb = max(v / 1024 / 1024 for v in rss_by_worker.values()) if rss_by_worker else 0.0
    except Exception:
        peak_worker_rss_mb = 0.0

    stats: Dict[str, Any] = {
        "wall_s":                    wall_s,
        "n_files":                   len(files_meta),
        "n_tasks":                   n_tasks_submitted,
        "task_types":                task_type_counts,
        "n_workers":                 n_workers_actual,
        "worker_nodes":              worker_nodes,
        "is_distributed":            is_distributed,
        "peak_worker_rss_mb":        round(peak_worker_rss_mb, 1),
        "n_memory_pressure_events":  _pause_handler.count,
        "load_cpu_s":                all_timing.get("load", 0.0),
        **{k: v for k, v in all_timing.items() if k.startswith("proc_")},
    }
    result_df = writer.finalize()
    if result_df is None:
        return None, stats  # parts on disk; caller uses save_parquet_from_parts
    return (result_df if len(result_df) > 0 else None), stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Streaming parquet writer  — used when parts were spilled to disk
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _null_columns_from_stats(parts: List[Path]) -> Set[str]:
    """Return column names that are all-null across every part, using parquet statistics.

    Reads only file metadata (no row data).  Columns whose statistics are absent
    are conservatively treated as non-null (kept).
    """
    null_count: Dict[str, int] = {}
    row_count:  Dict[str, int] = {}
    for p in parts:
        try:
            meta = pq.read_metadata(p)
        except Exception:
            return set()
        for rg_i in range(meta.num_row_groups):
            rg      = meta.row_group(rg_i)
            n_rows  = rg.num_rows
            for col_i in range(rg.num_columns):
                col_meta = rg.column(col_i)
                name     = col_meta.path_in_schema
                stats    = col_meta.statistics
                if stats is not None and stats.null_count is not None:
                    null_count[name] = null_count.get(name, 0) + stats.null_count
                    row_count[name]  = row_count.get(name,  0) + n_rows
    return {
        name for name, nc in null_count.items()
        if row_count.get(name, 0) > 0 and nc == row_count[name]
    }


def _int_shrink_from_stats(parts: List[Path], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Determine safe integer type shrinkage using parquet min/max statistics.

    Returns a mapping col_name → new_polars_dtype for columns where the global
    min/max fits a narrower integer type.  Columns without statistics are skipped.
    """
    int_cols = {c for c, dt in schema.items() if dt.is_integer() and c not in _INT_SHRINK_EXEMPT}
    if not int_cols:
        return {}
    global_min: Dict[str, Any] = {}
    global_max: Dict[str, Any] = {}
    for p in parts:
        try:
            meta = pq.read_metadata(p)
        except Exception:
            return {}
        for rg_i in range(meta.num_row_groups):
            rg = meta.row_group(rg_i)
            for col_i in range(rg.num_columns):
                col_meta = rg.column(col_i)
                name     = col_meta.path_in_schema
                if name not in int_cols:
                    continue
                stats = col_meta.statistics
                if stats is None or not stats.has_min_max:
                    continue
                try:
                    mn, mx = stats.min, stats.max
                    global_min[name] = min(global_min.get(name, mn), mn)
                    global_max[name] = max(global_max.get(name, mx), mx)
                except Exception:
                    continue
    result: Dict[str, Any] = {}
    for name, cur_dt in schema.items():
        if name not in global_min or not cur_dt.is_integer() or name in _INT_SHRINK_EXEMPT:
            continue
        test     = pl.Series(name, [global_min[name], global_max[name]], dtype=cur_dt)
        new_dt   = test.shrink_dtype().dtype
        if new_dt != cur_dt:
            result[name] = new_dt
    return result


def save_parquet_from_parts(
    parts:        List[Path],
    dest:         Path,
    metadata:     Any,        # ProjectMetadata
    row_group_size: Optional[int] = None,
) -> int:
    """Stream-merge part files into the final parquet, applying _post_process.

    Reads one part at a time so peak memory is ~max(part_size) not sum(parts).
    Handles diagonal_relaxed schema differences between parts: missing columns
    are filled with nulls, type conflicts resolved by the unified schema.

    Returns the total number of rows written.
    """

    if not parts:
        return 0

    # ── Step 1: unified schema via 0-row reads (reads only file metadata) ──
    empty_frames = [pl.read_parquet(p, n_rows=0) for p in parts]
    unified      = pl.concat(empty_frames, how="diagonal_relaxed")

    # ── Step 2: drop all-null columns (from stats, no data read) ──
    drop = _null_columns_from_stats(parts)
    keep = [c for c in unified.columns if c not in drop]
    unified = unified.select(keep)

    # ── Step 3: build target schema with dtype transforms ──
    int_targets = _int_shrink_from_stats(parts, dict(unified.schema))
    casts: List[Any] = []
    for col, dtype in unified.schema.items():
        if col in int_targets:
            casts.append(pl.col(col).cast(int_targets[col]))
        elif dtype == pl.Float64:
            casts.append(pl.col(col).cast(pl.Float32))
        elif col == "histogram_counts" and isinstance(dtype, pl.List):
            casts.append(pl.col(col).list.to_array(HISTOGRAM_BINS))
    if casts:
        unified = unified.with_columns(casts)

    # ── Step 4: column reorder (same logic as _post_process) ──
    dim_cols    = sorted(c for c in unified.columns if c.startswith("dim_"))
    blob_cols   = [c for c in unified.columns if _is_blob_dtype(unified.schema[c])]
    skip_set    = {"obs_level"} | set(dim_cols) | set(blob_cols)
    scalar_cols = [c for c in unified.columns if c not in skip_set]
    ordered     = [c for c in (["obs_level"] + dim_cols + scalar_cols + blob_cols)
                   if c in unified.columns]
    unified = unified.select(ordered)

    # ── Step 5: build target Arrow schema with footer metadata ──
    kv_meta      = metadata.to_parquet_meta()
    arrow_kv     = {k.encode(): v.encode() for k, v in kv_meta.items()}
    arrow_schema = unified.to_arrow().schema.with_metadata(arrow_kv)

    # ── Step 6: stream-write one part at a time ──
    rg_size    = row_group_size or 2048
    total_rows = 0
    with pq.ParquetWriter(dest, arrow_schema, compression="zstd") as writer:
        for p in parts:
            chunk = pl.read_parquet(p)

            # child_id → type annotation (same as _post_process step 1)
            if "child_id" in chunk.columns and "type" in chunk.columns:
                chunk = chunk.with_columns(
                    pl.when(pl.col("child_id").is_not_null())
                    .then(pl.lit("sub_file"))
                    .otherwise(pl.col("type"))
                    .alias("type")
                )

            # Fill columns missing from this part with nulls
            for col in ordered:
                if col not in chunk.columns:
                    chunk = chunk.with_columns(
                        pl.lit(None).cast(unified.schema[col]).alias(col)
                    )

            # Select ordered columns only (drop any extras from this part)
            chunk = chunk.select(ordered)

            # Cast to target schema (handles type widening from diagonal_relaxed)
            target_no_meta = arrow_schema.remove_metadata()
            arrow_chunk = chunk.to_arrow().cast(target_no_meta)
            writer.write_table(arrow_chunk, row_group_size=rg_size)
            total_rows += len(chunk)

    logger.info(
        "save_parquet_from_parts: %d rows, %d parts → '%s'",
        total_rows, len(parts), dest,
    )
    return total_rows


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cluster lifecycle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@contextmanager
def _get_or_create_client(config: ProcessingConfig) -> Generator[Tuple[Any, bool], None, None]:
    """Yield (client, is_distributed) where is_distributed=True for pre-existing clusters.

    Reuses an existing client if one is connected; otherwise spins up a temporary
    LocalCluster with config.max_workers workers and shuts it down on context exit.
    """
    try:
        client = get_client()
        logger.debug("_get_or_create_client: reusing existing client %s", client.scheduler_info()["address"])
        yield client, True
    except ValueError:
        logger.debug("_get_or_create_client: starting LocalCluster n_workers=%s", config.max_workers)
        # Ignore SIGINT before forking workers so they inherit SIG_IGN and don't
        # print tracebacks when the user presses Ctrl+C.  The parent restores its
        # own handler immediately after the cluster is started.
        _old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        cluster = LocalCluster(
            n_workers=config.max_workers,
            threads_per_worker=1,
            processes=True,
            env={
                "OMP_NUM_THREADS":      "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS":      "1",
                "NUMEXPR_MAX_THREADS":  "1",
                "NUMEXPR_NUM_THREADS":  "1",
            },
        )
        signal.signal(signal.SIGINT, _old_sigint)
        client = Client(cluster)
        client.run(_silence_numcodecs_warning)
        try:
            yield client, False
        finally:
            client.close()
            cluster.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _collect_file_metadata_only(
    bases:       List[Path],
    config:      ProcessingConfig,
    on_progress: Optional[Callable[[int, int], None]],
) -> Tuple[Optional[pl.DataFrame], Dict[str, Any]]:
    """No-loader mode: discover files and return their filesystem metadata only.

    Bypasses Dask entirely.  Each discovered file produces one row containing
    the standard file metadata columns
    """
    t0 = time.perf_counter()
    rows: List[dict] = []
    pbar = tqdm(unit=" files", unit_scale=False, desc="Scanning",
                disable=on_progress is not None, dynamic_ncols=True,
                bar_format="{desc}: {n_fmt}{unit} [{elapsed}, {rate_fmt}]")
    for _, file_meta in _discover_files(bases, config.selected_file_extensions):
        rows.append({"obs_level": 0, **file_meta})
        pbar.update(1)
        if on_progress:
            on_progress(len(rows), -1)
    pbar.close()

    stats: Dict[str, Any] = {
        "wall_s":                   time.perf_counter() - t0,
        "n_files":                  len(rows),
        "n_tasks":                  0,
        "task_types":               {},
        "n_workers":                0,
        "worker_nodes":             [],
        "is_distributed":           False,
        "peak_worker_rss_mb":       0.0,
        "n_memory_pressure_events": 0,
        "load_cpu_s":               0.0,
    }
    if not rows:
        return None, stats

    return _post_process(pl.from_dicts(rows)), stats


def build_records_df(
    bases:        List[Path],
    loader:       Any,
    processors:   List[Any],
    config:       Optional[ProcessingConfig] = None,
    parts_dir:    Optional[Path] = None,
    on_progress:  Optional[Callable[[int, int], None]] = None,
) -> Tuple[Optional[pl.DataFrame], Dict[str, Any]]:
    """Scan files, distribute processing across Dask workers, return (records_df, stats).

    When loader is None only filesystem metadata is collected.

    _discover_files, _plan_tasks, and the submit side of _coordinate_pipeline
    run concurrently via a streaming generator pipeline — workers receive their
    first tasks before the filesystem scan is complete.

    Args:
        bases:       Base directories to scan.
        loader:      Loader instance; must implement read_header, load, load_range.
                     Pass None for inventory-only mode (file metadata only).
        processors:  Processor instances to run on each record. Caller is responsible
                     for applying processors_included / processors_excluded filtering.
                     Ignored when loader is None.
        config:      Runtime options. Defaults used if None.
        parts_dir:   Directory for intermediate part files. None → in-memory only.
                     Ignored when loader is None.
        on_progress: Optional callback(done: int, total: int) called per completed record.
                     total is -1 until the full record count is known.

    Returns (records_df, stats_dict). records_df is None if no files found.
    stats_dict always contains wall_s, n_files, n_tasks, n_workers, load_cpu_s, proc_* keys.
    """
    cfg = config or ProcessingConfig()

    if loader is None:
        return _collect_file_metadata_only(bases, cfg, on_progress)

    with _get_or_create_client(cfg) as (client, is_distributed):
        n_workers = sum(client.nthreads().values())
        files_meta: List[dict] = []
        folder_exts = getattr(loader, "FOLDER_EXTENSIONS", None)
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
            n_workers=n_workers,
            is_distributed=is_distributed,
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
