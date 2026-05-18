import logging
import math
import os
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import List, Optional, Dict, Set, Iterable, Iterator, NamedTuple, Callable

import numpy as np
from tqdm.auto import tqdm
import gc

import dask.bag as db
from dask.diagnostics import ProgressBar

import polars as pl

from pixel_patrol_base.core.contracts import PixelPatrolLoader, PixelPatrolProcessor
from pixel_patrol_base.core.file_system import walk_filesystem
from pixel_patrol_base.plugin_registry import discover_loader, discover_processor_plugins
from pixel_patrol_base.utils.df_utils import (
    normalize_file_extension,
    postprocess_basic_file_metadata_df,
)
from pixel_patrol_base.core.specs import is_record_matching_processor
from pixel_patrol_base.config import COMBINE_HEADROOM_RATIO, MAX_INTERMEDIATE_FLUSHES
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.io.parquet_io import write_chunk


logger = logging.getLogger(__name__)


class _IndexedPath(NamedTuple):
    row_index: int
    path: str


def _process_file_batch(
    batch: List[_IndexedPath],
    loader_name: Optional[str],
    processor_classes: List[type],
) -> List[Dict[str, object]]:
    """Stateless bag worker: instantiate loader + processors and process a batch of files."""
    try:
        loader = discover_loader(loader_name) if loader_name else None
        if loader is None:
            logger.warning("Processing Core: no loader; skipping batch of %d file(s)", len(batch))
            return []
        processors = [cls() for cls in processor_classes]
        return _process_batch_locally(batch, loader, processors, False)
    except Exception:
        logger.exception("Processing Core: batch failed (first path: %s)",
                         batch[0].path if batch else "?")
        return []


class _BatchWorker:
    """Picklable callable for dask.distributed — avoids functools.partial serialisation issues."""

    def __init__(self, loader_name: Optional[str], processor_classes: List[type]):
        self.loader_name = loader_name
        self.processor_classes = processor_classes

    def __call__(self, batch: List[_IndexedPath]) -> List[Dict[str, object]]:
        import dask
        # Force synchronous scheduler for all internal da.compute() calls.
        # This worker runs inside a distributed worker process; without this,
        # internal dask computations (image loading, min/max) try to route back
        # through the distributed scheduler, which fails because tifffile's
        # internal locks cannot be serialised for network transport.
        with dask.config.set(scheduler='synchronous'):
            result = _process_file_batch(batch, self.loader_name, self.processor_classes)
        # Python's allocator holds onto freed numpy pages rather than returning them
        # to the OS, causing "unmanaged memory" to accumulate across batches.
        # gc.collect removes cyclic references; malloc_trim releases the held pages.
        gc.collect()
        try:
            import ctypes
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        return result


PATHS_DF_EXPECTED_SCHEMA = {  # TODO: delete or rename - as paths_df is retired.
    "path": pl.String,
    "name": pl.String,
    "type": pl.String,
    "parent": pl.String,
    "depth": pl.Int64,
    "size_bytes": pl.Int64,
    "modification_date": pl.Datetime(time_unit="us", time_zone=None),
    "file_extension": pl.String,
    "size_readable": pl.String,
    "imported_path": pl.String,
}


def _iter_indexed_batches(
    df: pl.DataFrame, batch_size: int
) -> Iterator[List[_IndexedPath]]:
    """Yield batches of indexed paths lazily to avoid materializing all rows when requesting batches from a large DataFrame.

    Args:
        df: Polars DataFrame with 'row_index' and 'path' columns.
        batch_size: Number of rows per batch.
    Yields:
        A list of _IndexedPath items representing the current batch.
    """
    if df.is_empty():
        return
    size = max(1, batch_size)
    for offset in range(0, df.height, size):
        slice_df = df.slice(offset, size).select(["row_index", "path"])
        batch = [_IndexedPath(int(row[0]), str(row[1])) for row in slice_df.iter_rows()]
        if batch:
            yield batch


def _process_batch_locally(
    batch: List[_IndexedPath],
    loader_instance: PixelPatrolLoader,
    processors: List[PixelPatrolProcessor],
    show_processor_progress: bool,
) -> List[Dict[str, object]]:
    """Run loader+processors on a batch inside the main process.

    Args:
        batch: A list of _IndexedPath items representing the current batch.
        loader_instance: An instance of PixelPatrolLoader to load files.
        processors: A list of PixelPatrolProcessor instances to process the loaded data.
        show_processor_progress: A boolean indicating whether to show progress during processing.
    Returns:
        A list of dictionaries containing the results from deep processing.
        The last entry is always a ``{"__timing__": {...}}`` sentinel for aggregation.
    """
    deep_rows: List[Dict[str, object]] = []
    timing: Dict[str, float] = {"load": 0.0, "n_files": 0}
    for item in batch:
        record_dicts = load_and_process_records_from_file(
            Path(item.path), loader_instance, processors, show_processor_progress,
            timing_out=timing,
        )
        timing["n_files"] = timing.get("n_files", 0) + 1
        for record_dict in record_dicts:
            if record_dict:
                deep_rows.append({"row_index": item.row_index, **record_dict})
    deep_rows.append({"__timing__": timing})
    return deep_rows


def _combine_batch_with_basic(
    basic: pl.DataFrame,
    batch: List[_IndexedPath],
    deep_rows: List[Dict[str, object]],
) -> pl.DataFrame:
    """Join deep batch results back to the corresponding basic rows by row_index.

    Args:
        basic: The full basic Polars DataFrame with all file metadata.
        batch: A list of _IndexedPath items representing the current batch.
        deep_rows: A list of dictionaries containing the results from deep processing.
    Returns:
        A Polars DataFrame combining the basic and deep data for the batch.
    """
    if not batch:
        return pl.DataFrame([])

    indices = pl.Series(
        (item.row_index for item in batch),
        dtype=pl.UInt32,
    )
    basic_batch = basic.select(pl.all().gather(indices))

    if not deep_rows:
        # Nothing deep to merge for this batch; return the basic rows as-is
        return basic_batch

    # polars will ignore our types here for all but lists/arrays
    deep_df = pl.DataFrame(
        deep_rows,
        nan_to_null=True,
        strict=False,
        infer_schema_length=None,
    )
    if deep_df.is_empty():
        return basic_batch

    deep_df = deep_df.with_columns(pl.col("row_index").cast(pl.Int64))
    overlap = [
        c for c in deep_df.columns if c in basic_batch.columns and c != "row_index"
    ]
    if overlap:
        # Drop overlapping columns from basic_batch so loader metadata takes precedence
        basic_batch = basic_batch.drop(overlap)

    return basic_batch.join(deep_df, on="row_index", how="left")


def _cleanup_partial_chunks_dir(flush_dir: Optional[Path], cleanup_combined_parquet: bool = True) -> None:
    """Remove intermediate parquet chunk files (records_batch_*.parquet) from a flush directory.

    Optionally also remove the combined ``records_df.parquet`` file when
    ``cleanup_combined_parquet`` is True. The function attempts to remove the
    directory if it becomes empty. Errors are logged but not raised.

    Args:
        flush_dir: Path to the directory containing partial chunk files. If ``None`` or
            the path does not exist, the function returns silently.
        cleanup_combined_parquet: If True, also attempt to remove ``records_df.parquet``.
    """
    # If the directory doesn't exist or no path provided, nothing to do.
    if not flush_dir or not flush_dir.exists():
        return

    removed_any = False
    for p in flush_dir.glob("records_batch_*.parquet"):
        try:
            p.unlink()
            removed_any = True
            logger.debug("Processing Core: removed partial chunk %s", p)
        except OSError as exc:
            logger.warning(
                "Processing Core: Could not remove partial chunk %s: %s", p, exc
            )

    removed_combined = False
    if cleanup_combined_parquet:
        combined_candidate = flush_dir / "records_df.parquet"
        try:
            if combined_candidate.exists():
                combined_candidate.unlink()
                removed_combined = True
                logger.info(
                    "Processing Core: removed combined records parquet %s", combined_candidate
                )
        except OSError as exc:
            logger.warning(
                "Processing Core: Could not remove combined records parquet %s: %s",
                combined_candidate,
                exc,
            )

    if removed_any or removed_combined:
        logger.info(
            "Processing Core: Removed partial records batches under %s", flush_dir
        )
        try:
            # try removing the dir if empty
            flush_dir.rmdir()
        except OSError:
            # If not empty or cannot remove, leave it in place.
            pass




def _hpc_open_array_info(path: str, loader_name: str) -> Optional[tuple]:
    """Open a file lazily (header only) and return (shape, dtype, dim_order_out), or None on failure."""
    try:
        loader = discover_loader(loader_name)
        result = loader.load(path)
        if result is None:
            return None
        record = result if not isinstance(result, dict) else next(iter(result.values()), None)
        if record is None:
            return None
        import dask.array as _da
        arr = _da.asarray(record.data)
        dim_order = [d.upper() for d in record.dim_order]
        y_ax, x_ax = dim_order.index('Y'), dim_order.index('X')
        arr = _da.moveaxis(arr, [y_ax, x_ax], [-2, -1])
        ns_dims = [d for d in dim_order if d not in ('Y', 'X')]
        return arr.shape, arr.dtype, ns_dims + ['Y', 'X']
    except Exception as exc:
        logger.warning('hpc plan: could not open %s — %s', path, exc)
        return None


def _hpc_build_tasks(
    basic: pl.DataFrame,
    loader_name: str,
    target_mb: float,
    slice_safe_classes: List[type],
    non_slice_safe_classes: List[type],
    max_files_per_batch: int = 50,
) -> tuple:
    """Route files into mixed task list. Small files batch; large files get a full-file task
    (non-slice-safe processors) plus per-slice tasks (slice-safe processors).

    Every file is opened lazily (header only) to get the actual array memory size.
    Returns (tasks, large_file_meta) where large_file_meta maps row_index to
    (full_shape, dim_order_out) needed for post-gather accumulation.
    """
    from pixel_patrol_image.plugins.processors.processor_block_utils import raster_slicing_plan as _sp

    tasks: List[Dict] = []
    large_file_meta: Dict[int, tuple] = {}
    batch: List[_IndexedPath] = []
    batch_size_mb = 0.0

    def _flush():
        if batch:
            tasks.append({'type': 'batch', 'batch': batch[:]})

    for row in basic.iter_rows(named=True):
        path, row_index = row['path'], int(row['row_index'])
        size_bytes = row.get('size_bytes') or 0

        info = _hpc_open_array_info(path, loader_name)
        if info is not None:
            shape, dtype, dim_order_out = info
            arr_mb = np.prod(shape) * dtype.itemsize / (1024 ** 2)
            logger.debug('hpc plan: %s  shape=%s  %.0f MB', path, shape, arr_mb)
            if arr_mb > target_mb:
                _flush(); batch.clear(); batch_size_mb = 0.0
                slices = _sp(shape, ''.join(dim_order_out), dtype, target_mb)
                large_file_meta[row_index] = (shape, dim_order_out)
                tasks.append({'type': 'full_file', 'row_index': row_index, 'path': path})
                for slc in slices:
                    origin = [s.start or 0 if isinstance(s, slice) else int(s) for s in slc]
                    tasks.append({'type': 'slice', 'row_index': row_index, 'path': path,
                                  'slc': slc, 'origin': origin, 'dim_order_out': dim_order_out,
                                  'full_shape': shape})
                continue

        batch.append(_IndexedPath(row_index, path))
        batch_size_mb += size_bytes / (1024 ** 2)
        if batch_size_mb >= target_mb or len(batch) >= max_files_per_batch:
            _flush(); batch.clear(); batch_size_mb = 0.0

    _flush()

    n_batch = sum(1 for t in tasks if t['type'] == 'batch')
    n_full = sum(1 for t in tasks if t['type'] == 'full_file')
    n_slice = sum(1 for t in tasks if t['type'] == 'slice')
    logger.debug('hpc_build_tasks: %d batch, %d full-file, %d slice tasks', n_batch, n_full, n_slice)
    return tasks, large_file_meta


class _HpcSliceWorker:
    """Process one tile chunk using all slice-safe processors via the run_slice interface."""

    def __init__(self, loader_name: str, slice_safe_classes: List[type]):
        self.loader_name = loader_name
        self.slice_safe_classes = slice_safe_classes

    def __call__(self, task: Dict) -> List[Dict]:
        import dask
        with dask.config.set(scheduler='synchronous'):
            result = self._process(task)
        gc.collect()
        try:
            import ctypes
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        return result

    def _process(self, task: Dict) -> List[Dict]:
        import dask.array as _da
        timing: Dict = {'load': 0.0, 'n_files': 1}
        rows: List[Dict] = []

        t0 = time.perf_counter()
        try:
            loader = discover_loader(self.loader_name)
            result = loader.load(task['path'])
        except Exception as exc:
            logger.info('hpc slice: loader failed %s: %s', task['path'], exc)
            rows.append({'__timing__': timing})
            return rows
        finally:
            timing['load'] += time.perf_counter() - t0

        if result is None:
            rows.append({'__timing__': timing})
            return rows
        record = result if not isinstance(result, dict) else next(iter(result.values()), None)
        if record is None:
            rows.append({'__timing__': timing})
            return rows

        arr = _da.asarray(record.data)
        dim_order = [d.upper() for d in record.dim_order]
        y_ax, x_ax = dim_order.index('Y'), dim_order.index('X')
        arr = _da.moveaxis(arr, [y_ax, x_ax], [-2, -1])

        slc = task['slc']
        origin = task['origin']
        dim_order_out = list(task['dim_order_out'])
        chunk = arr[slc].compute()

        for cls in self.slice_safe_classes:
            t_proc = time.perf_counter()
            try:
                tile_rows = cls().run_slice(chunk, list(origin), dim_order_out)
                for row in tile_rows:
                    row['__hpc_row_index__'] = task['row_index']
                    row['__proc__'] = cls.NAME
                rows.extend(tile_rows)
            except Exception as exc:
                logger.warning('hpc slice: %s failed for %s: %s', cls.NAME, task['path'], exc)
            finally:
                timing[f'proc_{cls.NAME}'] = timing.get(f'proc_{cls.NAME}', 0.0) + (time.perf_counter() - t_proc)

        rows.append({'__timing__': timing})
        return rows


def _build_deep_record_df(
    basic: pl.DataFrame,
    loader_instance: PixelPatrolLoader,
    processing_config: Optional[ProcessingConfig] = None,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
    flush_dir: Optional[Path] = None,
) -> tuple[pl.DataFrame, object]:
    if basic.is_empty():
        return pl.DataFrame([]), None

    config = processing_config or ProcessingConfig()
    total = basic.height
    processors = discover_processor_plugins()

    if processing_config:
        if processing_config.processors_included:
            processors = [p for p in processors if p.NAME in processing_config.processors_included]
        elif processing_config.processors_excluded:
            processors = [p for p in processors if p.NAME not in processing_config.processors_excluded]

    processor_classes = [proc.__class__ for proc in processors]
    loader_name = getattr(loader_instance, "NAME", None)
    target_mb = float(os.environ.get('PIXEL_PATROL_MAX_BLOCK_MB', '1024'))
    worker_count = _resolve_worker_count(config, total, target_mb=target_mb)
    flush_threshold = _resolve_flush_threshold(total, config)

    try:
        from dask.distributed import get_client as _get_client
        _get_client()
        _using_distributed = True
    except (ImportError, ValueError):
        _using_distributed = False

    slice_safe_classes = [cls for cls in processor_classes if getattr(cls, 'SLICE_SAFE', False)]
    non_slice_safe_classes = [cls for cls in processor_classes if not getattr(cls, 'SLICE_SAFE', False)]

    # Build HPC task plan when there are slice-safe processors and a loader is present.
    # Opens large files lazily (header only) to get actual array memory size; small files
    # are detected by disk size and batched without opening.
    _hpc_tasks: Optional[List[Dict]] = None
    _large_file_meta: Dict[int, tuple] = {}
    if loader_name and slice_safe_classes:
        try:
            target_mb = float(os.environ.get('PIXEL_PATROL_MAX_BLOCK_MB', '1024'))
            _hpc_tasks, _large_file_meta = _hpc_build_tasks(
                basic, loader_name, target_mb, slice_safe_classes, non_slice_safe_classes,
            )
        except Exception as _exc:
            logger.warning('hpc plan failed, falling back to regular batching: %s', _exc)
            _hpc_tasks = None

    _use_hpc = _hpc_tasks is not None and any(t['type'] == 'slice' for t in _hpc_tasks)

    # When using distributed without HPC plan, one task per file; locally, batch to amortise overhead.
    batch_size = 1 if (_using_distributed and not _use_hpc) else _resolve_batch_size(worker_count, total)

    accumulator = _RecordsAccumulator(flush_every_n=flush_threshold, flush_dir=flush_dir)
    if flush_dir:
        _cleanup_partial_chunks_dir(flush_dir, cleanup_combined_parquet=False)

    _tile_px = int(os.environ.get('PIXEL_PATROL_STATS_TILE_SIZE', '256'))
    _chunk_mb = float(os.environ.get('PIXEL_PATROL_MAX_BLOCK_MB', '1024'))

    if _use_hpc:
        n_batch_t = sum(1 for t in _hpc_tasks if t['type'] == 'batch')
        n_full_t  = sum(1 for t in _hpc_tasks if t['type'] == 'full_file')
        n_slice_t = sum(1 for t in _hpc_tasks if t['type'] == 'slice')
        parts = []
        if n_batch_t: parts.append(f"{n_batch_t} batch")
        if n_full_t:  parts.append(f"{n_full_t} full-file")
        if n_slice_t: parts.append(f"{n_slice_t} slice")
        print(f"  plan:   {total} file{'s' if total != 1 else ''}  →  {' + '.join(parts)} tasks"
              f"  ·  ≤{_chunk_mb:.0f} MB/chunk  ·  {_tile_px} px tiles")
        all_tasks = _hpc_tasks
    else:
        batches = list(_iter_indexed_batches(basic, batch_size))
        logger.debug("Processing Core: %d file(s), %d batch(es), %d worker(s)",
                     total, len(batches), worker_count)
        print(
            f"  plan:   {total} file{'s' if total != 1 else ''}"
            f"  ·  {len(batches)} task{'s' if len(batches) != 1 else ''}"
            f"  ·  {batch_size} file{'s' if batch_size != 1 else ''}/task"
            f"  ·  ≤{_chunk_mb:.0f} MB/chunk  ·  {_tile_px} px tiles"
        )
        all_tasks = [{'type': 'batch', 'batch': b} for b in batches]

    # Unified dispatch worker: routes each task to the right handler.
    class _DispatchWorker:
        def __init__(self, ln, all_cls, non_ss_cls, ss_cls):
            self.ln, self.all_cls, self.non_ss_cls = ln, all_cls, non_ss_cls
            self._slice_worker = _HpcSliceWorker(ln, ss_cls)

        def __call__(self, task):
            import dask
            if task['type'] == 'slice':
                return self._slice_worker(task)
            classes = self.non_ss_cls if task['type'] == 'full_file' else self.all_cls
            items = [_IndexedPath(task['row_index'], task['path'])] if task['type'] == 'full_file' \
                    else task['batch']
            with dask.config.set(scheduler='synchronous'):
                result = _process_file_batch(items, self.ln, classes)
            gc.collect()
            try:
                import ctypes
                ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
            except Exception:
                pass
            return result

    worker_fn = _DispatchWorker(loader_name, processor_classes, non_slice_safe_classes, slice_safe_classes)

    def _force_exit(sig, frame):
        raise SystemExit(1)

    t_wall_start = time.perf_counter()
    old_handler = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        bag = db.from_sequence(all_tasks, npartitions=len(all_tasks))
        try:
            if not _using_distributed:
                raise ValueError("no distributed client")
            from dask.distributed import get_client as _get_client, progress as _dist_progress
            _active_client = _get_client()
            futures = _active_client.map(worker_fn, all_tasks)
            _dist_progress(futures, notebook=False)
            # errors='skip' returns the exception object for killed/failed tasks
            # instead of raising, so partial results from surviving tasks are kept.
            all_results_raw = _active_client.gather(futures, errors='skip')
            n_failed = sum(1 for r in all_results_raw if isinstance(r, BaseException))
            if n_failed:
                logger.warning(
                    'gather: %d of %d tasks failed (workers killed or OOM); '
                    'partial results will be saved', n_failed, len(futures)
                )
            all_results: List[List[Dict]] = [
                r if not isinstance(r, BaseException) else []
                for r in all_results_raw
            ]
        except (ImportError, ValueError):
            scheduler = 'synchronous' if worker_count == 1 else 'processes'
            compute_kwargs = {} if worker_count == 1 else {'num_workers': worker_count}
            try:
                with ProgressBar(dt=1.0):
                    all_results = bag.map(worker_fn).compute(scheduler=scheduler, **compute_kwargs)
            except Exception as exc:
                logger.warning(
                    "Processing Core: parallel execution failed (%s). "
                    "Falling back to single-process execution.", exc,
                )
                with ProgressBar(dt=1.0):
                    all_results = bag.map(worker_fn).compute(scheduler='synchronous')
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, _force_exit)
        raise
    finally:
        signal.signal(signal.SIGINT, old_handler)

    wall_s = time.perf_counter() - t_wall_start

    # --- Post-gather: accumulate HPC slice rows, then merge and save ---
    all_timing: Dict[str, float] = {"load": 0.0, "n_files": 0}

    # Collect raw tile rows from slice tasks and strip timing sentinels from all results.
    raw_tile_rows_by_ridx: Dict[tuple, List[Dict]] = {}  # (row_index, proc_name) -> rows
    cleaned_results: List[tuple] = []   # (task, deep_rows)

    for task, raw_rows in zip(all_tasks, all_results):
        if raw_rows and isinstance(raw_rows[-1], dict) and "__timing__" in raw_rows[-1]:
            for k, v in raw_rows[-1]["__timing__"].items():
                if isinstance(v, (int, float)):
                    all_timing[k] = all_timing.get(k, 0.0) + v
            rows = raw_rows[:-1]
        else:
            rows = list(raw_rows)

        if task['type'] == 'slice':
            for row in rows:
                if "__hpc_row_index__" in row:
                    ridx = row.pop("__hpc_row_index__")
                    proc_name = row.pop("__proc__", "unknown")
                    raw_tile_rows_by_ridx.setdefault((ridx, proc_name), []).append(row)
        else:
            cleaned_results.append((task, rows))

    # Accumulate tile rows per (file, processor) — runs accumulate_power_set once per combo.
    # Groups by processor so each processor's rollup is correct and they're merged afterwards.
    accumulated_by_ridx: Dict[int, List[Dict]] = {}
    if raw_tile_rows_by_ridx:
        _ss_by_name = {cls.NAME: cls for cls in slice_safe_classes}
        for (ridx, proc_name), tile_rows in raw_tile_rows_by_ridx.items():
            proc_cls = _ss_by_name.get(proc_name)
            if proc_cls is None:
                continue
            shape, dim_order_out = _large_file_meta[ridx]
            try:
                acc_rows = proc_cls.accumulate_slice_rows(tile_rows, shape, dim_order_out)
            except Exception as exc:
                logger.warning('accumulate_slice_rows failed for %s/%s: %s', proc_name, ridx, exc)
                continue
            for row in acc_rows:
                row['row_index'] = ridx
            if ridx not in accumulated_by_ridx:
                accumulated_by_ridx[ridx] = acc_rows
            else:
                accumulated_by_ridx[ridx] = _merge_long_rows(accumulated_by_ridx[ridx], acc_rows)

    processed_count = 0
    with tqdm(total=total, desc="Saving results", unit="file",
              leave=True, colour="green", disable=progress_callback is not None) as progress:
        for task, deep_rows in cleaned_results:
            if task['type'] == 'full_file':
                ridx = task['row_index']
                if ridx in accumulated_by_ridx:
                    deep_rows = _merge_long_rows(deep_rows, accumulated_by_ridx[ridx])
                batch_items = [_IndexedPath(task['row_index'], task['path'])]
            else:
                batch_items = task['batch']

            batch_df = _combine_batch_with_basic(basic, batch_items, deep_rows)
            accumulator.add_batch(batch_df)
            progress.update(len(batch_items))
            processed_count += len(batch_items)
            if progress_callback:
                progress_callback(processed_count, total, Path(batch_items[-1].path))

    # Worker info: query scheduler right after gather while workers are still connected.
    n_workers_actual = 0
    worker_nodes: list = []
    if _using_distributed:
        try:
            from dask.distributed import get_client as _get_client
            _winfo = _get_client().scheduler_info().get('workers', {})
            n_workers_actual = len(_winfo)
            worker_nodes = sorted(set(
                a.split('//')[-1].split(':')[0] for a in _winfo
            ))
        except Exception:
            pass
    tasks_per_worker: dict = {}

    from pixel_patrol_base.core.processing_summary import ProcessingSummary
    summary = ProcessingSummary(
        n_files=total,
        wall_s=wall_s,
        worker_count=worker_count,
        is_distributed=_using_distributed,
        load_cpu_s=all_timing.get("load", 0.0),
        processor_cpu_s={k[5:]: v for k, v in all_timing.items() if k.startswith("proc_")},
        n_tasks=len(all_tasks),
        n_workers_actual=n_workers_actual,
        worker_nodes=worker_nodes,
        tasks_per_worker=tasks_per_worker,
    )

    return accumulator.finalize(), summary


def _row_coord_key(row: Dict) -> tuple:
    """Canonical coordinate tuple used to match rows from different processors.

    Only single-letter coordinate dim_* fields (dim_z, dim_c, dim_y, dim_x, …)
    are used — these are always exactly 5 characters ("dim_" + 1 letter) and carry
    scalar values (int or None).  Multi-character fields like dim_order (loader
    metadata) are excluded so they don't break dict-key hashing or row matching.
    """
    dim_items = tuple(sorted(
        (k, v) for k, v in row.items()
        if len(k) == 5 and k.startswith("dim_")
        and (v is None or isinstance(v, (int, float)))
    ))
    return (row.get("obs_level"),) + dim_items


def _merge_long_rows(existing: List[Dict], incoming: List[Dict]) -> List[Dict]:
    """Merge two lists of row dicts by coordinate key (obs_level + dim_* fields).

    Rows with matching coordinates are merged in place (incoming fields added to the
    existing row).  Unmatched incoming rows are appended.  This allows two processors
    that both return a long-format list — e.g. raster-image and channel-colocalization —
    to share the same rollup tree without one overwriting the other.
    """
    by_key: Dict[tuple, Dict] = {_row_coord_key(r): r for r in existing}
    result = list(existing)
    for r in incoming:
        key = _row_coord_key(r)
        if key in by_key:
            by_key[key].update(r)
        else:
            result.append(r)
    return result


def _extract_record_properties(
    rcd,
    processors: List[PixelPatrolProcessor],
    show_progress: bool,
    file_path: Optional[Path] = None,
    timing_out: Optional[Dict] = None,
) -> List[Dict]:
    """Run all matching processors on a single Record.

    Returns a list of long-format row dicts.  A processor returning a list
    provides the long-format rows; any other return is merged into every row
    as scalar metadata.
    """
    scalar: Dict = dict(rcd.meta)
    scalar.setdefault("obs_level", 0)
    long_rows: Optional[List[Dict]] = None
    global_only_scalar: Dict = {}  # from processors marked GLOBAL_ONLY — only goes in obs_level=0

    processor_iter: Iterable[PixelPatrolProcessor]
    if show_progress:
        processor_iter = tqdm(
            processors,
            desc="  Running processors for image: ",
            unit="proc",
            leave=False,
            colour="blue",
            position=1,
        )
    else:
        processor_iter = processors

    record_path = str(file_path) if file_path else "unknown"

    for P in processor_iter:
        if not is_record_matching_processor(rcd, P.INPUT):
            continue
        proc_name = P.NAME
        t0 = time.perf_counter()
        try:
            logger.debug("Processing Core: start processor=%s record=%s", proc_name, record_path)
            out = P.run(rcd)
            dt = time.perf_counter() - t0
            logger.debug(
                "Processing Core: done processor=%s record=%s (%.2fs)",
                proc_name,
                record_path,
                dt,
            )
            if isinstance(out, list):
                if out:  # empty list means "nothing to add" — don't discard prior rows
                    long_rows = _merge_long_rows(long_rows, out) if long_rows is not None else out
            elif getattr(P, 'GLOBAL_ONLY', False):
                global_only_scalar.update(out)
            else:
                scalar.update(out)
        except RuntimeError as err:
            logger.warning(
                "Processor %s skipped for record %s: %s",
                P.NAME,
                record_path,
                err,
            )
        except (ValueError, TypeError):
            logger.exception(
                "Processor %s failed for record %s",
                P.NAME,
                record_path,
            )
        finally:
            if timing_out is not None:
                k = f"proc_{proc_name}"
                timing_out[k] = timing_out.get(k, 0.0) + (time.perf_counter() - t0)

    if long_rows is not None:
        result = [{**scalar, **row} for row in long_rows]
        for row in result:
            if row.get('obs_level') == 0:
                row.update(global_only_scalar)
        return result
    return [{**scalar, **global_only_scalar}]


def load_and_process_records_from_file(
    file_path: Path,
    loader: PixelPatrolLoader,
    processors: List[PixelPatrolProcessor],
    show_processor_progress: bool = True,
    timing_out: Optional[Dict] = None,
) -> List[Dict]:
    """
    Load a file with the given loader, run all matching processors, and return
    combined metadata for each record in the file.

    Loaders may return:
    - A single Record (single-image files)
    - A Dict[str, Record] where keys are child identifiers for multi-image files - becomes `child_id` column

    Args:
        file_path: Path to the file to process.
        loader: An instance of PixelPatrolLoader to load the file.
        processors: A list of PixelPatrolProcessor instances to run on the loaded record(s).
        show_processor_progress: Is show progress bar in e.g. Cli
        timing_out: Optional mutable dict; loader and processor CPU-seconds are accumulated into it.
    Returns:
        A list of dicts with combined metadata from the loader and all applicable
        processors.  Empty list if the file cannot be loaded.
    """
    if not file_path.exists():
        logger.warning(f"File not found: '{file_path}'. Cannot extract metadata.")
        return []

    t_load = time.perf_counter()
    try:
        result = loader.load(str(file_path))
        if result is None:
            return []
    except Exception as e:
        logger.info(f"Loader '{loader.NAME}' failed with exception, skipping: {e}")
        return []
    finally:
        if timing_out is not None:
            timing_out["load"] = timing_out.get("load", 0.0) + (time.perf_counter() - t_load)

    result_list: List[Dict] = []

    if isinstance(result, dict):
        # Multi-image file: show progress over sub-images
        items = tqdm(
            result.items(),
            desc="  Processing sub-images",
            unit="img",
            leave=False,
            colour="blue",
            position=1,
        )
        for child_id, rcd in items:
            if isinstance(child_id, int):
                child_id = str(child_id)
            if not isinstance(child_id, str) or not child_id:
                logger.warning(
                    "Loader '%s' returned invalid child key %r for '%s'; skipping record.",
                    loader.NAME, child_id, file_path,
                )
                continue
            rows = _extract_record_properties(rcd, processors, False, file_path, timing_out=timing_out)
            for row in rows:
                row["child_id"] = child_id
            result_list.extend(rows)
    else:
        result_list.extend(_extract_record_properties(
            result, processors, show_processor_progress, file_path, timing_out=timing_out,
        ))

    return result_list

def build_records_df(
    bases: List[Path],
    loader: Optional[PixelPatrolLoader],
    processing_config: Optional[ProcessingConfig] = None,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
    flush_dir: Optional[Path] = None,
) -> tuple[Optional[pl.DataFrame], object]:
    """Build the full records DataFrame by scanning files and processing them.

    Args:
        bases: List of base directories to scan for files.
        loader: Optional PixelPatrolLoader instance to use for loading files.
        processing_config: Optional ProcessingConfig for processor selection.
        progress_callback: Optional callback `function(current: int, total: int, current_file: Path) -> None`
                            Called for each file processed during deep processing
        flush_dir: Optional Path to a directory for flushing intermediate parquet chunks during processing.
    Returns:
        Tuple of (records DataFrame or None, ProcessingSummary or None).
    """

    config = processing_config or ProcessingConfig()

    basic = _build_basic_file_df(
        bases, loader=loader, accepted_extensions=config.selected_file_extensions
    )
    if loader is None or basic is None:
        return basic, None

    basic = basic.with_row_index("row_index").with_columns(
        pl.col("row_index").cast(pl.Int64)
    )
    deep, summary = _build_deep_record_df(
        basic,
        loader,
        processing_config=config,
        progress_callback=progress_callback,
        flush_dir=flush_dir,
    )

    return deep, summary


def _build_basic_file_df(
    bases, loader, accepted_extensions
):
    """Build the basic file DataFrame by scanning the filesystem.

    Args:
        bases: list of base paths to scan
        loader: optional loader instance
        accepted_extensions: set or 'all'
    """
    basic = walk_filesystem(
        bases, loader=loader, accepted_extensions=accepted_extensions
    )
    if basic.is_empty():
        return None
    basic = postprocess_basic_file_metadata_df(normalize_file_extension(basic))

    return basic


# TODO: delete or rename as paths_df is retired
def count_file_extensions(paths_df: Optional[pl.DataFrame]) -> Dict[str, int]:
    """
    Counts file extensions from a Polars DataFrame containing file system paths.
    Returns a dictionary { 'extension': count, 'all_files': total_count } for all files.
    Files without extensions are discarded.

    Args:
        paths_df: The Polars DataFrame containing file system path data,
                  expected to have 'type' and 'file_extension' columns.
    Returns:
        A dictionary with file extension counts and a total count under 'all_files'.
        Returns {'all_files': 0} if paths_df is not available or empty.
    """
    if paths_df is None or paths_df.is_empty():
        logger.warning(
            "No paths DataFrame provided or it's empty. Cannot count file extensions."
        )
        return {"all_files": 0}

    required_cols = {"type", "file_extension"}
    if not required_cols.issubset(paths_df.columns):
        logger.error(
            f"Paths DataFrame is missing required columns for extension counting: {required_cols - set(paths_df.columns)}"
        )
        return {"all_files": paths_df.height}

    df_files = paths_df.filter(pl.col("type") == "file")
    df_files = df_files.filter(
        pl.col("file_extension").is_not_null() & (pl.col("file_extension") != "")
    )

    if df_files.is_empty():
        logger.info("No files with valid extensions found in the DataFrame.")
        return {"all_files": 0}

    grouped = df_files.group_by("file_extension").agg(pl.count().alias("count"))
    result = {
        row["file_extension"]: row["count"] for row in grouped.iter_rows(named=True)
    }
    result["all_files"] = df_files.height

    return result


_WORKER_MEMORY_OVERHEAD = 6.0  # conservative: input + processing intermediates + Python allocator headroom


def _resolve_worker_count(
    config: ProcessingConfig,
    total_rows: Optional[int] = None,
    target_mb: float = 1024.0,
) -> int:
    """Return the number of parallel workers, capped by CPUs, available RAM, and file count.

    When max_workers is not explicitly configured, the RAM limit is derived from
    available memory divided by (target_mb × overhead factor). This prevents OOM
    when many workers each load a large image chunk simultaneously.
    """
    cpu_count = os.cpu_count() or 1

    if config.processing_max_workers is not None:
        max_workers = max(1, min(config.processing_max_workers, cpu_count))
    else:
        ram_limit = _ram_worker_limit(target_mb)
        max_workers = max(1, min(cpu_count, ram_limit))
        logger.debug(
            "Processing Core: auto worker count = %d (CPUs=%d, RAM limit=%d at %.0f MB/worker)",
            max_workers, cpu_count, ram_limit, target_mb * _WORKER_MEMORY_OVERHEAD,
        )

    if total_rows is not None:
        if total_rows <= 1:
            return 1
        return min(max_workers, max(1, total_rows))
    return max_workers


def _ram_worker_limit(target_mb: float) -> int:
    """Max workers that fit in available RAM given the per-worker memory budget."""
    available = _estimate_available_memory_bytes()
    if available is None:
        return os.cpu_count() or 1
    budget_per_worker = target_mb * _WORKER_MEMORY_OVERHEAD * 1024 ** 2
    return max(1, int(available / budget_per_worker))


def _resolve_batch_size(worker_count: int, total_rows: int) -> int:
    """
    Choose how many files to pack into each dask task.

    Targets 4× worker_count tasks so the progress bar has enough steps to be
    meaningful while keeping batch sizes large enough to amortise the
    per-task pickle serialisation overhead.  Capped at 50 files per batch so
    results never grow large enough to slow down the process pool.

    Examples:
        57 files,  22 workers → batch=1  (57 tasks,  fine-grained progress)
        1k files,  22 workers → batch=11 (91 tasks,  ~4 per worker)
        200k files,22 workers → batch=50 (4k tasks, capped to avoid overhead)
    """
    target_tasks = worker_count * 4
    return max(1, min(50, total_rows // max(1, target_tasks)))


def _resolve_flush_threshold(total_rows: int, config: ProcessingConfig) -> int:
    """Sets the flush threshold from settings or defaults. Ensures non-negative and half the dataset size to enforce a checkpoint flush.

    Args:
        total_rows: Total number of rows/files to process.
        config: ProcessingConfig that may specify `records_flush_every_n`.
    Returns:
        An integer representing the flush threshold for records.
    """
    if total_rows <= 0:
        return 0
    value = config.records_flush_every_n
    # Preserve the ability to disable flushing when a non-positive value is configured.
    if value is None or value <= 0:
        return 0

    # Ensure at least one flush can occur for small datasets while capping at roughly half the dataset size.
    upper_bound = max(1, total_rows // 2)

    # Enforce a maximum count of intermediate flushes (e.g. <= MAX_INTERMEDIATE_FLUSHES).
    # If the user configured a very small flush threshold that would cause too many flushes,
    # we adjust it upward to keep the number of flushes reasonable and log a warning.
    min_allowed = max(1, math.ceil(total_rows / MAX_INTERMEDIATE_FLUSHES))
    if value < min_allowed:
        requested_flushes = math.ceil(total_rows / value) if value > 0 else float('inf')
        logger.warning(
            "Processing Core: Flushing this often on your dataset would result in %d flushes. "
            "This slows processing down. Your settings are being changed to result in %d flushes only.",
            requested_flushes,
            MAX_INTERMEDIATE_FLUSHES,
        )
        value = min_allowed

    return max(1, min(value, upper_bound))


def _estimate_available_memory_bytes() -> Optional[int]:
    """Estimate available system memory in bytes."""
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)

    except ImportError:
        # psutil not installed
        pass
    except (OSError, RuntimeError) as exc:
        logger.debug("psutil failed to read memory: %s", exc)

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * avail_pages)

    except (AttributeError, ValueError, OSError) as exc:
        logger.debug("sysconf failed to read memory: %s", exc)
        return None


def _reorder_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Splice dim_* columns immediately after obs_level; move binary/list columns last.

    Everything else stays in its natural insertion order, so loader and processor
    columns appear where they already land without any hardcoded name lists.
    """
    blob_cols = [
        c for c in df.columns
        if df[c].dtype == pl.Binary or isinstance(df[c].dtype, (pl.List, pl.Array, pl.Struct))
    ]
    blob_set = set(blob_cols)
    dim_cols = sorted(c for c in df.columns if c.startswith("dim_") and c not in blob_set)
    dim_set = set(dim_cols)

    ordered: list[str] = []
    for c in df.columns:
        if c in dim_set or c in blob_set:
            continue
        ordered.append(c)
        if c == "obs_level":
            ordered.extend(dim_cols)

    if dim_set and "obs_level" not in df.columns:
        ordered = dim_cols + ordered

    ordered.extend(blob_cols)
    return df.select(ordered)


class _RecordsAccumulator:
    """Collect rows, batch-convert to Polars, and optionally flush to disk."""

    def __init__(
        self,
        flush_every_n: int,
        flush_dir: Optional[Path],
        combine_headroom_ratio: float = COMBINE_HEADROOM_RATIO,
    ) -> None:
        self._flush_every_n = flush_every_n
        self._flush_dir = Path(flush_dir) if flush_dir else None
        self._active_df = pl.DataFrame([])
        self._written_files: List[Path] = []
        self._chunk_index = 0
        self._combine_headroom_ratio = combine_headroom_ratio
        if self._flush_dir:
            logger.debug(
                "Processing Core: buffering %s rows per chunk; partial batches go to '%s'",
                self._flush_every_n,
                self._flush_dir,
            )

    @property
    def flush_dir(self) -> Optional[Path]:
        """Public accessor for the configured flush directory (avoids accessing protected attribute)."""
        return self._flush_dir

    def add_batch(self, batch: pl.DataFrame) -> None:
        """
        Add a batch of records to the accumulative active dataframe.
        Flushes active dataframe to a Parquet chunk if the active dataframe exceeds `self._flush_every_n`.

        Args:
            batch (pl.DataFrame): A Polars DataFrame containing the batch of records to add.
        """
        if batch is None or batch.is_empty():
            return

        if self._active_df.is_empty():
            self._active_df = batch
        else:
            self._active_df = pl.concat(
                [self._active_df, batch],
                how="diagonal_relaxed",
                rechunk=False,
            )

        if (
            self._flush_dir
            and self._flush_every_n > 0
            and self._active_df.height >= self._flush_every_n
        ):
            self._flush_active_to_disk(force=False)

    def finalize(self) -> pl.DataFrame:
        """
        Finalize accumulation: flush remaining active data and combine all chunks into one Parquet file if applicable, while removing partial chunks.
        If insufficient memory is available for combining, skips the combine step and retains chunk files.

        Returns:
            A Polars DataFrame containing all accumulated records, or an empty DataFrame if combining was skipped.
        """
        if self._flush_dir:
            if not self._active_df.is_empty():
                self._flush_active_to_disk(force=True)
        elif not self._active_df.is_empty():
            return self._active_df

        if not self._written_files:
            return self._active_df

        total_size = sum(p.stat().st_size for p in self._written_files if p.exists())
        if not self._is_enough_memory_to_combine(total_size):
            return pl.DataFrame([])

        frames = [pl.read_parquet(path) for path in self._written_files]
        if not self._active_df.is_empty():
            frames.append(self._active_df)

        if not frames:
            return pl.DataFrame([])

        final_df = pl.concat(frames, how="diagonal_relaxed", rechunk=True)
        final_df = self.post_process_final_df(final_df)

        return final_df

    def _is_enough_memory_to_combine(self, total_size: int) -> bool:
        available_memory = _estimate_available_memory_bytes()
        if available_memory is None:
            return True
        projected = total_size * self._combine_headroom_ratio
        if projected > available_memory:
            logger.warning(
                "Processing Core: skipping final combine; chunk files remain at %s (total %.2f MB, available %.2f MB)",
                self._flush_dir,
                total_size / (1024 * 1024),
                available_memory / (1024 * 1024),
            )
            return False
        return True

    @staticmethod
    def post_process_final_df(final_df):
        if "row_index" in final_df.columns:
            final_df = final_df.drop("row_index")
        if "child_id" in final_df.columns:
            final_df = final_df.with_columns(
                pl.when(pl.col("child_id").is_not_null())
                .then(pl.lit("sub_file"))
                .otherwise(pl.col("type"))
                .alias("type")
            )
        cols_to_drop = [
            col for col in final_df.columns
            if final_df[col].is_null().all()
            or (final_df[col].dtype == pl.Utf8 and (final_df[col].is_null() | (final_df[col] == "")).all())
        ]
        if cols_to_drop:
            final_df = final_df.drop(cols_to_drop)
        try:
            final_df = optimize_dtypes(final_df)
        except Exception as exc:
            logger.exception("Processing Core: dtype shrinking failed; continuing without dtype shrink: %s", exc)
        final_df = _reorder_columns(final_df)
        return final_df

    def _flush_active_to_disk(self, force: bool) -> None:
        """Flushes the active dataframe to disk as a parquet chunk file.

        Args:
            force (bool): Whether to force flushing regardless of the active dataframe size.
        """
        if self._active_df.is_empty():
            return
        if self._flush_dir is None:
            logger.info("Flush directory is not set. Skipping flush to disk.")
            return
        if not force and self._active_df.height < self._flush_every_n:
            return

        logger.debug(f"Flushing active DataFrame to disk at chunk index {self._chunk_index}")
        chunk_filename = f"records_batch_{self._chunk_index:05d}.parquet"

        chunk_path = write_chunk(
            self._active_df,
            self._flush_dir / chunk_filename,
            compression="zstd",
        )
        if chunk_path is None:
            logger.warning(
                "Processing Core: failed to write partial chunk %s", chunk_filename
            )
            return

        logger.debug("Processing Core: Writing partial records chunk to %s", chunk_path)

        self._written_files.append(chunk_path)

        # Release reference to active dataframe and GC to free memory.
        self._active_df = pl.DataFrame([])
        try:
            gc.collect()
        except Exception as e:
            logger.warning(
                "Processing Core: unexpected gc.collect() error: %s",
                e,
                exc_info=True,
            )

        self._chunk_index += 1

    def load_existing_chunks(self) -> Set[int]:
        """Load existing parquet chunk files from the flush directory.

        Populates `_written_files` and `_chunk_index` and returns a set of already-processed
        `row_index` values so the caller can skip re-processing those rows.

        Returns:
            A set of integers representing the `row_index` values of already-processed records.
        """
        processed: Set[int] = set()
        if self._flush_dir is None or not self._flush_dir.exists():
            return processed

        files = sorted(self._flush_dir.glob("records_batch_*.parquet"))
        if not files:
            return processed

        # adopt existing files and set next chunk index
        valid_files: List[Path] = []
        max_idx = -1
        for f in files:
            stem = f.stem  # e.g. records_batch_00001
            try:
                idx = int(stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                # ignore files with unexpected names
                continue

            try:
                df = pl.read_parquet(f, columns=["row_index"])
                if not df.is_empty() and "row_index" in df.columns:
                    processed.update(int(val) for val in df["row_index"].to_list())
                valid_files.append(f)
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                logger.warning(
                    "Processing Core: could not read existing chunk %s. This file will be skipped; the chunk may be overwritten on subsequent runs. "
                    "If you still experience issues when viewing reports, consider removing or inspecting it manually.",
                    f,
                )

        self._written_files = valid_files

        self._chunk_index = max_idx + 1
        logger.info(
            "Processing Core: found %d existing partial chunk(s) (next chunk index=%d)",
            len(self._written_files),
            self._chunk_index,
        )
        return processed


def optimize_dtypes(
    df: pl.DataFrame,
    shrink_floats: bool = True,
) -> pl.DataFrame:
    """
    Shrink integer dtypes (via Series.shrink_dtype()) and downcast floats safely.
    """
    # Columns that must stay Int64 because their per-row values may fit in a
    # smaller type but aggregates (sums across many files) would overflow.
    _NO_SHRINK = {"size_bytes"}

    series_updates = []
    for col_name in df.columns:
        s = df[col_name]
        dtype = s.dtype

        if dtype.is_integer():
            if col_name not in _NO_SHRINK:
                shrunk = s.shrink_dtype()
                if shrunk.dtype != dtype:
                    series_updates.append(shrunk)
            continue

        if shrink_floats and dtype == pl.Float64:
            series_updates.append(s.cast(pl.Float32))
            continue

    return df.with_columns(series_updates) if series_updates else df


def cleanup_flush_dir(flush_dir: Optional[Path]) -> None:
    """Remove the flush directory after a successful save.

    Should be called by the caller (e.g. project.py) after save_parquet succeeds,
    ensuring chunks are only deleted once the final parquet is safely written.
    Args:
        flush_dir: Path to the directory to remove. No-op if None or does not exist.
    """
    if not flush_dir or not flush_dir.exists():
        return
    try:
        shutil.rmtree(flush_dir)
        logger.info("Processing Core: Removed intermediate batches directory '%s'.", flush_dir)
    except OSError as exc:
        logger.warning("Processing Core: Could not remove '%s': %s", flush_dir, exc)
