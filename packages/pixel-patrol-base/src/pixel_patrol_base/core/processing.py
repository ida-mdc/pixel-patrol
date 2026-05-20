from __future__ import annotations

import gc
import logging
import math
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Set

import polars as pl
from tqdm.auto import tqdm

from pixel_patrol_base.config import COMBINE_HEADROOM_RATIO, DEFAULT_CHUNK_EVERY_N
from pixel_patrol_base.core.contracts import PixelPatrolLoader, PixelPatrolProcessor
from pixel_patrol_base.core.file_system import walk_filesystem
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.specs import is_record_matching_processor
from pixel_patrol_base.io.parquet_io import write_chunk
from pixel_patrol_base.plugin_registry import discover_loader, discover_processor_plugins
from pixel_patrol_base.utils.df_utils import (
    normalize_file_extension,
    postprocess_basic_file_metadata_df,
)


logger = logging.getLogger(__name__)

# Per-process context populated by _process_worker_initializer in each subprocess.
_PROCESS_WORKER_CONTEXT: Dict[str, object] = {}


class _IndexedPath(NamedTuple):
    row_index: int
    path: str


# ---------------------------------------------------------------------------
# Subprocess worker machinery
# ---------------------------------------------------------------------------

def _process_worker_initializer(
    loader_id: Optional[str],
    processor_classes: List[type],
    chunks_dir: Path,
    chunk_every_n: int,
) -> None:
    global _PROCESS_WORKER_CONTEXT
    _PROCESS_WORKER_CONTEXT = {
        "loader":       discover_loader(loader_id) if loader_id else None,
        "processors":   [cls() for cls in processor_classes],
        "chunks_dir":   chunks_dir,
        "chunk_every_n": chunk_every_n,
    }


def _process_worker_files(worker_files: List[_IndexedPath], worker_id: int) -> None:
    """Subprocess entry point. Reads context set by initializer, delegates to _process_files."""
    ctx = _PROCESS_WORKER_CONTEXT
    loader = ctx.get("loader")
    if loader is None:
        logger.warning("Processing Core: worker has no loader; skipping %d file(s).", len(worker_files))
        return
    _process_files(
        worker_files,
        loader,
        ctx["processors"],
        ctx["chunks_dir"],
        ctx["chunk_every_n"],
        worker_id,
        show_progress=False,
    )


# ---------------------------------------------------------------------------
# File iteration and processing
# ---------------------------------------------------------------------------

def _assign_files_to_workers(
    df: pl.DataFrame, n_files_per_worker: int
) -> Iterator[List[_IndexedPath]]:
    """Yield equal-sized groups of files, one group per worker call."""
    if df.is_empty():
        return
    size = max(1, n_files_per_worker)
    for offset in range(0, df.height, size):
        rows = df.slice(offset, size).select(["row_index", "path"])
        group = [_IndexedPath(int(r[0]), str(r[1])) for r in rows.iter_rows()]
        if group:
            yield group


def _iter_file_rows(
    file_path: Path,
    loader: PixelPatrolLoader,
    processors: List[PixelPatrolProcessor],
    show_progress: bool,
) -> Iterator[Dict]:
    """Yield one processed row dict per record in the file.

    Supports streaming loaders (load_iter) and dict/single-record loaders (load).
    Yields nothing if the file cannot be opened or produces no records.
    """
    if not file_path.exists():
        logger.warning("Processing Core: file not found '%s'.", file_path)
        return

    if hasattr(loader, "load_iter"):
        try:
            items = tqdm(
                loader.load_iter(str(file_path)),
                desc="  Processing sub-images",
                unit="img",
                leave=False,
                colour="blue",
                position=1,
            )
            for child_id, rcd in items:
                child_id = str(child_id) if isinstance(child_id, int) else child_id
                if not isinstance(child_id, str) or not child_id:
                    logger.warning(
                        "Processing Core: loader '%s' returned invalid child key %r in '%s'; skipping.",
                        loader.NAME, child_id, file_path,
                    )
                    continue
                for row in _extract_record_properties(rcd, processors, False, file_path):
                    yield {**row, "child_id": child_id}
        except Exception:
            logger.exception("Processing Core: load_iter failed for '%s'; skipping file.", file_path)
        return

    try:
        result = loader.load(str(file_path))
    except Exception:
        logger.exception("Processing Core: loader failed for '%s'; skipping file.", file_path)
        return

    if result is None:
        return

    if isinstance(result, dict):
        items = tqdm(
            result.items(),
            desc="  Processing sub-images",
            unit="img",
            leave=False,
            colour="blue",
            position=1,
        )
        for child_id, rcd in items:
            child_id = str(child_id) if isinstance(child_id, int) else child_id
            if not isinstance(child_id, str) or not child_id:
                logger.warning(
                    "Processing Core: loader '%s' returned invalid child key %r in '%s'; skipping.",
                    loader.NAME, child_id, file_path,
                )
                continue
            for row in _extract_record_properties(rcd, processors, False, file_path):
                yield {**row, "child_id": child_id}
    else:
        yield from _extract_record_properties(result, processors, show_progress, file_path)


def _write_chunk(rows: List[Dict], path: Path) -> None:
    """Write a list of row dicts to a parquet chunk file."""
    df = pl.DataFrame(rows, nan_to_null=True, strict=False, infer_schema_length=None)
    result = write_chunk(df, path, compression="zstd")
    if result is None:
        logger.warning("Processing Core: failed to write chunk '%s'.", path)


def _process_files(
    worker_files: List[_IndexedPath],
    loader: PixelPatrolLoader,
    processors: List[PixelPatrolProcessor],
    chunks_dir: Path,
    chunk_every_n: int,
    worker_id: int,
    show_progress: bool = False,
    on_file_done: Optional[Callable[[Path], None]] = None,
) -> None:
    """Process files and write rows to chunk files in chunks_dir.

    Rows accumulate in a buffer; when the buffer reaches chunk_every_n rows it is
    flushed to disk and the buffer cleared. A final flush always happens at the end,
    so at least one chunk is written if any rows were produced. Returns None — no
    row data is transmitted back to the caller.
    """
    row_buffer: List[Dict] = []
    chunk_seq = 0

    for item in worker_files:
        for row in _iter_file_rows(Path(item.path), loader, processors, show_progress):
            row_buffer.append({"row_index": item.row_index, **row})
            if chunk_every_n > 0 and len(row_buffer) >= chunk_every_n:
                _write_chunk(row_buffer, chunks_dir / f"chunk_{worker_id:06d}_{chunk_seq:05d}.parquet")
                row_buffer.clear()
                gc.collect()
                chunk_seq += 1
        if on_file_done:
            on_file_done(Path(item.path))

    if row_buffer:
        _write_chunk(row_buffer, chunks_dir / f"chunk_{worker_id:06d}_{chunk_seq:05d}.parquet")
        gc.collect()


# ---------------------------------------------------------------------------
# Record extraction helpers (unchanged)
# ---------------------------------------------------------------------------

def _row_coord_key(row: Dict) -> tuple:
    dim_items = tuple(sorted((k, row.get(k)) for k in row if k.startswith("dim_")))
    return (row.get("obs_level"),) + dim_items


def _merge_long_rows(existing: List[Dict], incoming: List[Dict]) -> List[Dict]:
    """Merge two long-format row lists by coordinate key (obs_level + dim_* fields).

    Matching rows are merged in-place; unmatched incoming rows are appended.
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
) -> List[Dict]:
    """Run all matching processors on a single Record and return long-format row dicts."""
    scalar: Dict = dict(rcd.meta)
    scalar.setdefault("obs_level", 0)
    long_rows: Optional[List[Dict]] = None

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
            logger.debug("Processing Core: done processor=%s record=%s (%.2fs)", proc_name, record_path, dt)
            if isinstance(out, list):
                if out:
                    long_rows = _merge_long_rows(long_rows, out) if long_rows is not None else out
            else:
                scalar.update(out)
        except RuntimeError as err:
            logger.warning("Processor %s skipped for record %s: %s", P.NAME, record_path, err)
        except (ValueError, TypeError):
            logger.exception("Processor %s failed for record %s", P.NAME, record_path)

    if long_rows is not None:
        return [{**scalar, **row} for row in long_rows]
    return [dict(scalar)]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_records_df(
    bases: List[Path],
    loader: Optional[PixelPatrolLoader],
    processing_config: Optional[ProcessingConfig] = None,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
    chunks_dir: Optional[Path] = None,
) -> Optional[pl.DataFrame]:
    """Scan files, run loader + processors, and return the full records DataFrame."""
    config = processing_config or ProcessingConfig()
    basic = _build_basic_file_df(bases, loader=loader, accepted_extensions=config.selected_file_extensions)
    if loader is None or basic is None:
        return basic

    basic = basic.with_row_index("row_index").with_columns(pl.col("row_index").cast(pl.Int64))

    _tmp_dir = None
    if chunks_dir is None:
        _tmp_dir = tempfile.mkdtemp(prefix="pixel_patrol_chunks_")
        chunks_dir = Path(_tmp_dir)

    try:
        return _build_deep_record_df(
            basic,
            loader,
            processing_config=config,
            progress_callback=progress_callback,
            chunks_dir=chunks_dir,
        )
    finally:
        if _tmp_dir is not None:
            _cleanup_chunks_dir(Path(_tmp_dir))


def _build_basic_file_df(bases, loader, accepted_extensions):
    basic = walk_filesystem(bases, loader=loader, accepted_extensions=accepted_extensions)
    if basic.is_empty():
        return None
    return postprocess_basic_file_metadata_df(normalize_file_extension(basic))


# ---------------------------------------------------------------------------
# Deep processing: workers write chunks; main process combines at the end
# ---------------------------------------------------------------------------

def _build_deep_record_df(
    basic: pl.DataFrame,
    loader_instance: PixelPatrolLoader,
    processing_config: Optional[ProcessingConfig] = None,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
    chunks_dir: Optional[Path] = None,
) -> pl.DataFrame:
    if basic.is_empty():
        return pl.DataFrame([])

    config = processing_config or ProcessingConfig()
    total = basic.height

    processors = discover_processor_plugins()
    if config.processors_included:
        processors = [p for p in processors if p.NAME in config.processors_included]
    elif config.processors_excluded:
        processors = [p for p in processors if p.NAME not in config.processors_excluded]

    chunk_every_n = config.chunk_every_n or DEFAULT_CHUNK_EVERY_N
    worker_count = _resolve_worker_count(config, total_rows=total)

    chunks_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_chunks_dir(chunks_dir, cleanup_combined_parquet=False)

    all_files = [
        _IndexedPath(int(r[0]), str(r[1]))
        for r in basic.select(["row_index", "path"]).iter_rows()
    ]

    with tqdm(total=total, desc="Processing files", unit="file", leave=True, colour="green", position=0,
              disable=progress_callback is not None) as progress:

        if worker_count == 1:
            processed_count = 0

            def on_file_done(file_path: Path) -> None:
                nonlocal processed_count
                processed_count += 1
                progress.update(1)
                if progress_callback:
                    progress_callback(processed_count, total, file_path)

            _process_files(
                all_files, loader_instance, processors,
                chunks_dir, chunk_every_n,
                worker_id=0, show_progress=True, on_file_done=on_file_done,
            )

        else:
            processor_classes = [p.__class__ for p in processors]
            n_per_worker = max(1, math.ceil(total / worker_count))

            try:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_process_worker_initializer,
                    initargs=(getattr(loader_instance, "NAME", None), processor_classes, chunks_dir, chunk_every_n),
                    mp_context=multiprocessing.get_context("spawn"),
                ) as pool:
                    futures = {
                        pool.submit(_process_worker_files, group, worker_id): group
                        for worker_id, group in enumerate(_assign_files_to_workers(basic, n_per_worker))
                    }
                    processed_count = 0
                    for future in as_completed(futures):
                        group = futures[future]
                        try:
                            future.result()
                        except Exception:
                            logger.exception(
                                "Processing Core: worker failed for batch starting at '%s'.",
                                group[0].path,
                            )
                        processed_count += len(group)
                        progress.update(len(group))
                        if progress_callback:
                            progress_callback(processed_count, total, Path(group[-1].path))

            except Exception as exc:
                logger.warning(
                    "Processing Core: ProcessPoolExecutor unavailable (%s); falling back to threads.", exc
                )
                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    futures = {
                        pool.submit(
                            _process_files, group, loader_instance, processors,
                            chunks_dir, chunk_every_n, worker_id, False,
                        ): group
                        for worker_id, group in enumerate(_assign_files_to_workers(basic, n_per_worker))
                    }
                    processed_count = 0
                    for future in as_completed(futures):
                        group = futures[future]
                        try:
                            future.result()
                        except Exception:
                            logger.exception(
                                "Processing Core: thread worker failed for batch starting at '%s'.",
                                group[0].path,
                            )
                        processed_count += len(group)
                        progress.update(len(group))
                        if progress_callback:
                            progress_callback(processed_count, total, Path(group[-1].path))

    return _combine_all_chunks(chunks_dir, basic)


def _combine_all_chunks(chunks_dir: Path, basic_df: pl.DataFrame) -> pl.DataFrame:
    """Read all chunk files written by workers and join with basic file metadata.

    Chunks are not sorted before concatenation — this preserves a natural mix of
    files from different workers/conditions at the top of the resulting parquet,
    which gives the viewer a representative sample when it reads the first rows.

    Files that failed to process (no chunk rows) are included with null processed columns.
    """
    chunk_files = list(chunks_dir.glob("chunk_*.parquet"))

    if not chunk_files:
        return _post_process_final_df(basic_df)

    total_size = sum(f.stat().st_size for f in chunk_files if f.exists())
    available = _estimate_available_memory_bytes()
    if available is not None and total_size * COMBINE_HEADROOM_RATIO > available:
        logger.warning(
            "Processing Core: insufficient memory to combine chunks "
            "(%.1f MB chunks vs %.1f MB available); chunk files remain at '%s'.",
            total_size / 1e6, available / 1e6, chunks_dir,
        )
        return pl.DataFrame([])

    deep_df = pl.concat(
        [pl.read_parquet(f) for f in chunk_files],
        how="diagonal_relaxed",
        rechunk=True,
    )
    deep_df = deep_df.with_columns(pl.col("row_index").cast(pl.Int64))

    # Loader columns take precedence over basic for any overlapping names.
    overlapping = (set(basic_df.columns) & set(deep_df.columns)) - {"row_index"}
    basic_for_join = basic_df.drop(overlapping) if overlapping else basic_df

    # Left join: each deep row (including sub-images) gets its parent file's basic metadata.
    # One basic row expands to N deep rows for files with sub-images.
    merged = deep_df.join(basic_for_join, on="row_index", how="left")

    # Files that produced no rows (load failures) still appear with basic info only.
    processed_indices = set(deep_df["row_index"].unique().to_list())
    failed = basic_df.filter(~pl.col("row_index").is_in(processed_indices))
    if not failed.is_empty():
        merged = pl.concat([merged, failed], how="diagonal_relaxed", rechunk=False)

    gc.collect()
    return _post_process_final_df(merged)


def _post_process_final_df(df: pl.DataFrame) -> pl.DataFrame:
    """Clean up and finalize the merged DataFrame before returning to the caller."""
    if "row_index" in df.columns:
        df = df.drop("row_index")
    if "child_id" in df.columns:
        type_col = pl.col("type") if "type" in df.columns else pl.lit(None, dtype=pl.Utf8)
        df = df.with_columns(
            pl.when(pl.col("child_id").is_not_null())
            .then(pl.lit("sub_file"))
            .otherwise(type_col)
            .alias("type")
        )
    cols_to_drop = [
        col for col in df.columns
        if df[col].is_null().all()
        or (df[col].dtype == pl.Utf8 and (df[col].is_null() | (df[col] == "")).all())
    ]
    if cols_to_drop:
        df = df.drop(cols_to_drop)
    try:
        df = optimize_dtypes(df)
    except Exception as exc:
        logger.exception("Processing Core: dtype optimization failed; continuing as-is: %s", exc)
    return _reorder_columns(df)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _resolve_worker_count(config: ProcessingConfig, total_rows: Optional[int] = None) -> int:
    """Return the number of parallel workers to use (minimum 1).

    Capped by CPU count and, when provided, by total_rows so we never spin up
    more workers than there are files to process.
    """
    cpu_count = os.cpu_count() or 1

    if config.processing_max_workers is not None:
        max_workers = min(max(1, config.processing_max_workers), cpu_count)
    else:
        max_workers = max(1, cpu_count)

    if total_rows is not None:
        if total_rows <= 1:
            return 1
        return min(max_workers, max(1, total_rows))

    return max_workers


def _estimate_available_memory_bytes() -> Optional[int]:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except ImportError:
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
    """Move dim_* columns after obs_level; move binary/list columns to the end."""
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


def optimize_dtypes(df: pl.DataFrame, shrink_floats: bool = True) -> pl.DataFrame:
    """Shrink integer dtypes and downcast float64 → float32 where safe."""
    _NO_SHRINK = {"size_bytes"}  # must stay Int64 to avoid overflow in aggregations

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

    return df.with_columns(series_updates) if series_updates else df


# ---------------------------------------------------------------------------
# Chunk-directory lifecycle (called by project.py)
# ---------------------------------------------------------------------------

def _cleanup_chunks_dir(chunks_dir: Optional[Path], cleanup_combined_parquet: bool = True) -> None:
    """Remove chunk files from chunks_dir; optionally remove the combined parquet too."""
    if not chunks_dir or not chunks_dir.exists():
        return

    removed_any = False
    for p in chunks_dir.glob("chunk_*.parquet"):
        try:
            p.unlink()
            removed_any = True
            logger.debug("Processing Core: removed chunk %s", p)
        except OSError as exc:
            logger.warning("Processing Core: could not remove chunk %s: %s", p, exc)

    removed_combined = False
    if cleanup_combined_parquet:
        combined = chunks_dir / "records_df.parquet"
        try:
            if combined.exists():
                combined.unlink()
                removed_combined = True
                logger.info("Processing Core: removed combined records parquet %s", combined)
        except OSError as exc:
            logger.warning("Processing Core: could not remove combined parquet %s: %s", combined, exc)

    if removed_any or removed_combined:
        logger.info("Processing Core: removed chunks under %s", chunks_dir)
        try:
            chunks_dir.rmdir()
        except OSError:
            pass


def cleanup_chunks_dir(chunks_dir: Optional[Path]) -> None:
    """Remove the chunks directory after a successful parquet save."""
    if not chunks_dir or not chunks_dir.exists():
        return
    try:
        shutil.rmtree(chunks_dir)
        logger.info("Processing Core: removed chunks directory '%s'.", chunks_dir)
    except OSError as exc:
        logger.warning("Processing Core: could not remove '%s': %s", chunks_dir, exc)


# ---------------------------------------------------------------------------
# Legacy / misc helpers
# ---------------------------------------------------------------------------

PATHS_DF_EXPECTED_SCHEMA = {  # TODO: retire once paths_df is fully removed
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


def count_file_extensions(paths_df: Optional[pl.DataFrame]) -> Dict[str, int]:
    if paths_df is None or paths_df.is_empty():
        logger.warning("No paths DataFrame provided or it's empty. Cannot count file extensions.")
        return {"all_files": 0}

    required_cols = {"type", "file_extension"}
    if not required_cols.issubset(paths_df.columns):
        logger.error(
            "Paths DataFrame is missing required columns for extension counting: %s",
            required_cols - set(paths_df.columns),
        )
        return {"all_files": paths_df.height}

    df_files = paths_df.filter(pl.col("type") == "file")
    df_files = df_files.filter(pl.col("file_extension").is_not_null() & (pl.col("file_extension") != ""))

    if df_files.is_empty():
        return {"all_files": 0}

    grouped = df_files.group_by("file_extension").agg(pl.count().alias("count"))
    result = {row["file_extension"]: row["count"] for row in grouped.iter_rows(named=True)}
    result["all_files"] = df_files.height
    return result