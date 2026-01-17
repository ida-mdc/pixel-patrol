import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple, Iterable, Iterator, NamedTuple, Callable
from tqdm.auto import tqdm

import polars as pl

from pixel_patrol_base.core.contracts import PixelPatrolLoader, PixelPatrolProcessor
from pixel_patrol_base.core.file_system import walk_filesystem
from pixel_patrol_base.plugin_registry import (
    discover_processor_plugins,
    discover_loader,
)
from pixel_patrol_base.utils.df_utils import (
    normalize_file_extension,
    postprocess_basic_file_metadata_df,
)
from pixel_patrol_base.core.specs import is_record_matching_processor
from pixel_patrol_base.config import (
    DEFAULT_RECORDS_FLUSH_EVERY_N,
)
from pixel_patrol_base.core.project_settings import Settings
from pixel_patrol_base.io.parquet_utils import write_dataframe_to_parquet


logger = logging.getLogger(__name__)

# When combining multiple parquet chunks into a single one in memory, we might need more RAM than the raw size of the files.
COMBINE_HEADROOM_RATIO = 1.25

# Global per-process context for worker initializers
_PROCESS_WORKER_CONTEXT: Dict[str, object] = {}


class _IndexedPath(NamedTuple):
    """Represents a file path with its associated row index in the dataset.

    Args:
        NamedTuple: row_index (int): The index of the row in the dataset.
                    path (str): The file system path to the file.
    """
    row_index: int
    path: str


def _process_worker_initializer(
    loader_id: Optional[str], processor_classes: List[type]
) -> None:
    """Initialize per-process loader and processor instances.
    
    Args:
        loader_id: Optional string identifier for the loader to use.
        processor_classes: List of PixelPatrolProcessor classes to instantiate.
    """
    global _PROCESS_WORKER_CONTEXT
    loader_instance = discover_loader(loader_id) if loader_id else None
    processors = [cls() for cls in processor_classes]
    _PROCESS_WORKER_CONTEXT = {
        "loader": loader_instance,
        "processors": processors,
    }


def _process_batch_in_worker(batch: List[_IndexedPath]) -> List[Dict[str, object]]:
    """Worker entry point for processing a batch of indexed paths.

    Args:
        batch: A list of _IndexedPath items representing the current batch.
    Returns:
        A list of dictionaries containing the results from deep processing.
    """
    loader = _PROCESS_WORKER_CONTEXT.get("loader")
    processors = _PROCESS_WORKER_CONTEXT.get("processors", [])
    if loader is None:
        logger.warning(
            "Processing Core: worker lacks loader; skipping batch of size %s",
            len(batch),
        )
        return []

    return _process_batch_locally(batch, loader, processors, False)


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
    """
    deep_rows: List[Dict[str, object]] = []
    for item in batch:
        record_dicts = load_and_process_records_from_file(
            Path(item.path), loader_instance, processors, show_processor_progress
        )
        for record_dict in record_dicts:
            if record_dict:
                deep_rows.append({"row_index": item.row_index, **record_dict})
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

    row_indices = [int(item.row_index) for item in batch]
    basic_batch = basic.filter(
        pl.col("row_index").is_in(pl.Series(row_indices, dtype=pl.Int64))
    )

    if not deep_rows:
        # Nothing deep to merge for this batch; return the basic rows as-is
        return basic_batch

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
        deep_df = deep_df.drop(overlap)

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
        except Exception:
            # If not empty or cannot remove, leave it in place.
            pass


def _build_deep_record_df(
    basic: pl.DataFrame,
    loader_instance: PixelPatrolLoader,
    settings: Optional[Settings] = None,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
) -> pl.DataFrame:
    """Process each path (optionally in parallel) and build a Polars DataFrame.

    Args:
        basic: The basic Polars DataFrame with file paths and metadata.
        loader_instance: An instance of PixelPatrolLoader to load files.
        settings: Optional Settings object for processing configuration.
        progress_callback: Optional GUI callback for progress updates.
    Returns:
        A Polars DataFrame containing deep processed records, aka the results.
    """
    if basic.is_empty():
        return pl.DataFrame([])

    processors = discover_processor_plugins()
    processor_classes = [proc.__class__ for proc in processors]
    worker_count = _resolve_worker_count(settings)
    flush_threshold = _resolve_flush_threshold(basic.height, settings)
    batch_size = _resolve_batch_size(worker_count, flush_threshold, basic.height)
    show_processor_progress = worker_count == 1

    accumulator = _RecordsAccumulator(
        flush_every_n=flush_threshold,
        flush_dir=getattr(settings, "records_flush_dir", None),
    )

    processed_rows: Set[int] = set()
    if accumulator._flush_dir:
        if settings and getattr(settings, "resume", False):
            # Resume: adopt existing partial chunks and skip already-processed files
            processed_rows = accumulator.load_existing_chunks()
        else:
            # ensure fresh run
            _cleanup_partial_chunks_dir(accumulator._flush_dir, cleanup_combined_parquet=False)

    remaining = basic
    if processed_rows:
        remaining = basic.filter(
            ~pl.col("row_index").is_in(pl.Series(list(processed_rows), dtype=pl.Int64))
        )
    if remaining.is_empty():
        return pl.DataFrame([])

    remaining_count = remaining.height

    with tqdm(
        total=remaining_count,
        desc="Processing files",
        unit="file",
        leave=True,
        colour="green",
        position=0,
        disable=progress_callback is not None,
    ) as progress:
        if processed_rows:
            logger.info(
                "Processing Core: skipping %d already-processed files; resuming %d remaining files",
                len(processed_rows),
                remaining_count,
            )

        if worker_count == 1:
            for batch in _iter_indexed_batches(remaining, batch_size):
                deep_rows = _process_batch_locally(
                    batch, loader_instance, processors, show_processor_progress
                )
                batch_df = _combine_batch_with_basic(basic, batch, deep_rows)
                accumulator.add_batch(batch_df)
                progress.update(len(batch))
                if progress_callback:
                    progress_callback(len(batch), remaining_count, Path(batch[-1].path))
            return accumulator.finalize()

        loader_name = getattr(loader_instance, "NAME", None)
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_process_worker_initializer,
                initargs=(loader_name, processor_classes),
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                future_map: Dict = {}
                for batch in _iter_indexed_batches(remaining, batch_size):
                    fut = executor.submit(_process_batch_in_worker, batch)
                    future_map[fut] = batch

                for future in as_completed(future_map):
                    batch = future_map[future]
                    try:
                        deep_rows = future.result()
                    except Exception:  # pragma: no cover - logged for observability
                        logger.exception(
                            "Process worker failed for batch starting at %s",
                            batch[0].path,
                        )
                        deep_rows = []

                    batch_df = _combine_batch_with_basic(basic, batch, deep_rows)
                    accumulator.add_batch(batch_df)
                    progress.update(len(batch))
                    if progress_callback:
                        progress_callback(
                            len(batch), remaining_count, Path(batch[-1].path)
                        )
        except Exception as exc:
            logger.warning(
                "Processing Core: ProcessPoolExecutor unavailable (%s); falling back to threads.",
                exc,
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map: Dict = {}
                for batch in _iter_indexed_batches(remaining, batch_size):
                    fut = executor.submit(
                        _process_batch_locally,
                        batch,
                        loader_instance,
                        processors,
                        False,
                    )
                    future_map[fut] = batch

                for future in as_completed(future_map):
                    batch = future_map[future]
                    try:
                        deep_rows = future.result()
                    except Exception:  # pragma: no cover - logged for observability
                        logger.exception(
                            "Processor failed for batch starting at %s", batch[0].path
                        )
                        deep_rows = []

                    batch_df = _combine_batch_with_basic(basic, batch, deep_rows)
                    accumulator.add_batch(batch_df)
                    progress.update(len(batch))
                    if progress_callback:
                        progress_callback(
                            len(batch), remaining_count, Path(batch[-1].path)
                        )

        return accumulator.finalize()


def load_and_process_records_from_file(
    file_path: Path,
    loader: PixelPatrolLoader,
    processors: List[PixelPatrolProcessor],
    show_processor_progress: bool = True,
) -> List[Dict]:
    """
    Load a file with the given loader, run all matching processors, and return combined metadata.

    Args:
        file_path: Path to the file to process.
        loader: An instance of PixelPatrolLoader to load the file.
        processors: A list of PixelPatrolProcessor instances to run on the loaded record(s).
    Returns:
        A list of dictionaries containing combined data (metadata) from the loader and all applicable processors.
        Returns empty list if file cannot be loaded or processed.
    """
    if not file_path.exists():
        logger.warning(f"File not found: '{file_path}'. Cannot extract metadata.")
        return []

    try:
        result = loader.load(str(file_path))
        if result is None:
            records = []
        elif isinstance(result, list):
            records = result
        else:
            # Single record - wrap in list
            records = [result]
    except Exception as e:
        logger.info(f"Loader '{loader.NAME}' failed with exception, skipping: {e}")
        return []

    if not records:
        return []

    # Process each record through processors
    result_list = []
    processor_iter: Iterable[PixelPatrolProcessor]
    if show_processor_progress:
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

    for art in records:
        extracted_properties = {}
        metadata = dict(art.meta)
        extracted_properties.update(metadata)

        for P in processor_iter:
            if not is_record_matching_processor(art, P.INPUT):
                continue
            try:
                out = P.run(art)
                if isinstance(out, dict):
                    extracted_properties.update(out)
                else:
                    art = out  # chainable: processors may transform the record
                    extracted_properties.update(art.meta)
            except Exception as e:
                logger.warning(f"Processor {P} failed: {e}")

        result_list.append(extracted_properties)

    return result_list


def build_records_df(
    bases: List[Path],
    selected_extensions: Set[str] | str,
    loader: Optional[PixelPatrolLoader],
    settings: Optional[Settings] = None,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
) -> Optional[pl.DataFrame]:
    """Build the full records DataFrame by scanning files and processing them.

    Args:
        bases: List of base directories to scan for files.
        selected_extensions: Set of file extensions to include, or "all" for all supported ('{"tif", "png", "jpg", ...}', or '"all"').
        loader: Optional PixelPatrolLoader instance to use for loading files.
        settings: Optional Settings object for processing configuration.
        progress_callback: Optional callback `function(current: int, total: int, current_file: Path) -> None`
                            Called for each file processed during deep processing.
    Returns:
        A Polars DataFrame containing the full records, or None if no files were found.
    """

    basic = _build_basic_file_df(
        bases, loader=loader, accepted_extensions=selected_extensions
    )
    if loader is None or basic is None:
        return basic

    basic = basic.with_row_index("row_index").with_columns(
        pl.col("row_index").cast(pl.Int64)
    )
    deep = _build_deep_record_df(basic, loader, settings=settings, progress_callback=progress_callback)

    return deep


def _build_basic_file_df(bases, loader, accepted_extensions):
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


def _resolve_worker_count(settings: Optional[Settings]) -> int:
    """
    Determine the number of parallel workers to use for processing with a minimum of 1.
    If settings specify a value, use that within bounds of available workers; otherwise, default to CPU count.

    Args:
        settings: Optional Settings object that may specify `processing_max_workers`.
    Returns:
        An integer representing the number of parallel workers to use.
    """
    if settings and settings.processing_max_workers is not None:
        return (
            max(1, settings.processing_max_workers)
            if settings.processing_max_workers <= os.cpu_count()
            else os.cpu_count() or 1
        )
    return max(1, os.cpu_count())


def _resolve_batch_size(
    worker_count: int, flush_threshold: int, total_rows: int
) -> int:
    """
    Derive batch size from flush threshold and available workers.
    Ensures that there is at least one checkpoint flush per dataset processing.

    Args:
        worker_count: Number of parallel workers aka CPU cores.
        flush_threshold: Number of rows after which to flush records to disk.
        total_rows: Total number of rows/files in the dataset to process.
    Returns:
        An integer representing the batch size per worker for processing.
    """
    if flush_threshold <= 0:  # e.g. flush is completely disabled or not properly set
        return max(1, math.ceil(total_rows / max(1, worker_count)))
    return max(1, math.ceil(flush_threshold / max(1, worker_count)))


def _resolve_flush_threshold(total_rows: int, settings: Optional[Settings]) -> int:
    """Sets the flush threshold from settings or defaults. Ensures non-negative and half the dataset size to enforce a checkpoint flush.

    Args:
        total_rows: Total number of rows/files to process.
        settings: Optional Settings object that may specify `records_flush_every_n`.
    Returns:
        An integer representing the flush threshold for records.
    """
    if total_rows <= 0:
        return 0
    value = (
        settings.records_flush_every_n
        if settings and settings.records_flush_every_n is not None
        else DEFAULT_RECORDS_FLUSH_EVERY_N
    )
    # Preserve the ability to disable flushing when a non-positive value is configured.
    if value is None or value <= 0:
        return 0
    # Ensure at least one flush can occur for small datasets while capping at roughly half the dataset size.
    upper_bound = max(1, total_rows // 2)
    return max(1, min(value, upper_bound))


def _estimate_available_memory_bytes() -> Optional[int]:
    """Estimate of available system memory in bytes.

    Returns:
        An integer representing available memory in bytes, or None if it cannot be determined.
    """
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            return int(page_size * avail_pages)
        except Exception:
            return None


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
            logger.info(
                "Processing Core: buffering %s rows per chunk; partial batches go to '%s'",
                self._flush_every_n,
                self._flush_dir,
            )

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
        Finalize accumulation: flush remaining active data and combine all chunks into one Parquet file if applicable, while remopving partial chunks.
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
        available_memory = _estimate_available_memory_bytes()
        if available_memory is not None:
            projected = total_size * self._combine_headroom_ratio
            if projected > available_memory:
                logger.warning(
                    "Processing Core: skipping final combine; chunk files remain at %s (total %.2f MB, available %.2f MB)",
                    self._flush_dir,
                    total_size / (1024 * 1024),
                    available_memory / (1024 * 1024),
                )
                return pl.DataFrame([])

        frames = [pl.read_parquet(path) for path in self._written_files]
        if not self._active_df.is_empty():
            frames.append(self._active_df)

        if not frames:
            return pl.DataFrame([])

        final_df = pl.concat(frames, how="diagonal_relaxed", rechunk=True)

        # write combined parquet file and tidy up partial chunks
        if self._flush_dir:
            try:
                combined_path = write_dataframe_to_parquet(
                    final_df,
                    "records_df.parquet",
                    self._flush_dir,
                    compression="zstd",
                )
                if combined_path is None:
                    logger.warning(
                        "Processing Core: failed to write combined records parquet to %s",
                        self._flush_dir,
                    )
                    return final_df
                logger.info(
                    "Processing Core: writing combined records DataFrame to %s",
                    combined_path,
                )

                # tidy up (leave combined parquet intact)
                _cleanup_partial_chunks_dir(self._flush_dir, cleanup_combined_parquet=False)
                # reset written files list
                self._written_files = []
            except Exception:
                logger.exception(
                    "Processing Core: failed to finalize combined records parquet in %s",
                    self._flush_dir,
                )

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
        chunk_path = write_dataframe_to_parquet(
            self._active_df,
            chunk_filename,
            self._flush_dir,
            compression="zstd",
        )
        if chunk_path is None:
            logger.warning(
                "Processing Core: failed to write partial chunk %s", chunk_filename
            )
            return

        logger.debug("Processing Core: writing partial records chunk to %s", chunk_path)
        self._written_files.append(chunk_path)
        self._active_df = pl.DataFrame([])
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
            except Exception:
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
