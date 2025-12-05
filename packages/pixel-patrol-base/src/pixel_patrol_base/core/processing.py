import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Dict, Set, Tuple, Iterable, Callable
from tqdm.auto import tqdm

import polars as pl

from pixel_patrol_base.core.contracts import PixelPatrolLoader, PixelPatrolProcessor
from pixel_patrol_base.core.file_system import walk_filesystem
from pixel_patrol_base.plugin_registry import discover_processor_plugins, discover_loader
from pixel_patrol_base.utils.df_utils import normalize_file_extension, postprocess_basic_file_metadata_df
from pixel_patrol_base.core.specs import is_record_matching_processor
from pixel_patrol_base.config import (
    DEFAULT_PROCESSING_BATCH_SIZE,
    DEFAULT_RECORDS_FLUSH_EVERY_N,
)
from pixel_patrol_base.core.project_settings import Settings
from pixel_patrol_base.io.parquet_utils import _write_dataframe_to_parquet


logger = logging.getLogger(__name__)

_PROCESS_WORKER_CONTEXT: Dict[str, object] = {}


def _process_worker_initializer(loader_id: Optional[str], processor_classes: List[type]) -> None:
    """Initialize per-process loader and processor instances."""
    global _PROCESS_WORKER_CONTEXT
    loader_instance = discover_loader(loader_id) if loader_id else None
    processors = [cls() for cls in processor_classes]
    _PROCESS_WORKER_CONTEXT = {
        "loader": loader_instance,
        "processors": processors,
    }


def _process_path_in_worker(file_path: str) -> Dict:
    """Worker entry point for processing a single file path."""
    loader = _PROCESS_WORKER_CONTEXT.get("loader")
    processors = _PROCESS_WORKER_CONTEXT.get("processors", [])
    if loader is None:
        logger.warning("Processing Core: worker lacks loader; skipping %s", file_path)
        return {}
    return get_all_record_properties(
        Path(file_path),
        loader,
        processors,
        show_processor_progress=False,
    )


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
    "imported_path": pl.String
}

def _scan_dirs_for_extensions(
    bases: List[Path],
    accepted_extensions: Set[str]
) -> List[Tuple[Path, Path]]:
    """
    Walk each base dir, filter by extension, and return a list of (file_path, base_dir) tuples.
    """
    matched: List[Tuple[Path, Path]] = []
    for base in bases:
        for root, _, files in os.walk(base):
            for name in files:
                ext = Path(name).suffix.lower().lstrip('.')
                if ext in accepted_extensions:
                    matched.append((Path(root) / name, base))
    return matched


def _build_deep_record_df(
    paths: List[Path],
    loader_instance: PixelPatrolLoader,
    settings: Optional[Settings] = None,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
) -> pl.DataFrame:
    """Loop over paths, get_all_record_properties, return DataFrame (may be empty).
    Optimized to minimize Python loop overhead where possible.
    
    Args:
        paths: List of file paths to process
        loader_instance: Loader instance to use for loading files
        progress_callback: Optional callback function(current: int, total: int, current_file: Path) -> None
                          Called for each file processed. If provided, tqdm progress bar is disabled.
    """
    processors = discover_processor_plugins()
    processor_classes = [proc.__class__ for proc in processors]
    worker_count = _resolve_worker_count(settings)
    show_processor_progress = worker_count == 1

    accumulator = _RecordsAccumulator(
        batch_size=_resolve_batch_size(settings),
        flush_every_n=_resolve_flush_threshold(settings),
        flush_dir=getattr(settings, "records_flush_dir", None),
    )

    path_strings = [str(p) for p in paths]

    # If partial chunk files exist, load them and skip already-processed paths
    processed_paths: Set[str] = set()
    if accumulator._flush_dir:
        processed_paths = accumulator.load_existing_chunks()
    if processed_paths:
        logger.info("Processing Core: skipping %d already-processed files; resuming %d remaining files", len(processed_paths), len(to_process))

    to_process = [Path(p) for p in path_strings if p not in processed_paths]
    total = len(to_process)

    # Use progress callback if provided, otherwise use tqdm
    iterator = tqdm(
        to_process,
        desc="Processing files",
        unit="file",
        leave=True,
        colour="green",
        position=0,
        disable=progress_callback is not None,  # Disable CLI bar if UI callback exists
    )

    for idx, file_path in enumerate(iterator, start=1):
        if progress_callback is not None:
            progress_callback(idx, total, file_path)
        if worker_count == 1:
            for file_path in to_process:
                record_dict = get_all_record_properties(
                    file_path, loader_instance, processors, show_processor_progress
                )
                if record_dict:
                    accumulator.add_row({"path": file_path, **record_dict})
        else:
            loader_name = getattr(loader_instance, "NAME", None)
            try:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_process_worker_initializer,
                    initargs=(loader_name, processor_classes),
                ) as executor:
                    future_map = {
                        executor.submit(_process_path_in_worker, file_path): file_path
                        for file_path in to_process
                    }

                    for future in as_completed(future_map):
                        file_path = future_map[future]
                        try:
                            record_dict = future.result()
                        except Exception:  # pragma: no cover - logged for observability
                            logger.exception("Process worker failed for %s", file_path)
                            record_dict = {}

                        if record_dict:
                            accumulator.add_row({"path": file_path, **record_dict})
                        progress.update(1)
            except Exception as exc:
                logger.warning(
                    "Processing Core: ProcessPoolExecutor unavailable (%s); falling back to threads.",
                    exc,
                )
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_map = {
                        executor.submit(
                            get_all_record_properties,
                            Path(file_path),
                            loader_instance,
                            processors,
                            False,
                        ): file_path
                        for file_path in to_process
                    }

                    for future in as_completed(future_map):
                        file_path = future_map[future]
                        try:
                            record_dict = future.result()
                        except Exception:  # pragma: no cover - logged for observability
                            logger.exception("Processor failed for %s", file_path)
                            record_dict = {}

                    if record_dict:
                        accumulator.add_row({"path": file_path, **record_dict})

    return accumulator.finalize()


def get_all_record_properties(
    file_path: Path,
    loader: PixelPatrolLoader,
    processors: List[PixelPatrolProcessor],
    show_processor_progress: bool = True,
) -> Dict:
    """
    Load a file with the given loader, run all matching processors, and return combined metadata.
    Args:
        file_path: Path to the file to process.
        loader: An instance of PixelPatrolLoader to load the file.
        processors: A list of PixelPatrolProcessor instances to run on the loaded record.

    Returns:
        A dictionary of combined data (metadata) from the loader and all applicable processors.
    """
    if not file_path.exists():
        logger.warning(f"File not found: '{file_path}'. Cannot extract metadata.")
        return {}
    
    extracted_properties = {}
    try:
        art = loader.load(str(file_path))
        metadata = dict(art.meta)
    except Exception as e:
        logger.info(f"Loader '{loader.NAME}' failed with exception, skipping: {e}")
        return {}

    # Always process using Record; processors opt-in via INPUT spec
    extracted_properties.update(metadata)
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

    for P in processor_iter:
        if not is_record_matching_processor(art, P.INPUT):
            continue
        out = P.run(art)
        if isinstance(out, dict):
            extracted_properties.update(out)
        else:
            art = out  # chainable: processors may transform the record
            extracted_properties.update(art.meta)

    return extracted_properties


def build_records_df(
    bases: List[Path],
    selected_extensions: Set[str] | str,
    loader: Optional[PixelPatrolLoader],
    settings: Optional[Settings] = None,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
) -> Optional[pl.DataFrame]:
    """
    Build records dataframe from file system.
    
    Args:
        bases: List of base directories to scan
        selected_extensions: File extensions to include
        loader: Optional loader instance
        progress_callback: Optional callback function(current: int, total: int, current_file: Path) -> None
                          Called for each file processed during deep processing.
    """

    basic = _build_basic_file_df(bases, loader=loader, accepted_extensions=selected_extensions)
    if loader is None or basic is None: return basic

    deep = _build_deep_record_df(
        [Path(p) for p in basic["path"].to_list()], 
        loader,
        settings=settings,
        progress_callback=progress_callback
    )

    return basic.join(deep, on="path", how="left")


def _build_basic_file_df(bases, loader, accepted_extensions):
    basic = walk_filesystem(bases, loader=loader, accepted_extensions=accepted_extensions)
    if basic.is_empty(): return None
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
        logger.warning("No paths DataFrame provided or it's empty. Cannot count file extensions.")
        return {"all_files": 0}

    required_cols = {"type", "file_extension"}
    if not required_cols.issubset(paths_df.columns):
        logger.error(
            f"Paths DataFrame is missing required columns for extension counting: {required_cols - set(paths_df.columns)}")
        return {"all_files": paths_df.height}

    df_files = paths_df.filter(pl.col("type") == "file")
    df_files = df_files.filter(pl.col("file_extension").is_not_null() & (pl.col("file_extension") != ""))

    if df_files.is_empty():
        logger.info("No files with valid extensions found in the DataFrame.")
        return {"all_files": 0}

    grouped = df_files.group_by("file_extension").agg(pl.count().alias("count"))
    result = {row["file_extension"]: row["count"] for row in grouped.iter_rows(named=True)}
    result["all_files"] = df_files.height

    return result


def _resolve_worker_count(settings: Optional[Settings]) -> int:
    if settings and settings.processing_max_workers is not None:
        return max(1, settings.processing_max_workers)
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count)


def _resolve_batch_size(settings: Optional[Settings]) -> int:
    value = settings.processing_batch_size if settings else DEFAULT_PROCESSING_BATCH_SIZE
    return max(1, value)


def _resolve_flush_threshold(settings: Optional[Settings]) -> int:
    value = settings.records_flush_every_n if settings else DEFAULT_RECORDS_FLUSH_EVERY_N
    return max(0, value)


class _RecordsAccumulator:
    """Collect rows, batch-convert to Polars, and optionally flush to disk."""

    def __init__(
        self,
        batch_size: int,
        flush_every_n: int,
        flush_dir: Optional[Path],
    ) -> None:
        self._batch_size = batch_size
        self._flush_every_n = flush_every_n
        self._flush_dir = Path(flush_dir) if flush_dir else None
        self._buffer: List[Dict[str, object]] = []
        self._active_df = pl.DataFrame([])
        self._written_files: List[Path] = []
        self._chunk_index = 0
        if self._flush_dir:
            logger.info(
                "Processing Core: buffering %s rows per chunk; partial batches go to '%s'",
                self._batch_size,
                self._flush_dir,
            )

    def add_row(self, row: Dict[str, object]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self._batch_size:
            self._flush_buffer_to_active()

    def finalize(self) -> pl.DataFrame:
        if self._buffer:
            self._flush_buffer_to_active()

        if (
            self._flush_dir
            and self._flush_every_n > 0
            and self._written_files
            and not self._active_df.is_empty()
        ):
            self._flush_active_to_disk(force=True)

        if not self._written_files:
            return self._active_df

        frames = [pl.read_parquet(path) for path in self._written_files]
        if not self._active_df.is_empty():
            frames.append(self._active_df)

        if not frames:
            return pl.DataFrame([])

        final_df = pl.concat(frames, how="diagonal_relaxed", rechunk=True)

        # If a flush directory is configured, also write a single combined
        # Parquet file there for convenience and remove the per-chunk files
        # to keep the directory tidy. This makes the processed records
        # individually usable without needing to stitch chunk files later.
        if self._flush_dir:
            try:
                combined_path = _write_dataframe_to_parquet(
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

                # Remove the individual chunk files now that we have a combined file
                for p in list(self._written_files):
                    try:
                        p.unlink()
                    except Exception as exc:  # pragma: no cover - best-effort cleanup
                        logger.warning("Processing Core: could not remove partial chunk %s: %s", p, exc)
                # Reset written files list
                self._written_files = []
                # Attempt to remove the directory if it's empty
                try:
                    self._flush_dir.rmdir()
                except Exception:
                    # If not empty or cannot remove, leave it in place.
                    pass
            except Exception:
                logger.exception(
                    "Processing Core: failed to finalize combined records parquet in %s",
                    self._flush_dir,
                )

        return final_df

    def _flush_buffer_to_active(self) -> None:
        if not self._buffer:
            return

        chunk = pl.DataFrame(
            self._buffer,
            nan_to_null=True,
            strict=False,
            infer_schema_length=None,
        )
        self._buffer.clear()
        if chunk.is_empty():
            return

        if self._active_df.is_empty():
            self._active_df = chunk
        else:
            self._active_df = pl.concat(
                [self._active_df, chunk],
                how="diagonal_relaxed",
                rechunk=False,
            )

        if self._flush_dir and self._flush_every_n > 0 and self._active_df.height >= self._flush_every_n:
            self._flush_active_to_disk(force=False)

    def _flush_active_to_disk(self, force: bool) -> None:
        if self._active_df.is_empty():
            return
        if self._flush_dir is None:
            print("Flush directory is not set. Skipping flush to disk.")
            return
        if not force and self._active_df.height < self._flush_every_n:
            return
        print(f"Flushing active DataFrame to disk at chunk index {self._chunk_index}")
        chunk_filename = f"records_batch_{self._chunk_index:05d}.parquet"
        chunk_path = _write_dataframe_to_parquet(
            self._active_df,
            chunk_filename,
            self._flush_dir,
            compression="zstd",
        )
        if chunk_path is None:
            logger.warning("Processing Core: failed to write partial chunk %s", chunk_filename)
            return

        logger.info("Processing Core: writing partial records chunk to %s", chunk_path)
        self._written_files.append(chunk_path)
        self._active_df = pl.DataFrame([])
        self._chunk_index += 1

    def load_existing_chunks(self) -> Set[str]:
        """Load existing parquet chunk files from the flush directory.

        Populates `_written_files` and `_chunk_index` and returns a set of already-processed
        `path` values so the caller can skip re-processing those files.
        """
        processed: Set[str] = set()
        if self._flush_dir is None or not self._flush_dir.exists():
            return processed

        files = sorted(self._flush_dir.glob("records_batch_*.parquet"))
        if not files:
            return processed

        # adopt existing files and set next chunk index
        self._written_files = files.copy()
        max_idx = -1
        for f in files:
            stem = f.stem  # e.g. records_batch_00001
            try:
                idx = int(stem.rsplit("_", 1)[1])
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                # ignore files with unexpected names
                continue

            try:
                df = pl.read_parquet(f, columns=["path"])
                if not df.is_empty() and "path" in df.columns:
                    processed.update(df["path"].to_list())
            except Exception:
                logger.warning("Processing Core: could not read existing chunk %s", f)

        self._chunk_index = max_idx + 1
        logger.info("Processing Core: found %d existing partial chunk(s) (next chunk index=%d)", len(self._written_files), self._chunk_index)
        return processed
