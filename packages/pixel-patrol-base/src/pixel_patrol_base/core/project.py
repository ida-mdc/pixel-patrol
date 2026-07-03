import logging
from pathlib import Path
import dataclasses
from typing import Dict, List, Union, Iterable, Optional, Set, Callable
import polars as pl

from pixel_patrol_base.core import processing, validation
from pixel_patrol_base.core.contracts import PixelPatrolLoader, PixelPatrolSource
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.plugin_registry import discover_loader, discover_processor_plugins, select_source
from pixel_patrol_base.utils.path_utils import resolve_parquet_output_path
from pixel_patrol_base.io.parquet_io import save_parquet


logger = logging.getLogger(__name__)

class Project:

    def __init__(self, name: str, base_dir: Union[str, Path], loader: Optional[str]=None, output_path: Optional[Union[str, Path]]=None,
                 source: Optional[PixelPatrolSource]=None):

        validation.validate_project_name(name)
        self.name: str = name
        self.base_dir = base_dir

        if output_path is None:
            output_path = Path(self.base_dir) / f"{self.name}.parquet"
            logger.debug(f"Project Core: No output_path specified; inferring: '{output_path}'.")
        self.output_path: Path = resolve_parquet_output_path(output_path)

        self.loader: Optional[PixelPatrolLoader] = discover_loader(loader_id=loader) if loader else None
        # Optional explicit source; when None the source is auto-selected from the paths.
        self.source: Optional[PixelPatrolSource] = source
        self.paths: List[Path] = [self.base_dir]
        self.records_df: Optional[pl.DataFrame] = None

        if loader is None:
            logger.warning(f"Project Core: No loader specified for project '{self.name}'. Only basic file information will be extracted.")
        logger.debug(f"Project Core: Project '{self.name}' initialized with loader {self.loader.NAME if self.loader else 'None' } and base dir: {self.base_dir}.")


    @property
    def base_dir(self) -> Optional[Path]:
        return self._base_dir

    @base_dir.setter
    def base_dir(self, value: Union[str, Path]) -> None:
        """Set and validate the project base directory."""
        logger.debug(f"Project Core: Attempting to set project base directory to '{value}'.")
        resolved_base = validation.resolve_and_validate_base_dir(value)
        self._base_dir = resolved_base
        logger.debug(f"Project Core: Project base directory set to: '{self._base_dir}'.")


    def add_paths(self, paths: Union[str, Path, Iterable[Union[str, Path]]]) -> "Project":
        logger.debug(f"Project Core: Attempting to add paths to project '{self.name}'.")

        new_bases = [str(p) for p in validation.validate_paths_type(paths)]
        if not new_bases:
            return self

        # Validation and normalization of inputs belong to the source, not here:
        # drop the base_dir placeholder once explicit paths are added, pick the
        # source that handles these bases, and let it resolve them.
        existing = [] if self.paths == [self.base_dir] else [str(p) for p in self.paths]
        source = self.source or select_source([*existing, *new_bases])
        resolved = source.resolve_bases(new_bases, existing, self.base_dir)

        if not resolved:
            logger.info(f"Project Core: No valid or non-redundant paths provided to add to project '{self.name}'. No change.")
            return self

        self.paths = sorted(resolved, key=str)
        logger.debug(f"Project Core: Current project paths: {self.paths}")
        return self


    def _prepare_processing_config(
            self,
            processing_config: Optional[ProcessingConfig],
    ) -> tuple:
        """Resolve config, select processors, populate metadata, and log the run header.

        Returns (config, processors) so process_records stays a thin orchestrator.
        """
        config: ProcessingConfig = processing_config or ProcessingConfig()
        config = dataclasses.replace(
            config,
            selected_file_extensions=_resolve_extensions(config.selected_file_extensions, self.loader),
        )
        self.metadata = config.metadata.populate_from_project(self)

        processors = discover_processor_plugins()
        if config.processors_included:
            processors = [p for p in processors if p.NAME in config.processors_included]
        elif config.processors_excluded:
            processors = [p for p in processors if p.NAME not in config.processors_excluded]

        logger.info("Input:      %s", ", ".join(str(p) for p in self.paths))
        logger.info("Output:     %s", self.output_path)
        logger.info("Loader:     %s", self.loader.NAME if self.loader else "none")
        logger.info("Processors: %s", ", ".join(p.NAME for p in processors) or "none")

        return config, processors


    def process_records(
            self,
            processing_config: Optional[ProcessingConfig] = None,
            progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> "Project":
        """Process files and save the result parquet.

        Args:
            processing_config: Runtime options (extensions, processors, flush behaviour, …).
                               If None, defaults are used.
            progress_callback: Optional callback(done, total) called per completed record.
        """
        config, processors = self._prepare_processing_config(processing_config)
        parts_dir = self.output_path.parent / f"_parts_{self.output_path.stem}"
        processing.cleanup_chunks_dir(parts_dir)  # clear any stale parts from a previous run

        self.records_df, stats = processing.build_records_df(
            bases=self.paths,
            base_dir=self.base_dir,
            loader=self.loader,
            processors=processors,
            config=config,
            parts_dir=parts_dir,
            on_progress=progress_callback,
            source=self.source or select_source(self.paths),
        )

        self._save_result(stats, parts_dir, config)
        return self


    def _save_result(
            self,
            stats:     dict,
            parts_dir: Path,
            config:    ProcessingConfig,
    ) -> None:
        """Store stats, then write the final parquet via the appropriate path.

        Two save paths exist because build_records_df may spill to parts on disk
        (large datasets) or keep everything in memory (small datasets):
          - records_df is None  → parts on disk → save_parquet_from_parts (streaming)
          - records_df is set   → in-memory     → save_parquet
        """
        rgs_kwargs: Dict = {}
        if config.parquet_row_group_size is not None:
            rgs_kwargs["row_group_size"] = config.parquet_row_group_size

        if stats:
            self.metadata.processing_stats = stats
            _log_processing_summary(self.name, stats)

        if self.records_df is None:
            parts_on_disk = sorted(parts_dir.glob("part_*.parquet")) if parts_dir.exists() else []
            if not parts_on_disk:
                logger.warning("Project Core: No files found/processed. records_df will be None.")
                return
            logger.info("Project Core: streaming %d parts → '%s'", len(parts_on_disk), self.output_path)
            try:
                processing.save_parquet_from_parts(
                    parts_on_disk, self.output_path, self.metadata, **rgs_kwargs
                )
                processing.cleanup_chunks_dir(parts_dir)
                logger.info("Output → '%s'", self.output_path)
            except Exception as e:
                logger.warning("Project Core: Could not save parquet to '%s': %s", self.output_path, e)
            return

        if self.records_df.is_empty():
            logger.warning("Project Core: No files found/processed. records_df will be None.")
            self.records_df = None
            return

        try:
            save_parquet(self.records_df, self.output_path, self.metadata, **rgs_kwargs)
            processing.cleanup_chunks_dir(parts_dir)
            logger.info("Output → '%s'", self.output_path)
        except Exception as e:
            logger.warning("Project Core: Could not save parquet to '%s': %s", self.output_path, e)


    def get_name(self) -> str:
        """Get the project name."""
        return self.name

    def get_base_dir(self) -> Optional[Path]:
        return self.base_dir

    def get_paths(self) -> List[Path]:
        """Get the list of directory paths added to the project."""
        return self.paths

    def get_records_df(self) -> Optional[pl.DataFrame]:
        """Get the single DataFrame containing processed data."""
        return self.records_df

    def get_loader(self) -> PixelPatrolLoader:
        return self.loader

    def get_output_path(self) -> Path:
        return self.output_path


def _log_processing_summary(project_name: str, stats: dict) -> None:
    wall_s  = stats.get("wall_s", 0.0)
    n_files = stats.get("n_files", 0)
    n_tasks = stats.get("n_tasks", 0)
    n_w     = stats.get("n_workers", 0)
    load_s  = stats.get("load_cpu_s", 0.0)
    proc_s  = {k[5:]: v for k, v in stats.items() if k.startswith("proc_")}

    def _fmt_s(s: float) -> str:
        if s < 60:
            return f"{s:.1f} s"
        m, sec = divmod(int(s), 60)
        return f"{m}m {sec:02d}s" if m < 60 else f"{m // 60}h {m % 60:02d}m"

    throughput = n_files / wall_s if wall_s > 0 else 0.0
    logger.info("Done:       %d files in %s  ·  %.1f files/s", n_files, _fmt_s(wall_s), throughput)

    total_cpu = load_s + sum(proc_s.values())
    logger.debug("processing stats: wall=%s cpu=%s tasks=%d workers=%d",
                 _fmt_s(wall_s), _fmt_s(total_cpu), n_tasks, n_w)
    for stage, cpu_s in ([("loading", load_s)] + list(proc_s.items())):
        logger.debug("  %-20s %s  (%.1f s/file)", stage, _fmt_s(cpu_s),
                     cpu_s / n_files if n_files else 0)


def _resolve_extensions(
        proposed: Union[Set[str], str],
        loader: Optional[PixelPatrolLoader],
) -> Union[Set[str], str]:
    """
    Resolves selected_file_extensions against the loader's supported extensions.

    Rules:
    - "all" with loader    -> loader.SUPPORTED_EXTENSIONS
    - "all" without loader -> "all"
    - Set[str] with loader -> lowercased, filtered against SUPPORTED_EXTENSIONS (warns on unknowns)
    - Set[str] no loader   -> lowercased as-is
    - Empty set            -> empty set (caller decides whether to error)
    - Other type           -> TypeError
    """
    if isinstance(proposed, str) and proposed.lower() == "all":
        if loader is None:
            logger.debug("Project Core: All file extensions are selected.")
            return "all"
        else:
            logger.debug(f"Project Core: Using loader-supported extensions: {loader.SUPPORTED_EXTENSIONS}")
            return loader.SUPPORTED_EXTENSIONS

    if isinstance(proposed, set):
        proposed = {x.lower() for x in proposed if isinstance(x, str)}
        if not proposed:
            logger.warning("Project Core: selected_file_extensions is an empty set - no file will be processed.")
            return set()
        if loader is None:
            logger.debug(f"Project Core: File extensions selected: {proposed}")
            return proposed
        else:
            resolved = validation.validate_and_filter_extensions(proposed, loader.SUPPORTED_EXTENSIONS)
            if not resolved:
                logger.warning(
                    "Project Core: No loader-supported file extensions provided. No files will be processed.")
                return set()
            logger.debug(f"Project Core: File extensions set to: {resolved}.")
            return resolved

    logger.error(f"Project Core: Invalid type for selected_file_extensions: {type(proposed)}")
    raise TypeError("selected_file_extensions must be 'all' or a Set[str].")
