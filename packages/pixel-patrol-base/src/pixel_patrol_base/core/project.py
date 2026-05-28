import logging
from pathlib import Path
import dataclasses
from typing import Dict, List, Union, Iterable, Optional, Set, Callable
import polars as pl

from pixel_patrol_base.core import processing, validation
from pixel_patrol_base.core.contracts import PixelPatrolLoader
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.plugin_registry import discover_loader, discover_processor_plugins
from pixel_patrol_base.utils.path_utils import process_new_paths_for_redundancy, resolve_parquet_output_path
from pixel_patrol_base.io.parquet_io import save_parquet


logger = logging.getLogger(__name__)

class Project:

    def __init__(self, name: str, base_dir: Union[str, Path], loader: Optional[str]=None, output_path: Optional[Union[str, Path]]=None):

        validation.validate_project_name(name)
        self.name: str = name
        self.base_dir = base_dir

        if output_path is None:
            output_path = Path(self.base_dir) / f"{self.name}.parquet"
            logger.debug(f"Project Core: No output_path specified; inferring: '{output_path}'.")
        self.output_path: Path = resolve_parquet_output_path(output_path)

        self.loader: Optional[PixelPatrolLoader] = discover_loader(loader_id=loader) if loader else None
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

        paths_to_add_raw = validation.validate_paths_type(paths)

        validated_paths_to_process = []
        for p_input in paths_to_add_raw:
            validated_path = validation.resolve_and_validate_project_path(p_input, self.base_dir)
            if validated_path:
                validated_paths_to_process.append(validated_path)

        if not validated_paths_to_process:
            logger.info(f"Project Core: No valid or non-redundant paths provided to add to project '{self.name}'. No change.")
            return self

        initial_paths_set = set(self.paths)
        temp_final_paths_set = set(self.paths).copy()  # Start with current paths
        if len(self.paths) == 1 and self.paths[0] == self.base_dir:
            logger.debug(
                "Project Core: Explicit paths being added, removing base directory from initial paths set for redundancy check.")
            temp_final_paths_set.clear()

        updated_paths_set = process_new_paths_for_redundancy(
            validated_paths_to_process,
            temp_final_paths_set  # Use the potentially modified set
        )

        self.paths = sorted(list(updated_paths_set))

        if set(self.paths) != initial_paths_set:
            logger.debug(f"Project Core: Paths updated for project '{self.name}'. Total paths count: {len(self.paths)}.")
        else:
            logger.debug(
                f"Project Core: No change to project paths for '{self.name}'. Total paths count: {len(self.paths)}.")

        logger.debug(f"Project Core: Current project paths: {self.paths}")
        return self


    def delete_path(self, path: Union[str, Path]) -> "Project":
        logger.debug(f"Project Core: Attempting to delete path '{path}' from project '{self.name}'.")

        resolved_p_to_delete = validation.resolve_and_validate_project_path(path, self.base_dir)

        if resolved_p_to_delete is None:
            logger.error(f"Project Core: Invalid or inaccessible path '{path}' provided for deletion. Cannot proceed.")
            raise ValueError(f"Cannot delete path: '{path}' is invalid, inaccessible, or outside the project base.")

        if len(self.paths) == 1 and self.paths[0] == resolved_p_to_delete:
            self.paths = [self.base_dir]
            logger.info(
                f"Project Core: Last specific path '{resolved_p_to_delete}' deleted; re-added base directory '{self.base_dir}'.")
            return self

        initial_len = len(self.paths)
        self.paths = [p for p in self.paths if p != resolved_p_to_delete]

        if len(self.paths) < initial_len:
            logger.debug(f"Project Core: Successfully deleted path '{resolved_p_to_delete}' from project '{self.name}'.")

        else:
            logger.warning(
                f"Project Core: Path '{resolved_p_to_delete}' was not found in project '{self.name}' paths. No change.")

        return self


    def _prepare_processing_config(self, processing_config: Optional[ProcessingConfig]) -> ProcessingConfig:
        """
        Validates and fills in defaults on the provided ProcessingConfig:
        - Resolves selected_file_extensions against the loader if needed.
        """
        config: ProcessingConfig = processing_config or ProcessingConfig()

        config = dataclasses.replace(config,
                                     selected_file_extensions=_resolve_extensions(config.selected_file_extensions,
                                                                                  self.loader))

        self.metadata = config.metadata.populate_from_project(self)

        return config


    def process_records(
            self,
            processing_config: Optional[ProcessingConfig] = None,
            progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> "Project":
        """
        Processes files in the project, building records_df.

        Args:
            processing_config: Runtime options for slicing, processor selection, file
                               extensions, flush behaviour, etc. If None, defaults are used.
            progress_callback: Optional callback(done: int, total: int) called per completed record.
        """
        config = self._prepare_processing_config(processing_config)
        parts_dir = self.output_path.parent / f"_parts_{self.output_path.stem}"

        processors = discover_processor_plugins()
        if config.processors_included:
            processors = [p for p in processors if p.NAME in config.processors_included]
        elif config.processors_excluded:
            processors = [p for p in processors if p.NAME not in config.processors_excluded]

        self.records_df = processing.build_records_df(
            bases=self.paths,
            loader=self.loader,
            processors=processors,
            config=config,
            parts_dir=parts_dir,
            on_progress=progress_callback,
        )

        rgs_kwargs: Dict = {}
        if config.parquet_row_group_size is not None:
            rgs_kwargs["row_group_size"] = config.parquet_row_group_size

        if self.records_df is None:
            # Either nothing was processed, or parts were spilled to disk and
            # finalize() intentionally skipped the collect to avoid OOM.
            parts_on_disk = sorted(parts_dir.glob("part_*.parquet")) if parts_dir.exists() else []
            if not parts_on_disk:
                logger.warning("Project Core: No files found/processed. records_df will be None.")
                return self
            # Streaming path: merge parts without loading all into memory.
            logger.info(
                "Project Core: streaming %d parts → '%s'",
                len(parts_on_disk), self.output_path,
            )
            try:
                processing.save_parquet_from_parts(
                    parts_on_disk, self.output_path, self.metadata, **rgs_kwargs
                )
                processing.cleanup_chunks_dir(parts_dir)
            except Exception as e:
                logger.warning("Project Core: Could not save parquet to '%s': %s", self.output_path, e)
            return self

        if self.records_df.is_empty():
            logger.warning("Project Core: No files found/processed. records_df will be None.")
            self.records_df = None
            return self

        # In-memory path (small dataset, parts_dir=None).
        try:
            save_parquet(self.records_df, self.output_path, self.metadata, **rgs_kwargs)
            processing.cleanup_chunks_dir(parts_dir)
        except Exception as e:
            logger.warning("Project Core: Could not save parquet to '%s': %s", self.output_path, e)

        return self


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
