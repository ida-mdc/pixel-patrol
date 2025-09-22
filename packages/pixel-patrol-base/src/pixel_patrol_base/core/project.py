import logging
from pathlib import Path
from typing import List, Union, Iterable, Optional, Set

import polars as pl

from pixel_patrol_base.config import MIN_N_EXAMPLE_FILES, MAX_N_EXAMPLE_FILES
from pixel_patrol_base.core import processing, validation
from pixel_patrol_base.core.contracts import PixelPatrolLoader
from pixel_patrol_base.core.project_settings import Settings
from pixel_patrol_base.plugin_registry import discover_loader
from pixel_patrol_base.utils.path_utils import process_new_paths_for_redundancy

logger = logging.getLogger(__name__)

class Project:

    def __init__(self, name: str, base_dir: Union[str, Path], loader: Optional[str]=None):

        validation.validate_project_name(name)
        self.name: str = name

        self.base_dir = base_dir

        self.loader: Optional[PixelPatrolLoader] = discover_loader(loader_id=loader) if loader else None

        self.paths: List[Path] = [self.base_dir]
        self.settings: Settings = Settings()
        self.records_df: Optional[pl.DataFrame] = None

        if loader is None:
            logger.warning(f"Project Core: No loader specified for project '{self.name}'. Only basic file information will be extracted.")
        logger.info(f"Project Core: Project '{self.name}' initialized with loader {self.loader.NAME if self.loader else "None" } and base dir: {self.base_dir}.")


    @property
    def base_dir(self) -> Optional[Path]:
        """Get the project base directory."""
        return self._base_dir

    @base_dir.setter
    def base_dir(self, value: Union[str, Path]) -> None:
        """Set and validate the project base directory."""
        logger.info(f"Project Core: Attempting to set project base directory to '{value}'.")
        resolved_base = validation.resolve_and_validate_base_dir(value)
        self._base_dir = resolved_base
        logger.info(f"Project Core: Project base directory set to: '{self._base_dir}'.")


    def add_paths(self, paths: Union[str, Path, Iterable[Union[str, Path]]]) -> "Project":
        logger.info(f"Project Core: Attempting to add paths to project '{self.name}'.")

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
            logger.info(
                "Project Core: Explicit paths being added, removing base directory from initial paths set for redundancy check.")
            temp_final_paths_set.clear()

        updated_paths_set = process_new_paths_for_redundancy(
            validated_paths_to_process,
            temp_final_paths_set  # Use the potentially modified set
        )

        self.paths = sorted(list(updated_paths_set))

        if set(self.paths) != initial_paths_set:
            logger.info(f"Project Core: Paths updated for project '{self.name}'. Total paths count: {len(self.paths)}.")
        else:
            logger.info(
                f"Project Core: No change to project paths for '{self.name}'. Total paths count: {len(self.paths)}.")

        logger.debug(f"Project Core: Current project paths: {self.paths}")
        return self


    def delete_path(self, path: Union[str, Path]) -> "Project":
        logger.info(f"Project Core: Attempting to delete path '{path}' from project '{self.name}'.")

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
            logger.info(f"Project Core: Successfully deleted path '{resolved_p_to_delete}' from project '{self.name}'.")

        else:
            logger.warning(
                f"Project Core: Path '{resolved_p_to_delete}' was not found in project '{self.name}' paths. No change.")

        return self

    def set_settings(self, settings: Settings) -> "Project":
        logger.info(f"Project Core: Attempting to set project settings for '{self.name}'.")

        # Handle selected_file_extensions first.
        self._set_selected_file_extensions(settings)

        # Validate cmap: Must be a valid Matplotlib colormap
        if not validation.is_valid_colormap(settings.cmap):
            logger.error(f"Project Core: Invalid colormap name '{settings.cmap}'.")
            raise ValueError(f"Invalid colormap name: '{settings.cmap}'. It is not a recognized Matplotlib colormap.")

        if not isinstance(settings.n_example_files, int) or \
            settings.n_example_files < MIN_N_EXAMPLE_FILES or \
            settings.n_example_files >= MAX_N_EXAMPLE_FILES:
            logger.error(f"Project Core: Invalid n_example_files value: {settings.n_example_files}.")
            raise ValueError("Number of example files must be an integer between 1 and 19 (i.e., positive and below 20).")

        # All validations passed, apply the new settings.
        self.settings = settings
        logger.info(f"Project Core: Project settings updated for '{self.name}'.")
        return self


    def process_records(self, settings: Optional[Settings] = None) -> "Project":
        """
        Processes records (e.g. images) in the project, building `records_df`.
        Args:
            settings: An optional Settings object to apply to the project. If None, the project's current settings will be used.

        Returns:
            The Project instance with the `records_df` updated.
        """
        if settings is not None:
            logger.info("Project Core: Applying provided settings before processing files.")
            self.set_settings(settings)
        if not self.settings.selected_file_extensions:
            raise ValueError("No supported file extensions selected. Provide at least one valid extension.")
        exts = self.settings.selected_file_extensions

        self.records_df = processing.build_records_df(self.paths, exts, loader=self.loader)

        if self.records_df is None or self.records_df.is_empty():
            logger.warning(
                "Project Core: No files found/processed. records_df will be None.")
            self.records_df = None

        return self

    def get_name(self) -> str:
        """Get the project name."""
        return self.name

    def get_base_dir(self) -> Optional[Path]:
        return self.base_dir

    def get_paths(self) -> List[Path]:
        """Get the list of directory paths added to the project."""
        return self.paths

    def get_settings(self) -> Settings:
        """Get the current project settings."""
        return self.settings

    def get_records_df(self) -> Optional[pl.DataFrame]:
        """Get the single DataFrame containing processed data."""
        return self.records_df

    def get_loader(self) -> PixelPatrolLoader:
        return self.loader

    def _set_selected_file_extensions(self, new_settings: Settings) -> None:
        """
        Set `selected_file_extensions` on `new_settings`.
        Rules:
        - If already set on `self.settings`: keep as-is (immutable for this project instance).
        - If input == "all":
            * with loader  -> use `loader.SUPPORTED_EXTENSIONS`
            * without loader -> 'all'
        - If input is a Set[str]: lowercase, then
            * with loader -> filter against `SUPPORTED_EXTENSIONS`.
            * without loader -> use as-is.
        Raises:
        - TypeError: if input is neither "all" nor a Set[str].
        """

        existing_extensions = self.settings.selected_file_extensions
        proposed_extensions = new_settings.selected_file_extensions

        if existing_extensions:
            logger.info(f"Project Core: selected_file_extensions already set; keeping existing value: {existing_extensions}")
            new_settings.selected_file_extensions = existing_extensions
            return

        if isinstance(proposed_extensions, str) and proposed_extensions.lower() == 'all':
            if self.loader is None:
                new_settings.selected_file_extensions = 'all'
                logger.info("Project Core: All file extensions are selected")
                return
            else:
                new_settings.selected_file_extensions = self.loader.SUPPORTED_EXTENSIONS
                logger.info(f"Project Core: Using loader-supported extensions: {new_settings.selected_file_extensions}")
                return

        if isinstance(proposed_extensions, Set):
            proposed_extensions = {x.lower() for x in proposed_extensions if isinstance(x, str)}
            if not proposed_extensions:
                new_settings.selected_file_extensions = set()
                logger.warning(f"Project Core: selected_file_extensions is an empty set - no file will be processed")
                return
            if not self.loader:
                new_settings.selected_file_extensions = proposed_extensions
                logger.info(f"Project Core: File extensions are selected: {proposed_extensions}")
                return
            else:
                proposed_extensions = validation.validate_and_filter_extensions(proposed_extensions, self.loader.SUPPORTED_EXTENSIONS)
                if not proposed_extensions:
                    new_settings.selected_file_extensions = set()
                    logger.warning("Project Core: No loader supported file extensions provided. No files will be processed.")
                    return
                new_settings.selected_file_extensions = proposed_extensions
                logger.info(f"Project Core: Set file extensions to: {proposed_extensions}.")
                return
        else:
            logger.error(f"Project Core: Invalid type for selected_file_extensions: {type(proposed_extensions)}")
            raise TypeError("selected_file_extensions must be 'all' or a Set[str].")
