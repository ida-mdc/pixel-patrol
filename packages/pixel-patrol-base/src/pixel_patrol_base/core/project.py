import logging
from pathlib import Path
from typing import List, Union, Iterable, Optional, Set

import polars as pl

from pixel_patrol_base.config import DEFAULT_PRESELECTED_FILE_EXTENSIONS, MIN_N_EXAMPLE_IMAGES, MAX_N_EXAMPLE_IMAGES
from pixel_patrol_base.core import processing, validation
from pixel_patrol_base.core.project_settings import Settings
from pixel_patrol_base.utils.path_utils import process_new_paths_for_redundancy

logger = logging.getLogger(__name__)


class Project:

    def __init__(self, name: str, base_dir: Union[str, Path], loader: str):

        validation.validate_project_name(name)
        self.name: str = name

        self.base_dir = base_dir

        self.loader = loader

        self.paths: List[Path] = [self.base_dir]
        self.paths_df: Optional[pl.DataFrame] = None
        self.settings: Settings = Settings()
        self.images_df: Optional[pl.DataFrame] = None

        logger.info(f"Project Core: Project '{self.name}' initialized with loader {self.loader} and base dir: {self.base_dir}.")


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


    def process_paths(self) -> "Project":
        if not self.paths:
            logger.warning("No directory paths added to preprocess. paths_df will be None.")
            self.paths_df = None # Ensure it's None if no paths
        else:
            self.paths_df  = processing.build_paths_df(self.paths)
        return self


    def set_settings(self, settings: Settings) -> "Project":
        logger.info(f"Project Core: Attempting to set project settings for '{self.name}'.")

        # Handle selected_file_extensions first.
        self._set_selected_file_extensions(settings)

        # Validate cmap: Must be a valid Matplotlib colormap
        if not validation.is_valid_colormap(settings.cmap):
            logger.error(f"Project Core: Invalid colormap name '{settings.cmap}'.")
            raise ValueError(f"Invalid colormap name: '{settings.cmap}'. It is not a recognized Matplotlib colormap.")

        if not isinstance(settings.n_example_images, int) or \
            settings.n_example_images < MIN_N_EXAMPLE_IMAGES or \
            settings.n_example_images >= MAX_N_EXAMPLE_IMAGES:
            logger.error(f"Project Core: Invalid n_example_images value: {settings.n_example_images}.")
            raise ValueError("Number of example images must be an integer between 1 and 19 (i.e., positive and below 20).")

        # All validations passed, apply the new settings.
        self.settings = settings
        logger.info(f"Project Core: Project settings updated for '{self.name}'.")
        return self


    def process_images(self, settings: Optional[Settings] = None) -> "Project":
        """
        Processes images in the project, building `images_df`.
        - If `paths_df` does not exist, it performs a single, targeted file system scan to build `images_df` directly.
        - If `paths_df` already exists it leverages this existing DataFrame, filters it, and then extracts image metadata.

        Args:
            settings: An optional Settings object to apply to the project. If None, the project's current settings will be used.

        Returns:
            The Project instance with the `images_df` updated.
        """
        if settings is not None:
            logger.info("Project Core: Applying provided settings before processing images.")
            self.set_settings(settings)
        if not self.settings.selected_file_extensions:
            raise ValueError("No supported file extensions selected. Provide at least one valid extension.")
        exts = self.settings.selected_file_extensions

        if self.paths_df is None or self.paths_df.is_empty():
            self.images_df = processing.build_images_df_from_file_system(self.paths, exts, loader=self.loader)
        else:
            self.images_df = processing.build_images_df_from_paths_df(self.paths_df, exts, loader=self.loader)

        if self.images_df is None or self.images_df.is_empty():
            logger.warning(
                "Project Core: No image files found/processed. images_df will be None.")
            self.images_df = None

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

    def get_paths_df(self) -> Optional[pl.DataFrame]:
        """Get the single DataFrame containing preprocessed data."""
        return self.paths_df

    def get_images_df(self) -> Optional[pl.DataFrame]:
        """Get the single DataFrame containing processed data."""
        return self.images_df

    def get_loader(self) -> str:
        return self.loader

    def _set_selected_file_extensions(self, new_settings: Settings) -> None:
        """
        Handles the setting of `selected_file_extensions` within the Settings object.
        Performs validation and filtering against supported extensions.
        Raises ValueError if extensions are attempted to be changed after initial definition.
        """
        current_extensions_value = self.settings.selected_file_extensions
        new_extensions_input = new_settings.selected_file_extensions

        if bool(current_extensions_value):
            logger.info(
                f"Project Core: File extensions are already set to '{current_extensions_value}'. No changes allowed.")
            new_settings.selected_file_extensions = current_extensions_value
            return

        if isinstance(new_extensions_input, str) and new_extensions_input.lower() == "all":
            new_extensions_input = DEFAULT_PRESELECTED_FILE_EXTENSIONS
            new_settings.selected_file_extensions = new_extensions_input
            logger.info(
                f"Project Core: Selected file extensions set to 'all'. Using default preselected extensions: {new_extensions_input}.")
            return
        elif not isinstance(new_extensions_input, Set):
            logger.error(
                f"Project Core: Invalid type for selected_file_extensions: {type(new_extensions_input)}. Defaulting to empty set.")
            raise TypeError(
                "selected_file_extensions must be 'all' (string) or a Set of strings."
            )
        else:
            if not new_extensions_input:
                logger.warning(
                    "Project Core: No file extensions provided. Defaulting to empty set.")
                new_settings.selected_file_extensions = set()
                return
            else:
                new_extensions_input = validation.validate_and_filter_extensions(new_extensions_input)
                if not new_extensions_input:
                    new_settings.selected_file_extensions = set()
                    logger.warning(
                        "Project Core: No supported file extensions provided. The selected_file_extensions will be empty.")
                    return

        new_settings.selected_file_extensions = new_extensions_input
        logger.info(f"Project Core: Set file extensions to: {new_extensions_input}.")
