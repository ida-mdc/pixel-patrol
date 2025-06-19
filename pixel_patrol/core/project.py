from pathlib import Path
from typing import Any, List, Union, Iterable, Optional, Set
import logging


from pixel_patrol.core.project_settings import Settings
from  pixel_patrol.core import processing, report, validation
from pixel_patrol.utils.path_utils import is_subpath, is_superpath
from pixel_patrol.config import DEFAULT_PRESELECTED_FILE_EXTENSIONS
import polars as pl

from pixel_patrol.widgets.widget_interface import PixelPatrolWidget
from pixel_patrol.utils.widget import load_widgets

logger = logging.getLogger(__name__)

class Project:

    def __init__(self, name: str, base_dir: Union[str, Path]): # base_dir is now mandatory

        if not name or not name.strip():
            logger.error("Project Core: Project name cannot be empty or just whitespace.")
            raise ValueError("Project name cannot be empty or just whitespace.")
        self.name: str = name

        self.base_dir: Optional[Path] = None
        self.set_base_dir_internal(base_dir)

        self.paths: List[Path] = [self.base_dir]
        self.paths_df: Optional[pl.DataFrame] = None
        self.settings: Settings = Settings()

        self.widgets: List[PixelPatrolWidget] = load_widgets()
        logger.info(f"Project Core: Discovered and activated {len(self.widgets)} total widget types via entry points.")

        self.images_df: Optional[pl.DataFrame] = None
        self.results: Any = None # HTML file? All plots? # TODO: Define a better type once decided

        logger.info(f"Project Core: Project '{self.name}' initialized with base dir: {self.base_dir}.")


    def set_base_dir_internal(self, base_dir: Union[str, Path]) -> None:
        """Internal method to set and validate base_dir during initialization."""
        logger.info(f"Project Core: Attempting to set project base directory to '{base_dir}'.")
        resolved_base = Path(base_dir).resolve()

        if not resolved_base.exists():
            logger.error(f"Project Core: Specified project base directory not found: {resolved_base}.")
            raise FileNotFoundError(f"Project base directory not found: {resolved_base}")
        if not resolved_base.is_dir():
            logger.error(f"Project Core: Specified project base directory is not a directory: {resolved_base}.")
            raise ValueError(f"Project base directory is not a directory: {resolved_base}")

        self.base_dir = resolved_base
        logger.info(f"Project Core: Project base directory set to: '{self.base_dir}'.")


    def add_paths(self, paths: Union[str, Path, Iterable[Union[str, Path]]]) -> "Project":
        logger.info(f"Project Core: Attempting to add paths to project '{self.name}'.")

        if isinstance(paths, (str, Path)):
            paths_to_add_raw = [paths]
        elif isinstance(paths, Iterable):
            paths_to_add_raw = list(paths)
        else:
            logger.error("Project Core: Invalid paths type provided. Must be str, Path, or an iterable.")
            raise TypeError("Paths must be a string, Path, or an iterable of strings/Paths.")

        validated_paths_to_process = []
        for p_input in paths_to_add_raw:
            validated_path = self._is_valid_path_for_project(p_input)
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

        updated_paths_set = self._process_new_paths_for_redundancy(
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

        resolved_p_to_delete = self._is_valid_path_for_project(path)

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

        # Validate n_example_images: Must be a positive integer below 20
        # TODO: move hard coded values to config
        if not isinstance(settings.n_example_images, int) or \
           settings.n_example_images < 1 or \
           settings.n_example_images >= 20:
            logger.error(f"Project Core: Invalid n_example_images value: {settings.n_example_images}.")
            raise ValueError("Number of example images must be an integer between 1 and 19 (i.e., positive and below 20).")

        # All validations passed, apply the new settings.
        self.settings = settings
        logger.info(f"Project Core: Project settings updated for '{self.name}'.")
        return self

    def process_images(self, settings: Optional[Settings] = None) -> "Project":
        """
        Processes images in the project, building the `images_df`.

        This method offers an optimized workflow:
        - If `paths_df` does not exist, it performs a single, targeted file system scan
          to build `images_df` directly.
        - If `paths_df` already exists (meaning `process_paths()` was previously called),
          it leverages this existing DataFrame, filters it, and then extracts image metadata.

        Any `settings` provided as an argument will first be applied to the project
        (validated and persisted), and then the image processing will use the project's
        (potentially updated) settings for filtering and metadata extraction.

        Args:
            settings: An optional Settings object to apply to the project
                      before processing images. If None, the project's current
                      settings (self.settings) will be used without modification.

        Returns:
            The Project instance with the `images_df` updated.
        """
        # Apply settings if provided (this also handles validation and persistence)
        if settings is not None:
            logger.info("Project Core: Applying provided settings before processing images.")
            self.set_settings(settings)  # This method ensures validation and updates self.settings

        if self.paths_df is None:
            # Optimized path: paths_df is not built, so perform a targeted direct scan for images
            self.images_df = self._process_images_from_file_system()
        else:
            # Standard Path: paths_df exists, filter it and then extract deep metadata
            self.images_df = self._process_images_from_paths_df()

        return self


    def generate_report(self, dest: Path) -> None:
        report.generate_report(self.images_df, dest)

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

    def _is_valid_path_for_project(self, raw_path: Union[str, Path]) -> Optional[Path]:

        try:
            candidate_path = Path(raw_path)
            resolved_path = candidate_path.resolve() if candidate_path.is_absolute() else (self.base_dir / candidate_path).resolve()

            if not resolved_path.exists():
                logger.warning(
                    f"Project Core: Path not found and will be skipped: {raw_path} (resolved to {resolved_path})."
                )
                return None
            if not resolved_path.is_dir():
                logger.warning(
                    f"Project Core: Path is not a directory and will be skipped: {raw_path} (resolved to {resolved_path})."
                )
                return None

            try:
                resolved_path.relative_to(self.base_dir)  # TODO: I think this is inclusive, and thus confusing - maybe need to change.
            except ValueError:
                logger.warning(
                    f"Project Core: Path '{resolved_path}' is not within the project base directory "
                    f"'{resolved_path}' and will be skipped."
                )
                return None

            return resolved_path


        except Exception as e:
            logger.error(
                f"Project Core: Error validating path '{raw_path}': {e}", exc_info=True
            )
            return None

    @staticmethod
    def _process_new_paths_for_redundancy(validated_paths: List[Path], existing_paths_set: set[Path]) -> set[
        Path]:
        """
        Processes a list of new validated paths, handling redundancy (subpaths/superpaths)
        against a set of existing paths. Returns the updated set of paths.
        """
        final_paths_set = existing_paths_set.copy()

        for new_candidate_path in validated_paths:
            is_subpath_of_existing = False

            # Check against paths already in the final_paths_set (which includes existing + previous new candidates)
            for existing_path_in_set in list(final_paths_set): # Iterate over a copy to allow modification
                # If the new path is a subpath of an existing one, skip it
                if is_subpath(new_candidate_path, existing_path_in_set):
                    logger.warning(
                        f"Project Core: Path '{new_candidate_path}' is a subpath of existing project path "
                        f"'{existing_path_in_set}' and will be skipped to avoid redundancy."
                    )
                    is_subpath_of_existing = True
                    break
                # If the new path is a superpath of an existing one, remove the existing (redundant) path
                elif is_superpath(new_candidate_path, existing_path_in_set):
                    logger.info(
                        f"Project Core: Path '{new_candidate_path}' is a superpath of existing project path "
                        f"'{existing_path_in_set}'. Removing the subpath."
                    )
                    final_paths_set.remove(existing_path_in_set)

            if not is_subpath_of_existing:
                # Add the new candidate if it's not a subpath of any other existing/new path
                final_paths_set.add(new_candidate_path)

        return final_paths_set


    def _set_selected_file_extensions(self, new_settings: Settings) -> None:
        """
        Handles the setting of `selected_file_extensions` within the Settings object.
        Performs validation and filtering against supported extensions.
        Raises ValueError if extensions are attempted to be changed after initial definition.
        """
        current_extensions_value = self.settings.selected_file_extensions
        new_extensions_input = new_settings.selected_file_extensions

        if isinstance(new_extensions_input, str) and new_extensions_input.lower() == "all":
            new_extensions_input = DEFAULT_PRESELECTED_FILE_EXTENSIONS
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
                return
            else:
                new_extensions_input = self._validate_and_filter_extensions(new_extensions_input)
                if not new_extensions_input:
                    logger.warning(
                        "Project Core: No supported file extensions provided. The selected_file_extensions will be empty.")
                    return

        if (current_extensions_value
                and isinstance(current_extensions_value, Set)
                and current_extensions_value == new_extensions_input):
            logger.info(f"Project Core: File extensions remain unchanged as '{current_extensions_value}'.")
            return

        is_current_value_defined = bool(current_extensions_value)

        if is_current_value_defined:
            logger.error(
                f"Project Core: Attempted to change file extensions from '{current_extensions_value}' to '{new_extensions_input}'.")
            raise ValueError("File extensions cannot be changed once they have been defined for the project.")

        new_settings.selected_file_extensions = new_extensions_input
        logger.info(f"Project Core: Set file extensions to: {new_extensions_input}.")


    def _validate_and_filter_extensions(self, extensions: Set[str]) -> Set[str]:
        """
        Helper method to filter user-provided extensions against supported ones
        and log warnings for unsupported extensions.
        """
        supported_extensions = extensions.intersection(DEFAULT_PRESELECTED_FILE_EXTENSIONS)
        unsupported_extensions = extensions - supported_extensions

        if unsupported_extensions:
            logger.warning(
                f"Project Core: The following file extensions are not supported and will be ignored: "
                f"{', '.join(unsupported_extensions)}. "
                f"Supported extensions (after filtering): {', '.join(sorted(supported_extensions))}."
            )
        return supported_extensions

    def _process_images_from_file_system(self) -> Optional[pl.DataFrame]:
        """
        Private method to build images_df by performing a direct, single-pass scan
        of the file system for specified image extensions when paths_df is not available.
        """
        # self.settings.selected_file_extensions is guaranteed to be a Set[str] here
        actual_extensions_for_scan = self.settings.selected_file_extensions
        assert isinstance(actual_extensions_for_scan, Set), "selected_file_extensions must be a Set[str] at this point."

        if not actual_extensions_for_scan:
            logger.warning(
                "Project Core: No image file extensions selected. Cannot build images DataFrame via direct file system scan.")
            return None

        logger.info("Project Core: paths_df not built. Performing direct file system scan for images.")
        # This function now returns the full images_df directly
        images_df = processing.build_images_df_from_file_system(self.paths, actual_extensions_for_scan, self.widgets)

        if images_df is None or images_df.is_empty():
            logger.warning(
                "Project Core: No image files found via direct file system scan for the specified extensions. images_df will be None.")
            return None

        return images_df


    def _process_images_from_paths_df(self) -> Optional[pl.DataFrame]:
        """
        Private method to build images_df by leveraging an existing paths_df.
        It filters paths_df for image files and then extracts deep metadata.
        """
        logger.info("Project Core: paths_df exists. Filtering it for images and extracting metadata.")

        if self.paths_df.is_empty():
            logger.warning("Project Core: Existing paths_df is empty. Cannot filter for images. images_df will be None.")
            return None

        # self.settings.selected_file_extensions is guaranteed to be a Set[str] here
        actual_extensions_for_scan = self.settings.selected_file_extensions

        if not actual_extensions_for_scan:
            logger.warning("Project Core: No image file extensions selected. Cannot filter existing paths_df. images_df will be None.")
            return None

        # Filter the existing paths_df by the determined actual_extensions_for_scan
        filtered_paths_df_for_images = self.paths_df.filter(
            (pl.col("type") == "file") &
            (pl.col("file_extension").str.to_lowercase().is_in(list(actual_extensions_for_scan)))
        )

        if filtered_paths_df_for_images.is_empty():
            logger.warning(f"Project Core: No image files found in existing paths_df for extensions {actual_extensions_for_scan}. images_df will be None.")
            return None

        # Pass the already filtered DataFrame to _build_images_df_from_paths_df for deep metadata extraction
        return processing.build_images_df_from_paths_df(filtered_paths_df_for_images, self.widgets)

