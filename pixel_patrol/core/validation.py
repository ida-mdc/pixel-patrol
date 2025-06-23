import matplotlib.cm as cm
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Union, Optional, Tuple, Set
from pixel_patrol.config import DEFAULT_PRESELECTED_FILE_EXTENSIONS

logger = logging.getLogger(__name__)


def validate_project_name(name: str):
    if not isinstance(name, str):
        logger.error(f"Project name '{name}' is not a string type.")
        raise TypeError("Project name must be a string.")
    if not name or not name.strip():
        logger.error("Project name cannot be empty or just whitespace.")
        raise ValueError("Project name cannot be empty or just whitespace.")


def _is_valid_directory(path_to_check: Path) -> Tuple[bool, Optional[str]]:
    if not path_to_check.exists():
        return False, "not found"
    if not path_to_check.is_dir():
        return False, "not a directory"
    return True, None


def resolve_and_validate_base_dir(path_to_validate: Union[str, Path]) -> Path:
    """
    Validates if a project base directory exists and is a directory, raising errors on failure.
    """
    resolved_path = Path(path_to_validate).resolve()
    is_valid, reason = _is_valid_directory(resolved_path)

    if not is_valid:
        if reason == "not found":
            logger.error(f"Project base directory not found: {resolved_path}.")
            raise FileNotFoundError(f"Project base directory not found: {resolved_path}")
        elif reason == "not a directory":
            logger.error(f"Project base directory is not a directory: {resolved_path}.")
            raise ValueError(f"Project base directory is not a directory: {resolved_path}")
    return resolved_path


def resolve_and_validate_project_path(raw_path: Union[str, Path], base_dir: Path) -> Optional[Path]:
    """
    Resolves and validates a path for inclusion in a project.
    Logs warnings and returns None if invalid, or if outside base_dir.
    """
    try:
        candidate_path = Path(raw_path)
        resolved_path = candidate_path.resolve() if candidate_path.is_absolute() else (base_dir / candidate_path).resolve()

        is_valid, reason = _is_valid_directory(resolved_path)

        if not is_valid:
            logger.warning(
                f"Path not valid ('{reason}') and will be skipped: {raw_path} (resolved to {resolved_path})."
            )
            return None

        try:
            resolved_path.relative_to(base_dir)
        except ValueError:
            logger.warning(
                f"Path '{resolved_path}' is not within the project base directory "
                f"'{base_dir}' and will be skipped."
            )
            return None

        return resolved_path

    except Exception as e:
        logger.error(
            f"Error validating path '{raw_path}': {e}", exc_info=True
        )
        return None


def validate_paths_type(paths):
    if isinstance(paths, (str, Path)):
        return [paths]
    elif isinstance(paths, Iterable):
        return list(paths)
    else:
        logger.error("Project Core: Invalid paths type provided. Must be str, Path, or an iterable.")
        raise TypeError("Paths must be a string, Path, or an iterable of strings/Paths.")


def validate_and_filter_extensions(extensions: Set[str]) -> Set[str]:
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


def is_valid_colormap(cmap_name: str) -> bool:
    """
    Checks if a given string is a valid Matplotlib colormap name.
    """
    if not isinstance(cmap_name, str):
        logger.warning(f"Colormap name '{cmap_name}' is not a string type.")
        return False
    try:
        # Using get_cmap() and catching ValueError is the robust way to check.
        # This will work for both builtin and registered colormaps.
        cm.get_cmap(cmap_name)
        return True
    except ValueError:
        return False
    except Exception as e:
        logger.error(f"Unexpected error when checking colormap '{cmap_name}': {e}")
        return False