import zipfile
import yaml
import polars as pl
import tempfile
from pathlib import Path
from typing import Optional, Any, List, Dict, Tuple
import logging

from pixel_patrol.core.project import Project
from pixel_patrol.core.project_settings import Settings

logger = logging.getLogger(__name__)

METADATA_FILENAME = 'metadata.yml'
PATHS_DF_FILENAME = 'paths_df.parquet'
IMAGES_DF_FILENAME = 'images_df.parquet'


def _settings_to_dict(settings: Settings) -> dict:
    """
    Converts a Settings dataclass instance to a dictionary for YAML export.
    Converts 'selected_file_extensions' from a set to a sorted list for cleaner YAML output.
    """
    s_dict = settings.__dict__.copy()
    # CONVERT SET TO SORTED LIST FOR YAML READABILITY
    if 'selected_file_extensions' in s_dict and isinstance(s_dict['selected_file_extensions'], set):
        s_dict['selected_file_extensions'] = sorted(list(s_dict['selected_file_extensions']))
    return s_dict


def _dict_to_settings(settings_dict: dict) -> Settings:
    """Converts a dictionary from YAML import back into a Settings dataclass instance."""
    s_dict = settings_dict.copy()
    try:
        # If 'selected_file_extensions' was stored as a list, convert it back to a set
        if 'selected_file_extensions' in s_dict and isinstance(s_dict['selected_file_extensions'], list):
            s_dict['selected_file_extensions'] = set(s_dict['selected_file_extensions'])
        return Settings(**s_dict)
    except TypeError as e:
        logger.warning(
            f"Project IO: Could not fully reconstruct Settings from dictionary. Using default settings. Error: {e}")
        return Settings()


def _write_dataframe_to_parquet(
        df: Optional[pl.DataFrame],
        base_filename: str,
        tmp_path: Path,
) -> Optional[Path]:
    """Helper to write an optional Polars DataFrame to a Parquet file in a temporary path."""
    if df is None:
        return None
    file_path = tmp_path / base_filename
    data_name = file_path.stem
    try:
        df.write_parquet(file_path)
        return file_path
    except Exception as e:
        logger.warning(f"Project IO: Could not write {data_name} data ({base_filename}) to temporary file: {e}")
        return None


def _prepare_project_metadata(project: Project) -> Dict[str, Any]:
    metadata_content = {
        'name': project.name,  # Ensure name is first
        'base_dir': str(project.base_dir) if project.base_dir else None,
        'paths': [str(p) for p in project.paths],
        'settings': _settings_to_dict(project.settings),
    }
    return metadata_content


def _write_metadata_to_tmp(metadata_content: Dict[str, Any], tmp_path: Path) -> Path:
    """Writes the project metadata to a temporary YAML file."""
    metadata_file_path = tmp_path / METADATA_FILENAME
    try:
        with open(metadata_file_path, 'w') as f:
            yaml.dump(metadata_content, f, default_flow_style=False)
        return metadata_file_path
    except Exception as e:
        raise IOError(f"Could not write {METADATA_FILENAME} to temporary directory: {e}") from e


def _add_files_to_zip(
        zip_file_path: Path,
        files_to_add: List[Tuple[Path, str]]
) -> None:
    """
    Creates or updates a zip archive with specified files.
    Args:
        zip_file_path: The path to the zip archive to create/update.
        files_to_add: A list of tuples, where each tuple is (source_path_in_tmp, arcname_in_zip).
    """
    try:
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for source_path, arcname in files_to_add:
                if source_path.exists():
                    zf.write(source_path, arcname=arcname)
                else:
                    logger.warning(f"Project IO: Skipping missing file {source_path.name} for zip archive.")

    except Exception as e:
        raise IOError(f"Could not create or write to zip archive at {zip_file_path}: {e}") from e


def export_project(project: Project, dest: Path) -> None:
    """
    Exports the project state to a zip archive.
    Args:
        project: The Project object to export.
        dest: The destination path for the zip archive (e.g., 'my_project.zip').

    Archive contains:
    - metadata.yml: Project name, paths (as strings), settings.
    - paths_df.parquet (if exists): Preprocessed data.
    - images_df.parquet (if exists): Processed data.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        files_for_zip: List[Tuple[Path, str]] = []

        # 1. Prepare and write metadata
        metadata_content = _prepare_project_metadata(project)
        metadata_file_path = _write_metadata_to_tmp(metadata_content, tmp_path)
        files_for_zip.append((metadata_file_path, METADATA_FILENAME))

        # 2. Write DataFrames to temporary files and add to list for zipping
        paths_df_tmp_path = _write_dataframe_to_parquet(project.paths_df, PATHS_DF_FILENAME, tmp_path)
        if paths_df_tmp_path:
            files_for_zip.append((paths_df_tmp_path, PATHS_DF_FILENAME))

        images_df_tmp_path = _write_dataframe_to_parquet(project.images_df, IMAGES_DF_FILENAME, tmp_path)
        if images_df_tmp_path:
            files_for_zip.append((images_df_tmp_path, IMAGES_DF_FILENAME))

        # 3. Create the zip archive with all prepared files
        _add_files_to_zip(dest, files_for_zip)


def _read_dataframe_from_parquet(
        file_path: Path,
        src_archive: Path
) -> Optional[pl.DataFrame]:
    """Helper to read an optional Polars DataFrame from a Parquet file."""
    if not file_path.exists():
        return None
    data_name = file_path.stem
    try:
        df = pl.read_parquet(file_path)
        return df
    except Exception as e:
        logger.warning(f"Project IO: Could not read {data_name} data from '{file_path.name}' "
                       f"in archive '{src_archive.name}'. Data not loaded. Error: {e}")
        return None


def _validate_source_archive(src: Path) -> None:
    """Performs initial validation checks on the source archive path."""
    if not src.exists():
        raise FileNotFoundError(f"Archive not found: {src}")
    if not zipfile.is_zipfile(src):
        raise ValueError(f"Source file is not a valid zip archive: {src}")


def _extract_archive_contents(src: Path, tmp_path: Path) -> None:
    """Extracts the contents of the source archive to a temporary directory."""
    try:
        with zipfile.ZipFile(src, 'r') as zf:
            zf.extractall(tmp_path)
    except zipfile.BadZipFile:
        raise ValueError(f"Could not read zip archive: {src}. It might be corrupted.") from None
    except Exception as e:
        raise IOError(f"Error extracting archive {src}: {e}") from e


def _read_and_validate_metadata(tmp_path: Path, src_archive: Path) -> Dict[str, Any]:
    """Reads and validates the metadata.yml file from the temporary directory."""
    metadata_file = tmp_path / METADATA_FILENAME
    if not metadata_file.exists():
        raise ValueError(f"Archive is missing the required '{METADATA_FILENAME}' file: {src_archive}")
    try:
        with open(metadata_file, 'r') as f:
            metadata_content = yaml.safe_load(f)
        if not isinstance(metadata_content, dict):
            raise ValueError(f"{METADATA_FILENAME} content is not a dictionary.")
        return metadata_content
    except yaml.YAMLError as e:
        raise ValueError(f"Could not parse {METADATA_FILENAME} from archive {src_archive}: {e}") from e
    except Exception as e:
        raise IOError(f"Error reading {METADATA_FILENAME} from archive {src_archive}: {e}") from e


def _reconstruct_project_core_data(metadata_content: Dict[str, Any]) -> Project:
    name = metadata_content.get('name', 'Imported Project')
    base_dir_str = metadata_content.get('base_dir')
    paths_str_list = metadata_content.get('paths', [])
    settings_dict = metadata_content.get('settings', {})

    if not isinstance(paths_str_list, list):
        logger.warning(
            f"Project IO: 'paths' data in {METADATA_FILENAME} is not a list. Found type: {type(paths_str_list).__name__}")
        paths_str_list = []

    if not isinstance(settings_dict, dict):
        logger.warning(
            f"Project IO: 'settings' data in {METADATA_FILENAME} is not a dictionary. Found type: {type(settings_dict).__name__}")
        settings_dict = {}

    project = Project(name, Path.cwd())  # Initialize with a dummy base_dir first, then update
    if base_dir_str is not None:
        try:
            project.base_dir = Path(base_dir_str)  # Corrected line
        except (FileNotFoundError, ValueError) as e:
            logger.warning(
                f"Project IO: Could not set base directory '{base_dir_str}' for imported project: {e}. Project will retain initial base_dir.")

    project.paths = [Path(p_str) for p_str in paths_str_list]
    try:
        project.settings = _dict_to_settings(settings_dict)
    except Exception as e:
        logger.warning(
            f"Project IO: Could not fully reconstruct settings from metadata. Using default settings. Error: {e}")
        project.settings = Settings()

    return project


def import_project(src: Path) -> Project:
    """
    Imports a project state from a zip archive.
    Reconstructs and returns a Project object.
    Args:
        src: The path to the zip archive to import.
    Returns:
        A reconstructed Project object.
    """
    _validate_source_archive(src)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        _extract_archive_contents(src, tmp_path)

        metadata_content = _read_and_validate_metadata(tmp_path, src)

        project = _reconstruct_project_core_data(metadata_content)

        project.paths_df = _read_dataframe_from_parquet(
            tmp_path / PATHS_DF_FILENAME,
            src
        )
        project.images_df = _read_dataframe_from_parquet(
            tmp_path / IMAGES_DF_FILENAME,
            src
        )

        return project
