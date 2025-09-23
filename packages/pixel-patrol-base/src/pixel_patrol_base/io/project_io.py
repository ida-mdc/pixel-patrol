import zipfile
import numpy as np
import yaml
import polars as pl
import tempfile
from pathlib import Path
from typing import Optional, Any, List, Dict, Tuple
import logging
import dataclasses

from pixel_patrol_base.core.project import Project
from pixel_patrol_base.core.project_settings import Settings
from pixel_patrol_base.core import validation

logger = logging.getLogger(__name__)

METADATA_FILENAME = 'metadata.yml'
RECORDS_DF_FILENAME = 'records_df.parquet'


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
    """
    Converts a dictionary from YAML import back into a Settings dataclass instance.
    Handles cases where the input is not a dictionary or has malformed parts.
    """
    # Ensure settings_dict is actually a dictionary before attempting to copy or access
    if not isinstance(settings_dict, dict):
        logger.warning(
            f"Project IO: _dict_to_settings received non-dictionary input: {type(settings_dict).__name__}. Using default settings."
        )
        return Settings()

    s_dict = settings_dict.copy()

    # Handle 'selected_file_extensions' conversion from list to set
    try:
        if 'selected_file_extensions' in s_dict and isinstance(s_dict['selected_file_extensions'], list):
            s_dict['selected_file_extensions'] = set(s_dict['selected_file_extensions'])
    except Exception as e:
        logger.warning(
            f"Settings IO: Could not convert 'selected_file_extensions' back to set. Error: {e}.")

    # Reconstruct the Settings object
    # Filter out keys not present in Settings dataclass to avoid TypeError
    # This ensures robustness against older metadata formats or unexpected keys
    valid_settings_keys = {f.name for f in dataclasses.fields(Settings)}
    filtered_s_dict = {k: v for k, v in s_dict.items() if k in valid_settings_keys}

    try:
        return Settings(**filtered_s_dict)
    except Exception as e:
        logger.warning(
            f"Project IO: Could not fully reconstruct Settings from filtered dictionary. Using default settings. Error: {e}")
        return Settings()


def _serialize_ndarray_columns_dataframe(polars_df: pl.DataFrame) -> pl.DataFrame:
    """
    Serializes columns containing numpy ndarrays to lists of int64 for compatibility with Parquet.
    This is necessary because Polars does not support direct serialization of numpy ndarrays.
    Args:
        df: The Polars DataFrame to process.
    Returns:
        A Polars DataFrame with ndarray columns serialized to lists.
    """
    for col in polars_df.columns:
        if polars_df[col].dtype == pl.Object:
            try:
                # Attempt to convert ndarray columns to lists
                polars_df = polars_df.with_columns(
                    pl.col(col).map_elements(lambda x: x.tolist() if isinstance(x, np.ndarray) else x, return_dtype=pl.List(pl.Int64))
                )
                logger.info(f"Project IO: Successfully serialized column '{col}' from ndarray to list.")
            except Exception as e:
                logger.warning(f"Project IO: Failed to serialize column '{col}' to list. Error: {e}. This column will be excluded from the Parquet export.")
                polars_df = polars_df.drop(col)
    return polars_df

def _deserialize_ndarray_columns_dataframe(polars_df: pl.DataFrame) -> pl.DataFrame:
    """
    Deserializes columns containing lists of int64 back to numpy ndarrays.
    Args:
        polars_df: The Polars DataFrame to process.
    Returns:
        A Polars DataFrame with list columns deserialized to ndarrays.
    """
    for col in polars_df.columns:
        if polars_df[col].dtype == pl.List(pl.Int64):
            try:
                polars_df = polars_df.with_columns(
                    pl.col(col)
                    .map_elements(
                        lambda x: np.array(x) if isinstance(x, (list, pl.Series)) else x, # in case a series appears in a saved dataframe.
                        return_dtype=pl.Object,
                    )
                    .cast(pl.Object)
                )
                logger.info(f"Project IO: Successfully deserialized column '{col}' from list to ndarray.")
            except Exception as e:
                logger.warning(f"Project IO: Failed to deserialize column '{col}' to ndarray. Error: {e}. This column will be excluded from the DataFrame.")
                polars_df = polars_df.drop(col)
    return polars_df

def _write_dataframe_to_parquet(
        df: Optional[pl.DataFrame],
        base_filename: str,
        tmp_path: Path,
) -> Optional[Path]:
    """Helper to write an optional Polars DataFrame to a Parquet file in a temporary path."""
    if df is None:
        return None

    df = _serialize_ndarray_columns_dataframe(df)

    # Identify columns with empty Struct types
    empty_struct_cols = [
        name for name, dtype in df.schema.items()
        if isinstance(dtype, pl.Struct) and not dtype.fields
    ]

    # Add a dummy field to each problematic column
    for col in empty_struct_cols:
        df = df.with_columns(pl.lit(None).alias(col))

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
        'loader': getattr(getattr(project, "loader", None), "NAME", None),
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
    - records_df.parquet (if exists): Processed data.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        files_for_zip: List[Tuple[Path, str]] = []

        # 1. Prepare and write metadata
        metadata_content = _prepare_project_metadata(project)
        metadata_file_path = _write_metadata_to_tmp(metadata_content, tmp_path)
        files_for_zip.append((metadata_file_path, METADATA_FILENAME))

        records_df_tmp_path = _write_dataframe_to_parquet(project.records_df, RECORDS_DF_FILENAME, tmp_path)
        if records_df_tmp_path:
            files_for_zip.append((records_df_tmp_path, RECORDS_DF_FILENAME))

        # 2. Create the zip archive with all prepared files
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
        df = _deserialize_ndarray_columns_dataframe(df)
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


def _reconstruct_project_core_data(
    metadata_content: Dict[str, Any],
    has_records_df: bool
) -> Project:
    name = metadata_content.get('name', 'Imported Project')
    base_dir_str = metadata_content.get('base_dir')
    paths_str_list = metadata_content.get('paths', []) # This gets "not a list" from the malformed metadata
    settings_dict = metadata_content.get('settings', {})
    loader_id = metadata_content.get('loader')

    if not isinstance(paths_str_list, list): # This condition is TRUE for the test case
        logger.warning(
            f"Project IO: 'paths' data in {METADATA_FILENAME} is not a list. Found type: {type(paths_str_list).__name__}")
        paths_str_list = []

    # Initialize with a dummy base_dir first, will be updated based on validation
    if loader_id:
        try:
            project = Project(name, Path.cwd(), loader=loader_id)
        except Exception as e:
            logger.warning(
                f"Project IO: Loader '{loader_id}' from metadata could not be resolved. "
                f"Proceeding without a loader. Error: {e}"
            )
            project = Project(name, Path.cwd())
    else:
        project = Project(name, Path.cwd())

    # Handle base_dir
    if base_dir_str is not None:
        imported_base_dir = Path(base_dir_str)
        if has_records_df:
            # If records_df exists, set base_dir directly, warn if not found
            project._base_dir = imported_base_dir # Set directly to bypass setter validation
            if not imported_base_dir.exists():
                logger.warning(
                    f"Project IO: Imported project's base directory '{imported_base_dir}' does not exist on the file system. "
                    "This project has processed files data (records_df), so it can still be used for analysis, "
                    "but path-dependent operations may be limited."
                )
            else:
                logger.info(f"Project IO: Imported project base directory set to '{imported_base_dir}'.")
        else:
            # If records_df does NOT exist, base_dir MUST be valid
            try:
                project.base_dir = imported_base_dir # Use setter for full validation
                logger.info(f"Project IO: Imported project base directory set to '{project.base_dir}'.")
            except (FileNotFoundError, ValueError) as e:
                raise ValueError(
                    f"Project requires file system access but imported base directory '{imported_base_dir}' is invalid or inaccessible: {e}. "
                    "Cannot load project without processed data (e.g. image data)."
                ) from e
    else:
        # If base_dir was None in metadata, and no records_df, this is an issue.
        # If records_df exists, it's less critical, but still note it.
        if not has_records_df:
            raise ValueError(
                "Project requires file system access but no base directory was specified in the imported metadata. "
                "Cannot load project without processed data (e.g. image data)."
            )
        else:
            logger.warning("Project IO: No base directory specified in the imported metadata. "
                           "Project has processed files (records_df), but path-dependent operations may be limited.")
            project._base_dir = None


    # Handle paths
    reconstructed_paths: List[Path] = []
    if has_records_df:
        # If records_df exists, add paths as is, warn if not found
        for p_str in paths_str_list: # If paths_str_list is correctly [], this loop does not run.
            p = Path(p_str)
            if not p.exists():
                logger.warning(
                    f"Project IO: Imported path '{p}' does not exist on the file system. "
                    "This project has processed files (records_df), so it can still be used for analysis, "
                    "but path-dependent operations may fail."
                )
            reconstructed_paths.append(p)
        project.paths = reconstructed_paths
    else:
        # If records_df does NOT exist, paths MUST be valid
        for p_str in paths_str_list: # If paths_str_list is correctly [], this loop does not run.
            try:
                # Use resolve_and_validate_project_path for proper validation relative to base_dir
                validated_path = validation.resolve_and_validate_project_path(Path(p_str), project.base_dir)
                if validated_path:
                    reconstructed_paths.append(validated_path)
                else:
                    raise ValueError(f"Path '{p_str}' is invalid, inaccessible, or outside the project base directory.")
            except (FileNotFoundError, ValueError) as e:
                raise ValueError(
                    f"Project requires file system access but imported path '{p_str}' is invalid or inaccessible: {e}. "
                    "Cannot load project without processed files (e.g. image data)."
                ) from e
        project.paths = reconstructed_paths

    # Call _dict_to_settings with the potentially malformed settings_dict
    # _dict_to_settings is now robust to non-dictionary inputs.
    project.settings = _dict_to_settings(settings_dict)

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

        # First, try to read records_df to determine validation behavior
        imported_records_df = _read_dataframe_from_parquet(
            tmp_path / RECORDS_DF_FILENAME,
            src
        )
        has_records_df = imported_records_df is not None and not imported_records_df.is_empty()
        logger.info(f"Project IO: Imported archive {'HAS' if has_records_df else 'DOES NOT HAVE'} records_df.")

        project = _reconstruct_project_core_data(metadata_content, has_records_df)

        project.records_df = imported_records_df # Assign the already read records_df

        return project
