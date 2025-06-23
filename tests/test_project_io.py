import pytest
from pathlib import Path
import polars as pl
import zipfile
import yaml
import logging
import shutil # Required for shutil.rmtree

from pixel_patrol.core.project import Project
from pixel_patrol.core.project_settings import Settings
from pixel_patrol import api
from pixel_patrol.io.project_io import METADATA_FILENAME, PATHS_DF_FILENAME, IMAGES_DF_FILENAME
from pixel_patrol.io.project_io import _settings_to_dict # Helper for test assertions

# Configure logging for tests to capture warnings/errors
logging.basicConfig(level=logging.INFO)


# --- Tests for export_project ---

def test_export_project_empty(project_instance: Project, tmp_path: Path):
    """
    Test exporting a newly created project with no data or custom settings.
    An "empty" project here means it only contains its mandatory name and base_dir.
    """
    export_path = tmp_path / "empty_project.zip"
    api.export_project(project_instance, export_path)

    assert export_path.exists()
    assert zipfile.is_zipfile(export_path)

    # Verify content of the zip file
    with zipfile.ZipFile(export_path, 'r') as zf:
        namelist = zf.namelist()
        assert METADATA_FILENAME in namelist
        assert PATHS_DF_FILENAME not in namelist # Should not exist for empty project
        assert IMAGES_DF_FILENAME not in namelist # Should not exist for empty project

        # Verify metadata content
        with zf.open(METADATA_FILENAME) as meta_file:
            metadata = yaml.safe_load(meta_file)
            assert metadata['name'] == project_instance.name
            assert Path(metadata['base_dir']) == project_instance.base_dir
            # For a newly created project, its `paths` list contains only its `base_dir`
            assert [Path(p) for p in metadata['paths']] == [project_instance.base_dir]
            assert metadata['settings'] == _settings_to_dict(project_instance.settings)
            assert metadata['settings'] == _settings_to_dict(project_instance.settings)

def test_export_project_with_minimal_data(project_with_minimal_data: Project, tmp_path: Path):
    """Test exporting a project with base directory and paths_df."""
    export_path = tmp_path / "minimal_data_project.zip"
    api.export_project(project_with_minimal_data, export_path)

    assert export_path.exists()
    assert zipfile.is_zipfile(export_path)

    with zipfile.ZipFile(export_path, 'r') as zf:
        namelist = zf.namelist()
        assert METADATA_FILENAME in namelist
        assert PATHS_DF_FILENAME in namelist
        assert IMAGES_DF_FILENAME not in namelist # Not built in this fixture

        # Verify metadata
        with zf.open(METADATA_FILENAME) as meta_file:
            metadata = yaml.safe_load(meta_file)
            assert metadata['name'] == project_with_minimal_data.name
            assert Path(metadata['base_dir']) == project_with_minimal_data.base_dir
            assert [Path(p) for p in metadata['paths']] == project_with_minimal_data.paths
            assert metadata['settings'] == _settings_to_dict(project_with_minimal_data.settings)

        # Verify paths_df content
        with zf.open(PATHS_DF_FILENAME) as df_file:
            loaded_df = pl.read_parquet(df_file)
            assert loaded_df.equals(project_with_minimal_data.paths_df)

def test_export_project_with_all_data(project_with_all_data: Project, tmp_path: Path):
    """Test exporting a project with base_dir, paths_df, images_df, and custom settings."""
    export_path = tmp_path / "all_data_project.zip"
    api.export_project(project_with_all_data, export_path)

    assert export_path.exists()
    assert zipfile.is_zipfile(export_path)

    with zipfile.ZipFile(export_path, 'r') as zf:
        namelist = zf.namelist()
        assert METADATA_FILENAME in namelist
        assert PATHS_DF_FILENAME in namelist
        assert IMAGES_DF_FILENAME in namelist

        # Verify metadata
        with zf.open(METADATA_FILENAME) as meta_file:
            metadata = yaml.safe_load(meta_file)
            assert metadata['name'] == project_with_all_data.name
            assert Path(metadata['base_dir']) == project_with_all_data.base_dir
            assert [Path(p) for p in metadata['paths']] == project_with_all_data.paths
            assert metadata['settings'] == _settings_to_dict(project_with_all_data.settings)

        # Verify paths_df content
        with zf.open(PATHS_DF_FILENAME) as df_file:
            loaded_df = pl.read_parquet(df_file)
            assert loaded_df.equals(project_with_all_data.paths_df)

        # Verify images_df content
        with zf.open(IMAGES_DF_FILENAME) as df_file:
            loaded_df = pl.read_parquet(df_file)
            assert loaded_df.equals(project_with_all_data.images_df)

def test_export_project_creates_parent_directories(project_instance: Project, tmp_path: Path):
    """Test that `export_project` creates non-existent parent directories for the destination path."""
    nested_dir = tmp_path / "new_dir" / "sub_new_dir"
    export_path = nested_dir / "nested_project.zip"
    api.export_project(project_instance, export_path)
    assert export_path.exists()
    assert export_path.parent.exists() # Checks if sub_new_dir was created
    assert export_path.parent.parent.exists() # Checks if new_dir was created

# --- Tests for import_project ---

def test_import_project_empty(project_instance: Project, tmp_path: Path):
    """
    Test importing a project that was exported with no data.
    An "empty" project here means it only contains its mandatory name and base_dir
    An "empty" project here means it only contains its mandatory name and base_dir
    """
    export_path = tmp_path / "exported_empty_project.zip"
    api.export_project(project_instance, export_path) # Export an empty project first

    imported_project = api.import_project(export_path)

    assert imported_project.name == project_instance.name
    assert imported_project.base_dir == project_instance.base_dir
    assert imported_project.paths == project_instance.paths
    assert imported_project.settings == project_instance.settings
    assert imported_project.paths_df is None
    assert imported_project.images_df is None

def test_import_project_with_minimal_data(project_with_minimal_data: Project, tmp_path: Path):
    """Test importing a project with base directory and paths_df."""
    export_path = tmp_path / "exported_minimal_data_project.zip"
    api.export_project(project_with_minimal_data, export_path)

    imported_project = api.import_project(export_path)

    assert imported_project.name == project_with_minimal_data.name
    assert imported_project.base_dir == project_with_minimal_data.base_dir
    assert imported_project.paths == project_with_minimal_data.paths
    assert imported_project.settings == project_with_minimal_data.settings
    assert imported_project.paths_df is not None
    assert imported_project.paths_df.equals(project_with_minimal_data.paths_df)
    assert imported_project.images_df is None # Not built in this fixture

def test_import_project_with_all_data(project_with_all_data: Project, tmp_path: Path):
    """Test importing a project with base_dir, paths_df, images_df, and custom settings."""
    export_path = tmp_path / "exported_all_data_project.zip"
    api.export_project(project_with_all_data, export_path)

    imported_project = api.import_project(export_path)

    assert imported_project.name == project_with_all_data.name
    assert imported_project.base_dir == project_with_all_data.base_dir
    assert imported_project.paths == project_with_all_data.paths
    assert imported_project.settings == project_with_all_data.settings
    assert imported_project.paths_df is not None
    assert imported_project.paths_df.equals(project_with_all_data.paths_df)
    assert imported_project.images_df is not None
    assert imported_project.images_df.equals(project_with_all_data.images_df)

def test_import_project_non_existent_file(tmp_path: Path):
    """Test importing from a path that does not exist."""
    non_existent_path = tmp_path / "non_existent.zip"
    with pytest.raises(FileNotFoundError, match="Archive not found"):
        api.import_project(non_existent_path)

def test_import_project_non_zip_file(tmp_path: Path):
    """Test importing from a file that is not a valid zip archive."""
    non_zip_file = tmp_path / "not_a_zip.txt"
    non_zip_file.touch() # Create an empty file
    with pytest.raises(ValueError, match="Source file is not a valid zip archive"):
        api.import_project(non_zip_file)

def test_import_project_missing_metadata(tmp_path: Path):
    """Test importing from a zip file that is missing the required metadata.yml."""
    corrupted_zip_path = tmp_path / "missing_metadata.zip"
    with zipfile.ZipFile(corrupted_zip_path, 'w') as zf:
        # Add some dummy content, but intentionally omit METADATA_FILENAME
        dummy_file = tmp_path / "dummy.txt"
        dummy_file.touch()
        zf.write(dummy_file, arcname="dummy.txt")

    with pytest.raises(ValueError, match=f"Archive is missing the required '{METADATA_FILENAME}' file"):
        api.import_project(corrupted_zip_path)

def test_import_project_malformed_metadata_settings_not_dict(project_instance: Project, tmp_path: Path, caplog):
    pass


def test_import_project_malformed_metadata_paths_not_list(project_instance: Project, tmp_path: Path, caplog):
    """Test importing a project where paths in metadata.yml are not a list."""
    export_path = tmp_path / "malformed_paths_project.zip"
    tmp_staging_path = tmp_path / "temp_staging_paths"
    tmp_staging_path.mkdir()

    try:
        malformed_metadata = {
            'name': project_instance.name,
            'base_dir': str(project_instance.base_dir), # Ensure base_dir is present and a string
            'paths': "not a list", # Intentionally malformed paths
            'settings': _settings_to_dict(project_instance.settings),
        }
        metadata_file = tmp_staging_path / METADATA_FILENAME
        with open(metadata_file, 'w') as f:
            yaml.dump(malformed_metadata, f)

        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(metadata_file, arcname=METADATA_FILENAME)

        with caplog.at_level(logging.WARNING):
            imported_project = api.import_project(export_path)
            # Check that a warning was logged
            assert "Project IO: 'paths' data in metadata.yml is not a list. Found type: str" in caplog.text

        # Verify that paths are empty as a fallback for malformed input
        assert imported_project.paths == []
    finally:
        shutil.rmtree(tmp_staging_path, ignore_errors=True)


def test_import_project_base_dir_not_found_after_export(project_with_minimal_data: Project, tmp_path: Path, caplog):
    pass

def test_import_project_corrupted_dataframe_parquet(project_with_minimal_data: Project, tmp_path: Path, caplog):
    """
    Test importing a project where a DataFrame parquet file is corrupted.
    The project should load, but the corrupted DataFrame should be None, and a warning should be logged.
    """
    export_path = tmp_path / "corrupted_paths_df_project.zip"
    api.export_project(project_with_minimal_data, export_path) # Export a valid project first

    # Now, "corrupt" the paths_df.parquet inside the zip by replacing its content with invalid data
    with zipfile.ZipFile(export_path, 'a') as zf: # 'a' for append, but it can overwrite if same name
        # Write some non-parquet data instead of the actual parquet file
        zf.writestr(PATHS_DF_FILENAME, b"THIS IS NOT A VALID PARQUET FILE BUT JUNK DATA")

    with caplog.at_level(logging.WARNING):
        imported_project = api.import_project(export_path)
        # Check that a warning about not being able to read paths_df was logged
        assert "Project IO: Could not read paths_df data" in caplog.text

    # Verify that paths_df is None due to corruption, but other attributes are fine
    assert imported_project.name == project_with_minimal_data.name
    assert imported_project.paths_df is None
    assert imported_project.base_dir == project_with_minimal_data.base_dir
    assert imported_project.paths == project_with_minimal_data.paths
    assert imported_project.settings == project_with_minimal_data.settings
    assert imported_project.images_df is None # Still None for this fixture type

def test_import_project_missing_dataframe_files(project_instance: Project, tmp_path: Path):
    """
    Test importing a project where DataFrame files (paths_df.parquet, images_df.parquet)
    are legitimately missing (e.g., exported before they were built).
    The project should load successfully, and the DFs should be None.
    """
    export_path = tmp_path / "missing_dfs_project.zip"
    # Export an 'empty' project, which by default will not have DFs
    api.export_project(project_instance, export_path)

    imported_project = api.import_project(export_path)

    assert imported_project.name == project_instance.name
    assert imported_project.paths_df is None
    assert imported_project.images_df is None
    assert imported_project.base_dir == project_instance.base_dir
    assert imported_project.paths == project_instance.paths
    assert imported_project.settings == project_instance.settings

def test_export_import_project_full_cycle(project_with_all_data: Project, tmp_path: Path):
    """
    Performs a full export-import cycle with a project containing
    all possible data (base_dir, paths, settings, paths_df, images_df)
    and verifies integrity.
    """
    export_path = tmp_path / "full_cycle_project.zip"
    api.export_project(project_with_all_data, export_path)
    imported_project = api.import_project(export_path)

    # Verify all attributes are correctly preserved
    assert imported_project.name == project_with_all_data.name
    assert imported_project.base_dir == project_with_all_data.base_dir
    assert imported_project.paths == project_with_all_data.paths
    assert imported_project.settings == project_with_all_data.settings
    assert imported_project.paths_df is not None
    assert imported_project.paths_df.equals(project_with_all_data.paths_df)
    assert imported_project.images_df is not None
    assert imported_project.images_df.equals(project_with_all_data.images_df)
