import pytest
from pathlib import Path
import polars as pl
import zipfile
import yaml
import logging
import shutil

from pixel_patrol.core.project import Project
from pixel_patrol.core.project_settings import Settings
from pixel_patrol import api
from pixel_patrol.io.project_io import METADATA_FILENAME, PATHS_DF_FILENAME, IMAGES_DF_FILENAME
from pixel_patrol.io.project_io import _settings_to_dict # Helper for test assertions

# Configure logging for tests to capture warnings/errors
logging.basicConfig(level=logging.INFO)


@pytest.fixture # TODO: Should move to conftest.py - change name
def project_with_minimal_data(project_with_base: Project, temp_test_dirs: list[Path]) -> Project:
    """Provides a Project with base_dir and paths_df (minimal data) using conftest fixtures."""
    project = project_with_base # Already has base_dir set by fixture
    project = api.add_paths(project, temp_test_dirs)
    project = api.process_paths(project)
    return project

@pytest.fixture # TODO: Should move to conftest.py - change name
def project_with_all_data(project_with_minimal_data: Project) -> Project: # TODO: change those to not be hard coded
    """
    Provides a Project with base_dir, paths_df, images_df, and custom settings.
    Builds upon project_with_minimal_data.
    """
    project = project_with_minimal_data # Already has base_dir, paths, paths_df

    # Set some custom settings for image processing that include actual image extensions
    new_settings = Settings(
        cmap="viridis",
        n_example_images=5,
        selected_file_extensions={"jpg", "png", "tif"} # Match extensions in temp_test_dirs
    )
    project = api.set_settings(project, new_settings)

    # Build images_df based on the dummy files created in temp_test_dirs
    # This will now correctly find .jpg and .png files
    try:
        project = api.process_images(project)
    except ValueError as e:
        # If no images are found (e.g., due to specific test setup or filtering),
        # log and optionally provide a mock images_df for consistent testing.
        logging.warning(f"Could not build images_df for fixture project_with_all_data: {e}")
        # Fallback to a mock images_df if real one couldn't be built
        project.images_df = pl.DataFrame({
            "image_path": [str(project.base_dir / "test_dir1" / "fileA.jpg"), str(project.base_dir / "test_dir2" / "fileB.png")],
            "width": [100, 200],
            "height": [150, 250],
            "size_bytes": [1024, 2048],
            "is_corrupt": [False, False]
        })
    return project

# --- Tests for export_project ---

def test_export_project_empty(named_project: Project, tmp_path: Path): # TODO: We probably shouldn't allow this kind of export in the future.
    """Test exporting a newly created project with no data or custom settings."""
    export_path = tmp_path / "empty_project.zip"
    api.export_project(named_project, export_path)

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
            assert metadata['name'] == named_project.name
            assert metadata['base_dir'] is None
            assert metadata['paths'] == []
            assert metadata['settings'] == _settings_to_dict(named_project.settings)
            assert metadata['results'] is None

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
            # Use Polars' frame_equal for robust DataFrame comparison
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
            assert loaded_df.frame_equal(project_with_all_data.paths_df)

        # Verify images_df content
        with zf.open(IMAGES_DF_FILENAME) as df_file:
            loaded_df = pl.read_parquet(df_file)
            assert loaded_df.frame_equal(project_with_all_data.images_df)

def test_export_project_creates_parent_directories(named_project: Project, tmp_path: Path):
    """Test that `export_project` creates non-existent parent directories for the destination path."""
    nested_dir = tmp_path / "new_dir" / "sub_new_dir"
    export_path = nested_dir / "nested_project.zip"
    api.export_project(named_project, export_path)
    assert export_path.exists()
    assert export_path.parent.exists() # Checks if sub_new_dir was created
    assert export_path.parent.parent.exists() # Checks if new_dir was created

# --- Tests for import_project ---

def test_import_project_empty(named_project: Project, tmp_path: Path):
    """Test importing a project that was exported with no data."""
    export_path = tmp_path / "exported_empty_project.zip"
    api.export_project(named_project, export_path) # Export an empty project first

    imported_project = api.import_project(export_path)

    assert imported_project.name == named_project.name
    assert imported_project.base_dir is None
    assert imported_project.paths == []
    assert imported_project.settings == named_project.settings
    assert imported_project.paths_df is None
    assert imported_project.images_df is None
    assert imported_project.results is None

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
    assert imported_project.paths_df.frame_equal(project_with_minimal_data.paths_df)
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
    assert imported_project.paths_df.frame_equal(project_with_all_data.paths_df)
    assert imported_project.images_df is not None
    assert imported_project.images_df.frame_equal(project_with_all_data.images_df)
    assert imported_project.results == project_with_all_data.results # Assuming results is None or comparable

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

def test_import_project_malformed_metadata_settings_not_dict(named_project: Project, tmp_path: Path, caplog):
    """Test importing a project where settings in metadata.yml are not a dictionary."""
    export_path = tmp_path / "malformed_settings_project.zip"
    tmp_staging_path = tmp_path / "temp_staging_settings"
    tmp_staging_path.mkdir()

    try:
        # Manually create malformed metadata
        malformed_metadata = {
            'name': named_project.name,
            'base_dir': None,
            'paths': [],
            'settings': ["not", "a", "dict"], # Intentionally malformed settings
            'results': None
        }
        metadata_file = tmp_staging_path / METADATA_FILENAME
        with open(metadata_file, 'w') as f:
            yaml.dump(malformed_metadata, f)

        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(metadata_file, arcname=METADATA_FILENAME)

        with caplog.at_level(logging.WARNING):
            imported_project = api.import_project(export_path)
            # Check that relevant warnings were logged
            assert "Project IO: 'settings' data in metadata.yml is not a dictionary." in caplog.text
            assert "Project IO: Could not fully reconstruct Settings from dictionary. Using default settings." in caplog.text

        # Verify that default settings were applied as a fallback
        assert imported_project.settings == Settings()
    finally:
        # Clean up the temporary staging directory
        shutil.rmtree(tmp_staging_path, ignore_errors=True)


def test_import_project_malformed_metadata_paths_not_list(named_project: Project, tmp_path: Path, caplog):
    """Test importing a project where paths in metadata.yml are not a list."""
    export_path = tmp_path / "malformed_paths_project.zip"
    tmp_staging_path = tmp_path / "temp_staging_paths"
    tmp_staging_path.mkdir()

    try:
        malformed_metadata = {
            'name': named_project.name,
            'base_dir': None,
            'paths': "not a list", # Intentionally malformed paths
            'settings': _settings_to_dict(named_project.settings),
            'results': None
        }
        metadata_file = tmp_staging_path / METADATA_FILENAME
        with open(metadata_file, 'w') as f:
            yaml.dump(malformed_metadata, f)

        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(metadata_file, arcname=METADATA_FILENAME)

        with caplog.at_level(logging.WARNING):
            imported_project = api.import_project(export_path)
            # Check that a warning was logged
            assert "Project IO: 'paths' data in metadata.yml is not a list." in caplog.text

        # Verify that paths are empty as a fallback for malformed input
        assert imported_project.paths == []
    finally:
        shutil.rmtree(tmp_staging_path, ignore_errors=True)


def test_import_project_base_dir_not_found_after_export(project_with_minimal_data: Project, tmp_path: Path, caplog):
    """
    Test importing a project where the base_dir defined in metadata no longer exists on disk.
    The project should still load, but base_dir should be None, and a warning should be logged.
    """
    export_path = tmp_path / "project_with_deleted_base.zip"
    api.export_project(project_with_minimal_data, export_path)

    # Simulate base_dir being deleted or moved after export
    if project_with_minimal_data.base_dir and project_with_minimal_data.base_dir.exists():
        shutil.rmtree(project_with_minimal_data.base_dir)

    with caplog.at_level(logging.WARNING):
        imported_project = api.import_project(export_path)
        # Check that a warning about missing base_dir was logged
        assert f"Project IO: Could not set base directory '{project_with_minimal_data.base_dir}' for imported project:" in caplog.text

    # Verify that base_dir is None, but other project attributes are still loaded
    assert imported_project.base_dir is None
    assert imported_project.name == project_with_minimal_data.name
    # paths_df should still be loaded from the zip file as it's self-contained
    assert imported_project.paths_df is not None
    assert imported_project.paths_df.frame_equal(project_with_minimal_data.paths_df)
    # The 'paths' list itself is reconstructed from metadata, which holds the string paths
    assert [str(p) for p in imported_project.paths] == [str(p) for p in project_with_minimal_data.paths]
    assert imported_project.settings == project_with_minimal_data.settings

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

def test_import_project_missing_dataframe_files(named_project: Project, tmp_path: Path):
    """
    Test importing a project where DataFrame files (paths_df.parquet, images_df.parquet)
    are legitimately missing (e.g., exported before they were built).
    The project should load successfully, and the DFs should be None.
    """
    export_path = tmp_path / "missing_dfs_project.zip"
    # Export an 'empty' project, which by default will not have DFs
    api.export_project(named_project, export_path)

    imported_project = api.import_project(export_path)

    assert imported_project.name == named_project.name
    assert imported_project.paths_df is None
    assert imported_project.images_df is None
    # Other attributes should still match the empty project's state
    assert imported_project.base_dir is None
    assert imported_project.paths == []
    assert imported_project.settings == named_project.settings

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
    assert imported_project.paths_df.frame_equal(project_with_all_data.paths_df)
    assert imported_project.images_df is not None
    assert imported_project.images_df.frame_equal(project_with_all_data.images_df)
    assert imported_project.results == project_with_all_data.results # Assuming results is None or comparable