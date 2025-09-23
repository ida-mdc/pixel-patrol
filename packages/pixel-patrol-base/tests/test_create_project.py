import pytest
from pathlib import Path
import os

from pixel_patrol_base import api
from pixel_patrol_base.core.project import Project
from pixel_patrol_base.core.project_settings import Settings

def test_create_project_basic(mock_project_name: str, tmp_path: Path):
    project = api.create_project(mock_project_name, tmp_path)
    assert isinstance(project, Project)
    assert project.name == mock_project_name
    assert project.base_dir == tmp_path.resolve() # Assert base_dir is set
    assert project.paths == [project.base_dir]
    assert project.records_df is None
    assert isinstance(project.settings, Settings)

def test_create_project_empty_name_not_allowed(tmp_path: Path): # Add tmp_path fixture
    with pytest.raises(ValueError, match="Project name cannot be empty or just whitespace."):
        api.create_project("", tmp_path) # Provide a dummy base_dir

def test_create_project_whitespace_name_not_allowed(tmp_path: Path): # Add tmp_path fixture
    with pytest.raises(ValueError, match="Project name cannot be empty or just whitespace."):
        api.create_project("   ", tmp_path) # Provide a dummy base_dir

def test_create_project_non_existent_base_dir(mock_project_name: str, tmp_path: Path):
    non_existent_dir = tmp_path / "no_such_dir"
    with pytest.raises(FileNotFoundError, match="Project base directory not found"):
        api.create_project(mock_project_name, non_existent_dir)

def test_create_project_base_dir_not_a_directory(mock_project_name: str, tmp_path: Path):
    test_file = tmp_path / "test_file.txt"
    test_file.touch()
    with pytest.raises(ValueError, match="Project base directory is not a directory"):
        api.create_project(mock_project_name, test_file)

def test_create_project_invalid_base_dir_type(mock_project_name: str):
    with pytest.raises(TypeError) as excinfo:
        api.create_project(mock_project_name, 12345)  # An integer is an invalid type

    actual_error_message = str(excinfo.value)

    assert "str" in actual_error_message
    assert "os.PathLike object" in actual_error_message
    assert "not 'int'" in actual_error_message

    if actual_error_message.startswith("expected str, bytes"):
        # This is the Python 3.10 format (observed on GitHub Actions)
        assert actual_error_message == "expected str, bytes or os.PathLike object, not int"
    elif actual_error_message.startswith("argument should be a str"):
        # This is the Python 3.12 format (observed locally)
        assert actual_error_message == "argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'int'"
    else:
        # If neither format is matched, fail the test with the unexpected message
        pytest.fail(f"Unexpected TypeError message format: '{actual_error_message}'")



def test_create_project_base_dir_with_relative_path(mock_project_name: str, tmp_path: Path):
    relative_dir_name = "my_relative_project_base"
    (tmp_path / relative_dir_name).mkdir()
    relative_base_dir = Path(relative_dir_name)

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        project = api.create_project(mock_project_name, relative_base_dir)
        assert isinstance(project, Project)
        # The base_dir should be resolved to the absolute path relative to tmp_path
        assert project.base_dir == (tmp_path / relative_dir_name).resolve()
    finally:
        os.chdir(original_cwd)  # Restore original CWD