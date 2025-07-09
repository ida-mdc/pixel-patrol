import pytest
from pathlib import Path
import os

from pixel_patrol import api
from pixel_patrol.core.project import Project
from pixel_patrol.core.project_settings import Settings

def test_create_project_basic(mock_project_name: str, tmp_path: Path):
    project = api.create_project(mock_project_name, tmp_path)
    assert isinstance(project, Project)
    assert project.name == mock_project_name
    assert project.base_dir == tmp_path.resolve() # Assert base_dir is set
    assert project.paths == [project.base_dir]
    assert project.paths_df is None
    assert project.images_df is None
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
    expected_error_regex = r"(?:argument should be a |expected )str(?:, bytes)? or (?:an )?os\.PathLike object(?: where __fspath__ returns a str)?, not 'int'" 
    with pytest.raises(TypeError, match=expected_error_regex):
        api.create_project(mock_project_name, 12345)  # An integer is an invalid type

    # You might also consider a more general regex if different types produce slightly different messages,
    # or separate tests for different invalid types if their error messages vary significantly.
    # For a list, pathlib will also raise a TypeError:
    expected_error_regex_list = "argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'list'"
    with pytest.raises(TypeError, match=expected_error_regex_list):
        api.create_project(mock_project_name, ["/invalid/path"])  # A list is an invalid type


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