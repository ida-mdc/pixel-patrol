import pytest
from pathlib import Path

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
