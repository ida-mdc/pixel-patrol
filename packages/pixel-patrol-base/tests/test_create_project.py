import pytest
from pathlib import Path
import os

from pixel_patrol_base import api
from pixel_patrol_base.core.project import Project

def test_create_project_basic(mock_project_name: str, tmp_path: Path):
    project = api.create_project(mock_project_name, tmp_path)
    assert isinstance(project, Project)
    assert project.name == mock_project_name
    assert project.base_dir == tmp_path.resolve() # Assert base_dir is set
    assert project.paths == [project.base_dir]
    assert project.records_df is None

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
    assert "not 'int'" in actual_error_message or "not int" in actual_error_message


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

def test_output_path_inferred_when_not_provided(mock_project_name: str, tmp_path: Path):
    """When no output_path is given, it should default to base_dir/name.parquet."""
    project = api.create_project(mock_project_name, tmp_path)
    assert project.output_path == tmp_path.resolve() / f"{mock_project_name}.parquet"


def test_output_path_explicit_absolute(mock_project_name: str, tmp_path: Path):
    """An explicit absolute path is stored as-is (resolved and normalised)."""
    output = tmp_path / "results" / "my_output.parquet"
    project = api.create_project(mock_project_name, tmp_path, output_path=output)
    assert project.output_path == output.resolve()


def test_output_path_suffix_corrected(mock_project_name: str, tmp_path: Path):
    """A path without .parquet extension gets the suffix added automatically."""
    output = tmp_path / "my_output"
    project = api.create_project(mock_project_name, tmp_path, output_path=output)
    assert project.output_path.suffix == ".parquet"
    assert project.output_path == (tmp_path / "my_output.parquet").resolve()


def test_output_path_parent_dir_created(mock_project_name: str, tmp_path: Path):
    """resolve_parquet_output_path should mkdir -p the parent directory."""
    output = tmp_path / "deep" / "nested" / "output.parquet"
    assert not output.parent.exists()
    project = api.create_project(mock_project_name, tmp_path, output_path=output)
    assert output.parent.exists()
    assert project.output_path == output.resolve()


def test_output_path_relative_resolved(mock_project_name: str, tmp_path: Path):
    """A relative output_path is resolved against the current working directory."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        project = api.create_project(mock_project_name, tmp_path, output_path=Path("relative_output.parquet"))
        assert project.output_path == (tmp_path / "relative_output.parquet").resolve()
    finally:
        os.chdir(original_cwd)


def test_output_path_is_always_absolute(mock_project_name: str, tmp_path: Path):
    """output_path must always be an absolute path regardless of input."""
    output = tmp_path / "out.parquet"
    project = api.create_project(mock_project_name, tmp_path, output_path=output)
    assert project.output_path.is_absolute()


def test_output_path_inferred_is_always_absolute(mock_project_name: str, tmp_path: Path):
    """The inferred default output_path must also be absolute."""
    project = api.create_project(mock_project_name, tmp_path)
    assert project.output_path.is_absolute()


def test_output_path_string_input_accepted(mock_project_name: str, tmp_path: Path):
    """output_path can be supplied as a plain string."""
    output = str(tmp_path / "from_string.parquet")
    project = api.create_project(mock_project_name, tmp_path, output_path=output)
    assert project.output_path == Path(output).resolve()
    assert project.output_path.suffix == ".parquet"


def test_output_path_independent_of_base_dir(mock_project_name: str, tmp_path: Path):
    """output_path does not have to live inside base_dir."""
    base = tmp_path / "base"
    base.mkdir()
    output = tmp_path / "elsewhere" / "out.parquet"
    project = api.create_project(mock_project_name, base, output_path=output)
    assert project.output_path == output.resolve()
    assert not project.output_path.is_relative_to(base)