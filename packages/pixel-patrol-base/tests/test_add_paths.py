import pytest
import re
from pathlib import Path
from typing import List
import logging

from pixel_patrol_base import api
from pixel_patrol_base.core.project import Project


# Fixture for project_instance is already in conftest.py, using that.

def test_project_init_paths_with_base_dir(project_instance: Project, tmp_path: Path):
    """Test that Project is initialized with base_dir in its paths list."""
    assert project_instance.paths == [tmp_path.resolve()]
    assert len(project_instance.paths) == 1


def test_add_paths_to_initial_base_dir_only(project_instance: Project, temp_test_dirs: List[Path]):
    """
    Test adding paths when only base_dir was initially in project.paths.
    Base_dir should be replaced by the specific paths.
    """
    assert project_instance.paths == [project_instance.base_dir]  # Initial state from fixture

    specific_path_to_add = temp_test_dirs[0]
    updated_project = api.add_paths(project_instance, specific_path_to_add)

    assert updated_project is project_instance
    assert project_instance.base_dir not in updated_project.paths  # Base dir should be removed
    assert updated_project.paths == [specific_path_to_add.resolve()]
    assert len(updated_project.paths) == 1


def test_add_paths_multiple_to_initial_base_dir_only(project_instance: Project, temp_test_dirs: List[Path]):
    """
    Test adding multiple paths when only base_dir was initially in project.paths.
    Base_dir should be replaced by the specific paths.
    """
    assert project_instance.paths == [project_instance.base_dir]  # Initial state from fixture

    paths_to_add = temp_test_dirs  # List of Path objects
    updated_project = api.add_paths(project_instance, paths_to_add)

    expected_paths = sorted([p.resolve() for p in temp_test_dirs])
    assert updated_project is project_instance
    assert project_instance.base_dir not in updated_project.paths
    assert sorted(updated_project.paths) == expected_paths
    assert len(updated_project.paths) == len(temp_test_dirs)


def test_add_paths_no_change_if_invalid_paths_and_only_base_dir(project_instance: Project, tmp_path: Path, caplog):
    """
    Test that if only base_dir is present and invalid paths are added,
    the paths list remains unchanged (i.e., still contains only base_dir).
    """
    assert project_instance.paths == [project_instance.base_dir]  # Initial state from fixture

    non_existent_path = tmp_path / "non_existent_dir_2"
    with caplog.at_level(logging.INFO):
        api.add_paths(project_instance, str(non_existent_path))

    assert project_instance.paths == [project_instance.base_dir]  # Should still only contain base_dir
    assert "No valid or non-redundant paths provided" in caplog.text


def test_add_paths_single_string(project_instance: Project, temp_test_dirs: List[Path]):
    # Reset paths to simulate having other paths already, not just base_dir
    # This test is now slightly different to reflect current behavior where base_dir is init path
    api.add_paths(project_instance, temp_test_dirs[0])  # Add one specific path
    initial_specific_path = temp_test_dirs[0].resolve()
    assert project_instance.paths == [initial_specific_path]

    # Now add another specific path
    dir_to_add = str(temp_test_dirs[1])
    updated_project = api.add_paths(project_instance, dir_to_add)
    assert updated_project is project_instance
    assert sorted(updated_project.paths) == sorted([initial_specific_path, Path(dir_to_add).resolve()])


def test_add_paths_single_path_object(project_instance: Project, temp_test_dirs: List[Path]):
    # Reset paths to simulate having other paths already
    api.add_paths(project_instance, temp_test_dirs[0])
    initial_specific_path = temp_test_dirs[0].resolve()

    dir_to_add = temp_test_dirs[1]
    updated_project = api.add_paths(project_instance, dir_to_add)
    assert updated_project is project_instance
    assert sorted(updated_project.paths) == sorted([initial_specific_path, dir_to_add.resolve()])


def test_add_paths_multiple_mixed_types(project_instance: Project, temp_test_dirs: List[Path]):
    # Reset paths to simulate having other paths already
    api.add_paths(project_instance, temp_test_dirs[0])
    initial_specific_path = temp_test_dirs[0].resolve()

    dir_list = [str(temp_test_dirs[1]), temp_test_dirs[0]]  # Add existing and new
    updated_project = api.add_paths(project_instance, dir_list)
    expected_paths = sorted([initial_specific_path, temp_test_dirs[1].resolve()])
    assert sorted(p.as_posix() for p in updated_project.paths) == sorted(p.as_posix() for p in expected_paths)
    assert len(updated_project.paths) == 2  # No new path was added from temp_test_dirs[0] which was already there


def test_add_paths_non_existent_path_is_skipped(project_instance: Project, tmp_path: Path, caplog):
    # This test should now verify that if base_dir is present and invalid paths are added, base_dir remains.
    assert project_instance.paths == [project_instance.base_dir]
    non_existent_path = tmp_path / "non_existent_dir_3"
    with caplog.at_level(logging.WARNING):
        api.add_paths(project_instance, str(non_existent_path))
    assert project_instance.paths == [project_instance.base_dir]  # Path should not be added, base_dir preserved
    assert "Path not valid ('not found') and will be skipped" in caplog.text


def test_add_paths_file_is_skipped(project_instance: Project, tmp_path: Path, caplog):
    # This test should now verify that if base_dir is present and files are added, base_dir remains.
    assert project_instance.paths == [project_instance.base_dir]
    test_file = tmp_path / "test_file_2.txt"
    test_file.touch()
    with caplog.at_level(logging.WARNING):
        api.add_paths(project_instance, str(test_file))
    assert project_instance.paths == [project_instance.base_dir]  # File should not be added, base_dir preserved
    assert "Path not valid ('not a directory') and will be skipped" in caplog.text


def test_add_paths_accepts_local_csv_manifest(project_instance: Project, tmp_path: Path):
    # A .csv is claimed by a specialized source (ManifestSource), so add_paths
    # accepts it verbatim even though it is a file, not a directory.
    manifest = tmp_path / "load_data.csv"
    manifest.write_text("URL_DNA\ns3://b/x.tiff\n")
    api.add_paths(project_instance, str(manifest))
    assert str(manifest) in [str(p) for p in project_instance.paths]


def test_add_paths_path_outside_base_is_skipped(project_instance: Project, tmp_path: Path, caplog):
    # This test should now verify that if base_dir is present and outside paths are added, base_dir remains.
    assert project_instance.paths == [project_instance.base_dir]
    outside_dir = tmp_path.parent / "outside_project_dir_2"
    outside_dir.mkdir(exist_ok=True)
    with caplog.at_level(logging.WARNING):
        api.add_paths(project_instance, str(outside_dir))
    assert project_instance.paths == [project_instance.base_dir]  # Path should not be added, base_dir preserved
    assert "is not within the project base directory" in caplog.text
    outside_dir.rmdir()


def test_add_paths_superpath_replaces_subpath(project_instance: Project, tmp_path: Path, caplog):
    parent_dir = tmp_path / "parent_dir"
    parent_dir.mkdir()
    sub_dir = parent_dir / "sub_dir"
    sub_dir.mkdir()
    another_sub_dir = parent_dir / "another_sub_dir"
    another_sub_dir.mkdir()

    # Add sub_dir and another_sub_dir first. This will remove base_dir from project.paths.
    api.add_paths(project_instance, [sub_dir, another_sub_dir])
    assert sorted(project_instance.paths) == sorted([sub_dir.resolve(), another_sub_dir.resolve()])

    with caplog.at_level(logging.INFO):  # Logging for replacement is INFO
        api.add_paths(project_instance, parent_dir)

    # parent_dir should have replaced sub_dir and another_sub_dir
    assert project_instance.paths == [parent_dir.resolve()]
    assert "is a superpath of existing project path" in caplog.text


def test_add_paths_subpath_is_skipped(project_instance: Project, tmp_path: Path, caplog):
    parent_dir = tmp_path / "parent_to_sub"
    parent_dir.mkdir()
    sub_dir = parent_dir / "sub_to_skip"
    sub_dir.mkdir()

    # Add parent_dir first. This will remove base_dir from project.paths.
    api.add_paths(project_instance, parent_dir)
    assert project_instance.paths == [parent_dir.resolve()]

    # Now try to add its sub_dir - it should be skipped
    with caplog.at_level(logging.WARNING):
        api.add_paths(project_instance, sub_dir)

    assert project_instance.paths == [parent_dir.resolve()]  # Should remain unchanged
    assert "is a subpath of existing project path" in caplog.text


def test_add_paths_empty_input_preserves_current_paths(project_instance: Project):
    # Test when only base_dir is present
    initial_paths = project_instance.paths.copy()  # Should be [base_dir]
    updated_project = api.add_paths(project_instance, [])
    assert updated_project.paths == initial_paths

    # Test when specific paths are present
    subdir_x = project_instance.base_dir / "subdir_x"
    subdir_x.mkdir()
    api.add_paths(project_instance, subdir_x)
    assert project_instance.paths == [subdir_x.resolve()]  # confirm subdir_x was actually added
    initial_paths = project_instance.paths.copy()
    updated_project = api.add_paths(project_instance, [])
    assert updated_project.paths == initial_paths

def test_add_base_dir_when_subpaths_exist(project_instance: Project, tmp_path: Path, caplog):
    # Create sub-directories within tmp_path (which is project_instance.base_dir)
    sub_dir_a = tmp_path / "sub_dir_a"
    sub_dir_a.mkdir()
    sub_dir_b = tmp_path / "sub_dir_b"
    sub_dir_b.mkdir()

    # Add these sub-directories to the project, which will replace the initial base_dir
    api.add_paths(project_instance, [sub_dir_a, sub_dir_b])
    assert sorted(project_instance.paths) == sorted([sub_dir_a.resolve(), sub_dir_b.resolve()])

    # Now, add the base_dir itself
    with caplog.at_level(logging.INFO):
        updated_project = api.add_paths(project_instance, project_instance.base_dir)

    assert updated_project is project_instance
    # The base_dir should now be the only path
    assert updated_project.paths == [project_instance.base_dir.resolve()]
    # Check for the log message indicating a superpath replacement
    assert "is a superpath of existing project path" in caplog.text
