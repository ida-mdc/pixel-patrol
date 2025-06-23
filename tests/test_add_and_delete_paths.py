import pytest
from pathlib import Path
from typing import List
import logging

from pixel_patrol import api
from pixel_patrol.core.project import Project


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
    api.add_paths(project_instance, project_instance.base_dir / "subdir_x")
    initial_paths = project_instance.paths.copy()  # Should be [base_dir / "subdir_x"]
    updated_project = api.add_paths(project_instance, [])
    assert updated_project.paths == initial_paths


def test_delete_path_existing(project_instance: Project, temp_test_dirs: List[Path]):
    api.add_paths(project_instance, temp_test_dirs)  # Adds specific paths, removes base_dir
    path_to_delete = str(temp_test_dirs[0])
    updated_project = api.delete_path(project_instance, path_to_delete)
    assert updated_project is project_instance
    assert Path(path_to_delete).resolve() not in updated_project.paths
    assert len(updated_project.paths) == len(temp_test_dirs) - 1
    # Ensure base_dir is NOT re-added if other paths remain
    assert updated_project.paths == [temp_test_dirs[1].resolve()]  # Assuming temp_test_dirs has 2 items


def test_delete_path_resulting_in_empty_list_re_adds_base_dir(project_instance: Project, tmp_path: Path, caplog):
    # Start with only base_dir in paths
    assert project_instance.paths == [project_instance.base_dir]

    # Add a single specific path
    single_specific_path = tmp_path / "single_item_dir"
    single_specific_path.mkdir()
    api.add_paths(project_instance, single_specific_path)
    assert project_instance.paths == [single_specific_path.resolve()]  # Base_dir should be removed now

    # Delete that single specific path
    with caplog.at_level(logging.INFO):
        updated_project = api.delete_path(project_instance, str(single_specific_path))

    assert updated_project is project_instance
    assert single_specific_path.resolve() not in updated_project.paths
    assert updated_project.paths == [project_instance.base_dir]  # Base_dir should be re-added
    assert "re-added base directory" in caplog.text
    single_specific_path.rmdir()


def test_delete_path_base_dir_cannot_be_deleted_if_only_path(project_instance: Project, caplog):
    """
    Tests that the base_dir cannot be deleted if it is the only path in the list,
    and that the paths remain unchanged, while also checking the log message.
    """
    assert project_instance.paths == [project_instance.base_dir]  # Initial state from fixture

    with caplog.at_level(logging.INFO):
        updated_project = api.delete_path(project_instance, str(project_instance.base_dir))

    assert updated_project is project_instance
    assert updated_project.paths == [project_instance.base_dir]

    assert f"Project Core: Last specific path '{project_instance.base_dir}' deleted; re-added base directory '{project_instance.base_dir}'." in caplog.text


def test_delete_path_valid_but_not_in_project(project_instance: Project, tmp_path: Path, caplog):
    """
    Tests deleting a valid, existing directory within the base,
    but which was never explicitly added to the project's paths list (beyond initial base_dir).
    """
    # Project paths should contain only base_dir from fixture initially
    assert project_instance.paths == [project_instance.base_dir]

    test_dir_in_base = tmp_path / "some_dir_in_base_not_in_project"
    test_dir_in_base.mkdir()

    initial_paths = project_instance.paths.copy()  # Should be [base_dir]

    with caplog.at_level(logging.WARNING):
        updated_project = api.delete_path(project_instance, str(test_dir_in_base))

    assert updated_project.paths == initial_paths  # Should remain unchanged ([base_dir])
    assert f"Path '{test_dir_in_base.resolve()}' was not found in project" in caplog.text

    test_dir_in_base.rmdir()


def test_delete_path_invalid_or_outside_base(project_instance: Project, tmp_path: Path):
    """
    Tests deleting a path that is either invalid (e.g., non-existent, not a dir)
    or outside the project's base, which should raise a ValueError.
    """
    # Scenario 1: Path outside project base
    outside_dir = tmp_path.parent / "outside_project_dir_to_delete"
    outside_dir.mkdir(exist_ok=True)
    with pytest.raises(ValueError,
                       match=f"Cannot delete path: '{outside_dir}' is invalid, inaccessible, or outside the project base."):
        api.delete_path(project_instance, str(outside_dir))
    outside_dir.rmdir()  # Clean up

    # Scenario 2: Non-existent path within project base (but not added)
    non_existent_path = project_instance.get_base_dir() / "non_existent_path_to_delete"
    with pytest.raises(ValueError,
                       match=f"Cannot delete path: '{non_existent_path}' is invalid, inaccessible, or outside the project base."):
        api.delete_path(project_instance, str(non_existent_path))

    # Scenario 3: A file (not a directory) within project base
    test_file = project_instance.get_base_dir() / "test_file_to_delete.txt"
    test_file.touch()
    with pytest.raises(ValueError,
                       match=f"Cannot delete path: '{test_file}' is invalid, inaccessible, or outside the project base."):
        api.delete_path(project_instance, str(test_file))
    test_file.unlink()  # Clean up


def test_delete_non_existent_path_with_other_specific_paths(project_instance: Project, tmp_path: Path, caplog):
    """
    Tests deleting a non-existent path when the project already has other
    specific paths listed (not just base_dir). Should result in no change
    to the project's paths and a warning.
    """
    existing_dir = tmp_path / "existing_dir_1"
    existing_dir.mkdir()
    another_existing_dir = tmp_path / "existing_dir_2"
    another_existing_dir.mkdir()

    # Add specific paths, which should remove base_dir from project.paths
    api.add_paths(project_instance, [existing_dir, another_existing_dir])
    initial_paths = sorted([existing_dir.resolve(), another_existing_dir.resolve()])
    assert sorted(project_instance.paths) == initial_paths

    non_existent_path_to_delete = tmp_path / "non_existent_dir_to_delete"
    # Create it temporarily so _is_valid_path_for_project resolves it,
    # but it's not in project_instance.paths
    non_existent_path_to_delete.mkdir()

    with caplog.at_level(logging.WARNING):
        updated_project = api.delete_path(project_instance, str(non_existent_path_to_delete))

    assert updated_project is project_instance
    # Paths should remain unchanged
    assert sorted(updated_project.paths) == initial_paths
    assert f"Path '{non_existent_path_to_delete.resolve()}' was not found in project" in caplog.text

    non_existent_path_to_delete.rmdir() # Clean up

def test_delete_subpath_when_superpath_is_present(project_instance: Project, tmp_path: Path, caplog):
    """
    Tests deleting a subpath when its superpath is present in project.paths.
    The superpath should remain, and the subpath should not be found.
    """
    super_path = tmp_path / "super_dir"
    super_path.mkdir()
    sub_path = super_path / "sub_dir"
    sub_path.mkdir()

    # Add super_path to project.paths
    api.add_paths(project_instance, super_path)
    assert project_instance.paths == [super_path.resolve()]

    # Attempt to delete the sub_path
    with caplog.at_level(logging.WARNING):
        updated_project = api.delete_path(project_instance, str(sub_path))

    assert updated_project is project_instance
    # super_path should still be there, sub_path was never in the list
    assert updated_project.paths == [super_path.resolve()]
    assert f"Path '{sub_path.resolve()}' was not found in project" in caplog.text


