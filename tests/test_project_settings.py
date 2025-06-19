import pytest
import logging
from pathlib import Path

from pixel_patrol.core.project import Project
from pixel_patrol.core.project_settings import Settings
from pixel_patrol import api
from pixel_patrol.config import DEFAULT_PRESELECTED_FILE_EXTENSIONS

# Set up basic logging for tests to capture messages
logging.basicConfig(level=logging.INFO)


@pytest.fixture
def named_project_with_base_dir(tmp_path: Path) -> Project:
    """Provides a Project instance with a base directory set."""
    return api.create_project("TestProject", tmp_path)


def test_get_settings_initial(named_project_with_base_dir: Project):
    """Test retrieving default settings from a newly created project."""
    settings = api.get_settings(named_project_with_base_dir)
    assert isinstance(settings, Settings)
    assert settings.cmap == "rainbow"
    assert settings.n_example_images == 9
    assert settings.selected_file_extensions == set()


def test_set_settings_valid(named_project_with_base_dir: Project):
    """Test setting and retrieving valid new settings."""
    new_settings = Settings(cmap="viridis", n_example_images=5, selected_file_extensions={"jpg", "png"})
    updated_project = api.set_settings(named_project_with_base_dir, new_settings)
    retrieved_settings = api.get_settings(updated_project)

    assert retrieved_settings.cmap == "viridis"
    assert retrieved_settings.n_example_images == 5
    assert retrieved_settings.selected_file_extensions == {"jpg", "png"}


def test_set_settings_invalid_cmap(named_project_with_base_dir: Project):
    """Test setting settings with an invalid colormap name."""
    invalid_settings = Settings(cmap="non_existent_cmap")
    with pytest.raises(ValueError, match="Invalid colormap name"):
        api.set_settings(named_project_with_base_dir, invalid_settings)


def test_set_settings_invalid_n_example_images(named_project_with_base_dir: Project):
    """Test setting n_example_images with invalid values (too low, too high, wrong type)."""
    # Test too low
    invalid_settings_low = Settings(n_example_images=0)
    with pytest.raises(ValueError, match="Number of example images must be an integer between 1 and 19"):
        api.set_settings(named_project_with_base_dir, invalid_settings_low)

    # Test too high
    invalid_settings_high = Settings(n_example_images=20)
    with pytest.raises(ValueError, match="Number of example images must be an integer between 1 and 19"):
        api.set_settings(named_project_with_base_dir, invalid_settings_high)

    # Test not an integer
    invalid_settings_type = Settings(n_example_images=9.5)
    with pytest.raises(ValueError, match="Number of example images must be an integer between 1 and 19"):
        api.set_settings(named_project_with_base_dir, invalid_settings_type)


# --- NEW/ENHANCED TESTS FOR selected_file_extensions ---

def test_set_settings_set_selected_file_extensions_empty_initially(named_project_with_base_dir: Project, caplog):
    """Test that selected file extensions can be set to an empty set initially."""



def test_set_settings_set_selected_file_extensions_with_unsupported(named_project_with_base_dir: Project, caplog):
    """Test setting extensions including unsupported types."""
    mixed_extensions = {"jpg", "xyz", "tiff"} # jpg, tiff are supported, xyz is not
    expected_extensions = {"jpg", "tiff"}
    with caplog.at_level(logging.WARNING):
        new_settings = Settings(selected_file_extensions=mixed_extensions)
        updated_project = api.set_settings(named_project_with_base_dir, new_settings)
        assert api.get_settings(updated_project).selected_file_extensions == expected_extensions
        assert "The following file extensions are not supported and will be ignored: xyz." in caplog.text


def test_set_settings_set_selected_file_extensions_only_unsupported(named_project_with_base_dir: Project, caplog):
    """Test setting extensions with only unsupported types results in empty set."""


def test_set_settings_set_selected_file_extensions_to_all(named_project_with_base_dir: Project, caplog):
    """Test setting selected_file_extensions to the string 'all'."""


def test_set_settings_invalid_string_for_extensions(named_project_with_base_dir: Project, caplog):
    """Test setting selected_file_extensions to an invalid string (not 'all')."""


def test_set_settings_invalid_type_for_extensions(named_project_with_base_dir: Project, caplog):
    """Test setting selected_file_extensions to an invalid type."""
    invalid_settings = Settings(selected_file_extensions=["jpg", "png"]) # List instead of Set
    with pytest.raises(TypeError, match="selected_file_extensions must be 'all' \\(string\\) or a Set of strings."):
        with caplog.at_level(logging.ERROR):
            api.set_settings(named_project_with_base_dir, invalid_settings)
            assert "Invalid type for selected_file_extensions: <class 'list'>." in caplog.text


def test_set_settings_change_selected_file_extensions_after_initial_set_different_set(named_project_with_base_dir: Project, caplog):
    """Test that selected file extensions cannot be changed to a different set after they've been set once."""
    # Set initially
    initial_settings = Settings(selected_file_extensions={"jpg"})
    project_with_ext = api.set_settings(named_project_with_base_dir, initial_settings)
    assert api.get_settings(project_with_ext).selected_file_extensions == {"jpg"}

    # Try to change to a different set
    changed_settings = Settings(selected_file_extensions={"png"})
    with pytest.raises(ValueError, match="File extensions cannot be changed once they have been defined"):
        with caplog.at_level(logging.ERROR):
            api.set_settings(project_with_ext, changed_settings)
            assert "Attempted to change file extensions from" in caplog.text
            assert "to {'png'}." in caplog.text
    # Ensure it remains unchanged
    assert api.get_settings(project_with_ext).selected_file_extensions == {"jpg"}


def test_set_settings_change_selected_file_extensions_after_initial_set_to_empty(named_project_with_base_dir: Project, caplog):
    """Test that selected file extensions cannot be changed to an empty set if previously defined."""


def test_set_settings_change_selected_file_extensions_from_all_to_set(named_project_with_base_dir: Project, caplog):
    """Test that selected file extensions cannot be changed from 'all' to a specific set."""
    # Set initially to 'all'
    initial_settings = Settings(selected_file_extensions="all")
    project_with_ext = api.set_settings(named_project_with_base_dir, initial_settings)
    assert api.get_settings(project_with_ext).selected_file_extensions == DEFAULT_PRESELECTED_FILE_EXTENSIONS

    # Try to change to a specific set
    changed_settings = Settings(selected_file_extensions={"jpg"})
    with pytest.raises(ValueError, match="File extensions cannot be changed once they have been defined"):
        with caplog.at_level(logging.ERROR):
            api.set_settings(project_with_ext, changed_settings)
            assert "Attempted to change file extensions from" in caplog.text
            assert "to {'jpg'}." in caplog.text
    # Ensure it remains unchanged
    assert api.get_settings(project_with_ext).selected_file_extensions == DEFAULT_PRESELECTED_FILE_EXTENSIONS


def test_set_settings_change_selected_file_extensions_from_set_to_all(named_project_with_base_dir: Project, caplog):
    """Test that selected file extensions cannot be changed from a specific set to 'all'."""
    # Set initially to a specific set
    initial_settings = Settings(selected_file_extensions={"jpg"})
    project_with_ext = api.set_settings(named_project_with_base_dir, initial_settings)
    assert api.get_settings(project_with_ext).selected_file_extensions == {"jpg"}

    # Try to change to 'all'
    changed_settings = Settings(selected_file_extensions="all")
    with pytest.raises(ValueError, match="File extensions cannot be changed once they have been defined"):
        with caplog.at_level(logging.ERROR):
            api.set_settings(project_with_ext, changed_settings)
            assert "Attempted to change file extensions from" in caplog.text
            assert "to 'all'." in caplog.text # Note: the log message will show 'all' as a string input
    # Ensure it remains unchanged
    assert api.get_settings(project_with_ext).selected_file_extensions == {"jpg"}


def test_set_settings_set_selected_file_extensions_to_same_set_already_defined(named_project_with_base_dir: Project, caplog):
    """Test setting selected_file_extensions to the same set when already defined (should not raise error)."""
    # Set initially
    initial_settings = Settings(selected_file_extensions={"jpg"})
    project_with_ext = api.set_settings(named_project_with_base_dir, initial_settings)
    assert api.get_settings(project_with_ext).selected_file_extensions == {"jpg"}

    # Try to set to the exact same set
    same_settings = Settings(selected_file_extensions={"jpg"})
    with caplog.at_level(logging.INFO):
        updated_project = api.set_settings(project_with_ext, same_settings)
        assert api.get_settings(updated_project).selected_file_extensions == {"jpg"}
        assert "File extensions remain unchanged as" in caplog.text
        assert "{'jpg'}" in caplog.text # Log should reflect the set value


def test_set_settings_set_selected_file_extensions_to_all_when_already_default_set(named_project_with_base_dir: Project, caplog):
    """Test setting 'all' string when extensions are already the default preselected set."""



def test_set_settings_set_selected_file_extensions_to_default_set_when_already_all_string(named_project_with_base_dir: Project, caplog):
    """Test setting the default preselected set when extensions were initially set as 'all' string."""
    # Set initially as 'all' string
    initial_settings = Settings(selected_file_extensions="all")
    project_with_ext = api.set_settings(named_project_with_base_dir, initial_settings)
    assert api.get_settings(project_with_ext).selected_file_extensions == DEFAULT_PRESELECTED_FILE_EXTENSIONS

    # Try to set to the actual DEFAULT_PRESELECTED_FILE_EXTENSIONS set
    same_as_default_set_settings = Settings(selected_file_extensions=DEFAULT_PRESELECTED_FILE_EXTENSIONS)
    with caplog.at_level(logging.INFO):
        updated_project = api.set_settings(project_with_ext, same_as_default_set_settings)
        assert api.get_settings(updated_project).selected_file_extensions == DEFAULT_PRESELECTED_FILE_EXTENSIONS
        assert "File extensions remain unchanged as" in caplog.text
        # Log should reflect the set value
        assert str(DEFAULT_PRESELECTED_FILE_EXTENSIONS) in caplog.text