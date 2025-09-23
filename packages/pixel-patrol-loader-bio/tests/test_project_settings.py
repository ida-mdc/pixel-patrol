import pytest
import logging
from pathlib import Path

from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader
from pixel_patrol_base.core.project import Project
from pixel_patrol_base.core.project_settings import Settings
from pixel_patrol_base import api

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def named_project_with_base_dir(tmp_path: Path) -> Project:
    """Provides a Project instance with a base directory set."""
    return api.create_project("TestProject", tmp_path, loader="bioio")


def test_get_settings_initial(named_project_with_base_dir: Project):
    """Test retrieving default settings from a newly created project."""
    settings = api.get_settings(named_project_with_base_dir)
    assert isinstance(settings, Settings)
    assert settings.cmap == "rainbow"
    assert settings.n_example_files == 9
    assert settings.selected_file_extensions == set()


def test_set_settings_valid(named_project_with_base_dir: Project):
    """Test setting and retrieving valid new settings."""
    new_settings = Settings(cmap="viridis", n_example_files=5, selected_file_extensions={"jpg", "png"})
    updated_project = api.set_settings(named_project_with_base_dir, new_settings)
    retrieved_settings = api.get_settings(updated_project)
    assert retrieved_settings.cmap == "viridis"
    assert retrieved_settings.n_example_files == 5
    assert retrieved_settings.selected_file_extensions == {"jpg", "png"}


def test_set_settings_invalid_cmap(named_project_with_base_dir: Project):
    """Test setting settings with an invalid colormap name."""
    invalid_settings = Settings(cmap="non_existent_cmap")
    with pytest.raises(ValueError, match="Invalid colormap name"):
        api.set_settings(named_project_with_base_dir, invalid_settings)


def test_set_settings_invalid_n_example_images(named_project_with_base_dir: Project):
    """Test setting n_example_images with invalid values (too low, too high, wrong type)."""
    invalid_settings_low = Settings(n_example_files=0)
    with pytest.raises(ValueError, match="Number of example files must be an integer between 1 and 19"):
        api.set_settings(named_project_with_base_dir, invalid_settings_low)

    invalid_settings_high = Settings(n_example_files=20)
    with pytest.raises(ValueError, match="Number of example files must be an integer between 1 and 19"):
        api.set_settings(named_project_with_base_dir, invalid_settings_high)

    invalid_settings_type = Settings(n_example_files=9.5)
    with pytest.raises(ValueError, match="Number of example files must be an integer between 1 and 19"):
        api.set_settings(named_project_with_base_dir, invalid_settings_type)


def test_set_settings_set_selected_file_extensions_empty_initially(named_project_with_base_dir: Project, caplog):
    """Test that selected file extensions can be set to an empty set initially."""
    new_settings = Settings(selected_file_extensions=set())
    with caplog.at_level(logging.WARNING):
        updated_project = api.set_settings(named_project_with_base_dir, new_settings)
        assert api.get_settings(updated_project).selected_file_extensions == set()
        assert "selected_file_extensions is an empty set - no file will be processed" in caplog.text


def test_set_settings_set_selected_file_extensions_with_unsupported(named_project_with_base_dir: Project, caplog):
    """Test setting extensions including unsupported types."""
    mixed_extensions = {"jpg", "xyz", "tiff"}  # jpg, tiff are supported, xyz is not
    expected_extensions = {"jpg", "tiff"}
    with caplog.at_level(logging.WARNING):
        new_settings = Settings(selected_file_extensions=mixed_extensions)
        updated_project = api.set_settings(named_project_with_base_dir, new_settings)
        assert api.get_settings(updated_project).selected_file_extensions == expected_extensions
        assert "The following file extensions are not supported and will be ignored: xyz." in caplog.text


def test_set_settings_set_selected_file_extensions_only_unsupported(named_project_with_base_dir: Project, caplog):
    """Test setting extensions with only unsupported types results in empty set."""
    unsupported_extensions = {"xyz", "abc"}
    with caplog.at_level(logging.WARNING):
        new_settings = Settings(selected_file_extensions=unsupported_extensions)
        updated_project = api.set_settings(named_project_with_base_dir, new_settings)
        assert api.get_settings(updated_project).selected_file_extensions == set()
        assert "The following file extensions are not supported and will be ignored:" in caplog.text
        assert "abc" in caplog.text
        assert "xyz" in caplog.text
        assert "No loader supported file extensions provided. No files will be processed." in caplog.text

def test_set_settings_set_selected_file_extensions_to_all(named_project_with_base_dir: Project, caplog):
    """Test setting selected_file_extensions to the string 'all'."""
    new_settings = Settings(selected_file_extensions="all")
    with caplog.at_level(logging.INFO):
        updated_project = api.set_settings(named_project_with_base_dir, new_settings)
        assert api.get_settings(updated_project).selected_file_extensions == BioIoLoader.SUPPORTED_EXTENSIONS
        assert "Using loader-supported extensions:" in caplog.text


def test_set_settings_invalid_string_for_extensions(named_project_with_base_dir: Project, caplog):
    """Test setting selected_file_extensions to an invalid string (not 'all')."""
    invalid_settings = Settings(selected_file_extensions="invalid_string")
    with pytest.raises(TypeError, match=r"selected_file_extensions must be 'all' or a Set\[str\]\."):
        with caplog.at_level(logging.ERROR):
            api.set_settings(named_project_with_base_dir, invalid_settings)
            assert "Invalid type for selected_file_extensions: <class 'str'>." in caplog.text


def test_set_settings_invalid_type_for_extensions(named_project_with_base_dir: Project, caplog):
    """Test setting selected_file_extensions to an invalid type."""
    invalid_settings = Settings(selected_file_extensions=["jpg", "png"])  # List instead of Set
    with pytest.raises(TypeError, match=r"selected_file_extensions must be 'all' or a Set\[str\]\."):
        with caplog.at_level(logging.ERROR):
            api.set_settings(named_project_with_base_dir, invalid_settings)
            assert "Invalid type for selected_file_extensions: <class 'list'>." in caplog.text


def test_set_settings_change_selected_file_extensions_after_initial_set_different_set(
        named_project_with_base_dir: Project, caplog):
    initial_settings = Settings(selected_file_extensions={"jpg"})
    project_with_ext = api.set_settings(named_project_with_base_dir, initial_settings)
    assert api.get_settings(project_with_ext).selected_file_extensions == {"jpg"}

    changed_settings = Settings(selected_file_extensions={"png"})

    with caplog.at_level(logging.INFO):
        updated_project = api.set_settings(project_with_ext, changed_settings)
        assert "selected_file_extensions already set; keeping existing value:" in caplog.text

    assert api.get_settings(updated_project).selected_file_extensions == {"jpg"}


def test_set_settings_change_selected_file_extensions_after_initial_set_to_empty(named_project_with_base_dir: Project,
                                                                                 caplog):
    initial_settings = Settings(selected_file_extensions={"jpg"})
    project_with_ext = api.set_settings(named_project_with_base_dir, initial_settings)
    assert api.get_settings(project_with_ext).selected_file_extensions == {"jpg"}

    changed_settings = Settings(selected_file_extensions=set())

    with caplog.at_level(logging.INFO):
        updated_project = api.set_settings(project_with_ext, changed_settings)
        assert "selected_file_extensions already set; keeping existing value:" in caplog.text

    assert api.get_settings(updated_project).selected_file_extensions == {"jpg"}


def test_set_settings_change_selected_file_extensions_from_all_to_set(named_project_with_base_dir: Project, caplog):
    initial_settings = Settings(selected_file_extensions="all")
    project_with_ext = api.set_settings(named_project_with_base_dir, initial_settings)
    assert api.get_settings(project_with_ext).selected_file_extensions == BioIoLoader.SUPPORTED_EXTENSIONS

    changed_settings = Settings(selected_file_extensions={"jpg"})

    with caplog.at_level(logging.INFO):
        updated_project = api.set_settings(project_with_ext, changed_settings)
        # INFO message asserted in other tests; keeping behavior consistent.


def test_set_settings_change_selected_file_extensions_from_set_to_all(named_project_with_base_dir: Project, caplog):
    initial_settings = Settings(selected_file_extensions={"jpg"})
    project_with_ext = api.set_settings(named_project_with_base_dir, initial_settings)
    assert api.get_settings(project_with_ext).selected_file_extensions == {"jpg"}

    changed_settings = Settings(selected_file_extensions="all")

    with caplog.at_level(logging.INFO):
        updated_project = api.set_settings(project_with_ext, changed_settings)
        assert "selected_file_extensions already set; keeping existing value:" in caplog.text

    assert api.get_settings(updated_project).selected_file_extensions == {"jpg"}


def test_set_settings_set_selected_file_extensions_to_same_set_already_defined(named_project_with_base_dir: Project,
                                                                               caplog):
    initial_settings = Settings(selected_file_extensions={"jpg"})
    project_with_ext = api.set_settings(named_project_with_base_dir, initial_settings)
    assert api.get_settings(project_with_ext).selected_file_extensions == {"jpg"}

    same_settings = Settings(selected_file_extensions={"jpg"})
    with caplog.at_level(logging.INFO):
        updated_project = api.set_settings(project_with_ext, same_settings)
        assert api.get_settings(updated_project).selected_file_extensions == {"jpg"}
        assert "selected_file_extensions already set; keeping existing value:" in caplog.text


def test_set_settings_set_selected_file_extensions_to_all_when_already_default_set(named_project_with_base_dir: Project,
                                                                                   caplog):
    initial_settings = Settings(selected_file_extensions=BioIoLoader.SUPPORTED_EXTENSIONS)
    project_with_ext = api.set_settings(named_project_with_base_dir, initial_settings)
    assert api.get_settings(project_with_ext).selected_file_extensions == BioIoLoader.SUPPORTED_EXTENSIONS

    new_settings = Settings(selected_file_extensions="all")
    with caplog.at_level(logging.INFO):
        updated_project = api.set_settings(project_with_ext, new_settings)
        assert api.get_settings(updated_project).selected_file_extensions == BioIoLoader.SUPPORTED_EXTENSIONS
        assert "selected_file_extensions already set; keeping existing value:" in caplog.text


def test_set_settings_set_selected_file_extensions_to_default_set_when_already_all_string(
        named_project_with_base_dir: Project, caplog):
    initial_settings = Settings(selected_file_extensions="all")
    project_with_ext = api.set_settings(named_project_with_base_dir, initial_settings)
    assert api.get_settings(project_with_ext).selected_file_extensions == BioIoLoader.SUPPORTED_EXTENSIONS

    same_as_default_set_settings = Settings(selected_file_extensions=BioIoLoader.SUPPORTED_EXTENSIONS)
    with caplog.at_level(logging.INFO):
        updated_project = api.set_settings(project_with_ext, same_as_default_set_settings)
        assert api.get_settings(updated_project).selected_file_extensions == BioIoLoader.SUPPORTED_EXTENSIONS
        assert "selected_file_extensions already set; keeping existing value:" in caplog.text
