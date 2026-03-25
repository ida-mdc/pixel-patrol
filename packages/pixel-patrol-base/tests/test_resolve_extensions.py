"""
Tests for _resolve_extensions and _prepare_processing_config.
"""
import pytest
import logging
from pathlib import Path

from pixel_patrol_base.core.project import Project, _resolve_extensions
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base import api

logging.basicConfig(level=logging.INFO)


# --- Fixtures ---

class _StubLoader:
    """Minimal loader stub satisfying the PixelPatrolLoader Protocol."""
    NAME = "stub"
    SUPPORTED_EXTENSIONS = {"tif", "tiff", "png", "jpg"}
    OUTPUT_SCHEMA = {}
    OUTPUT_SCHEMA_PATTERNS = []
    FOLDER_EXTENSIONS = set()

    def is_folder_supported(self, path: Path) -> bool:
        return False

    def load(self, source: str):
        raise NotImplementedError("stub")


@pytest.fixture
def stub_loader() -> _StubLoader:
    return _StubLoader()


@pytest.fixture
def project_no_loader(tmp_path: Path) -> Project:
    return api.create_project("TestProject", tmp_path)


@pytest.fixture
def project_with_loader(tmp_path: Path, stub_loader: _StubLoader) -> Project:
    project = api.create_project("TestProject", tmp_path)
    project.loader = stub_loader
    return project


# =============================================================================
# _resolve_extensions
# =============================================================================

# --- "all" ---

def test_all_no_loader_returns_all_string(caplog):
    with caplog.at_level(logging.INFO):
        result = _resolve_extensions("all", loader=None)
    assert result == "all"
    assert "All file extensions are selected" in caplog.text


def test_all_with_loader_returns_supported_extensions(stub_loader, caplog):
    with caplog.at_level(logging.INFO):
        result = _resolve_extensions("all", loader=stub_loader)
    assert result == stub_loader.SUPPORTED_EXTENSIONS
    assert "Using loader-supported extensions" in caplog.text


def test_all_case_insensitive_no_loader():
    assert _resolve_extensions("ALL", loader=None) == "all"
    assert _resolve_extensions("All", loader=None) == "all"


def test_all_case_insensitive_with_loader(stub_loader):
    assert _resolve_extensions("ALL", loader=stub_loader) == stub_loader.SUPPORTED_EXTENSIONS


# --- set, no loader ---

def test_set_no_loader_returns_lowercased(caplog):
    with caplog.at_level(logging.INFO):
        result = _resolve_extensions({"PNG", "JPG"}, loader=None)
    assert result == {"png", "jpg"}
    assert "File extensions selected" in caplog.text


def test_set_no_loader_not_filtered(caplog):
    """Without a loader, any extension is accepted as-is — no filtering."""
    result = _resolve_extensions({"xyz", "abc"}, loader=None)
    assert result == {"xyz", "abc"}


def test_empty_set_no_loader_warns(caplog):
    with caplog.at_level(logging.WARNING):
        result = _resolve_extensions(set(), loader=None)
    assert result == set()
    assert "empty set" in caplog.text


# --- set, with loader ---

def test_set_with_loader_lowercased(stub_loader):
    result = _resolve_extensions({"TIF", "PNG"}, loader=stub_loader)
    assert result == {"tif", "png"}


def test_set_with_loader_filters_unsupported(stub_loader, caplog):
    with caplog.at_level(logging.WARNING):
        result = _resolve_extensions({"tif", "xyz"}, loader=stub_loader)
    assert result == {"tif"}
    assert "xyz" in caplog.text


def test_set_with_loader_only_unsupported_returns_empty(stub_loader, caplog):
    with caplog.at_level(logging.WARNING):
        result = _resolve_extensions({"xyz", "abc"}, loader=stub_loader)
    assert result == set()
    assert "No loader-supported file extensions provided" in caplog.text


def test_empty_set_with_loader_warns(stub_loader, caplog):
    with caplog.at_level(logging.WARNING):
        result = _resolve_extensions(set(), loader=stub_loader)
    assert result == set()
    assert "empty set" in caplog.text


# --- invalid types ---

def test_invalid_string_raises():
    with pytest.raises(TypeError, match="selected_file_extensions must be 'all' or a Set"):
        _resolve_extensions("not_all", loader=None)


def test_list_raises():
    with pytest.raises(TypeError, match="selected_file_extensions must be 'all' or a Set"):
        _resolve_extensions(["png", "jpg"], loader=None)


# =============================================================================
# _prepare_processing_config
# =============================================================================

# --- no loader ---

def test_prepare_no_loader_defaults_to_all(project_no_loader: Project):
    config = project_no_loader._prepare_processing_config(None)
    assert config.selected_file_extensions == "all"


def test_prepare_no_loader_explicit_set_accepted(project_no_loader: Project):
    pc = ProcessingConfig(selected_file_extensions={"png", "tif"})
    config = project_no_loader._prepare_processing_config(pc)
    assert config.selected_file_extensions == {"png", "tif"}


def test_prepare_no_loader_empty_set_warns(project_no_loader: Project, caplog):
    pc = ProcessingConfig(selected_file_extensions=set())
    with caplog.at_level(logging.WARNING):
        config = project_no_loader._prepare_processing_config(pc)
    assert config.selected_file_extensions == set()
    assert "empty set" in caplog.text


# --- with loader ---

def test_prepare_with_loader_defaults_to_supported(project_with_loader: Project, stub_loader):
    config = project_with_loader._prepare_processing_config(None)
    assert config.selected_file_extensions == stub_loader.SUPPORTED_EXTENSIONS


def test_prepare_with_loader_explicit_set_filtered(project_with_loader: Project, caplog):
    pc = ProcessingConfig(selected_file_extensions={"tif", "xyz"})
    with caplog.at_level(logging.WARNING):
        config = project_with_loader._prepare_processing_config(pc)
    assert config.selected_file_extensions == {"tif"}
    assert "xyz" in caplog.text


def test_prepare_with_loader_only_unsupported_warns(project_with_loader: Project, caplog):
    pc = ProcessingConfig(selected_file_extensions={"xyz", "abc"})
    with caplog.at_level(logging.WARNING):
        config = project_with_loader._prepare_processing_config(pc)
    assert config.selected_file_extensions == set()
    assert "No loader-supported file extensions provided" in caplog.text


def test_prepare_with_loader_empty_set_warns(project_with_loader: Project, caplog):
    pc = ProcessingConfig(selected_file_extensions=set())
    with caplog.at_level(logging.WARNING):
        config = project_with_loader._prepare_processing_config(pc)
    assert config.selected_file_extensions == set()
    assert "empty set" in caplog.text


def test_prepare_invalid_string_raises(project_with_loader: Project):
    with pytest.raises(TypeError, match="selected_file_extensions must be 'all' or a Set"):
        project_with_loader._prepare_processing_config(
            ProcessingConfig(selected_file_extensions="invalid_string")
        )


# --- flush dir ---

def test_prepare_infers_flush_dir(project_no_loader: Project):
    config = project_no_loader._prepare_processing_config(None)
    assert config.records_flush_dir == project_no_loader.base_dir / f"{project_no_loader.name}_batches"


def test_prepare_respects_explicit_flush_dir(project_no_loader: Project, tmp_path: Path):
    custom_dir = tmp_path / "custom_batches"
    pc = ProcessingConfig(records_flush_dir=custom_dir)
    config = project_no_loader._prepare_processing_config(pc)
    assert config.records_flush_dir == custom_dir


def test_prepare_stamps_flush_dir_on_project(project_no_loader: Project):
    project_no_loader._prepare_processing_config(None)
    assert project_no_loader.records_flush_dir == project_no_loader.base_dir / f"{project_no_loader.name}_batches"