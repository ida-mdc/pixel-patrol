import pytest
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.config import DEFAULT_RECORDS_FLUSH_EVERY_N


# --- Defaults ---

def test_defaults():
    config = ProcessingConfig()
    assert config.slicing_enabled is True
    assert config.slicing_dimensions_included == set()
    assert config.slicing_dimensions_excluded == {"X", "Y"}
    assert config.processors_included == set()
    assert config.processors_excluded == set()
    assert config.selected_file_extensions == "all"
    assert config.pixel_patrol_flavor == ""
    assert config.processing_max_workers is None
    assert config.records_flush_every_n == DEFAULT_RECORDS_FLUSH_EVERY_N
    assert config.records_flush_dir is None


# --- slicing_enabled ---

def test_slicing_enabled_false():
    config = ProcessingConfig(slicing_enabled=False)
    assert config.slicing_enabled is False


def test_slicing_enabled_not_bool_raises():
    with pytest.raises(TypeError, match="slicing_enabled must be a bool"):
        ProcessingConfig(slicing_enabled="yes")


# --- slicing_dimensions_included / excluded conflict ---

def test_slicing_dims_included_only():
    config = ProcessingConfig(slicing_dimensions_included={"T", "C"})
    assert config.slicing_dimensions_included == {"T", "C"}


def test_slicing_dims_excluded_only():
    config = ProcessingConfig(slicing_dimensions_excluded={"X", "Y", "Z"})
    assert config.slicing_dimensions_excluded == {"X", "Y", "Z"}


def test_slicing_dims_both_set_warns(caplog):
    with caplog.at_level("WARNING"):
        ProcessingConfig(
            slicing_dimensions_included={"T"},
            slicing_dimensions_excluded={"Z"},
        )
    assert "slicing_dimensions_excluded will be ignored" in caplog.text


def test_slicing_dims_included_with_default_excluded_no_warning(caplog):
    # Default excluded is {"X", "Y"} — should not warn
    with caplog.at_level("WARNING"):
        ProcessingConfig(slicing_dimensions_included={"T"})
    assert "slicing_dimensions_excluded will be ignored" not in caplog.text


# --- processors_included / excluded conflict ---

def test_processors_included_only():
    config = ProcessingConfig(processors_included={"BasicStats"})
    assert config.processors_included == {"BasicStats"}


def test_processors_excluded_only():
    config = ProcessingConfig(processors_excluded={"HeavyProcessor"})
    assert config.processors_excluded == {"HeavyProcessor"}


def test_processors_both_set_warns(caplog):
    with caplog.at_level("WARNING"):
        ProcessingConfig(
            processors_included={"BasicStats"},
            processors_excluded={"HeavyProcessor"},
        )
    assert "processors_excluded will be ignored" in caplog.text


# --- selected_file_extensions ---

def test_extensions_all():
    config = ProcessingConfig(selected_file_extensions="all")
    assert config.selected_file_extensions == "all"


def test_extensions_set():
    config = ProcessingConfig(selected_file_extensions={"tif", "png"})
    assert config.selected_file_extensions == {"tif", "png"}


def test_extensions_empty_set():
    config = ProcessingConfig(selected_file_extensions=set())
    assert config.selected_file_extensions == set()


# --- processing_max_workers ---

def test_max_workers_none():
    config = ProcessingConfig(processing_max_workers=None)
    assert config.processing_max_workers is None


def test_max_workers_valid():
    config = ProcessingConfig(processing_max_workers=4)
    assert config.processing_max_workers == 4


def test_max_workers_zero_raises():
    with pytest.raises(ValueError, match="processing_max_workers must be a positive integer"):
        ProcessingConfig(processing_max_workers=0)


def test_max_workers_negative_raises():
    with pytest.raises(ValueError, match="processing_max_workers must be a positive integer"):
        ProcessingConfig(processing_max_workers=-1)


def test_max_workers_non_int_raises():
    with pytest.raises(ValueError, match="processing_max_workers must be a positive integer"):
        ProcessingConfig(processing_max_workers=1.5)


# --- records_flush_every_n ---

def test_flush_every_n_valid():
    config = ProcessingConfig(records_flush_every_n=50)
    assert config.records_flush_every_n == 50


def test_flush_every_n_zero_raises():
    with pytest.raises(ValueError, match="records_flush_every_n must be a positive integer"):
        ProcessingConfig(records_flush_every_n=0)


def test_flush_every_n_negative_raises():
    with pytest.raises(ValueError, match="records_flush_every_n must be a positive integer"):
        ProcessingConfig(records_flush_every_n=-10)


# --- records_flush_dir ---

def test_flush_dir_path(tmp_path):
    config = ProcessingConfig(records_flush_dir=tmp_path)
    assert config.records_flush_dir == tmp_path


def test_flush_dir_none():
    config = ProcessingConfig(records_flush_dir=None)
    assert config.records_flush_dir is None
