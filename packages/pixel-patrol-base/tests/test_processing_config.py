import pytest
from pixel_patrol_base.core.processing import _MAX_SIZE_EXPANSION_FACTOR, _task_budget_mb
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.config import DEFAULT_MB_PER_WORKER, DEFAULT_ROWS_PER_PART


# --- Defaults ---

def test_defaults():
    config = ProcessingConfig()
    assert config.processors_included == set()
    assert config.processors_excluded == set()
    assert config.selected_file_extensions == "all"
    assert config.max_workers is None
    assert config.mb_per_worker is None
    assert config.slice_size is None
    assert config.rows_per_part == DEFAULT_ROWS_PER_PART


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


# --- max_workers ---

def test_max_workers_none():
    config = ProcessingConfig(max_workers=None)
    assert config.max_workers is None


def test_max_workers_valid():
    config = ProcessingConfig(max_workers=4)
    assert config.max_workers == 4


def test_max_workers_zero_raises():
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        ProcessingConfig(max_workers=0)


def test_max_workers_negative_raises():
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        ProcessingConfig(max_workers=-1)


# --- mb_per_worker ---

def test_mb_per_worker_valid():
    config = ProcessingConfig(mb_per_worker=2048.0)
    assert config.mb_per_worker == 2048.0


def test_effective_mb_per_worker_falls_back_to_default():
    assert ProcessingConfig().effective_mb_per_worker() == DEFAULT_MB_PER_WORKER
    assert ProcessingConfig(mb_per_worker=2048.0).effective_mb_per_worker() == 2048.0


def test_task_budget_is_worker_ram_over_expansion_factor():
    assert _task_budget_mb(ProcessingConfig()) == DEFAULT_MB_PER_WORKER / _MAX_SIZE_EXPANSION_FACTOR
    assert _task_budget_mb(ProcessingConfig(mb_per_worker=800.0)) == 800.0 / _MAX_SIZE_EXPANSION_FACTOR


def test_mb_per_worker_zero_raises():
    with pytest.raises(ValueError, match="mb_per_worker must be positive"):
        ProcessingConfig(mb_per_worker=0)


def test_mb_per_worker_negative_raises():
    with pytest.raises(ValueError, match="mb_per_worker must be positive"):
        ProcessingConfig(mb_per_worker=-1.0)


# --- slice_size ---

def test_slice_size_none():
    config = ProcessingConfig(slice_size=None)
    assert config.slice_size is None


def test_slice_size_valid():
    config = ProcessingConfig(slice_size={"Z": 1, "Y": -1, "X": -1})
    assert config.slice_size == {"Z": 1, "Y": -1, "X": -1}


def test_slice_size_invalid_raises():
    with pytest.raises(ValueError):
        ProcessingConfig(slice_size={"Z": 0})


def test_slice_size_lowercase_normalised_to_uppercase():
    config = ProcessingConfig(slice_size={"z": 2, "y": -1})
    assert config.slice_size == {"Z": 2, "Y": -1}


def test_slice_size_mixed_case_normalised():
    config = ProcessingConfig(slice_size={"z": 2, "X": -1})
    assert config.slice_size == {"Z": 2, "X": -1}


# --- rows_per_part ---

def test_rows_per_part_valid():
    config = ProcessingConfig(rows_per_part=50)
    assert config.rows_per_part == 50


def test_rows_per_part_zero_raises():
    with pytest.raises(ValueError, match="rows_per_part must be a positive integer"):
        ProcessingConfig(rows_per_part=0)


def test_rows_per_part_negative_raises():
    with pytest.raises(ValueError, match="rows_per_part must be a positive integer"):
        ProcessingConfig(rows_per_part=-10)
