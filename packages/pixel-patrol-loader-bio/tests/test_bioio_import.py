from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pytest

from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader
from pixel_patrol_base.core.processing import build_records_df
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.plugin_registry import discover_processor_plugins


@pytest.fixture(scope="module")
def loader():
    return BioIoLoader()


@pytest.fixture(scope="module")
def processors():
    return discover_processor_plugins()


def _process_single_file(
    file_path: Path,
    loader: BioIoLoader,
    processors: list,
) -> List[dict]:
    """Run build_records_df on a single file; return obs_level=0 rows as dicts."""
    if not file_path.exists():
        return []
    ext = file_path.suffix.lower().lstrip(".")
    df = build_records_df(
        bases=[file_path.parent],
        loader=loader,
        processors=processors,
        config=ProcessingConfig(selected_file_extensions={ext}),
    )
    if df is None:
        return []
    return df.filter(
        (df["path"] == str(file_path)) & (df["obs_level"] == 0)
    ).to_dicts()


def get_image_files_from_data_dir(test_data_dir: Path):
    return [f for f in test_data_dir.rglob("*")
            if f.is_file() and f.name != "not_an_image.txt"]


def test_nonexistent_path_raises(tmp_path, loader, processors):
    missing = tmp_path / "nope.tiny_png"
    assert _process_single_file(missing, loader=loader, processors=processors) == []


def test_unsupported_file(test_data_dir: Path, loader, processors):
    non_image_file = test_data_dir / "not_an_image.txt"
    result = _process_single_file(non_image_file, loader=loader, processors=processors)
    assert result == [], f"Expected empty list for non-image file, got {result}"


def test_empty_or_corrupt_image(tmp_path, loader, processors):
    f = tmp_path / "zero.tif"
    f.write_bytes(b"")
    assert _process_single_file(f, loader=loader, processors=processors) == []


def test_bioio_image_properties_per_file(
    image_file_path: Path,
    loader, processors,
    expected_image_data: Dict[str, Dict[str, Any]],
):
    file_name = image_file_path.name
    result = _process_single_file(image_file_path, loader=loader, processors=processors)

    assert result is not None, f"Failed to get properties for {file_name}"
    assert result != [], f"Properties list is empty for {file_name}"

    actual_properties = result[0]
    assert "num_pixels" in actual_properties, f"Missing 'num_pixels' for {file_name}"
    assert actual_properties["num_pixels"] > 0, f"num_pixels is zero for {file_name}"
    assert "min_intensity" in actual_properties, f"Missing 'min_intensity' for {file_name}"
    assert "max_intensity" in actual_properties, f"Missing 'max_intensity' for {file_name}"


def test_all_image_files_load_and_standardize(
    image_file_path: Path,
    loader, processors,
):
    file_name = image_file_path.name
    result = _process_single_file(image_file_path, loader=loader, processors=processors)

    assert result is not None, f"Failed to get properties for {file_name}"
    assert result != [], f"Properties list is empty for {file_name}"

    for i, properties in enumerate(result):
        record_label = f"{file_name}[{i}]" if len(result) > 1 else file_name
        assert properties.get("obs_level") == 0, f"Expected obs_level=0 for {record_label}"
        assert "num_pixels" in properties, f"Missing num_pixels for {record_label}"
        assert "min_intensity" in properties, f"Missing min_intensity for {record_label}"
        assert "max_intensity" in properties, f"Missing max_intensity for {record_label}"
