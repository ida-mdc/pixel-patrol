from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import shutil

import polars as pl
import pytest

from pixel_patrol_base.core.processing import build_records_df
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.plugin_registry import discover_processor_plugins
from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _example_bioio_dir() -> Path:
    return _repo_root() / "examples" / "datasets" / "bioio"


def _iter_image_files(root: Path) -> Iterable[Path]:
    supported_extensions = {ext.lower().lstrip(".") for ext in BioIoLoader.SUPPORTED_EXTENSIONS}
    for p in root.rglob("*"):
        if not p.is_file() or p.name == "not_an_image.txt":
            continue
        if p.suffix.lower().lstrip(".") in supported_extensions:
            yield p


def _copy_sample_images(src_files: List[Path], dst_dir: Path) -> List[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for src in src_files:
        dst = dst_dir / src.name
        shutil.copyfile(src, dst)
        copied.append(dst)
    return copied


def test_build_records_df_multiprocessing_on_real_images(tmp_path):
    data_dir = _example_bioio_dir()
    if not data_dir.exists():
        pytest.skip(f"Example data directory not found: {data_dir}")

    src_files = list(_iter_image_files(data_dir))
    if len(src_files) < 2:
        pytest.skip("Not enough sample images found for multiprocessing integration test")

    sample_dir = tmp_path / "sample"
    copied = _copy_sample_images(src_files[:2], sample_dir)

    loader = BioIoLoader()
    config = ProcessingConfig(
        max_workers=2,
        rows_per_part=1,
        selected_file_extensions=loader.SUPPORTED_EXTENSIONS,
    )

    df, _ = build_records_df(
        bases=[sample_dir],
        loader=loader,
        processors=discover_processor_plugins(),
        config=config,

    )

    assert df is not None
    assert df.filter(pl.col("obs_level") == 0).height == len(copied)
    assert "path" in df.columns


def test_build_records_df_multiprocessing_reports_progress(tmp_path):
    data_dir = _example_bioio_dir()
    if not data_dir.exists():
        pytest.skip(f"Example data directory not found: {data_dir}")

    src_files = list(_iter_image_files(data_dir))
    if len(src_files) < 3:
        pytest.skip("Not enough sample images found for progress callback integration test")

    sample_dir = tmp_path / "sample"
    copied = _copy_sample_images(src_files[:3], sample_dir)

    loader = BioIoLoader()
    config = ProcessingConfig(max_workers=2, rows_per_part=1)

    progress_calls = []

    def progress_callback(current: int, total: int) -> None:
        progress_calls.append((current, total))

    df, _ = build_records_df(
        bases=[sample_dir],
        loader=loader,
        processors=discover_processor_plugins(),
        config=config,

        on_progress=progress_callback,
    )

    assert df is not None
    assert df.filter(pl.col("obs_level") == 0).height == len(copied)
    assert progress_calls, "progress_callback was not called"
    assert progress_calls[-1][0] == len(copied)


def test_build_records_df_multiprocessing_includes_processor_outputs(tmp_path):
    data_dir = _example_bioio_dir()
    if not data_dir.exists():
        pytest.skip(f"Example data directory not found: {data_dir}")

    src_files = list(_iter_image_files(data_dir))
    if len(src_files) < 2:
        pytest.skip("Not enough sample images found for processor output integration test")

    sample_dir = tmp_path / "sample"
    copied = _copy_sample_images(src_files[:2], sample_dir)

    loader = BioIoLoader()
    config = ProcessingConfig(max_workers=2, rows_per_part=1)

    df, _ = build_records_df(
        bases=[sample_dir],
        loader=loader,
        processors=discover_processor_plugins(),
        config=config,

    )

    assert df is not None
    assert df.filter(pl.col("obs_level") == 0).height == len(copied)
    assert "mean_intensity" in df.columns
    assert any(val is not None for val in df["mean_intensity"].to_list())
