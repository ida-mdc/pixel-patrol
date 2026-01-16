from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import shutil

import pytest

from pixel_patrol_base.core.processing import build_records_df
from pixel_patrol_base.core.project_settings import Settings
from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader


def _repo_root() -> Path:
    """Resolve the repository root directory from this test file."""
    return Path(__file__).resolve().parents[3]


def _example_bioio_dir() -> Path:
    """Locate the example bioio dataset directory."""
    return _repo_root() / "examples" / "datasets" / "bioio"


def _iter_image_files(root: Path) -> Iterable[Path]:
    """Iterate over loader-supported image files under the dataset root."""
    supported_extensions = {ext.lower().lstrip(".") for ext in BioIoLoader.SUPPORTED_EXTENSIONS}
    for p in root.rglob("*"):
        if not p.is_file() or p.name == "not_an_image.txt":
            continue
        if p.suffix.lower().lstrip(".") in supported_extensions:
            yield p


def _copy_sample_images(src_files: List[Path], dst_dir: Path) -> List[Path]:
    """Copy source images into a temporary directory for isolated processing."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for src in src_files:
        dst = dst_dir / src.name
        shutil.copyfile(src, dst)
        copied.append(dst)
    return copied


def test_build_records_df_multiprocessing_on_real_images(tmp_path):
    """Run real multiprocessing over a tiny image subset and verify flush output."""
    data_dir = _example_bioio_dir()
    if not data_dir.exists():
        pytest.skip(f"Example data directory not found: {data_dir}")

    src_files = list(_iter_image_files(data_dir))
    if len(src_files) < 2:
        pytest.skip("Not enough sample images found for multiprocessing integration test")

    sample_dir = tmp_path / "sample"
    copied = _copy_sample_images(src_files[:2], sample_dir)

    loader = BioIoLoader()
    settings = Settings(
        processing_max_workers=2,
        records_flush_every_n=1,
        records_flush_dir=tmp_path / "batches",
    )

    df = build_records_df(
        bases=[sample_dir],
        selected_extensions=loader.SUPPORTED_EXTENSIONS,
        loader=loader,
        settings=settings,
    )

    assert df is not None
    assert df.height == len(copied)
    assert "path" in df.columns
    assert (tmp_path / "batches" / "records_df.parquet").exists()
#
#
# def test_build_records_df_multiprocessing_reports_progress(tmp_path):
#     """Progress callback should reflect total processed files with multiprocessing."""
#     data_dir = _example_bioio_dir()
#     if not data_dir.exists():
#         pytest.skip(f"Example data directory not found: {data_dir}")
#
#     src_files = list(_iter_image_files(data_dir))
#     if len(src_files) < 3:
#         pytest.skip("Not enough sample images found for progress callback integration test")
#
#     sample_dir = tmp_path / "sample"
#     copied = _copy_sample_images(src_files[:3], sample_dir)
#
#     loader = BioIoLoader()
#     settings = Settings(
#         processing_max_workers=2,
#         records_flush_every_n=1,
#         records_flush_dir=tmp_path / "batches",
#     )
#
#     progress_calls = []
#
#     def progress_callback(current: int, total: int, current_file: Path) -> None:
#         progress_calls.append((current, total, current_file))
#
#     df = build_records_df(
#         bases=[sample_dir],
#         selected_extensions=loader.SUPPORTED_EXTENSIONS,
#         loader=loader,
#         settings=settings,
#         progress_callback=progress_callback,
#     )
#
#     assert df is not None
#     assert df.height == len(copied)
#     assert sum(call[0] for call in progress_calls) == len(copied)
#
#
# def test_build_records_df_multiprocessing_includes_processor_outputs(tmp_path):
#     """Processor outputs should appear when multiprocessing loads real images."""
#     data_dir = _example_bioio_dir()
#     if not data_dir.exists():
#         pytest.skip(f"Example data directory not found: {data_dir}")
#
#     src_files = list(_iter_image_files(data_dir))
#     if len(src_files) < 2:
#         pytest.skip("Not enough sample images found for processor output integration test")
#
#     sample_dir = tmp_path / "sample"
#     copied = _copy_sample_images(src_files[:2], sample_dir)
#
#     loader = BioIoLoader()
#     settings = Settings(
#         processing_max_workers=2,
#         records_flush_every_n=1,
#         records_flush_dir=tmp_path / "batches",
#     )
#
#     df = build_records_df(
#         bases=[sample_dir],
#         selected_extensions=loader.SUPPORTED_EXTENSIONS,
#         loader=loader,
#         settings=settings,
#     )
#
#     assert df is not None
#     assert df.height == len(copied)
#     assert "mean_intensity" in df.columns
#     assert any(val is not None for val in df["mean_intensity"].to_list())
