"""A file that fails to load inside an otherwise-good batch must be counted
as failed, not silently discarded along with its batch siblings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from pixel_patrol_base.core.processing import build_records_df
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader


def test_corrupted_file_in_batch_counts_as_failed(tmp_path: Path):
    for name in ("a.tif", "bad.tif", "c.tif"):
        arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        tifffile.imwrite(tmp_path / name, arr)

    # Truncate pixel data (but not the header) so read_header succeeds during
    # planning and the failure only surfaces when the batch actually loads it.
    bad_path = tmp_path / "bad.tif"
    data = bad_path.read_bytes()
    bad_path.write_bytes(data[: len(data) // 2])

    config = ProcessingConfig(max_workers=1, mb_per_task=100)
    df, stats = build_records_df(
        [tmp_path], loader=BioIoLoader(), processors=[], config=config, base_dir=tmp_path,
    )

    assert stats["n_images_processed"] == 2
    assert stats["n_images_failed"] == 1


def test_two_corrupted_files_in_one_batch_both_count_as_failed(tmp_path: Path):
    """Regression guard: failures must be counted per file, not per batch task -
    a single batch covering 2 bad files out of 4 must report n_images_failed == 2, not 1."""
    for name in ("a.tif", "bad1.tif", "bad2.tif", "d.tif"):
        arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        tifffile.imwrite(tmp_path / name, arr)

    for name in ("bad1.tif", "bad2.tif"):
        bad_path = tmp_path / name
        data = bad_path.read_bytes()
        bad_path.write_bytes(data[: len(data) // 2])

    config = ProcessingConfig(max_workers=1, mb_per_task=100)
    df, stats = build_records_df(
        [tmp_path], loader=BioIoLoader(), processors=[], config=config, base_dir=tmp_path,
    )

    assert stats["n_images_processed"] == 2
    assert stats["n_images_failed"] == 2
