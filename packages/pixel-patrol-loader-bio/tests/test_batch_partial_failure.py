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
