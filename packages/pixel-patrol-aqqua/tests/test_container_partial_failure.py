"""A sub-image that fails to load inside an otherwise-good container must be
counted as failed, not silently dropped from every count."""
from __future__ import annotations

from pathlib import Path

import lmdb
import numpy as np

from pixel_patrol_aqqua.plugins.loaders.lmdb_loader import LmdbLoader
from pixel_patrol_base.core.processing import build_records_df
from pixel_patrol_base.core.processing_config import ProcessingConfig

from conftest import _write_lmdb


def _corrupt_entry(lmdb_path: Path, index: int) -> None:
    env = lmdb.open(str(lmdb_path), map_size=50 * 1024 * 1024, max_dbs=2)
    db = env.open_db(key=b"image_data", integerkey=True, create=False)
    with env.begin(db=db, write=True) as txn:
        txn.put(index.to_bytes(8, byteorder="little"), b"not a valid blosc2 cframe")
    env.close()


def test_corrupted_sub_image_counts_as_failed(tmp_path: Path):
    lmdb_path = tmp_path / "test.lmdb"
    rng = np.random.default_rng(0)
    images = [
        (rng.integers(0, 255, (8, 8), dtype=np.uint8), {"image-uuid": f"uuid-{i}"})
        for i in range(5)
    ]
    _write_lmdb(lmdb_path, images)
    _corrupt_entry(lmdb_path, 3)  # break one image outside read_header's 3-sample peek

    config = ProcessingConfig(max_workers=1, mb_per_task=100)
    df, stats = build_records_df(
        [tmp_path], loader=LmdbLoader(), processors=[], config=config, base_dir=tmp_path,
    )

    assert stats["n_images_processed"] == 4
    assert stats["n_images_failed"] == 1
