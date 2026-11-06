from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pixel_patrol_base.core.compression import compression_factor, compression_type
from pixel_patrol_base.core.processing import _FileLedger, _ResultsWriter, _extract_image_meta, save_parquet_from_parts
from pixel_patrol_base.core.record import record_from


@pytest.mark.parametrize("algorithm,expected", [
    ("none", "uncompressed"), ("bytes(little)", "uncompressed"),
    ("deflate", "lossless"), ("deflate + horizontal predictor", "lossless"), ("blosc(zstd, bitshuffle) level 3", "lossless"),
    ("zstd1", "lossless"), ("gzip level 4 + shuffle", "lossless"), ("bytes(little) + zstd", "lossless"),
    ("JPEG", "lossy"), ("zfpy", "lossy"), ("deflate + JPEG", "lossy"),
    ("JPEG XR", "lossy or lossless"), ("jpeg2000", "lossy or lossless"), ("JPEG XR + zstd0", "lossy or lossless"),
    ("frobnicate", "unknown"), (None, None),
])
def test_compression_type(algorithm, expected):
    assert compression_type(algorithm) == expected


@pytest.mark.parametrize("pixel_bytes,size,expected", [(1000, 250, 4.0), (1000, 1000, 1.0), (1000, 1400, 1.0), (0, 10, None), (1000, 0, None)])
def test_compression_factor_is_capped_at_one_for_files_larger_than_their_pixels(pixel_bytes, size, expected):
    assert compression_factor(pixel_bytes, size) == expected


def test_image_meta_gets_the_type_of_whatever_algorithm_the_loader_reported():
    image = record_from(np.zeros((4, 4), np.uint8), {"dim_order": "YX", "compression": "JPEG"}, kind="intensity")
    plain = record_from(np.zeros((4, 4), np.uint8), {"dim_order": "YX"}, kind="intensity")
    assert _extract_image_meta(image)["compression_type"] == "lossy"
    assert "compression_type" not in _extract_image_meta(plain)


def _image_row(num_pixels, dtype="uint16"):
    return {"num_pixels": num_pixels, "dtype": dtype}


FILES = [{"path": "one.tif", "size_bytes": 500}, {"path": "scenes.czi", "size_bytes": 625}]


def test_a_file_gets_the_factor_of_all_its_images():
    ledger = _FileLedger()
    ledger.add(1, _image_row(1000), n_images=2)
    ledger.add(1, _image_row(500, "uint8"), n_images=2)
    ledger.add(0, _image_row(250), n_images=1)
    assert ledger.compression_factors(FILES).sort("path").rows() == [("one.tif", 1.0), ("scenes.czi", 4.0)]


def test_a_file_with_failed_images_or_no_pixel_info_gets_no_factor():
    ledger = _FileLedger()
    ledger.add(1, _image_row(1000), n_images=2)
    ledger.add(0, {"num_pixels": 10}, n_images=1)
    assert ledger.compression_factors(FILES).is_empty()


def test_the_ledger_keeps_a_few_numbers_per_file_not_the_images():
    ledger = _FileLedger()
    for _ in range(1000):
        ledger.add(0, _image_row(10), n_images=1000)
    assert len(ledger._files) == 1


def _rows(*paths):
    return pl.DataFrame({"path": list(paths), "obs_level": [0] * len(paths)})


def test_in_memory_result_gets_the_factor_by_path():
    writer = _ResultsWriter(rows_per_part=100, parts_dir=None)
    writer.add(_rows("scenes.czi", "scenes.czi", "other.tif"))
    factors = pl.DataFrame({"path": ["scenes.czi"], "compression_factor": [4.0]})
    result = writer.finalize(factors)
    assert result["compression_factor"].to_list() == [4.0, 4.0, None]


def test_streamed_result_gets_the_per_file_factor(tmp_path: Path):
    from pixel_patrol_base.core.processing_config import ProcessingConfig
    from pixel_patrol_base.core.project import Project
    (tmp_path / "images").mkdir()
    metadata = ProcessingConfig().metadata.populate_from_project(Project("t", tmp_path / "images", output_path=tmp_path / "o.parquet"))
    parts = tmp_path / "parts"
    parts.mkdir()
    _rows("scenes.czi").write_parquet(parts / "part_0000.parquet")
    _rows("scenes.czi", "other.tif").write_parquet(parts / "part_0001.parquet")
    file_columns = pl.DataFrame({"path": ["scenes.czi"], "compression_factor": [4.0]})

    dest = tmp_path / "merged.parquet"
    save_parquet_from_parts(sorted(parts.glob("part_*.parquet")), dest, metadata, file_columns=file_columns)

    merged = pl.read_parquet(dest)
    assert merged.sort("path")["compression_factor"].to_list() == [None, 4.0, 4.0]
    assert merged.schema["compression_factor"] == pl.Float32
