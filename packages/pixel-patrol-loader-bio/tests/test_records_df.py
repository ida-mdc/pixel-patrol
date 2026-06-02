from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tifffile
from PIL import Image

from pixel_patrol_base.core.processing import build_records_df
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.plugin_registry import discover_processor_plugins
from pixel_patrol_base.utils.df_utils import postprocess_basic_file_metadata_df
import zarr
from zarr.storage import LocalStore
from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader
from pixel_patrol_loader_bio.plugins.loaders.tifffile_loader import TifffileLoader
from pixel_patrol_loader_bio.plugins.loaders.zarr_loader import ZarrLoader
from datetime import datetime


@pytest.fixture
def loader():
    return BioIoLoader()


@pytest.fixture
def processors():
    return discover_processor_plugins()


def test_build_records_df_returns_none_when_no_matching_files(tmp_path, loader, processors):
    (tmp_path / "document.txt").write_text("not an image")
    result, _ = build_records_df(
        bases=[tmp_path],
        loader=loader,
        processors=processors,
        config=ProcessingConfig(selected_file_extensions={"png", "jpg"}),
    )
    assert result is None


def test_postprocess_basic_file_metadata_df_adds_modification_month(tmp_path):
    base = tmp_path
    df = pl.DataFrame({
        "path": [str(base / "sub" / "a.txt"), str(base / "sub" / "b.txt")],
        "name": ["a.txt", "b.txt"],
        "type": ["file", "file"],
        "parent": [str(base / "sub"), str(base / "sub")],
        "depth": [2, 2],
        "size_bytes": [1024, 2048],
        "modification_date": [
            datetime(2025, 3, 15, 12, 0),
            datetime(2025, 7, 1, 9, 30),
        ],
        "file_extension": ["txt", "txt"],
        "size_readable": ["", ""],
        "imported_path": [str(base), str(base)],
    })

    out = postprocess_basic_file_metadata_df(df)

    assert "modification_month" in out.columns
    assert "size_readable" in out.columns
    assert out["modification_month"].to_list() == [3, 7]
    assert out["size_readable"].to_list() == ["1.0 KB", "2.0 KB"]


def test_full_records_df_computes_real_mean_intensity(tmp_path, loader, processors):
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()

    a = np.zeros((2, 2, 1), dtype=np.uint8)
    Image.fromarray(a.squeeze(), mode="L").save(img_dir / "zero.png")

    b = np.full((2, 2, 1), 255, dtype=np.uint8)
    Image.fromarray(b.squeeze(), mode="L").save(img_dir / "full.png")

    df, _ = build_records_df(
        bases=[img_dir],
        loader=loader,
        processors=processors,
        config=ProcessingConfig(selected_file_extensions={"png"}),
    )
    assert isinstance(df, pl.DataFrame)
    paths = df["path"].to_list()
    assert sorted(Path(p).name for p in paths) == ["full.png", "zero.png"]

    assert "mean_intensity" in df.columns

    mip = {Path(p).name: v for p, v in zip(df["path"].to_list(), df["mean_intensity"].to_list())}
    assert mip["zero.png"] == 0.0
    assert mip["full.png"] == 255.0


def test_full_records_df_handles_5d_tif_t_z_c_dimensions(tmp_path, loader, processors):
    t_size, c_size, z_size, y_size, x_size = 2, 3, 4, 2, 2
    arr = np.zeros((t_size, c_size, z_size, y_size, x_size), dtype=np.uint8)
    for t in range(t_size):
        for c in range(c_size):
            for z in range(z_size):
                arr[t, c, z, ...] = (t * z_size + z) * 10 + c * 5

    path = tmp_path / "5d.tif"
    tifffile.imwrite(str(path), arr, photometric='minisblack')

    df, _ = build_records_df(
        bases=[tmp_path],
        loader=loader,
        processors=processors,
        config=ProcessingConfig(selected_file_extensions={"tif"}),
    )

    assert {"dim_t", "dim_c", "dim_z", "mean_intensity"}.issubset(set(df.columns))

    def mean_at(t, c, z):
        row = df.filter(
            (pl.col("obs_level") == 3) &
            (pl.col("dim_t") == t) &
            (pl.col("dim_c") == c) &
            (pl.col("dim_z") == z)
        )
        assert len(row) == 1, f"Expected 1 row for t={t} c={c} z={z}, got {len(row)}"
        return row["mean_intensity"][0]

    for t in range(t_size):
        for z in range(z_size):
            for c in range(c_size):
                expected = (t * z_size + z) * 10 + c * 5
                assert mean_at(t, c, z) == expected, f"t={t} c={c} z={z}: expected {expected}"

    def mean_single(obs, **dims):
        mask = pl.col("obs_level") == obs
        for k, v in dims.items():
            mask = mask & (pl.col(f"dim_{k}") == v)
        row = df.filter(mask)
        assert len(row) == 1, f"Expected 1 row for obs={obs} dims={dims}"
        return row["mean_intensity"][0]

    for t in range(t_size):
        block_vals = [(t * z_size + z) * 10 + c * 5 for c in range(c_size) for z in range(z_size)]
        assert mean_single(1, t=t) == pytest.approx(sum(block_vals) / len(block_vals))

    for c in range(c_size):
        block_vals = [(t * z_size + z) * 10 + c * 5 for t in range(t_size) for z in range(z_size)]
        assert mean_single(1, c=c) == pytest.approx(sum(block_vals) / len(block_vals))

    for z in range(z_size):
        block_vals = [(t * z_size + z) * 10 + c * 5 for t in range(t_size) for c in range(c_size)]
        assert mean_single(1, z=z) == pytest.approx(sum(block_vals) / len(block_vals))

    all_vals = [
        (t * z_size + z) * 10 + c * 5
        for t in range(t_size) for c in range(c_size) for z in range(z_size)
    ]
    assert mean_single(0) == pytest.approx(sum(all_vals) / len(all_vals))


def test_full_records_df_handles_png_gray(tmp_path, loader, processors):
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    arr[..., 0] = 10
    arr[..., 1] = 20
    arr[..., 2] = 30

    path = tmp_path / "rgb.png"
    Image.fromarray(arr).save(str(path))

    df, _ = build_records_df(
        bases=[tmp_path],
        loader=loader,
        processors=processors,
        config=ProcessingConfig(selected_file_extensions={"png"}),
    )

    assert "mean_intensity" in df.columns
    raw_gray = np.mean(arr)
    expected_gray = np.uint8(raw_gray)
    global_row = df.filter(pl.col("obs_level") == 0)["mean_intensity"][0]
    assert global_row == expected_gray


def test_build_records_df_tifffile(tmp_path):
    im = np.zeros((4, 8, 8), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "img.tif", im, imagej=True, metadata={"axes": "CYX"})
    df, _ = build_records_df(
        bases=[tmp_path],
        loader=TifffileLoader(),
        processors=discover_processor_plugins(),
        config=ProcessingConfig(selected_file_extensions={"tif"}),
    )
    assert df is not None
    assert len(df) > 0


def test_build_records_df_zarr(tmp_path):
    zarr_path = tmp_path / "img.zarr"
    store = LocalStore(str(zarr_path))
    root = zarr.group(store=store)
    arr = root.create_array("0", shape=(2, 8, 8), chunks=(2, 8, 8), dtype="uint16", overwrite=True)
    arr[:] = np.zeros((2, 8, 8), dtype="uint16")
    root.attrs.put({"multiscales": [{"version": "0.4", "datasets": [{"path": "0"}],
        "axes": [{"name": "c"}, {"name": "y"}, {"name": "x"}]}]})
    df, _ = build_records_df(
        bases=[tmp_path],
        loader=ZarrLoader(),
        processors=discover_processor_plugins(),
        config=ProcessingConfig(selected_file_extensions={"zarr"}),
    )
    assert df is not None
    assert len(df) > 0
