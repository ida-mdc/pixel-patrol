from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import tifffile
from PIL import Image

from pixel_patrol_base.config import STANDARD_DIM_ORDER
from pixel_patrol_base.core import processing
from pixel_patrol_base.core.image_operations_and_metadata import get_all_image_properties
from pixel_patrol_base.core.loaders.bioio_loader import BioIoLoader
from pixel_patrol_base.core.processing import (
    build_images_df_from_paths_df,
    build_images_df_from_file_system,
    _scan_dirs_for_extensions,
    _filter_paths_df,
    _get_deep_image_df,
    PATHS_DF_EXPECTED_SCHEMA,
    _postprocess_basic_file_metadata_df
)
from pixel_patrol_base.plugins import discover_processor_plugins


@pytest.fixture(scope="module")
def loader():
    """Define a set of all available image properties we expect to extract."""
    return BioIoLoader()

@pytest.fixture(scope="module")
def processors():
    """Define a set of all available image properties we expect to extract."""
    return discover_processor_plugins()

def test_build_images_df_from_paths_df_empty(mock_empty_paths_df):
    assert build_images_df_from_paths_df(mock_empty_paths_df, {"png", "jpg"}, loader="bioio") is None


def test_build_images_df_from_paths_df_with_only_non_image_rows_returns_none():
    # Only folder or non-image extensions
    df = pl.DataFrame({
        "path": ["a.txt", "folder", "b.doc"],
        "type": ["file", "folder", "file"],
        "file_extension": ["txt", None, "doc"],
    })
    assert build_images_df_from_paths_df(df, {"jpg", "png"}, "bioio") is None


def test_build_images_df_from_file_system_no_images(tmp_path):
    non_image_dir = tmp_path / "non_image_files"
    non_image_dir.mkdir()
    (non_image_dir / "document.txt").write_text("This is a text file.")
    (non_image_dir / "data.csv").write_text("1,2,3\n4,5,6")

    paths = [non_image_dir]
    extensions = {"png", "jpg"}

    with patch('pixel_patrol_base.core.processing._get_deep_image_df', return_value=pl.DataFrame()) as mock_get_deep_image_df:
        images_df = build_images_df_from_file_system(paths, extensions, "bioio")
        assert images_df is None
        mock_get_deep_image_df.assert_not_called()


def test_scan_dirs_for_extensions_filters_correct_extensions(tmp_path):
    dir1 = tmp_path / "dir1"; dir1.mkdir()
    dir2 = tmp_path / "dir2"; dir2.mkdir()
    (dir1 / "a.jpg").write_bytes(b"")
    (dir1 / "b.png").write_bytes(b"")
    (dir1 / "c.txt").write_bytes(b"")
    (dir2 / "d.JPG").write_bytes(b"")
    result = _scan_dirs_for_extensions([dir1, dir2], {"jpg", "png"})
    expected = {
        (dir1 / "a.jpg", dir1),
        (dir1 / "b.png", dir1),
        (dir2 / "d.JPG", dir2),
    }
    assert set(result) == expected

def test_scan_dirs_for_extensions_handles_empty_directory_list():
    result = _scan_dirs_for_extensions([], {"jpg", "png"})
    assert result == []


def test_filter_paths_df_returns_only_file_rows_with_accepted_extensions():
    df = pl.DataFrame({
        "path": ["a.jpg", "b.png", "c.txt", "d.jpg", "e"],
        "type": ["file", "file", "file", "folder", "file"],
        "file_extension": ["jpg", "PNG", "txt", None, ""],
    })
    result = _filter_paths_df(df, {"jpg", "png"})
    # Only rows 0 and 1 match type 'file' and extensions jpg or png (case-insensitive)
    assert result.select("path").to_series().to_list() == ["a.jpg", "b.png"]


def test_filter_paths_df_excludes_folders_and_unaccepted_extensions():
    df = pl.DataFrame({
        "path": ["img1.jpg", "img2.png", "docs", "img3.gif", "folder.jpg"],
        "type": ["file", "file", "folder", "file", "folder"],
        "file_extension": ["jpg", "png", None, "gif", "jpg"],
    })
    result = _filter_paths_df(df, {"jpg", "png"})
    assert result.select("path").to_series().to_list() == ["img1.jpg", "img2.png"]


def test_get_deep_image_df_returns_dataframe_with_required_columns(tmp_path, monkeypatch):
    p1 = tmp_path / "img1.jpg"; p1.write_bytes(b"")
    p2 = tmp_path / "img2.png"; p2.write_bytes(b"")
    paths = [p1, p2]
    required_cols = ["width", "height"]

    def fake_get_all_image_properties(_path, read_pixel_data, loader, processors):
        assert loader.id() == "bioio"
        assert read_pixel_data == True
        return {"width": 100, "height": 200}
    monkeypatch.setattr(processing, "get_all_image_properties", fake_get_all_image_properties)

    df = _get_deep_image_df(paths, "bioio", read_pixel_data=True)

    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == {"path", "width", "height"}
    assert df.height == 2
    assert df["width"].to_list() == [100, 100]
    assert df["height"].to_list() == [200, 200]


def test_get_all_image_properties_returns_empty_for_nonexistent_file(tmp_path, loader, processors):
    missing = tmp_path / "no.png"
    assert get_all_image_properties(missing, read_pixel_data=True, loader=loader, processors=processors) == {}


def test_get_all_image_properties_returns_empty_if_loading_fails(tmp_path, monkeypatch, loader, processors):
    img_file = tmp_path / "img.jpg"
    img_file.write_bytes(b"not really an image")
    monkeypatch.setattr("pixel_patrol_base.core.loaders.bioio_loader._load_bioio_image", lambda p: None)
    assert get_all_image_properties(img_file, read_pixel_data=True, loader=loader, processors=processors) == {}


class DummyImg:
    dims = type("D", (), {"order": STANDARD_DIM_ORDER})()
    physical_pixel_sizes = type("PS", (), {"X": 0.5, "Y": 0.75, "Z": None, "T": None})()
    channel_names = ["ch1", "ch2"]
    ome_metadata = {}
    shape = tuple(
        2 if d in ("Y", "X") else 1
        for d in STANDARD_DIM_ORDER
    )
    def get_image_data(self):
        return np.zeros(self.shape, dtype=np.uint8)

def test_get_all_image_properties_extracts_standard_and_requested_metadata(tmp_path, monkeypatch, loader, processors):
    img_file = tmp_path / "img.png"
    img_file.write_bytes(b"")

    monkeypatch.setattr(
        "pixel_patrol_base.core.loaders.bioio_loader._load_bioio_image",
        lambda p: DummyImg()
    )
    props = get_all_image_properties(
        img_file, read_pixel_data=False, loader=loader, processors=processors
    )
    expected_shape = [
        2 if d in ("Y", "X") else 1
        for d in STANDARD_DIM_ORDER]

    assert props["shape"].tolist() == expected_shape
    assert props["ndim"] == len(STANDARD_DIM_ORDER)

    assert props["pixel_size_X"] == 0.5
    assert props["channel_names"] == ["ch1", "ch2"]


def test_get_deep_image_df_ignores_paths_with_no_metadata(tmp_path, monkeypatch, loader, processors):
    p_valid = tmp_path / "valid.jpg"; p_valid.write_bytes(b"")
    p_invalid = tmp_path / "invalid.png"; p_invalid.write_bytes(b"")

    def fake_get_all_image_properties(path, _read_pixels, _loader, _processors):
        return {"width": 10, "height": 20} if path == p_valid else {}

    monkeypatch.setattr(
        "pixel_patrol_base.core.processing.get_all_image_properties",
        fake_get_all_image_properties
    )

    df = _get_deep_image_df([p_valid, p_invalid], loader="bioio", read_pixel_data=True)

    assert isinstance(df, pl.DataFrame)
    assert df.height == 1
    assert df["path"].to_list() == [str(p_valid)]
    assert df["width"].to_list() == [10]
    assert df["height"].to_list() == [20]


def test_build_images_df_from_paths_df_with_valid_paths_df_returns_expected_columns_and_values(
    mock_paths_df_content, monkeypatch
):
    paths_df = mock_paths_df_content

    ext_set = set(paths_df["file_extension"].to_list())
    filtered = _filter_paths_df(paths_df, ext_set)
    valid_paths = filtered["path"].to_list()

    deep_df = pl.DataFrame({
        "path": valid_paths,
        "width": [100 + i*100 for i in range(len(valid_paths))],
        "height": [150 + i*100 for i in range(len(valid_paths))],
    })
    monkeypatch.setattr(
        "pixel_patrol_base.core.processing._get_deep_image_df",
        lambda paths, cols: deep_df
    )

    result = build_images_df_from_paths_df(paths_df, ext_set, "bioio")

    expected_cols = set(PATHS_DF_EXPECTED_SCHEMA.keys()) | {"width", "height"}
    assert expected_cols.issubset(set(result.columns))

    assert result["path"].to_list() == deep_df["path"].to_list()
    assert result["width"].to_list() == deep_df["width"].to_list()
    assert result["height"].to_list() == deep_df["height"].to_list()


def test_build_images_df_from_paths_df__returns_nulls_for_missing_deep_metadata(
    mock_paths_df_content, monkeypatch
):

    paths_df = mock_paths_df_content
    extensions = {"jpg", "png"}

    filtered = _filter_paths_df(paths_df, extensions)
    basic_paths = [*filtered["path"].to_list()]

    stubbed = pl.DataFrame({
        "path": [basic_paths[0]],
        "width": [123],
        "height": [456],
    })
    monkeypatch.setattr(
        "pixel_patrol_base.core.processing._get_deep_image_df",
        lambda paths, cols: stubbed
    )

    result = build_images_df_from_paths_df(paths_df, extensions, "bioio")

    assert len(result) == 2
    first_row = result.filter(pl.col("path") == basic_paths[0])
    second_row = result.filter(pl.col("path") == basic_paths[1])

    assert first_row["width"].item() == 123
    assert first_row["height"].item() == 456

    assert second_row["width"].is_null().all()
    assert second_row["height"].is_null().all()


def test_build_images_df_from_file_system_with_images_returns_expected_columns_and_values(tmp_path, monkeypatch):

    base = tmp_path / "root"
    base.mkdir()
    img1 = base / "graphic.png"; img1.write_text("dummy")
    img2 = base / "photo1.jpg";  img2.write_text("dummy")
    (base / "notes.txt").write_text("not an image")

    expected_paths = [str(img1), str(img2)]

    deep_df = pl.DataFrame({
        "path": expected_paths,
        "width": [64, 128],
        "height": [48, 256],
    })
    monkeypatch.setattr(
        "pixel_patrol_base.core.processing._get_deep_image_df",
        lambda paths, cols: deep_df
    )

    result = build_images_df_from_file_system(
        bases=[base],
        selected_extensions={"jpg", "png"},
        loader="bioio"
    )

    assert result is not None

    expected_cols = set(PATHS_DF_EXPECTED_SCHEMA.keys()) | {"width", "height"}
    assert expected_cols.issubset(set(result.columns))

    assert set(result["path"].to_list()) == set(expected_paths)

    # Check that width/height values match for each path
    result_dict = { 
        row["path"]: (row["width"], row["height"]) for row in result.iter_rows(named=True)
    }
    expected_dict = dict(zip(expected_paths, zip([64, 128], [48, 256])))
    assert result_dict == expected_dict

def test_build_images_df_from_file_system_merges_basic_and_deep_metadata_correctly(tmp_path, monkeypatch):

    base = tmp_path / "root"
    base.mkdir()
    img1 = base / "one.jpg";  img1.write_text("x")
    img2 = base / "two.png";  img2.write_text("y")

    deep_df = pl.DataFrame({
        "path": [str(img1), str(img2)],
        "width": [10, 20],
        "height": [15, 25],
    })
    monkeypatch.setattr(
        "pixel_patrol_base.core.processing._get_deep_image_df",
        lambda paths, cols: deep_df
    )

    result = build_images_df_from_file_system(
        bases=[base],
        selected_extensions={"jpg", "png"},
        loader="bioio"
    )

    expected_cols = set(PATHS_DF_EXPECTED_SCHEMA.keys()) | {"width", "height"}
    assert expected_cols.issubset(set(result.columns))

    df = result.sort("path")
    assert df["path"].to_list() == [str(img1), str(img2)]
    assert df["width"].to_list() == [10, 20]
    assert df["height"].to_list() == [15, 25]


def test_postprocess_basic_file_metadata_df_adds_modification_month_and_imported_path_short(tmp_path):
    from pathlib import Path
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
        # size_readable will be recomputed by the function
        "size_readable": ["", ""],
        "imported_path": [str(base), str(base)],
    })

    out = _postprocess_basic_file_metadata_df(df)

    assert set(PATHS_DF_EXPECTED_SCHEMA.keys()).issubset(set(out.columns))
    assert out["modification_month"].to_list() == [3, 7]
    # imported_path_short may be the full base path or just the last part, depending on OS/Polars
    actual_short = out["imported_path_short"].to_list()
    expected_full = [str(base), str(base)]
    expected_last = [Path(base).name, Path(base).name]
    assert actual_short == expected_full or actual_short == expected_last
    assert out["size_readable"].to_list() == ["1.0 KB", "2.0 KB"]

def test_full_images_df_computes_real_mean_intensity(tmp_path):
    # Create a directory with two 2×2 images of known values
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()

    # Image A: all zeros → mean_intensity == 0
    a = np.zeros((2,2,1), dtype=np.uint8)
    Image.fromarray(a.squeeze(), mode="L").save(img_dir / "zero.png")

    # Image B: all 255 → mean_intensity == 255
    b = np.full((2,2,1), 255, dtype=np.uint8)
    Image.fromarray(b.squeeze(), mode="L").save(img_dir / "full.png")

    # Build without any monkeypatching
    df = build_images_df_from_file_system(
        bases=[img_dir],
        selected_extensions={"png"},
        loader="bioio"
    )
    assert isinstance(df, pl.DataFrame)
    paths = df["path"].to_list()
    assert sorted(Path(p).name for p in paths) == ["full.png", "zero.png"]

    assert "mean_intensity" in df.columns

    mip = { Path(p).name: v for p, v in zip(df["path"].to_list(), df["mean_intensity"].to_list()) }
    assert mip["zero.png"] == 0.0
    assert mip["full.png"] == 255.0


def test_full_images_df_handles_5d_tif_t_z_c_dimensions(tmp_path):

    t_size, c_size, z_size, y_size, x_size = 2, 3, 4, 2, 2
    arr = np.zeros((t_size, c_size, z_size, y_size, x_size), dtype=np.uint8)
    for t in range(t_size):
        for c in range(c_size):
            for z in range(z_size):
                arr[t, c, z, ...] = (t*z_size + z)*10 + c*5

    path = tmp_path / "5d.tif"
    tifffile.imwrite(str(path), arr, photometric='minisblack')

    df = build_images_df_from_file_system(
        bases=[tmp_path],
        selected_extensions={"tif"},
        loader="bioio"
    )

    expected_cols = {
        f"mean_intensity_t{t}_c{c}_z{z}"
        for t in range(t_size) for z in range(z_size) for c in range(c_size)
    }
    assert expected_cols.issubset(set(df.columns))

    actual = {col: df[col][0] for col in expected_cols}
    for t in range(t_size):
        for z in range(z_size):
            for c in range(c_size):
                key = f"mean_intensity_t{t}_c{c}_z{z}"
                expected = (t*z_size + z)*10 + c*5
                assert actual[key] == expected, f"{key} was {actual[key]}, expected {expected}"

    for t in range(t_size):
        col = f"mean_intensity_t{t}"
        assert col in df.columns
        # average of all blocks at this t:
        block_vals = [(t * z_size + z) * 10 + c * 5
                      for c in range(c_size) for z in range(z_size)]
        expected = sum(block_vals) / len(block_vals)
        assert df[0, col] == expected

    for c in range(c_size):
        col = f"mean_intensity_c{c}"
        assert col in df.columns
        block_vals = [(t * z_size + z) * 10 + c * 5
                      for t in range(t_size) for z in range(z_size)]
        expected = sum(block_vals) / len(block_vals)
        assert df[0, col] == expected

    for z in range(z_size):
        col = f"mean_intensity_z{z}"
        assert col in df.columns
        block_vals = [(t * z_size + z) * 10 + c * 5
                      for t in range(t_size) for c in range(c_size)]
        expected = sum(block_vals) / len(block_vals)
        assert df[0, col] == expected

    assert "mean_intensity" in df.columns
    all_vals = [(t * z_size + z) * 10 + c * 5
                for t in range(t_size) for c in range(c_size) for z in range(z_size)]
    overall_expected = sum(all_vals) / len(all_vals)
    assert df[0,"mean_intensity"] == overall_expected


def test_full_images_df_handles_png_gray(tmp_path):

    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    arr[..., 0] = 10  # R
    arr[..., 1] = 20  # G
    arr[..., 2] = 30  # B

    path = tmp_path / "rgb.png"
    Image.fromarray(arr).save(str(path))

    df = build_images_df_from_file_system(
        bases=[tmp_path],
        selected_extensions={"png"},
        loader="bioio"
    )

    assert "mean_intensity" in df.columns

    # TODO we need to re-add the functionality to weight the S dimension properly to support RGB correctly.
    # raw_gray = np.dot(np.array([10, 20, 30], np.float32), np.array(RGB_WEIGHTS, np.float32))
    raw_gray = np.mean(arr)
    expected_gray = np.uint8(raw_gray)
    assert df["mean_intensity"][0] == expected_gray
