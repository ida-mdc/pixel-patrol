import pickle
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from pixel_patrol_base.core.contracts import SkipFile

from pixel_patrol_tomography.mrc_loader import MrcLoader


def write_mrc(path: Path, data: np.ndarray, *, stack: bool = False, voxel_size: float | None = None) -> Path:
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(data)
        if stack:
            mrc.set_image_stack()
        if voxel_size is not None:
            mrc.voxel_size = voxel_size
        mrc.add_label("unit test label")
    return path


@pytest.fixture
def tilt_series_path(tmp_path: Path) -> Path:
    data = np.arange(5 * 8 * 6, dtype=np.uint16).reshape(5, 8, 6)
    return write_mrc(tmp_path / "tilt.st", data, stack=True, voxel_size=12.5)


@pytest.fixture
def volume_path(tmp_path: Path) -> Path:
    data = np.arange(4 * 8 * 6, dtype=np.int16).reshape(4, 8, 6) - 50
    return write_mrc(tmp_path / "volume.rec", data, voxel_size=10.0)


def test_read_header_returns_tilt_axis_for_stacks(tilt_series_path: Path) -> None:
    info = MrcLoader().read_header(tilt_series_path)
    assert info.shape == (5, 8, 6), ("unexpected shape", info.shape)
    assert info.dtype == np.uint16, ("unexpected dtype", info.dtype)
    assert info.dim_order == "AYX", ("unexpected dim order", info.dim_order)


def test_read_header_returns_z_for_volumes(volume_path: Path) -> None:
    info = MrcLoader().read_header(volume_path)
    assert info.dim_order == "ZYX", ("unexpected dim order", info.dim_order)


def test_load_tilt_series_returns_lazy_record_with_meta(tilt_series_path: Path) -> None:
    record = MrcLoader().load(tilt_series_path)
    assert record.dim_order == "AYX", ("unexpected dim order", record.dim_order)
    assert record.kind == "intensity", ("unexpected kind", record.kind)
    assert record.data.chunks == ((1,) * 5, (8,), (6,)), ("expected one chunk per section", record.data.chunks)
    expected = np.arange(5 * 8 * 6, dtype=np.uint16).reshape(5, 8, 6)
    np.testing.assert_array_equal(record.data.compute(), expected)
    assert record.meta["mrc_space_group"] == 0, ("stack must have ispg 0", record.meta["mrc_space_group"])
    assert record.meta["pixel_size_X"] == pytest.approx(12.5), ("unexpected pixel size", record.meta)
    assert record.meta["pixel_size_Y"] == pytest.approx(12.5), ("unexpected pixel size", record.meta)
    assert "pixel_size_A" not in record.meta, ("tilt axis must not get a pixel size", record.meta)
    assert "unit test label" in record.meta["mrc_labels"], ("label missing", record.meta["mrc_labels"])


def test_load_volume_returns_native_int16(volume_path: Path) -> None:
    record = MrcLoader().load(volume_path)
    assert record.dim_order == "ZYX", ("unexpected dim order", record.dim_order)
    assert "spatial-3d" in record.capabilities, ("volume must be spatial-3d", record.capabilities)
    assert record.meta["pixel_size_Z"] == pytest.approx(10.0), ("unexpected pixel size", record.meta)
    assert record.data.compute().min() == -50, ("signed data not preserved", record.data.compute().min())


def test_load_2d_image(tmp_path: Path) -> None:
    path = write_mrc(tmp_path / "single.mrc", np.ones((8, 6), dtype=np.float32))
    record = MrcLoader().load(path)
    assert record.dim_order == "YX", ("unexpected dim order", record.dim_order)
    assert record.data.shape == (8, 6), ("unexpected shape", record.data.shape)


def test_load_big_endian_file_converts_to_native(tmp_path: Path) -> None:
    data = np.arange(3 * 4 * 5, dtype=">u2").reshape(3, 4, 5)
    path = write_mrc(tmp_path / "big_endian.st", data, stack=True)
    record = MrcLoader().load(path)
    assert record.data.dtype.isnative, ("dtype must be native", record.data.dtype)
    np.testing.assert_array_equal(record.data.compute(), data.astype(np.uint16))


def test_loaded_record_data_is_picklable(tilt_series_path: Path) -> None:
    record = MrcLoader().load(tilt_series_path)
    restored = pickle.loads(pickle.dumps(record.data))
    np.testing.assert_array_equal(restored.compute(), record.data.compute())


def test_complex_mode_is_skipped(tmp_path: Path) -> None:
    path = write_mrc(tmp_path / "fft.mrc", np.ones((2, 4, 4), dtype=np.complex64))
    with pytest.raises(SkipFile):
        MrcLoader().read_header(path)
