"""Tests for NiftiLoader."""

import json
from pathlib import Path

import dask.array as da
import nibabel as nib
import numpy as np
import pytest

from pixel_patrol_base.core.contracts import SkipFile
from pixel_patrol_loader_medical.plugins.loaders.nifti_loader import NiftiLoader


@pytest.fixture
def loader():
    return NiftiLoader()


def _write_nifti(path: Path, arr: np.ndarray, zooms=None) -> Path:
    affine = np.eye(4)
    img = nib.Nifti1Image(arr, affine)
    if zooms:
        img.header.set_zooms(zooms)
    nib.save(img, str(path))
    return path


def test_name_and_extensions(loader):
    assert loader.NAME == "nifti"
    assert "nii" in loader.SUPPORTED_EXTENSIONS
    assert "gz" in loader.SUPPORTED_EXTENSIONS
    assert not loader.is_folder_supported(Path("."))


def test_read_header_3d(tmp_path, loader):
    path = tmp_path / "vol.nii"
    _write_nifti(path, np.zeros((10, 20, 30), dtype=np.int16))
    info = loader.read_header(path)
    assert info.shape == (10, 20, 30)
    assert info.dim_order == "XYZ"
    assert info.n_images == 1
    assert info.dtype == np.dtype("int16")


def test_read_header_4d(tmp_path, loader):
    path = tmp_path / "func.nii"
    _write_nifti(path, np.zeros((64, 64, 40, 120), dtype=np.float32))
    info = loader.read_header(path)
    assert info.shape == (64, 64, 40, 120)
    assert info.dim_order == "XYZT"
    assert info.n_images == 1


def test_read_header_non_nifti_gz_skips(tmp_path, loader):
    path = tmp_path / "data.gz"
    path.write_bytes(b"not a nifti file at all")
    with pytest.raises(SkipFile):
        loader.read_header(path)


def test_load_3d_roundtrip(tmp_path, loader):
    rng = np.random.default_rng(0)
    arr = rng.integers(-1000, 3000, (10, 20, 30), dtype=np.int16)
    path = tmp_path / "vol.nii"
    _write_nifti(path, arr)
    rec = loader.load(path)
    assert rec.dim_order == "XYZ"
    assert tuple(rec.data.shape) == (10, 20, 30)
    assert "spatial-2d" in rec.capabilities
    assert "spatial-3d" in rec.capabilities
    assert isinstance(rec.data, da.Array)
    np.testing.assert_array_equal(rec.data.compute(), arr)


def test_load_nii_gz(tmp_path, loader):
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 4096, (8, 8, 8), dtype=np.uint16)
    path = tmp_path / "vol.nii.gz"
    _write_nifti(path, arr)
    rec = loader.load(path)
    assert tuple(rec.data.shape) == (8, 8, 8)
    np.testing.assert_array_equal(rec.data.compute(), arr)


def test_pixel_sizes(tmp_path, loader):
    arr = np.zeros((5, 6, 7), dtype=np.float32)
    path = tmp_path / "sized.nii"
    _write_nifti(path, arr, zooms=(1.5, 1.5, 3.0))
    rec = loader.load(path)
    assert rec.meta["pixel_size_X"] == pytest.approx(1.5)
    assert rec.meta["pixel_size_Y"] == pytest.approx(1.5)
    assert rec.meta["pixel_size_Z"] == pytest.approx(3.0)


def test_pixel_sizes_4d_with_tr(tmp_path, loader):
    arr = np.zeros((64, 64, 30, 10), dtype=np.float32)
    path = tmp_path / "func.nii"
    _write_nifti(path, arr, zooms=(2.0, 2.0, 3.0, 1.5))
    rec = loader.load(path)
    assert rec.meta["pixel_size_T"] == pytest.approx(1.5)


def test_non_nifti_gz_skips(tmp_path, loader):
    path = tmp_path / "data.gz"
    path.write_bytes(b"not a nifti file at all")
    with pytest.raises(SkipFile):
        loader.load(path)


def test_bids_sidecar(tmp_path, loader):
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    path = tmp_path / "sub-01_T1w.nii"
    _write_nifti(path, arr)
    (tmp_path / "sub-01_T1w.json").write_text(
        json.dumps({"RepetitionTime": 2.0, "EchoTime": 0.03, "nested": {"key": "val"}})
    )
    rec = loader.load(path)
    assert rec.meta["RepetitionTime"] == pytest.approx(2.0)
    assert rec.meta["EchoTime"] == pytest.approx(0.03)
    assert "nested" not in rec.meta  # nested objects are excluded


def test_capabilities_4d(tmp_path, loader):
    path = tmp_path / "func.nii"
    _write_nifti(path, np.zeros((4, 5, 6, 20), dtype=np.float32))
    rec = loader.load(path)
    assert "temporal" in rec.capabilities


def test_complex_dtype_raises_skip_file(tmp_path, loader):
    arr = np.zeros((4, 4, 4), dtype=np.complex64)
    path = tmp_path / "mrs.nii"
    _write_nifti(path, arr)
    with pytest.raises(SkipFile, match="complex"):
        loader.read_header(path)
    with pytest.raises(SkipFile, match="complex"):
        loader.load(path)
