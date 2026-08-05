"""Tests for DicomLoader."""

from pathlib import Path
from typing import List

import numpy as np
import pydicom
import pydicom.uid
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence

from pixel_patrol_base.core.contracts import SkipFile
from pixel_patrol_loader_medical.plugins.loaders.dicom_loader import DicomLoader


@pytest.fixture
def loader():
    return DicomLoader()


def _write_dicom_slice(
    path: Path,
    pixel_array: np.ndarray,
    series_uid: str,
    instance_number: int = 1,
    z_pos: float = 0.0,
    modality: str = "MR",
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
) -> None:
    rows, cols = pixel_array.shape
    sop_uid = pydicom.uid.generate_uid()

    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)

    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.Modality = modality
    ds.SeriesDescription = "Test Series"
    ds.InstanceNumber = instance_number
    ds.ImagePositionPatient = [0.0, 0.0, z_pos]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0
    ds.RescaleSlope = rescale_slope
    ds.RescaleIntercept = rescale_intercept

    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pixel_array.astype(np.uint16).tobytes()

    pydicom.dcmwrite(str(path), ds)


def _write_series(folder: Path, n_slices: int, rows: int = 16, cols: int = 16) -> str:
    """Write n_slices into folder with a shared series UID. Returns the series UID."""
    series_uid = pydicom.uid.generate_uid()
    rng = np.random.default_rng(42)
    for i in range(n_slices):
        arr = rng.integers(0, 4096, (rows, cols), dtype=np.uint16)
        _write_dicom_slice(
            folder / f"slice_{i:03d}.dcm",
            arr,
            series_uid=series_uid,
            instance_number=i + 1,
            z_pos=float(i),
        )
    return series_uid


def test_name_and_extensions(loader):
    assert loader.NAME == "dicom"
    assert "dcm" in loader.SUPPORTED_EXTENSIONS
    assert "dicom" in loader.SUPPORTED_EXTENSIONS


def test_read_header_single_file(tmp_path, loader):
    path = tmp_path / "slice.dcm"
    series_uid = pydicom.uid.generate_uid()
    arr = np.zeros((32, 32), dtype=np.uint16)
    _write_dicom_slice(path, arr, series_uid=series_uid)

    info = loader.read_header(path)
    assert info.shape == (32, 32)
    assert info.dim_order == "YX"
    assert info.n_images == 1


def test_load_single_file_roundtrip(tmp_path, loader):
    path = tmp_path / "slice.dcm"
    series_uid = pydicom.uid.generate_uid()
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 4096, (16, 16), dtype=np.uint16)
    _write_dicom_slice(path, arr, series_uid=series_uid)

    rec = loader.load(path)
    assert rec.dim_order == "YX"
    assert tuple(rec.data.shape) == (16, 16)
    assert "spatial-2d" in rec.capabilities
    np.testing.assert_array_equal(rec.data.compute(), arr)


def test_load_single_file_modality_and_description(tmp_path, loader):
    path = tmp_path / "slice.dcm"
    series_uid = pydicom.uid.generate_uid()
    _write_dicom_slice(path, np.zeros((8, 8), dtype=np.uint16), series_uid=series_uid, modality="CT")
    rec = loader.load(path)
    assert rec.meta["Modality"] == "CT"
    assert rec.meta["SeriesDescription"] == "Test Series"


def test_load_single_file_pixel_sizes(tmp_path, loader):
    path = tmp_path / "slice.dcm"
    series_uid = pydicom.uid.generate_uid()
    _write_dicom_slice(path, np.zeros((8, 8), dtype=np.uint16), series_uid=series_uid)
    rec = loader.load(path)
    assert rec.meta["pixel_size_X"] == pytest.approx(1.0)
    assert rec.meta["pixel_size_Y"] == pytest.approx(1.0)
    assert rec.meta["pixel_size_Z"] == pytest.approx(1.0)


def test_is_folder_supported(tmp_path, loader):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert not loader.is_folder_supported(empty_dir)

    dicom_dir = tmp_path / "series"
    dicom_dir.mkdir()
    _write_series(dicom_dir, n_slices=3)
    assert loader.is_folder_supported(dicom_dir)


def test_read_header_folder_single_series(tmp_path, loader):
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    _write_series(series_dir, n_slices=5, rows=16, cols=16)

    info = loader.read_header(series_dir)
    assert info.shape == (5, 16, 16)
    assert info.dim_order == "ZYX"
    assert info.n_images == 1


def test_read_header_folder_multi_series(tmp_path, loader):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    uid1 = pydicom.uid.generate_uid()
    uid2 = pydicom.uid.generate_uid()
    for i in range(3):
        _write_dicom_slice(mixed / f"s1_{i}.dcm", np.zeros((8, 8), dtype=np.uint16), uid1, i + 1, float(i))
        _write_dicom_slice(mixed / f"s2_{i}.dcm", np.zeros((8, 8), dtype=np.uint16), uid2, i + 1, float(i))

    info = loader.read_header(mixed)
    assert info.n_images == 2


def test_load_folder_assembles_volume(tmp_path, loader):
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    rng = np.random.default_rng(7)
    series_uid = pydicom.uid.generate_uid()
    slices = []
    for i in range(4):
        arr = rng.integers(0, 1000, (8, 8), dtype=np.uint16)
        slices.append(arr)
        _write_dicom_slice(series_dir / f"s{i}.dcm", arr, series_uid, instance_number=i + 1, z_pos=float(i))

    rec = loader.load(series_dir)
    assert tuple(rec.data.shape) == (4, 8, 8)
    assert rec.dim_order == "ZYX"
    assert "spatial-3d" in rec.capabilities

    vol = rec.data.compute()
    # Slices should be sorted by z_pos (== instance_number order here)
    for i, sl in enumerate(slices):
        np.testing.assert_array_equal(vol[i], sl)


def test_load_range_multi_series(tmp_path, loader):
    folder = tmp_path / "multi"
    folder.mkdir()
    uid1 = pydicom.uid.generate_uid()
    uid2 = pydicom.uid.generate_uid()
    for i in range(3):
        _write_dicom_slice(folder / f"a{i}.dcm", np.zeros((4, 4), dtype=np.uint16), uid1, i + 1, float(i))
        _write_dicom_slice(folder / f"b{i}.dcm", np.ones((4, 4), dtype=np.uint16), uid2, i + 1, float(i))

    results = dict(loader.load_range(folder, 0, 2))
    assert len(results) == 2
    assert set(results.keys()) == {uid1, uid2}
    for rec in results.values():
        assert tuple(rec.data.shape) == (3, 4, 4)


def test_rescale_applied(tmp_path, loader):
    path = tmp_path / "ct.dcm"
    series_uid = pydicom.uid.generate_uid()
    arr = np.full((4, 4), 1024, dtype=np.uint16)
    _write_dicom_slice(path, arr, series_uid, rescale_slope=1.0, rescale_intercept=-1024.0)

    rec = loader.load(path)
    assert rec.data.dtype == np.dtype("float32")
    np.testing.assert_array_equal(rec.data.compute(), arr.astype(np.float32) - 1024.0)


def test_load_returns_lazy_dask_array(tmp_path, loader):
    import dask.array as da
    path = tmp_path / "slice.dcm"
    _write_dicom_slice(path, np.zeros((8, 8), dtype=np.uint16), pydicom.uid.generate_uid())
    rec = loader.load(path)
    assert isinstance(rec.data, da.Array)


def test_sr_file_raises_skip_file(tmp_path, loader):
    # Write a DICOM file with no Rows/Columns (mimics SR, KO, PR files).
    sop_uid = pydicom.uid.generate_uid()
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.88.22"
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    ds = FileDataset(str(tmp_path / "sr.dcm"), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.22"
    ds.SOPInstanceUID = sop_uid
    ds.Modality = "SR"
    pydicom.dcmwrite(str(tmp_path / "sr.dcm"), ds)

    with pytest.raises(SkipFile):
        loader.read_header(tmp_path / "sr.dcm")


def test_scan_series_nested_directories(tmp_path, loader):
    # Series files nested two levels deep, as in typical DICOM exports.
    nested = tmp_path / "patient" / "study" / "series"
    nested.mkdir(parents=True)
    series_uid = _write_series(nested, n_slices=3, rows=8, cols=8)

    # is_folder_supported finds them recursively
    assert loader.is_folder_supported(tmp_path)

    # read_header on the top-level folder assembles the series
    info = loader.read_header(tmp_path)
    assert info.shape == (3, 8, 8)
    assert info.n_images == 1
