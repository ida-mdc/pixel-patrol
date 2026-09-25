import json
import pickle
from pathlib import Path

import dask.array as da
import h5py
import numpy as np
import pytest

from pixel_patrol_loader_bio.plugins.loaders.h5_loader import H5Loader


@pytest.fixture
def loader():
    return H5Loader()


@pytest.fixture
def plain_h5(tmp_path: Path) -> Path:
    """Two image datasets at different depths plus one non-image dataset."""
    path = tmp_path / "plain.h5"
    with h5py.File(path, "w") as f:
        dset = f.create_dataset("raw", data=np.arange(2 * 4 * 6, dtype="uint16").reshape(2, 4, 6))
        dset.attrs["element_size_um"] = np.array([0.5, 0.25, 0.25])
        f.create_group("nested").create_dataset(
            "labels", data=np.zeros((4, 6), dtype="uint8"), chunks=(2, 3)
        )
        f.create_dataset("timestamps", data=np.arange(5, dtype="float64"))  # 1-D: not an image
        f.attrs["experiment"] = "test-run"
    return path


@pytest.fixture
def axistags_h5(tmp_path: Path) -> Path:
    """An ilastik/vigra-style file carrying axis labels as JSON."""
    path = tmp_path / "axistags.h5"
    axistags = json.dumps({"axes": [{"key": "z"}, {"key": "y"}, {"key": "x"}]})
    with h5py.File(path, "w") as f:
        dset = f.create_dataset("exported_data", data=np.zeros((3, 4, 5), dtype="uint8"))
        dset.attrs["axistags"] = axistags
    return path


@pytest.fixture
def bdv_h5(tmp_path: Path) -> Path:
    """A minimal BDV file: two timepoints x one setup, with a mipmap level and sidecar XML."""
    path = tmp_path / "dataset.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("s00/resolutions", data=np.array([[1, 1, 1], [2, 2, 1]], dtype="float64"))
        f.create_dataset("s00/subdivisions", data=np.array([[16, 16, 8]], dtype="int32"))
        for timepoint in ("t00000", "t00001"):
            f.create_dataset(f"{timepoint}/s00/0/cells", data=np.ones((8, 16, 16), dtype="uint16"))
            f.create_dataset(f"{timepoint}/s00/1/cells", data=np.ones((8, 8, 8), dtype="uint16"))
    path.with_suffix(".xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
        <SpimData version="0.2">
          <SequenceDescription>
            <ViewSetups>
              <ViewSetup>
                <id>0</id>
                <name>GFP</name>
                <size>16 16 8</size>
                <voxelSize>
                  <unit>micron</unit>
                  <size>0.25 0.25 1.5</size>
                </voxelSize>
              </ViewSetup>
            </ViewSetups>
          </SequenceDescription>
        </SpimData>
        """
    )
    return path


def _imaris_text(value: str) -> np.ndarray:
    """Encode text the way many Imaris files store HDF5 attributes."""
    return np.frombuffer(value.encode("utf-8") + b"\x00", dtype="S1")


@pytest.fixture
def imaris_h5(tmp_path: Path) -> Path:
    """Minimal two-timepoint, two-channel, pyramidal Imaris 5.5 file."""
    path = tmp_path / "modern.ims"
    with h5py.File(path, "w") as f:
        f.attrs["ImarisDataSet"] = _imaris_text("ImarisDataSet")
        f.attrs["ImarisVersion"] = _imaris_text("5.5.0")

        image = f.create_group("DataSetInfo/Image")
        image.attrs["Unit"] = _imaris_text("um")
        for index, (minimum, maximum) in enumerate(((0, 2), (0, 3), (0, 4))):
            image.attrs[f"ExtMin{index}"] = _imaris_text(str(minimum))
            image.attrs[f"ExtMax{index}"] = _imaris_text(str(maximum))
        for channel, (name, em, ex) in enumerate([("DAPI", "460 nm", "405 nm"), ("GFP", "525 nm", "488 nm")]):
            grp = f.create_group(f"DataSetInfo/Channel {channel}")
            grp.attrs["Name"] = _imaris_text(name)
            grp.attrs["LSMEmissionWavelength"] = _imaris_text(em)
            grp.attrs["LSMExcitationWavelength"] = _imaris_text(ex)

        for timepoint in range(2):
            for channel in range(2):
                value = timepoint * 10 + channel
                f.create_dataset(
                    f"DataSet/ResolutionLevel 0/TimePoint {timepoint}/Channel {channel}/Data",
                    data=np.full((2, 3, 4), value, dtype=np.uint16),
                    chunks=(1, 3, 4),
                )
                f.create_dataset(
                    f"DataSet/ResolutionLevel 1/TimePoint {timepoint}/Channel {channel}/Data",
                    data=np.full((1, 2, 2), value, dtype=np.uint16),
                )
        f.create_dataset("Thumbnail/Data", data=np.zeros((8, 8), dtype=np.uint8))
    return path


# ── plain HDF5 ───────────────────────────────────────────────────────────────

def test_read_header_counts_image_datasets_only(plain_h5: Path, loader):
    info = loader.read_header(plain_h5)
    assert info.n_images == 2  # 'raw' and 'nested/labels'; the 1-D dataset is skipped
    assert info.shape == (2, 4, 6)  # largest dataset is safest for task sizing
    assert info.dtype == np.dtype("uint16")
    assert info.dim_order == "AYX"


def test_load_returns_first_dataset_lazily(plain_h5: Path, loader):
    record = loader.load(plain_h5)
    assert isinstance(record.data, da.Array)
    assert record.dim_order == "YX"
    assert record.meta["h5_dataset_path"] == "/nested/labels"
    assert record.meta["chunks"] == (2, 3)
    assert np.array_equal(record.data.compute(), np.zeros((4, 6), dtype="uint8"))


def test_load_range_yields_every_dataset(plain_h5: Path, loader):
    items = list(loader.load_range(plain_h5, 0, 2))
    child_ids = [child_id for child_id, _ in items]
    assert child_ids == ["nested/labels", "raw"]

    _, raw = items[1]
    assert raw.dim_order == "AYX"  # no axis attribute: inferred, trailing YX
    assert tuple(raw.data.shape) == (2, 4, 6)
    assert raw.meta["size_A"] == 2
    assert raw.meta["num_pixels"] == 48
    assert np.array_equal(raw.data.compute(), np.arange(48, dtype="uint16").reshape(2, 4, 6))


@pytest.mark.parametrize("start, end, expected", [(1, 2, ["raw"]), (2, 5, [])])
def test_load_range_honours_slice_bounds(plain_h5: Path, loader, start, end, expected):
    assert [cid for cid, _ in loader.load_range(plain_h5, start, end)] == expected


def test_root_attributes_are_merged_into_meta(plain_h5: Path, loader):
    record = loader.load(plain_h5)
    assert record.meta["h5_attributes"]["experiment"] == "test-run"


def test_element_size_um_sets_pixel_sizes(plain_h5: Path, loader):
    items = list(loader.load_range(plain_h5, 0, 2))
    _, raw = items[1]
    assert raw.meta["pixel_size_Z"] == pytest.approx(0.5)
    assert raw.meta["pixel_size_Y"] == pytest.approx(0.25)
    assert raw.meta["pixel_size_X"] == pytest.approx(0.25)
    assert raw.meta["pixel_size_unit"] == "um"


def test_axistags_attribute_sets_dim_order(axistags_h5: Path, loader):
    record = loader.load(axistags_h5)
    assert record.dim_order == "ZYX"
    assert record.meta["size_Z"] == 3
    assert record.meta["size_X"] == 5


def test_dataset_proxy_survives_pickling(plain_h5: Path, loader):
    record = loader.load(plain_h5)
    revived = pickle.loads(pickle.dumps(record.data))
    assert np.array_equal(revived.compute(), record.data.compute())


def test_no_image_dataset_raises(tmp_path: Path, loader):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("scalars", data=np.arange(3))
    with pytest.raises(RuntimeError, match="no image-like dataset"):
        loader.read_header(path)


# ── BigDataViewer ────────────────────────────────────────────────────────────

def test_bdv_read_header_skips_mipmaps_and_helper_datasets(bdv_h5: Path, loader):
    info = loader.read_header(bdv_h5)
    assert info.n_images == 2  # two timepoints, level 0 only
    assert info.shape == (8, 16, 16)
    assert info.dim_order == "ZYX"


def test_bdv_load_range_child_ids_and_voxel_size(bdv_h5: Path, loader):
    items = list(loader.load_range(bdv_h5, 0, 2))
    assert [child_id for child_id, _ in items] == ["t00000/s00", "t00001/s00"]

    _, record = items[0]
    assert record.dim_order == "ZYX"
    assert record.meta["bdv_timepoint"] == "t00000"
    assert record.meta["bdv_setup"] == "s00"
    assert record.meta["bdv_setup_name"] == "GFP"
    assert record.meta["pixel_size_X"] == 0.25
    assert record.meta["pixel_size_Z"] == 1.5
    assert record.meta["pixel_size_unit"] == "micron"


def test_bdv_without_xml_still_loads(bdv_h5: Path, loader):
    bdv_h5.with_suffix(".xml").unlink()
    record = loader.load(bdv_h5)
    assert record.dim_order == "ZYX"
    assert "pixel_size_X" not in record.meta
    assert record.meta["bdv_setup"] == "s00"


# ── Imaris IMS ──────────────────────────────────────────────────────────────

def test_imaris_header_is_one_tczyx_image(imaris_h5: Path, loader):
    info = loader.read_header(imaris_h5)
    assert info.n_images == 1
    assert info.shape == (2, 2, 2, 3, 4)
    assert info.dtype == np.dtype("uint16")
    assert info.dim_order == "TCZYX"


def test_imaris_load_combines_time_and_channels_lazily(imaris_h5: Path, loader):
    record = loader.load(imaris_h5)
    assert isinstance(record.data, da.Array)
    assert record.dim_order == "TCZYX"
    assert record.data.shape == (2, 2, 2, 3, 4)
    assert np.all(record.data[0, 0].compute() == 0)
    assert np.all(record.data[1, 1].compute() == 11)


def test_imaris_uses_only_full_resolution_and_extracts_metadata(imaris_h5: Path, loader):
    record = loader.load(imaris_h5)
    assert record.meta["h5_dataset_path"] == "/DataSet/ResolutionLevel 0"
    assert record.meta["channel_names"] == ["DAPI", "GFP"]
    assert record.meta["imaris_format_version"] == "5.5.0"
    assert record.meta["pixel_size_unit"] == "um"
    assert record.meta["pixel_size_X"] == pytest.approx(0.5)
    assert record.meta["pixel_size_Y"] == pytest.approx(1.0)
    assert record.meta["pixel_size_Z"] == pytest.approx(2.0)
    assert record.meta["emission_wavelengths"] == ["460 nm", "525 nm"]
    assert record.meta["excitation_wavelengths"] == ["405 nm", "488 nm"]


def test_legacy_non_hdf5_ims_has_clear_error(tmp_path: Path, loader):
    path = tmp_path / "legacy.ims"
    path.write_bytes(b"not an HDF5 file")
    with pytest.raises(RuntimeError, match="legacy Imaris 2.7/3"):
        loader.read_header(path)
