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
        f.create_dataset("raw", data=np.arange(2 * 4 * 6, dtype="uint16").reshape(2, 4, 6))
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


def test_name_and_extensions(loader):
    assert loader.NAME == "h5"
    assert loader.SUPPORTED_EXTENSIONS == {"h5", "hdf5"}
    assert loader.CONTAINER_EXTENSIONS == {"h5", "hdf5"}
    assert loader.FOLDER_EXTENSIONS == set()


def test_is_folder_supported_is_always_false(tmp_path: Path, loader):
    assert loader.is_folder_supported(tmp_path) is False


# ── plain HDF5 ───────────────────────────────────────────────────────────────

def test_read_header_counts_image_datasets_only(plain_h5: Path, loader):
    info = loader.read_header(plain_h5)
    assert info.n_images == 2  # 'raw' and 'nested/labels'; the 1-D dataset is skipped
    assert info.shape == (4, 6)  # first by sorted path: 'nested/labels'
    assert info.dtype == np.dtype("uint8")
    assert info.dim_order == "YX"


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


def test_load_range_honours_slice_bounds(plain_h5: Path, loader):
    assert [cid for cid, _ in loader.load_range(plain_h5, 1, 2)] == ["raw"]
    assert list(loader.load_range(plain_h5, 2, 5)) == []


def test_root_attributes_are_merged_into_meta(plain_h5: Path, loader):
    record = loader.load(plain_h5)
    assert record.meta["h5_attributes"]["experiment"] == "test-run"


def test_axistags_attribute_sets_dim_order(axistags_h5: Path, loader):
    record = loader.load(axistags_h5)
    assert record.dim_order == "ZYX"
    assert record.meta["size_Z"] == 3
    assert record.meta["size_X"] == 5


def test_dataset_proxy_survives_pickling(plain_h5: Path, loader):
    """Records are built in the coordinator and computed in worker processes."""
    record = loader.load(plain_h5)
    revived = pickle.loads(pickle.dumps(record.data))
    assert np.array_equal(revived.compute(), record.data.compute())


def test_no_image_dataset_raises(tmp_path: Path, loader):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("scalars", data=np.arange(3))
    with pytest.raises(RuntimeError, match="no image-like dataset"):
        loader.read_header(path)


def test_missing_file_raises(tmp_path: Path, loader):
    with pytest.raises(Exception):
        loader.load(tmp_path / "nonexistent.h5")


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
