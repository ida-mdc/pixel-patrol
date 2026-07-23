"""Unit tests for BioIoLoader - fast, no pipeline."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from pixel_patrol_loader_bio.plugins.loaders.bioio_loader import BioIoLoader


@pytest.fixture
def loader():
    return BioIoLoader()


def test_name_and_extensions(loader):
    assert loader.NAME == "bioio"
    assert "tif" in loader.SUPPORTED_EXTENSIONS
    assert "czi" in loader.SUPPORTED_EXTENSIONS
    assert "png" in loader.SUPPORTED_EXTENSIONS


def test_load_png(tmp_path: Path, loader):
    arr = np.full((16, 16), 128, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(tmp_path / "gray.png")

    rec = loader.load(tmp_path / "gray.png")
    assert "Y" in rec.dim_order
    assert "X" in rec.dim_order
    assert rec.data.compute().dtype == np.uint8


def test_load_tif_cyx(tmp_path: Path, loader):
    arr = np.zeros((3, 8, 8), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "cyx.tif", arr, imagej=True, metadata={"axes": "CYX"})

    rec = loader.load(tmp_path / "cyx.tif")
    assert rec.dim_order == "CYX"
    assert rec.meta["size_C"] == 3


def test_read_header(tmp_path: Path, loader):
    arr = np.zeros((2, 8, 8), dtype=np.uint16)
    tifffile.imwrite(tmp_path / "cyx.tif", arr, imagej=True, metadata={"axes": "CYX"})

    info = loader.read_header(tmp_path / "cyx.tif")
    assert info.shape == (2, 8, 8)
    assert info.n_images >= 1
    assert info.dim_order == "CYX"


def test_load_unsupported_raises(tmp_path: Path, loader):
    (tmp_path / "file.xyz").write_bytes(b"not an image")
    with pytest.raises(Exception):
        loader.load(tmp_path / "file.xyz")


def test_load_range_one_bad_scene_does_not_lose_its_siblings(monkeypatch, loader):
    class _FakeImage:
        scenes = ["s0", "s1", "s2"]

        def set_scene(self, scene):
            if scene == "s1":
                raise RuntimeError("simulated decode failure")
            self._current = scene

    monkeypatch.setattr(loader, "_build_record", lambda img: img._current)
    monkeypatch.setattr(
        "pixel_patrol_loader_bio.plugins.loaders.bioio_loader._load_bioio_image",
        lambda file_path: _FakeImage(),
    )

    results = dict(loader.load_range(Path("unused.lif"), 0, 3))
    assert set(results.keys()) == {"s0", "s1", "s2"}
    assert results["s0"] == "s0"
    assert results["s1"] is None
    assert results["s2"] == "s2"
