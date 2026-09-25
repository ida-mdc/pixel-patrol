"""Unit tests for BioIoLoader - fast, no pipeline."""

import itertools
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


@pytest.fixture
def create_czi(monkeypatch):
    from pylibCZIrw import czi as pyczi

    # pylibCZIrw only passes T/Z/C/S on to libCZI; let V and I through too.
    monkeypatch.setattr(pyczi.CziWriter, "_create_plane",
                        staticmethod(lambda plane, scene: {"T": 0, "Z": 0, "C": 0, **(plane or {}), "S": scene}))
    return pyczi.create_czi


def test_load_czi_with_views_and_illuminations(tmp_path: Path, loader, create_czi):
    with create_czi(str(tmp_path / "views.czi")) as czi:
        for v, i, z in itertools.product(range(2), range(2), range(3)):
            czi.write(np.full((6, 8, 1), 100 * v + 10 * i + z, dtype=np.uint16), plane={"V": v, "I": i, "Z": z})

    rec = loader.load(tmp_path / "views.czi")

    assert rec.dim_order == "VITCZYX"
    assert rec.data[1, 0, 0, 0, 2, 0, 0].compute() == 102


def test_load_czi_mosaic(tmp_path: Path, loader, create_czi):
    with create_czi(str(tmp_path / "mosaic.czi")) as czi:
        for m, location in enumerate([(0, 0), (8, 0), (0, 6), (8, 6)]):
            czi.write(np.full((6, 8, 1), m + 1, dtype=np.uint16), location=location)

    rec = loader.load(tmp_path / "mosaic.czi")

    assert rec.dim_order == "TCZYX"
    assert rec.data.shape == (1, 1, 1, 12, 16)
    assert set(np.unique(rec.data.compute())) == {1, 2, 3, 4}


def test_load_czi_line_scan(tmp_path: Path, loader, create_czi):
    with create_czi(str(tmp_path / "line.czi")) as czi:
        for t in range(4):
            czi.write(np.full((1, 16, 1), t, dtype=np.uint8), plane={"T": t})

    rec = loader.load(tmp_path / "line.czi")

    assert rec.data.shape == (4, 1, 1, 1, 16)
    assert rec.meta["size_Y"] == 1
