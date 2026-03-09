"""Tests for VideoLoader."""
import math
from pathlib import Path

import numpy as np
import pytest

from pixel_patrol_loader_video.plugins.loaders.video_loader import VideoLoader


@pytest.fixture()
def tiny_mp4(tmp_path) -> Path:
    """Create a tiny synthetic mp4 (10 frames, 32×32, RGB) using av."""
    av = pytest.importorskip("av")
    out = tmp_path / "test.mp4"
    with av.open(str(out), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "ultrafast"}
        for i in range(10):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), i * 25, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return out


def test_loader_metadata(tiny_mp4):
    loader = VideoLoader()
    record = loader.load(str(tiny_mp4))
    meta = record.meta

    assert meta["dim_order"] == "TYXC"
    assert meta["n_frames"] == 10
    assert meta["Y_size"] == 32
    assert meta["X_size"] == 32
    assert meta["C_size"] == 3
    assert meta["fps"] == pytest.approx(10.0, abs=1.0)
    assert meta["dtype"] == "uint8"


def test_loader_array_shape(tiny_mp4):
    loader = VideoLoader()
    record = loader.load(str(tiny_mp4))
    arr = record.data.compute()

    assert arr.shape == (10, 32, 32, 3)
    assert arr.dtype == np.uint8


def test_supported_extensions():
    loader = VideoLoader()
    assert "mp4" in loader.SUPPORTED_EXTENSIONS
    assert "avi" in loader.SUPPORTED_EXTENSIONS
    assert "gif" in loader.SUPPORTED_EXTENSIONS


def test_is_folder_supported(tmp_path):
    loader = VideoLoader()
    assert not loader.is_folder_supported(tmp_path)
