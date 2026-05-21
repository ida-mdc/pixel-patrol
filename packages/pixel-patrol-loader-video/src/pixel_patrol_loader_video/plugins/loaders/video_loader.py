import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Set

import av
import dask
import dask.array as da
import numpy as np

from pixel_patrol_base.core.record import record_from

logger = logging.getLogger(__name__)


def _read_video_frames(source: str, n_channels: int) -> np.ndarray:
    """Decode all frames from a video file into a numpy array."""
    fmt = "rgb24" if n_channels == 3 else "gray"
    frames = []
    with av.open(source) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format=fmt))
    return np.stack(frames, axis=0)


def _probe_video(source: str) -> Dict[str, Any]:
    """Open the file and extract video stream metadata without decoding frames."""
    with av.open(source) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in: {source}")

        stream = container.streams.video[0]
        ctx = stream.codec_context

        fps: float = float(stream.average_rate) if stream.average_rate else 0.0
        duration_s: float | None = (
            float(container.duration) / av.time_base if container.duration else None
        )

        # n_frames: prefer the stored value; fall back to duration * fps
        n_frames: int = stream.frames or 0
        if n_frames == 0 and duration_s and fps:
            n_frames = math.ceil(duration_s * fps)

        width: int = stream.width
        height: int = stream.height

        # Determine colour-ness from pixel format
        pix_fmt = str(stream.pix_fmt) if stream.pix_fmt else "yuv420p"
        is_color = not any(t in pix_fmt for t in ("gray", "mono", "pal8"))
        n_channels = 3 if is_color else 1

        codec_name: str = ctx.name if ctx else "unknown"

        meta: Dict[str, Any] = {
            "fps": fps,
            "codec": codec_name,
            "n_frames": n_frames,
            "duration_seconds": duration_s,
            "n_channels": n_channels,
            "T_size": n_frames,
            "Y_size": height,
            "X_size": width,
        }

        if is_color:
            meta["dim_order"] = "TYXC"
            meta["dim_names"] = ["T", "Y", "X", "C"]
            meta["C_size"] = n_channels
            meta["shape"] = np.array([n_frames, height, width, n_channels])
        else:
            meta["dim_order"] = "TYX"
            meta["dim_names"] = ["T", "Y", "X"]
            meta["shape"] = np.array([n_frames, height, width])

        meta["ndim"] = len(meta["shape"])
        meta["num_pixels"] = int(np.prod(meta["shape"]))
        meta["dtype"] = "uint8"

        return meta


class VideoLoader:
    """
    Loader that reads video files (mp4, avi, mov, mkv, …) via PyAV (FFmpeg).

    Returns a Record backed by a lazy dask array of shape:
      - (T, Y, X, C)  for colour video  – dim_order = "TYXC"
      - (T, Y, X)     for grayscale     – dim_order = "TYX"
    """

    NAME = "video"

    SUPPORTED_EXTENSIONS: Set[str] = {
        "mp4",
        "avi",
        "mov",
        "mkv",
        "webm",
        "mts",
        "m2ts",
        "m4v",
        "flv",
        "wmv",
        "gif",
    }

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "dim_order": str,
        "dim_names": list,
        "n_frames": int,
        "num_pixels": int,
        "shape": list,
        "ndim": int,
        "dtype": str,
        "fps": float,
        "codec": str,
        "duration_seconds": float,
        "n_channels": int,
    }

    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]] = [
        (r"^[A-Za-z]_size$", int),
    ]

    FOLDER_EXTENSIONS: Set[str] = set()

    def is_folder_supported(self, path: Path) -> bool:
        return False

    def load(self, source: str):
        try:
            meta = _probe_video(source)
        except Exception as exc:
            raise RuntimeError(f"VideoLoader: cannot probe '{source}': {exc}") from exc

        shape = tuple(meta["shape"].tolist())
        n_channels = meta["n_channels"]

        delayed_frames = dask.delayed(_read_video_frames)(source, n_channels)
        arr = da.from_delayed(delayed_frames, shape=shape, dtype=np.uint8)

        return record_from(arr, meta, kind="intensity")
