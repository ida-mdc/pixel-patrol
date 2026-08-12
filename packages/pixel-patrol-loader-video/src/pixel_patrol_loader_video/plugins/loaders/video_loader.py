import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import av
import dask
import dask.array as da
import numpy as np

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.loader_schema import (
    RASTER_IMAGE_LOADER_SCHEMA,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
)
from pixel_patrol_base.core.record import record_from, Record

logger = logging.getLogger(__name__)

# Default number of frames per dask chunk for lazy loading.
DEFAULT_CHUNK_SIZE = 32

# Target maximum bytes per chunk (uncompressed). Chunk size is scaled down
# automatically for high-resolution video so that one chunk never exceeds
# this amount. Default: 256 MB.
#   4K RGB  (3840×2160×3 = ~25 MB/frame) → chunk_size = 10
#   1080p RGB (1920×1080×3 = ~6 MB/frame) → chunk_size = 32 (capped at default)
#   480p RGB  (854×480×3  = ~1 MB/frame)  → chunk_size = 32 (capped at default)
TARGET_CHUNK_BYTES = 256 * 1024 * 1024


def _resolve_chunk_size(meta: Dict[str, Any], max_chunk_size: int) -> int:
    """
    Return the number of frames per chunk, capped so that one chunk stays
    under TARGET_CHUNK_BYTES of uncompressed data.
    """
    bytes_per_frame = meta["Y_size"] * meta["X_size"] * meta["n_channels"]
    size_limited = max(1, TARGET_CHUNK_BYTES // bytes_per_frame)
    chunk_size = min(max_chunk_size, size_limited)
    if chunk_size < max_chunk_size:
        logger.debug(
            f"Reducing chunk_size from {max_chunk_size} to {chunk_size} "
            f"for {meta['X_size']}x{meta['Y_size']} video "
            f"({bytes_per_frame / 1024**2:.1f} MB/frame)."
        )
    return chunk_size


# Formats where n_frames is unreliable or the file is typically small enough
# that eager (all-at-once) loading is safe and simpler.
#   gif  – palette-based, frame count often missing/wrong in container header
#   apng – animated PNG, no container-level frame count
#   webp – animated WebP, same issue
EAGER_FORMATS: Set[str] = {"gif", "apng", "webp"}

# Formats where PyAV seeking is unreliable; these fall back to a purely
# sequential decode even in the chunked path (no seek optimisation).
#   mts / m2ts – MPEG-2 Transport Stream; PTS discontinuities make seeking fragile
#   wmv        – Windows Media Video; seek support is codec-dependent
NO_SEEK_FORMATS: Set[str] = {"mts", "m2ts", "wmv"}


def _pix_fmt_to_channels(pix_fmt: str) -> int:
    """Return 1 for grayscale pixel formats, 3 for everything else."""
    return 1 if any(t in pix_fmt for t in ("gray", "mono", "pal8")) else 3


def _count_packets(source: str) -> int:
    """Count video packets without decoding — fast even for large files."""
    with av.open(source) as container:
        stream = container.streams.video[0]
        return sum(1 for pkt in container.demux(stream) if pkt.pts is not None)


def _probe_video(source: str) -> Dict[str, Any]:
    """Open the file and extract video-stream metadata without decoding frames."""
    with av.open(source) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in: {source}")

        stream = container.streams.video[0]
        ctx = stream.codec_context

        fps: float = float(stream.average_rate) if stream.average_rate else 0.0
        duration_s: Optional[float] = (
            float(container.duration) / av.time_base if container.duration else None
        )

        # n_frames: prefer the stored value; fall back to duration * fps.
        # For eager formats (gif, apng, webp) this may still be 0 – that is
        # handled at load time by counting frames during the eager decode.
        n_frames: int = stream.frames or 0
        if n_frames == 0 and duration_s and fps:
            estimated = math.ceil(duration_s * fps)
            actual = _count_packets(source)
            if abs(actual - estimated) > 1:
                raise ValueError(
                    f"VFR (variable frame rate) video not yet supported: "
                    f"container stores no frame count; estimated {estimated} frames "
                    f"from duration×fps but found {actual} encoded frames."
                )
            n_frames = actual

        width: int = stream.width
        height: int = stream.height

        pix_fmt = str(stream.pix_fmt) if stream.pix_fmt else "yuv420p"
        n_channels = _pix_fmt_to_channels(pix_fmt)
        is_color = n_channels == 3

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
            "dtype": "uint8",
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

        return meta


def _av_fmt(n_channels: int) -> str:
    # rgb24: 8 bits per channel, 24 bits per pixel; gray: 8-bit single channel
    return "rgb24" if n_channels == 3 else "gray"


def _read_all_frames(source: str, n_channels: int) -> np.ndarray:
    """
    Decode every frame from *source* into a single (T, Y, X[, C]) array.
    Used for eager formats (gif, apng, webp) where frame count is unreliable.
    The returned array's T dimension reflects the true frame count.
    """
    fmt = _av_fmt(n_channels)
    frames = []
    with av.open(source) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format=fmt))
    if not frames:
        raise RuntimeError(f"No frames decoded from '{source}'")
    return np.stack(frames, axis=0)


def _read_frame_chunk(
    source: str,
    start: int,
    end: int,
    n_channels: int,
    use_seek: bool,
) -> np.ndarray:
    """
    Decode frames [start, end) from *source* and return a (chunk, Y, X[, C]) array.

    If *use_seek* is True the container is seeked to approximately the right
    position before decoding, avoiding a full sequential scan for every chunk.
    Seeking is skipped for formats where it is unreliable (NO_SEEK_FORMATS).

    Frame index tracking uses a manual counter — PyAV's VideoFrame does not
    expose a reliable per-frame index attribute across all versions/formats.
    After seeking we skip frames until we reach *start* using PTS comparison;
    on the no-seek path we simply count from 0.
    """
    fmt = _av_fmt(n_channels)
    frames: List[np.ndarray] = []

    with av.open(source) as container:
        stream = container.streams.video[0]
        seek_succeeded = False

        if use_seek and start > 0:
            # Convert frame index to a PTS estimate and seek a little early
            # (0.5 s) to ensure we land before the target frame, then skip
            # forward — this handles keyframe alignment without relying on
            # frame.index which is not available in all PyAV versions.
            fps = float(stream.average_rate) if stream.average_rate else 25.0
            target_sec = max(0.0, (start / fps) - 0.5)
            target_pts = int(target_sec / stream.time_base)
            try:
                container.seek(target_pts, stream=stream)
                seek_succeeded = True
            except Exception:
                # Seek failed — fall back to sequential decode from the start.
                try:
                    container.seek(0)
                except Exception:
                    pass

        # Estimate the PTS of the target start frame so we can skip pre-seek
        # frames without a counter (more robust than counting after an imprecise seek).
        if seek_succeeded and stream.average_rate:
            fps = float(stream.average_rate)
            start_pts_estimate = int((start / fps) / stream.time_base) if stream.time_base else 0
        else:
            start_pts_estimate = None

        sequential_idx = 0  # counts frames decoded since container open / seek

        for frame in container.decode(video=0):
            # Determine the absolute frame index for this frame.
            # After a seek, use PTS to figure out where we are; on the plain
            # sequential path just count from 0.
            if seek_succeeded and start_pts_estimate is not None and frame.pts is not None:
                # Approximate absolute frame index from PTS
                fps = float(stream.average_rate) if stream.average_rate else 25.0
                abs_idx = int(frame.pts * float(stream.time_base) * fps + 0.5)
            else:
                abs_idx = sequential_idx

            sequential_idx += 1

            if abs_idx < start:
                continue
            if abs_idx >= end:
                break

            frames.append(frame.to_ndarray(format=fmt))

            if len(frames) == (end - start):
                break

    if not frames:
        # Return an empty array so dask can still assemble the graph.
        # This can happen at the tail of a video when n_frames was slightly
        # over-estimated from duration * fps.
        return np.empty((0,), dtype=np.uint8)

    return np.stack(frames, axis=0)


def _build_chunked_array(
    source: str,
    meta: Dict[str, Any],
    chunk_size: int,
    use_seek: bool,
) -> da.Array:
    """
    Build a lazy dask array by splitting the video into chunks of *chunk_size*
    frames, each backed by a separate delayed call to *_read_frame_chunk*.
    """
    n_frames = meta["n_frames"]
    n_channels = meta["n_channels"]
    height = meta["Y_size"]
    width = meta["X_size"]

    chunk_arrays = []
    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        actual_chunk = end - start

        if n_channels == 1:
            chunk_shape = (actual_chunk, height, width)
        else:
            chunk_shape = (actual_chunk, height, width, n_channels)

        delayed_chunk = dask.delayed(_read_frame_chunk)(
            source, start, end, n_channels, use_seek
        )
        chunk_arrays.append(
            da.from_delayed(delayed_chunk, shape=chunk_shape, dtype=np.uint8)
        )

    return da.concatenate(chunk_arrays, axis=0)


def _build_eager_array(source: str, meta: Dict[str, Any]) -> da.Array:
    """
    Build a lazy dask array backed by a single delayed call that decodes all
    frames at once. Used for formats where per-chunk loading is unreliable
    (gif, apng, webp). The shape uses meta["shape"] as a hint; if the true
    frame count differs at compute time the array will still be correct because
    dask will reshape accordingly — but we log a warning if n_frames was 0
    (unknown) so the caller knows the T dimension is only an estimate.
    """
    n_channels = meta["n_channels"]
    shape = tuple(meta["shape"].tolist())

    if meta["n_frames"] == 0:
        logger.warning(
            f"n_frames could not be determined for '{source}'; "
            "the T dimension will be resolved at compute time. "
            "This is normal for GIF/APNG/WebP."
        )

    delayed_frames = dask.delayed(_read_all_frames)(source, n_channels)
    return da.from_delayed(delayed_frames, shape=shape, dtype=np.uint8)


class VideoLoader:
    """
    Loader that reads video files (mp4, avi, mov, mkv, …) via PyAV (FFmpeg).

    Returns a Record backed by a lazy dask array of shape:
      - (T, Y, X, C)  for colour video  – dim_order = "TYXC"
      - (T, Y, X)     for grayscale     – dim_order = "TYX"

    Loading strategy
    ----------------
    Most formats are loaded lazily in chunks of DEFAULT_CHUNK_SIZE frames so
    that only the frames actually needed are decoded into memory.  Seeking is
    used where reliable to avoid O(N²) decode cost when chunks are accessed
    non-sequentially.

    Formats in EAGER_FORMATS (gif, apng, webp) fall back to a single eager
    decode because their container headers do not store reliable frame counts,
    making chunk boundary calculation impossible.  These formats are typically
    small (palette-based or short loops) so the memory cost is acceptable.

    Formats in NO_SEEK_FORMATS (mts, m2ts, wmv) use chunked loading but
    without the seek optimisation because seeking in those containers is
    unreliable.
    """

    NAME = "video"
    DESCRIPTION = "Loads video files (mp4, avi, mov, mkv, gif, …) via PyAV (FFmpeg), returning frames as a lazy dask array."

    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = {
        "n_frames": "Total number of frames in the video.",
        "n_channels": "Number of colour channels per frame.",
        "fps": "Frames per second of the video stream.",
        "codec": "Video codec name (e.g. 'h264', 'vp9').",
        "duration_seconds": "Duration of the video in seconds.",
    }
    OUTPUT_SCHEMA_PATTERN_DESCRIPTIONS: Dict[str, str] = {
        r"^[A-Za-z]_size$": "Size of the named axis in pixels (e.g. X_size, Y_size, T_size).",
    }

    SUPPORTED_EXTENSIONS: Set[str] = {
        # Common web / consumer formats
        "mp4",
        "m4v",
        "mov",
        "avi",
        "mkv",
        "webm",
        "flv",
        "wmv",
        # Broadcast / capture formats
        "mts",
        "m2ts",
        "ts",
        # Animated image formats (eager path)
        "gif",
        "apng",
        "webp",
    }

    # Adopt the common raster-image-loader schema (dim_order, shape, dtype, ...)
    # and add only the fields specific to video streams.
    OUTPUT_SCHEMA: Dict[str, Any] = {
        **RASTER_IMAGE_LOADER_SCHEMA,
        "n_frames": int,
        "n_channels": int,
        "fps": float,
        "codec": str,  # video codec name (e.g. "h264", "vp9")
        "duration_seconds": float,
    }

    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]] = [
        *RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
        (r"^[A-Za-z]_size$", int),
    ]

    FOLDER_EXTENSIONS: Set[str] = set()

    # Video files compress heavily; on-disk size is a poor proxy for uncompressed
    # size. Listing them here forces read_header to always be called so that
    # processing.py uses the true uncompressed size for routing.
    # Eager formats (gif, apng, webp) are omitted -- they are typically small and
    # are decoded all-at-once anyway, so forced header reading doesn't help them.
    CONTAINER_EXTENSIONS: Set[str] = {
        "mp4", "m4v", "mov", "avi", "mkv", "webm", "flv", "wmv",
        "mts", "m2ts", "ts",
    }

    # Tuneable chunk size (frames per dask chunk).
    chunk_size: int = DEFAULT_CHUNK_SIZE

    def is_folder_supported(self, _path: Path) -> bool:
        return False

    def read_header(self, file_path: Path) -> FileInfo:
        """Read container/stream metadata without decoding any frames."""
        meta = _probe_video(str(file_path))
        shape = tuple(int(x) for x in meta["shape"])
        dim_order = meta["dim_order"]
        dtype = np.dtype(meta["dtype"])
        # Colour channels are packed per-frame and decoded together; defer them in
        # memory-chunk splitting so temporal dims absorb the reduction first.
        deferred = "".join(d for d in dim_order if d not in "TYX") or None
        return FileInfo(shape=shape, dtype=dtype, dim_order=dim_order, n_images=1,
                        deferred_dims=deferred)

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        """Yield the single (child_id, record) pair for a video file.

        Video files always have exactly one image (n_images == 1), so this is
        only ever called with start=0, stop=1.
        """
        for _ in range(start, stop):
            yield "0", self.load(str(file_path))

    def load(self, source: str):
        try:
            meta = _probe_video(source)
        except Exception as exc:
            raise RuntimeError(f"VideoLoader: cannot probe '{source}': {exc}") from exc

        ext = Path(source).suffix.lstrip(".").lower()

        if ext in EAGER_FORMATS:
            logger.debug(f"VideoLoader: using eager path for '{source}' (format: {ext})")
            arr = _build_eager_array(source, meta)
        else:
            use_seek = ext not in NO_SEEK_FORMATS
            chunk_size = _resolve_chunk_size(meta, self.chunk_size)
            logger.debug(
                f"VideoLoader: using chunked path for '{source}' "
                f"(chunk_size={chunk_size}, seek={use_seek})"
            )
            arr = _build_chunked_array(source, meta, chunk_size, use_seek)

        return record_from(arr, meta, kind="intensity")