"""Columns derived from the compression algorithm a loader reports and from the file size."""

import re
from typing import Optional

UNCOMPRESSED = "uncompressed"
LOSSLESS = "lossless"
LOSSY = "lossy"
EITHER = "lossy or lossless"
UNKNOWN = "unknown"

_RAW_STAGE = re.compile(r"^(none|bytes)\b")
_LOSSY_STAGE = re.compile(r"\b(lossy|ojpeg|zfp\w*|fixedscaleoffset|quantize|bitround)\b|^jpeg$")
_EITHER_STAGE = re.compile(r"xr|2000|jpegxl|jpeg xl|webp|lerc|jpeg.?ls|jxr|avif|heic|camera|system")
_LOSSLESS_STAGE = re.compile(
    r"\b(lossless|deflate|zlib|gzip|lzw|packbits|zstd\d*|lzma|lz4|bz2|bzip2|snappy|brotli|blosc|shuffle|bitshuffle"
    r"|delta|predictor|crc32c|transpose|sharding\w*)\b")

# A chain of stages (Zarr filters, mixed CZI sub-blocks) is as lossy as its worst stage.
_WORST_FIRST = (LOSSY, EITHER, UNKNOWN, LOSSLESS, UNCOMPRESSED)


def _stage_type(stage: str) -> str:
    label = stage.strip().lower()
    if _RAW_STAGE.search(label):
        return UNCOMPRESSED
    if _LOSSY_STAGE.search(label):
        return LOSSY
    if _EITHER_STAGE.search(label):
        return EITHER
    return LOSSLESS if _LOSSLESS_STAGE.search(label) else UNKNOWN


def compression_type(algorithm: Optional[str]) -> Optional[str]:
    """Whether a compression algorithm stores the pixels raw, losslessly or lossily; None when not recorded."""
    if algorithm is None:
        return None
    types = {_stage_type(stage) for stage in str(algorithm).split("+")}
    return next(t for t in _WORST_FIRST if t in types)


def compression_factor(uncompressed_bytes: int, size_bytes: int) -> Optional[float]:
    """Uncompressed size over size on disk. A file larger than its pixels (metadata, padding) counts as 1."""
    if uncompressed_bytes <= 0 or size_bytes <= 0:
        return None
    return uncompressed_bytes / min(size_bytes, uncompressed_bytes)
