import collections
import logging
import struct
from pathlib import Path
from typing import Dict, Iterable, Optional

import h5py
import nd2
import tifffile
import zarr

logger = logging.getLogger(__name__)

CZI_COMPRESSION = {0: "none", 1: "JPEG", 2: "LZW", 4: "JPEG XR", 5: "camera-specific",
                   6: "system-specific", 101: "zstd0", 102: "zstd1"}
TIFF_NAMES = {"ADOBE_DEFLATE": "deflate", "DEFLATE": "deflate", "NONE": "none"}
FIXED_BY_FORMAT = {".png": "deflate", ".jpg": "JPEG", ".jpeg": "JPEG", ".bmp": "none", ".lif": "none"}


def compression_of(path: Path) -> Optional[str]:
    """The codec a file stores its pixels with, read from the header alone; None when unknown."""
    try:
        if path.is_dir():
            return zarr_compression(path)
        suffix = path.suffix.lower()
        if suffix in FIXED_BY_FORMAT:
            return FIXED_BY_FORMAT[suffix]
        reader = HEADER_READERS.get(suffix)
        return reader(path) if reader else None
    except Exception as exc:
        logger.debug("compression of %s unknown: %s", path, exc)
    return None


def tiff_compression(path: Path) -> str:
    with tifffile.TiffFile(path) as tf:
        page = tf.pages.first
        codec = tifffile.COMPRESSION(page.compression).name
        predictor = tifffile.PREDICTOR(page.predictor).name
    label = TIFF_NAMES.get(codec, codec.lower().replace("_", " "))
    return label if predictor == "NONE" else f"{label} + {predictor.lower()} predictor"


def first_zarr_array(path: Path) -> zarr.Array:
    node = zarr.open(path, mode="r")
    if isinstance(node, zarr.Array):
        return node
    return min(node.arrays(), key=lambda named: named[0])[1]


def zarr_compression(path: Path) -> str:
    metadata = first_zarr_array(path).metadata
    if metadata.zarr_format == 2:
        return metadata.compressor.codec_id if metadata.compressor else "none"
    stages = [zarr_codec_label(codec.to_dict()) for codec in metadata.codecs if codec.to_dict()["name"] != "bytes"]
    return " + ".join(stages) or "none"


def zarr_codec_label(codec: Dict) -> str:
    config = codec.get("configuration", {})
    detail = [str(config[k]) for k in ("cname", "shuffle") if k in config]
    level = f" level {config['clevel']}" if "clevel" in config else ""
    return f"{codec['name']}({', '.join(detail)}){level}" if detail else codec["name"]


def czi_directory_position(handle) -> int:
    header = handle.read(32 + 8 + 8 + 16 + 16 + 4 + 8)
    if not header.startswith(b"ZISRAWFILE"):
        raise ValueError("not a CZI file")
    return struct.unpack("<q", header[-8:])[0]


def czi_codecs(handle) -> Iterable[int]:
    handle.seek(czi_directory_position(handle))
    handle.read(32)
    count = struct.unpack("<i", handle.read(4))[0]
    handle.read(124)
    for _ in range(count):
        handle.read(2)
        _pixel, _position, _part, compression = struct.unpack("<iqii", handle.read(20))
        handle.read(6)
        handle.read(20 * struct.unpack("<i", handle.read(4))[0])
        yield compression


def czi_compression(path: Path) -> str:
    with open(path, "rb") as handle:
        counts = collections.Counter(czi_codecs(handle))
    return " + ".join(CZI_COMPRESSION.get(code, str(code)) for code, _ in counts.most_common())


def nd2_compression(path: Path) -> str:
    with nd2.ND2File(path) as handle:
        return handle.attributes.compressionType or "none"


def hdf5_compression(dataset: h5py.Dataset) -> str:
    if dataset.compression is None:
        return "none"
    level = f" level {dataset.compression_opts}" if dataset.compression == "gzip" and dataset.compression_opts else ""
    return f"{dataset.compression}{level}" + (" + shuffle" if dataset.shuffle else "")


HEADER_READERS = {".czi": czi_compression, ".nd2": nd2_compression, ".tif": tiff_compression, ".tiff": tiff_compression}
