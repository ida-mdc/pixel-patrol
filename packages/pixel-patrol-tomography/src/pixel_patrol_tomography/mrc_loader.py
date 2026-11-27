"""MRC loader (tilt series, image stacks and reconstructed volumes) using mrcfile."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import dask.array as da
import mrcfile
import numpy as np
from pixel_patrol_base.core.contracts import FileInfo, SkipFile
from pixel_patrol_base.core.loader_schema import (
    RASTER_IMAGE_LOADER_SCHEMA,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
)
from pixel_patrol_base.core.record import Record, record_from

TILT_AXIS_NAME = "A"

# MRC modes 3 and 4 are complex-valued (Fourier transforms), which the pixel-patrol pipeline does not handle.
_COMPLEX_ARRAY_DATA_TO_SKIP = {3, 4}


def _decode_bytes(value: Any) -> str:
    return bytes(value).decode("ascii", errors="replace").rstrip("\x00").strip()


def _get_dim_order(mrc: mrcfile.mrcfile.MrcFile) -> str:
    """Dim order of mrc.data: stacks (ispg == 0, e.g. IMOD .st tilt series) get the tilt axis, volumes get Z."""
    ndim = mrc.data.ndim
    if ndim == 2:
        return "YX"
    if mrc.is_image_stack():
        return TILT_AXIS_NAME + "YX"
    return "ZYX"


def _check_supported(mrc: mrcfile.mrcfile.MrcFile, file_path: Path) -> None:
    header = mrc.header
    mode = int(header.mode)
    if mode in _COMPLEX_ARRAY_DATA_TO_SKIP:
        raise SkipFile(f"complex MRC mode {mode} is not supported: {file_path}")
    axis_mapping = (int(header.mapc), int(header.mapr), int(header.maps))
    if axis_mapping != (1, 2, 3):
        error_msg = f"non-standard MRC axis mapping (mapc, mapr, maps) = {axis_mapping} is not supported: {file_path}"
        raise SkipFile(error_msg)
    if mrc.data.ndim not in (2, 3):
        raise SkipFile(f"only 2D and 3D MRC volume are supported! not shape {mrc.data.shape}: {file_path}")


def _native_dtype(dtype: np.dtype) -> np.dtype:
    return dtype.newbyteorder("=")


def _extract_meta(mrc: mrcfile.mrcfile.MrcFile, dim_order: str) -> dict[str, Any]:
    header = mrc.header
    meta: dict[str, Any] = {
        "dim_order": dim_order,
        "dtype": str(_native_dtype(mrc.data.dtype)),
        "mrc_mode": int(header.mode),
        "mrc_space_group": int(header.ispg),
        "mrc_nversion": int(header.nversion),
        "mrc_exttyp": _decode_bytes(header.exttyp),
        "mrc_extended_header_bytes": int(header.nsymbt),
        "mrc_origin": [float(header.origin.x), float(header.origin.y), float(header.origin.z)],
        "mrc_header_min": float(header.dmin),
        "mrc_header_max": float(header.dmax),
        "mrc_header_mean": float(header.dmean),
        "mrc_header_rms": float(header.rms),
        "mrc_labels": [_decode_bytes(label) for label in header.label[: int(header.nlabl)]],
    }

    # The tilt axis has no physical size; voxel sizes are in Angstrom by MRC convention.
    voxel_size = mrc.voxel_size
    for axis in dim_order:
        if axis == TILT_AXIS_NAME:
            continue
        size = float(getattr(voxel_size, axis.lower()))
        meta[f"pixel_size_{axis}"] = size
    return meta


class _MrcSectionReader:
    """Picklable, lazily-reading array-like over an MRC file, for use with ``dask.array.from_array``.

    Holds only the path, shape and dtype; every indexing call re-opens the file as a memory map and copies just the
    requested region (converted to native byte order), so no file handle outlives a read.
    """

    def __init__(self, path: str, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.path = path
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)

    def __getitem__(self, key: Any) -> np.ndarray:
        with mrcfile.mmap(self.path, mode="r", permissive=True) as mrc:
            return np.array(mrc.data[key], dtype=self.dtype)


class MrcLoader:
    """Load MRC files - IMOD/SerialEM tilt series (.st, .ali), stacks (.mrcs) and reconstructions (.rec, .mrc).

    Files with space group 0 and more than one section are image stacks: their first axis is the tilt/projection
    index and is labelled ``A`` (dim order ``AYX``). Volumes (space group >= 1) get dim order ``ZYX``. Pixel data is
    read lazily, one section per dask chunk, via a memory map.
    """

    NAME = "mrc"
    DESCRIPTION = "Loads MRC files (tomography tilt series, image stacks and reconstructed volumes) via mrcfile."

    SUPPORTED_EXTENSIONS: set[str] = {"mrc", "mrcs", "st", "ali", "preali", "rec", "map"}
    FOLDER_EXTENSIONS: set[str] = set()
    CONTAINER_EXTENSIONS: set[str] = set()

    OUTPUT_SCHEMA: dict[str, Any] = {
        **RASTER_IMAGE_LOADER_SCHEMA,
        "mrc_mode": int,
        "mrc_space_group": int,
        "mrc_nversion": int,
        "mrc_exttyp": str,
        "mrc_extended_header_bytes": int,
        "mrc_origin": list,
        "mrc_header_min": float,
        "mrc_header_max": float,
        "mrc_header_mean": float,
        "mrc_header_rms": float,
        "mrc_labels": list,
    }

    OUTPUT_SCHEMA_DESCRIPTIONS: dict[str, str] = {
        "mrc_mode": "MRC data mode (0: int8, 1: int16, 2: float32, 6: uint16, 12: float16, ...).",
        "mrc_space_group": "MRC space group (ispg); 0 marks an image stack such as a tilt series, >= 1 a volume.",
        "mrc_nversion": "MRC format version (e.g. 20140).",
        "mrc_exttyp": "Extended header type (e.g. 'SERI', 'FEI1'); empty if not set.",
        "mrc_extended_header_bytes": "Size of the extended header in bytes (nsymbt).",
        "mrc_origin": "Origin [x, y, z] from the header.",
        "mrc_header_min": "Minimum pixel value as stored in the header (dmin); may be stale.",
        "mrc_header_max": "Maximum pixel value as stored in the header (dmax); may be stale.",
        "mrc_header_mean": "Mean pixel value as stored in the header (dmean); may be stale.",
        "mrc_header_rms": "RMS deviation as stored in the header; negative if not computed.",
        "mrc_labels": "Processing history labels from the header (e.g. 'TILT: Tomographic reconstruction').",
    }
    OUTPUT_SCHEMA_PATTERNS: list[tuple] = list(RASTER_IMAGE_LOADER_SCHEMA_PATTERNS)

    def is_folder_supported(self, path: Path) -> bool:
        return False

    def read_header(self, file_path: Path) -> FileInfo:
        with mrcfile.mmap(file_path, mode="r", permissive=True) as mrc:
            _check_supported(mrc, file_path)
            shape = tuple(mrc.data.shape)
            dtype = _native_dtype(mrc.data.dtype)
            dim_order = _get_dim_order(mrc)
        return FileInfo(shape=shape, dtype=dtype, dim_order=dim_order, n_images=1)

    def load(self, file_path: Path) -> Record:
        with mrcfile.mmap(file_path, mode="r", permissive=True) as mrc:
            _check_supported(mrc, file_path)
            shape = tuple(mrc.data.shape)
            dtype = _native_dtype(mrc.data.dtype)
            dim_order = _get_dim_order(mrc)
            meta = _extract_meta(mrc, dim_order)

        reader = _MrcSectionReader(str(file_path), shape, dtype)
        chunks = (1, *shape[1:]) if len(shape) == 3 else shape
        data = da.from_array(reader, chunks=chunks, name=f"mrc-{file_path}")
        return record_from(data, meta, kind="intensity")

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[tuple[str, Record]]:
        raise NotImplementedError("MrcLoader does not support container files")
