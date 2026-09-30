"""NIfTI loader using nibabel."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

import dask
import dask.array as da
import nibabel as nib
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured

from pixel_patrol_base.core.contracts import FileInfo, SkipFile
from pixel_patrol_base.core.loader_schema import (
    RASTER_IMAGE_LOADER_SCHEMA,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
)
from pixel_patrol_base.core.record import Record, record_from

logger = logging.getLogger(__name__)


def _nifti_dim_order(ndim: int) -> str:
    # NIfTI axis order: X, Y, Z, T for dims 1-4
    base = "XYZT"
    if ndim <= 4:
        return base[:ndim]
    return base + "ABCDEFGHIJ"[: ndim - 4]


def _channel_field_dtype(dtype: np.dtype) -> np.dtype | None:
    """Shared scalar dtype of a uniform structured dtype (e.g. NIfTI DT_RGB24), else None."""
    if dtype.names is None:
        return None
    field_dtypes = {dtype.fields[name][0] for name in dtype.names}
    return field_dtypes.pop() if len(field_dtypes) == 1 else None


def _load_bids_sidecar(nifti_path: Path) -> Dict[str, Any]:
    """Read the BIDS JSON sidecar alongside a .nii or .nii.gz file, if present.

    Keeps original BIDS key names. Skips arrays and nested objects.
    """
    name = nifti_path.name
    if name.lower().endswith(".nii.gz"):
        stem = nifti_path.with_name(name[:-7])  # strip .nii.gz
    else:
        stem = nifti_path.with_suffix("")        # strip .nii
    json_path = stem.with_suffix(".json")
    if not json_path.exists():
        return {}
    try:
        with open(json_path) as f:
            raw = json.load(f)
    except Exception as exc:
        logger.warning("NiftiLoader: could not read sidecar '%s': %s", json_path.name, exc)
        return {}
    return {
        k: v for k, v in raw.items()
        if isinstance(v, (str, int, float, bool))
    }


def _extract_meta(img: Any, dim_order: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "dim_order": dim_order,
        "dtype": str(np.dtype(img.get_data_dtype())),
    }
    try:
        zooms = img.header.get_zooms()
        for i, ax in enumerate(dim_order):
            if i < len(zooms) and ax in "XYZT":
                val = float(zooms[i])
                if val > 0:
                    meta[f"pixel_size_{ax}"] = val
    except Exception:
        pass
    try:
        intent_code, _, _ = img.header.get_intent()
        if intent_code and intent_code != "none":
            meta["nifti_intent"] = str(intent_code)
    except Exception:
        pass
    try:
        space, time = img.header.get_xyzt_units()
        if space:
            meta["voxel_unit"] = space
        if time:
            meta["time_unit"] = time
    except Exception:
        pass
    try:
        descrip = img.header["descrip"].tobytes().decode("latin-1").rstrip("\x00").strip()
        if descrip:
            meta["descrip"] = descrip
    except Exception:
        pass
    return meta


def _load_nifti_array(path_str: str) -> np.ndarray:
    img = nib.load(path_str)
    arr = np.asarray(img.dataobj)
    if arr.dtype.names is not None:
        arr = structured_to_unstructured(arr)  # e.g. RGB24 struct -> (..., C) plain array
    return arr


class NiftiLoader:
    """Load NIfTI images (.nii, .nii.gz) via nibabel."""

    NAME = "nifti"
    DESCRIPTION = "Loads NIfTI images (.nii, .nii.gz), reading voxel data and header metadata."

    SUPPORTED_EXTENSIONS: Set[str] = {"nii", "nii.gz"}
    FOLDER_EXTENSIONS:    Set[str] = set()
    CONTAINER_EXTENSIONS: Set[str] = {"nii.gz"}  # compressed; on-disk size understates uncompressed

    OUTPUT_SCHEMA: Dict[str, Any] = {**RASTER_IMAGE_LOADER_SCHEMA, "nifti_intent": str}
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = {
        "nifti_intent": "NIfTI intent code describing the data type (e.g. 'NIFTI_INTENT_NONE', 'NIFTI_INTENT_LABEL').",
    }
    OUTPUT_SCHEMA_PATTERNS: List[tuple] = list(RASTER_IMAGE_LOADER_SCHEMA_PATTERNS)

    def is_folder_supported(self, path: Path) -> bool:
        return False

    def read_header(self, file_path: Path) -> FileInfo:
        img = nib.load(str(file_path))
        shape = img.shape
        dtype = np.dtype(img.get_data_dtype())
        if np.issubdtype(dtype, np.complexfloating):
            raise SkipFile(f"complex dtype ({dtype}): NIfTI-MRS spectroscopy data is not supported")
        channel_dtype = _channel_field_dtype(dtype)
        if dtype.names is not None and channel_dtype is None:
            raise SkipFile(f"unsupported structured dtype ({dtype}): mixed-width fields")
        if channel_dtype is not None:
            dim_order = _nifti_dim_order(len(shape)) + "C"
            shape = (*shape, len(dtype.names))
            dtype = channel_dtype
        else:
            dim_order = _nifti_dim_order(len(shape))
        return FileInfo(shape=shape, dtype=dtype, dim_order=dim_order, n_images=1)

    def load(self, file_path: Path) -> Record:
        info = self.read_header(file_path)  # raises SkipFile for complex/unsupported dtype
        img = nib.load(str(file_path))
        shape, dtype, dim_order = info.shape, info.dtype, info.dim_order
        # sidecar merged first so header values take precedence on any key conflict
        meta = {**_load_bids_sidecar(file_path), **_extract_meta(img, dim_order)}
        raw_names = np.dtype(img.get_data_dtype()).names
        if raw_names is not None:
            meta["channel_names"] = list(raw_names)  # e.g. ['R', 'G', 'B']; only tagged rgb:C if they match
            meta["dtype"] = str(dtype)  # per-channel dtype, not the original structured dtype
        data = da.from_delayed(
            dask.delayed(_load_nifti_array)(str(file_path)),
            shape=shape,
            dtype=dtype,
        )
        return record_from(data, meta, kind="intensity")

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        raise NotImplementedError("NiftiLoader does not support container files")
