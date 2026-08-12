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

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.loader_schema import (
    RASTER_IMAGE_LOADER_SCHEMA,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
)
from pixel_patrol_base.core.record import Record, record_from

logger = logging.getLogger(__name__)


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _dim_order(ndim: int) -> str:
    # NIfTI axis order: X, Y, Z, T for dims 1-4
    base = "XYZT"
    if ndim <= 4:
        return base[:ndim]
    return base + "ABCDEFGHIJ"[: ndim - 4]


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
    return np.asarray(img.dataobj)


class NiftiLoader:
    """Load NIfTI images (.nii, .nii.gz) via nibabel."""

    NAME = "nifti"
    DESCRIPTION = "Loads NIfTI images (.nii, .nii.gz), reading voxel data and header metadata."

    # "gz" covers .nii.gz files (path.suffix == ".gz"); read_header/load validate _is_nifti.
    # gz is in CONTAINER_EXTENSIONS because .nii.gz on-disk size is much smaller than
    # uncompressed; without this the pipeline underestimates memory for small compressed files.
    SUPPORTED_EXTENSIONS: Set[str] = {"nii", "gz"}
    FOLDER_EXTENSIONS:    Set[str] = set()
    CONTAINER_EXTENSIONS: Set[str] = {"gz"}

    OUTPUT_SCHEMA: Dict[str, Any] = {**RASTER_IMAGE_LOADER_SCHEMA, "nifti_intent": str}
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = {
        "nifti_intent": "NIfTI intent code describing the data type (e.g. 'NIFTI_INTENT_NONE', 'NIFTI_INTENT_LABEL').",
    }
    OUTPUT_SCHEMA_PATTERNS: List[tuple] = list(RASTER_IMAGE_LOADER_SCHEMA_PATTERNS)

    def is_folder_supported(self, path: Path) -> bool:
        return False

    def read_header(self, file_path: Path) -> FileInfo:
        if not _is_nifti(file_path):
            raise ValueError(f"Not a NIfTI file: {file_path}")
        img = nib.load(str(file_path))
        shape = img.shape
        dtype = np.dtype(img.get_data_dtype())
        dim_order = _dim_order(len(shape))
        return FileInfo(shape=shape, dtype=dtype, dim_order=dim_order, n_images=1)

    def load(self, file_path: Path) -> Record:
        if not _is_nifti(file_path):
            raise ValueError(f"Not a NIfTI file: {file_path}")
        img = nib.load(str(file_path))
        shape = img.shape
        dtype = np.dtype(img.get_data_dtype())
        dim_order = _dim_order(len(shape))
        # sidecar merged first so header values take precedence on any key conflict
        meta = {**_load_bids_sidecar(file_path), **_extract_meta(img, dim_order)}
        data = da.from_delayed(
            dask.delayed(_load_nifti_array)(str(file_path)),
            shape=shape,
            dtype=dtype,
        )
        return record_from(data, meta, kind="intensity")

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        raise NotImplementedError("NiftiLoader does not support container files")
