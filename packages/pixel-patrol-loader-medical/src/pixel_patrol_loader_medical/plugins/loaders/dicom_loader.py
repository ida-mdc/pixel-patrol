"""DICOM loader using pydicom."""

from __future__ import annotations

import logging
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

import dask
import dask.array as da
import numpy as np
import pydicom

from pixel_patrol_base.core.contracts import FileInfo, SkipFile
from pixel_patrol_base.core.loader_schema import (
    RASTER_IMAGE_LOADER_SCHEMA,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
)
from pixel_patrol_base.core.record import Record, record_from

logger = logging.getLogger(__name__)


def _is_dicom(path: Path) -> bool:
    """Check by extension first, then by DICM magic bytes at offset 128."""
    ext = path.suffix.lower().lstrip(".")
    if ext in ("dcm", "dicom"):
        return True
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def _iter_dicom_files(folder: Path) -> Iterator[Path]:
    # Recurse so series nested in subdirectories (common in DICOM exports) are found.
    # DICOMDIR is a directory-index file, not image data; skip it by name.
    for f in folder.rglob("*"):
        if f.is_file() and f.name != "DICOMDIR" and _is_dicom(f):
            yield f


def _slice_sort_key(ds: pydicom.Dataset) -> float:
    pos = getattr(ds, "ImagePositionPatient", None)
    if pos is not None:
        try:
            return float(pos[2])
        except (ValueError, IndexError):
            pass
    return float(getattr(ds, "InstanceNumber", 0) or 0)


@lru_cache(maxsize=8)
def _scan_series(folder: Path) -> Dict[str, List[Path]]:
    """Return {SeriesInstanceUID: [sorted slice paths]} for all DICOM series in folder."""
    per_uid: Dict[str, List[Tuple[float, Path]]] = defaultdict(list)
    for f in _iter_dicom_files(folder):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            if not hasattr(ds, "Rows"):
                continue  # skip non-image DICOM files (SR, KO, PR, etc.)
            uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
            per_uid[uid].append((_slice_sort_key(ds), f))
        except Exception as exc:
            logger.warning("Skipping unreadable DICOM file %s: %s", f.name, exc)
    return {uid: [f for _, f in sorted(entries)] for uid, entries in per_uid.items()}


def _output_dtype(ds: pydicom.Dataset) -> np.dtype:
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    if slope != 1.0 or intercept != 0.0:
        return np.dtype("float32")
    bits = int(getattr(ds, "BitsAllocated", 16))
    signed = int(getattr(ds, "PixelRepresentation", 0)) == 1
    if bits == 8:
        return np.dtype("int8" if signed else "uint8")
    if bits == 32:
        return np.dtype("int32" if signed else "uint32")
    return np.dtype("int16" if signed else "uint16")


def _apply_rescale(arr: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    if slope == 1.0 and intercept == 0.0:
        return arr
    return arr.astype(np.float32) * slope + intercept


def _pixel_sizes(ds: pydicom.Dataset) -> Dict[str, float]:
    out: Dict[str, float] = {}
    ps = getattr(ds, "PixelSpacing", None)
    if ps is not None and len(ps) >= 2:
        out["pixel_size_Y"] = float(ps[0])
        out["pixel_size_X"] = float(ps[1])
    st = getattr(ds, "SliceThickness", None)
    if st is not None:
        try:
            out["pixel_size_Z"] = float(st)
        except (ValueError, TypeError):
            pass
    return out


# Acquisition/scanner tags to extract (original DICOM keyword names).
# PHI tags (PatientName, PatientID, PatientBirthDate, etc.) are deliberately excluded.
_DICOM_TAGS = [
    "Modality", "Manufacturer", "ManufacturerModelName", "SoftwareVersions",
    "BodyPartExamined", "SeriesDescription", "ProtocolName",
    "ScanningSequence", "SequenceVariant", "ScanOptions", "MRAcquisitionType", "SequenceName",
    "MagneticFieldStrength", "ImagingFrequency", "EchoTime", "RepetitionTime",
    "FlipAngle", "SAR", "EchoTrainLength", "PixelBandwidth",
    "SpacingBetweenSlices", "PatientPosition",
    "InPlanePhaseEncodingDirection", "ParallelReductionFactorInPlane",
]


def _series_meta(ds: pydicom.Dataset, dim_order: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "dim_order": dim_order,
        "dtype": str(_output_dtype(ds)),
    }
    meta.update(_pixel_sizes(ds))
    for tag in _DICOM_TAGS:
        val = getattr(ds, tag, None)
        if val is None:
            continue
        if isinstance(val, (pydicom.sequence.Sequence,)):
            continue
        # Convert pydicom types (DSfloat, IS, MultiValue, etc.) to plain Python scalars.
        # MultiValue is a list subclass; join with DICOM's backslash separator.
        if isinstance(val, (int, float, bool, str)):
            meta[tag] = val
        elif isinstance(val, pydicom.multival.MultiValue):
            meta[tag] = "\\".join(str(v) for v in val)
        elif hasattr(val, "__float__"):
            meta[tag] = float(val)
        elif hasattr(val, "__int__"):
            meta[tag] = int(val)
        else:
            s = str(val).strip()
            if s:
                meta[tag] = s
    return meta


def _load_series_array(file_strs: List[str]) -> np.ndarray:
    slices = []
    for path_str in file_strs:
        ds = pydicom.dcmread(path_str)
        slices.append(_apply_rescale(ds.pixel_array, ds))
    if len(slices) == 1:
        return slices[0]
    return np.stack(slices, axis=0)


def _load_single_dicom(path_str: str) -> np.ndarray:
    ds = pydicom.dcmread(path_str)
    return _apply_rescale(ds.pixel_array, ds)


def _build_series_record(files: List[Path], ref_ds: pydicom.Dataset) -> Record:
    n = len(files)
    rows, cols = int(ref_ds.Rows), int(ref_ds.Columns)
    shape = (n, rows, cols) if n > 1 else (rows, cols)
    dim_order = "ZYX" if n > 1 else "YX"
    dtype = _output_dtype(ref_ds)
    meta = _series_meta(ref_ds, dim_order)
    data = da.from_delayed(
        dask.delayed(_load_series_array)([str(f) for f in files]),
        shape=shape,
        dtype=dtype,
    )
    return record_from(data, meta, kind="intensity")


class DicomLoader:
    """Load DICOM images (.dcm, .dicom) via pydicom.

    # TODO: pipeline currently processes individual .dcm slices rather than assembling
    # series into 3D volumes; folder-based series assembly requires a discovery fix.
    """

    NAME = "dicom"
    DESCRIPTION = "Loads DICOM images (.dcm, .dicom), assembling multi-slice series from folders into 3D volumes."

    SUPPORTED_EXTENSIONS: Set[str] = {"dcm", "dicom"}
    FOLDER_EXTENSIONS:    Set[str] = {"dcm", "dicom"}
    CONTAINER_EXTENSIONS: Set[str] = set()

    OUTPUT_SCHEMA: Dict[str, Any] = {**RASTER_IMAGE_LOADER_SCHEMA}
    OUTPUT_SCHEMA_PATTERNS: List[tuple] = list(RASTER_IMAGE_LOADER_SCHEMA_PATTERNS)

    def is_folder_supported(self, path: Path) -> bool:
        return any(True for _ in _iter_dicom_files(path))

    def read_header(self, file_path: Path) -> FileInfo:
        if file_path.is_dir():
            return self._folder_header(file_path)
        return self._file_header(file_path)

    def _file_header(self, file_path: Path) -> FileInfo:
        ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
        if not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
            raise SkipFile(f"non-image DICOM (SR/RT/PR): {file_path.name}")
        rows, cols = int(ds.Rows), int(ds.Columns)
        n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        shape = (n_frames, rows, cols) if n_frames > 1 else (rows, cols)
        dim_order = "ZYX" if n_frames > 1 else "YX"
        return FileInfo(shape=shape, dtype=_output_dtype(ds), dim_order=dim_order, n_images=1)

    def _folder_header(self, folder: Path) -> FileInfo:
        series = _scan_series(folder)
        if not series:
            raise ValueError(f"No DICOM series found in: {folder}")
        n_images = len(series)
        first_files = next(iter(series.values()))
        ds = pydicom.dcmread(str(first_files[0]), stop_before_pixels=True)
        rows, cols = int(ds.Rows), int(ds.Columns)
        n = len(first_files)
        shape = (n, rows, cols) if n > 1 else (rows, cols)
        dim_order = "ZYX" if n > 1 else "YX"
        return FileInfo(shape=shape, dtype=_output_dtype(ds), dim_order=dim_order, n_images=n_images)

    def load(self, file_path: Path) -> Record:
        if file_path.is_dir():
            return self._load_folder_first_series(file_path)
        return self._load_single_file(file_path)

    def _load_folder_first_series(self, folder: Path) -> Record:
        series = _scan_series(folder)
        if not series:
            raise ValueError(f"No DICOM series found in: {folder}")
        if len(series) > 1:
            logger.warning(
                "DICOM folder %s has %d series; loading first only. "
                "Organise series into separate folders to process all.",
                folder.name, len(series),
            )
        _, files = next(iter(series.items()))
        ref_ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
        return _build_series_record(files, ref_ds)

    def _load_single_file(self, file_path: Path) -> Record:
        ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
        n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        rows, cols = int(ds.Rows), int(ds.Columns)
        shape = (n_frames, rows, cols) if n_frames > 1 else (rows, cols)
        dim_order = "ZYX" if n_frames > 1 else "YX"
        dtype = _output_dtype(ds)
        meta = _series_meta(ds, dim_order)
        data = da.from_delayed(
            dask.delayed(_load_single_dicom)(str(file_path)),
            shape=shape,
            dtype=dtype,
        )
        return record_from(data, meta, kind="intensity")

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        series = _scan_series(file_path)
        series_list = list(series.items())
        for i in range(start, min(stop, len(series_list))):
            uid, files = series_list[i]
            ref_ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
            yield uid, _build_series_record(files, ref_ds)
