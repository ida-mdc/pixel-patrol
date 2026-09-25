"""HDF5 loader supporting three layouts:

* **plain HDF5** - every numeric 2D+ dataset; axes from axistags/dim_order/_ARRAY_DIMENSIONS/axes, inferred as trailing YX otherwise.
* **Imaris IMS 5.5+** - ResolutionLevel 0 assembled into one TCZYX array; pyramids/thumbnails/histograms excluded.
* **BigDataViewer/BDV** - tXXXXX/sYY/0/cells; ZYX axes; voxel sizes from sibling XML.

Pixel data is lazy via ``_H5DatasetProxy`` (picklable, no open file handles in workers).
"""

import json
import logging
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import dask.array as da
import h5py
import numpy as np

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.loader_schema import (
    RASTER_IMAGE_LOADER_SCHEMA,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
)
from pixel_patrol_base.core.record import record_from, Record
from pixel_patrol_loader_bio.plugins.loaders._utils import infer_dim_order

logger = logging.getLogger(__name__)

_MIN_IMAGE_NDIM = 2
_NUMERIC_DTYPE_KINDS = "uif"

# Axis attributes written by common HDF5 producers, in order of preference.
_AXIS_ATTR_KEYS = ("dim_order", "_ARRAY_DIMENSIONS", "axes")

_BDV_TIMEPOINT_RE = re.compile(r"^t\d+$")
_BDV_SETUP_RE = re.compile(r"^s\d+$")
_BDV_FULL_RESOLUTION = "0"
_BDV_CELLS = "cells"
_BDV_DIM_ORDER = "ZYX"

_IMARIS_DATASET = "DataSet"
_IMARIS_INFO = "DataSetInfo"
_IMARIS_LEVEL = "ResolutionLevel 0"
_IMARIS_DATA = "Data"
_IMARIS_TIMEPOINT_RE = re.compile(r"^TimePoint (\d+)$")
_IMARIS_CHANNEL_RE = re.compile(r"^Channel (\d+)$")
_IMARIS_DIM_ORDER = "TCZYX"


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def _open_h5(file_path: Path) -> h5py.File:
    """Open read-only without HDF5 file locking (needed for network fs and parallel workers)."""
    try:
        return h5py.File(str(file_path), "r", locking=False)
    except TypeError:  # h5py too old to know the locking kwarg
        return h5py.File(str(file_path), "r")


# ---------------------------------------------------------------------------
# Lazy pixel access
# ---------------------------------------------------------------------------


class _H5DatasetProxy:
    """Picklable proxy for one HDF5 dataset; opens the file per-block so workers hold no handles."""

    def __init__(self, file_path: Path, dataset_path: str, shape: Tuple[int, ...], dtype: Any):
        self.file_path = str(file_path)
        self.dataset_path = dataset_path
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)

    def __getitem__(self, key: Any) -> np.ndarray:
        with _open_h5(Path(self.file_path)) as h5:
            return h5[self.dataset_path][key]


def _as_dask(file_path: Path, dset: h5py.Dataset) -> da.Array:
    proxy = _H5DatasetProxy(file_path, dset.name, dset.shape, dset.dtype)
    return da.from_array(
        proxy,
        chunks=dset.chunks or "auto",
        meta=np.empty((0,) * proxy.ndim, dtype=proxy.dtype),
    )


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def _is_image_dataset(obj: Any) -> bool:
    """An array of numbers with at least two dimensions - a plausible image."""
    return (
        isinstance(obj, h5py.Dataset)
        and obj.ndim >= _MIN_IMAGE_NDIM
        and obj.dtype.kind in _NUMERIC_DTYPE_KINDS
    )


def _bdv_dataset_paths(h5: h5py.File) -> Optional[List[str]]:
    """Full-resolution cells paths if BDV layout detected, else None."""
    timepoints = sorted(k for k in h5.keys() if _BDV_TIMEPOINT_RE.match(k))
    if not timepoints:
        return None
    paths: List[str] = []
    for timepoint in timepoints:
        for setup in sorted(k for k in h5[timepoint].keys() if _BDV_SETUP_RE.match(k)):
            path = f"{timepoint}/{setup}/{_BDV_FULL_RESOLUTION}/{_BDV_CELLS}"
            if path in h5:
                paths.append(path)
            else:
                logger.warning("H5Loader: BDV setup '%s/%s' has no level-0 cells", timepoint, setup)
    return paths


def _plain_dataset_paths(h5: h5py.File) -> List[str]:
    paths: List[str] = []
    h5.visititems(lambda name, obj: paths.append(name) if _is_image_dataset(obj) else None)
    return sorted(paths)


def _dataset_paths(h5: h5py.File) -> Tuple[List[str], bool]:
    bdv = _bdv_dataset_paths(h5)
    if bdv is not None:
        return bdv, True
    return _plain_dataset_paths(h5), False


def _child_id(dataset_path: str, is_bdv: bool) -> str:
    if is_bdv:
        timepoint, setup, *_ = dataset_path.split("/")
        return f"{timepoint}/{setup}"
    return dataset_path


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


def _as_text(value: Any) -> Optional[str]:
    """Decode an HDF5 attribute that may be a scalar string, bytes, or S1 char array."""
    if isinstance(value, np.ndarray) and value.dtype.kind in "SU":
        if value.ndim == 0:
            value = value.item()
        elif value.dtype.kind == "S":
            value = b"".join(bytes(v) for v in value.flat)
        else:
            value = "".join(str(v) for v in value.flat)
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8", errors="replace")
    if isinstance(value, (str, np.str_)):
        return str(value)
    return None


def _jsonable(value: Any) -> Any:
    """Coerce an HDF5 attribute value into something the table can hold."""
    if isinstance(value, (bytes, np.bytes_)):
        return _as_text(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_attrs(obj: Any) -> Dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in obj.attrs.items()}


def _attr_text(obj: Any, key: str) -> Optional[str]:
    text = _as_text(obj.attrs.get(key))
    return text.strip() if text is not None else None


def _attr_float(obj: Any, key: str) -> Optional[float]:
    value = obj.attrs.get(key)
    text = _as_text(value)
    try:
        return float(text if text is not None else value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Axis labels
# ---------------------------------------------------------------------------


def _as_axis_letters(value: Any, ndim: int) -> Optional[str]:
    if isinstance(value, (str, bytes, np.bytes_, np.str_)):
        text = _as_text(value) or ""
        labels = list(text) if len(text) == ndim else text.split()
    elif isinstance(value, (list, tuple, np.ndarray)):
        labels = [_as_text(v) for v in value]
    else:
        return None

    if len(labels) != ndim or not all(label and len(label) == 1 and label.isalpha() for label in labels):
        return None
    return "".join(labels).upper()


def _dim_order_from_axistags(value: Any, ndim: int) -> Optional[str]:
    """vigra/ilastik axistags JSON: {"axes": [{"key": "z"}, ...]}."""
    text = _as_text(value)
    if not text:
        return None
    try:
        axes = json.loads(text).get("axes", [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        logger.warning("H5Loader: could not parse 'axistags' attribute; ignoring it")
        return None
    return _as_axis_letters([axis.get("key") for axis in axes if isinstance(axis, dict)], ndim)


def _dim_order_from_attrs(attrs: Dict[str, Any], ndim: int) -> Optional[str]:
    from_axistags = _dim_order_from_axistags(attrs.get("axistags"), ndim)
    if from_axistags:
        return from_axistags
    for key in _AXIS_ATTR_KEYS:
        letters = _as_axis_letters(attrs.get(key), ndim)
        if letters:
            return letters
    return None


def _dim_order_for(dset: h5py.Dataset, attrs: Dict[str, Any], is_bdv: bool) -> str:
    if is_bdv:
        return _BDV_DIM_ORDER
    return _dim_order_from_attrs(attrs, dset.ndim) or infer_dim_order(dset.ndim)


# ---------------------------------------------------------------------------
# Imaris IMS
# ---------------------------------------------------------------------------


def _is_imaris_file(h5: h5py.File) -> bool:
    """Whether this is an HDF5-backed Imaris 5.5+ dataset."""
    marker = _attr_text(h5, "ImarisDataSet")
    return marker == "ImarisDataSet" and _IMARIS_DATASET in h5


def _numbered_children(group: h5py.Group, pattern: re.Pattern[str]) -> List[Tuple[int, str]]:
    found: List[Tuple[int, str]] = []
    for name in group.keys():
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), name))
    return sorted(found)


def _imaris_dataset_grid(h5: h5py.File) -> List[List[str]]:
    level_path = f"{_IMARIS_DATASET}/{_IMARIS_LEVEL}"
    if level_path not in h5:
        raise RuntimeError("Imaris IMS has no ResolutionLevel 0")

    level = h5[level_path]
    timepoints = _numbered_children(level, _IMARIS_TIMEPOINT_RE)
    if not timepoints:
        raise RuntimeError("Imaris IMS has no time points at ResolutionLevel 0")

    grid: List[List[str]] = []
    expected_channel_ids: Optional[List[int]] = None
    expected_shape: Optional[Tuple[int, ...]] = None
    expected_dtype: Optional[np.dtype] = None

    for _time_id, time_name in timepoints:
        time_group = level[time_name]
        channels = _numbered_children(time_group, _IMARIS_CHANNEL_RE)
        channel_ids = [channel_id for channel_id, _ in channels]
        if not channels:
            raise RuntimeError(f"Imaris IMS time point '{time_name}' has no channels")
        if expected_channel_ids is None:
            expected_channel_ids = channel_ids
        elif channel_ids != expected_channel_ids:
            raise RuntimeError("Imaris IMS time points do not contain the same channels")

        row: List[str] = []
        for _channel_id, channel_name in channels:
            path = f"{level_path}/{time_name}/{channel_name}/{_IMARIS_DATA}"
            if path not in h5 or not isinstance(h5[path], h5py.Dataset):
                raise RuntimeError(f"Imaris IMS is missing pixel dataset '{path}'")
            dset = h5[path]
            if dset.ndim not in (2, 3) or dset.dtype.kind not in _NUMERIC_DTYPE_KINDS:
                raise RuntimeError(f"Imaris IMS pixel dataset '{path}' is not a 2D/3D numeric array")
            shape = tuple(int(size) for size in dset.shape)
            dtype = np.dtype(dset.dtype)
            if expected_shape is None:
                expected_shape, expected_dtype = shape, dtype
            elif shape != expected_shape or dtype != expected_dtype:
                raise RuntimeError("Imaris IMS channels/time points have inconsistent shapes or dtypes")
            row.append(path)
        grid.append(row)
    return grid


def _imaris_channel_attr_list(h5: h5py.File, channel_count: int, attr_key: str) -> List[Optional[str]]:
    info = h5.get(_IMARIS_INFO)
    result: List[Optional[str]] = []
    for i in range(channel_count):
        group = info.get(f"Channel {i}") if isinstance(info, h5py.Group) else None
        result.append(_attr_text(group, attr_key) if group is not None else None)
    return result


def _imaris_channel_names(h5: h5py.File, channel_count: int) -> List[str]:
    raw = _imaris_channel_attr_list(h5, channel_count, "Name")
    return [name or f"Channel {i}" for i, name in enumerate(raw)]


def _imaris_spatial_metadata(h5: h5py.File, zyx_shape: Tuple[int, int, int]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    info = h5.get(_IMARIS_INFO)
    image = info.get("Image") if isinstance(info, h5py.Group) else None
    if image is None:
        return meta

    unit = _attr_text(image, "Unit")
    if unit:
        meta["pixel_size_unit"] = unit

    sizes = {"X": zyx_shape[2], "Y": zyx_shape[1], "Z": zyx_shape[0]}
    for index, axis in enumerate("XYZ"):
        minimum = _attr_float(image, f"ExtMin{index}")
        maximum = _attr_float(image, f"ExtMax{index}")
        if minimum is not None and maximum is not None and sizes[axis] > 0:
            meta[f"pixel_size_{axis}"] = (maximum - minimum) / sizes[axis]
    return meta


def _imaris_metadata(h5: h5py.File, grid: List[List[str]]) -> Dict[str, Any]:
    first = h5[grid[0][0]]
    source_shape = tuple(int(size) for size in first.shape)
    zyx_shape = source_shape if len(source_shape) == 3 else (1, *source_shape)
    n_channels = len(grid[0])
    meta: Dict[str, Any] = {
        "h5_dataset_path": f"/{_IMARIS_DATASET}/{_IMARIS_LEVEL}",
        "dim_order": _IMARIS_DIM_ORDER,
        "channel_names": _imaris_channel_names(h5, n_channels),
        "emission_wavelengths": _imaris_channel_attr_list(h5, n_channels, "LSMEmissionWavelength"),
        "excitation_wavelengths": _imaris_channel_attr_list(h5, n_channels, "LSMExcitationWavelength"),
        "imaris_format_version": _attr_text(h5, "ImarisVersion"),
        "h5_attributes": _read_attrs(h5),
    }
    meta.update(_imaris_spatial_metadata(h5, zyx_shape))
    return meta


def _build_imaris_record(file_path: Path, h5: h5py.File) -> Record:
    """Build one lazy TCZYX record from an HDF5-backed IMS file."""
    grid = _imaris_dataset_grid(h5)
    time_arrays: List[da.Array] = []
    for row in grid:
        channel_arrays: List[da.Array] = []
        for path in row:
            array = _as_dask(file_path, h5[path])
            if array.ndim == 2:
                array = array[None, ...]
            channel_arrays.append(array)
        time_arrays.append(da.stack(channel_arrays, axis=0))
    data = da.stack(time_arrays, axis=0)
    return record_from(data, _imaris_metadata(h5, grid), kind="intensity")


# ---------------------------------------------------------------------------
# BDV XML sidecar
# ---------------------------------------------------------------------------


def _parse_bdv_voxel_size(voxel: ET.Element) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    unit = voxel.findtext("unit")
    if unit and unit.strip():
        meta["pixel_size_unit"] = unit.strip()
    sizes = (voxel.findtext("size") or "").split()
    for axis, raw in zip("XYZ", sizes):
        try:
            meta[f"pixel_size_{axis}"] = float(raw)
        except ValueError:
            logger.warning("H5Loader: unparsable BDV voxel size %r", raw)
    return meta


def _parse_bdv_view_setup(setup: ET.Element) -> Optional[Tuple[int, Dict[str, Any]]]:
    raw_id = (setup.findtext("id") or "").strip()
    if not raw_id.isdigit():
        return None

    meta: Dict[str, Any] = {}
    name = setup.findtext("name")
    if name and name.strip():
        meta["bdv_setup_name"] = name.strip()
    voxel = setup.find("voxelSize")
    if voxel is not None:
        meta.update(_parse_bdv_voxel_size(voxel))
    return int(raw_id), meta


@lru_cache(maxsize=16)
def _load_bdv_setup_meta(file_path: Path) -> Dict[int, Dict[str, Any]]:
    """Parse the sibling BDV XML once per worker process; missing XML returns {}."""
    xml_path = file_path.with_suffix(".xml")
    if not xml_path.exists():
        logger.debug("H5Loader: no BDV XML next to '%s'", file_path.name)
        return {}
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        logger.warning("H5Loader: could not parse BDV XML '%s': %s", xml_path.name, exc)
        return {}

    setups: Dict[int, Dict[str, Any]] = {}
    for element in root.iter("ViewSetup"):
        parsed = _parse_bdv_view_setup(element)
        if parsed is not None:
            setup_id, meta = parsed
            setups[setup_id] = meta
    return setups


def _bdv_meta(file_path: Path, dataset_path: str) -> Dict[str, Any]:
    timepoint, setup, *_ = dataset_path.split("/")
    meta: Dict[str, Any] = {"bdv_timepoint": timepoint, "bdv_setup": setup}
    setup_id = setup[1:]
    if setup_id.isdigit():
        meta.update(_load_bdv_setup_meta(file_path).get(int(setup_id), {}))
    return meta


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _element_size_meta(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """pixel_size_Z/Y/X from a PlantSeg/Fiji/StarDist element_size_um attribute."""
    raw = attrs.get("element_size_um")
    if not isinstance(raw, list) or not (2 <= len(raw) <= 3):
        return {}
    meta: Dict[str, Any] = {"pixel_size_unit": "um"}
    for axis, size in zip("ZYX"[-len(raw):], raw):
        try:
            meta[f"pixel_size_{axis}"] = float(size)
        except (TypeError, ValueError):
            return {}
    return meta


def _extract_metadata(dset: h5py.Dataset, dim_order: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "h5_dataset_path": dset.name,
        "dim_order": dim_order,
        "dtype": str(dset.dtype),
        "shape": [int(s) for s in dset.shape],
        "ndim": int(dset.ndim),
        "num_pixels": int(np.prod(dset.shape)),
    }
    if dset.chunks is not None:
        meta["chunks"] = tuple(int(c) for c in dset.chunks)
    if attrs:
        meta["h5_attributes"] = attrs
    for axis, size in zip(dim_order, dset.shape):
        meta[f"size_{axis}"] = int(size)
    meta.update(_element_size_meta(attrs))
    return meta


def _build_record(
    file_path: Path, h5: h5py.File, dataset_path: str, is_bdv: bool, root_attrs: Dict[str, Any]
) -> Record:
    dset = h5[dataset_path]
    attrs = {**root_attrs, **_read_attrs(dset)}
    dim_order = _dim_order_for(dset, attrs, is_bdv)
    meta = _extract_metadata(dset, dim_order, attrs)
    if is_bdv:
        meta.update(_bdv_meta(file_path, dataset_path))
    return record_from(_as_dask(file_path, dset), meta, kind="intensity")


def _require_dataset_paths(h5: h5py.File, file_path: Path) -> Tuple[List[str], bool]:
    paths, is_bdv = _dataset_paths(h5)
    if not paths:
        raise RuntimeError(f"H5Loader: no image-like dataset found in '{file_path}'")
    return paths, is_bdv


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class H5Loader:
    """Loader for generic HDF5, BigDataViewer, and HDF5-backed Imaris IMS."""

    NAME = "h5"
    DESCRIPTION = (
        "Loads HDF5, BigDataViewer/BDV, and modern Imaris IMS files lazily. "
        "Format-aware readers exclude pyramid and helper datasets; generic HDF5 files "
        "treat each image-like dataset as a sub-image."
    )

    SUPPORTED_EXTENSIONS: Set[str] = {"h5", "hdf5", "ims"}

    # Always a container: one file holds many datasets; compressed size understates memory cost.
    FOLDER_EXTENSIONS:    Set[str] = set()
    CONTAINER_EXTENSIONS: Set[str] = {"h5", "hdf5", "ims"}

    OUTPUT_SCHEMA: Dict[str, Any] = {
        **RASTER_IMAGE_LOADER_SCHEMA,
        "h5_dataset_path": str,
        "h5_attributes": dict,
    }

    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = {
        "h5_dataset_path": "Path of the dataset within the HDF5 file.",
        "h5_attributes": "Raw HDF5 attributes of the dataset, merged with file root attributes.",
    }

    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]] = list(RASTER_IMAGE_LOADER_SCHEMA_PATTERNS)

    def is_folder_supported(self, path: Path) -> bool:
        return False

    def read_header(self, file_path: Path) -> FileInfo:
        """Report representative shape/dtype and the number of logical images."""
        try:
            h5 = _open_h5(file_path)
        except OSError as exc:
            if file_path.suffix.lower() == ".ims":
                raise RuntimeError(
                    "This .ims file is not HDF5-backed; legacy Imaris 2.7/3 files "
                    "are not supported by the native h5 loader"
                ) from exc
            raise
        with h5:
            if _is_imaris_file(h5):
                grid = _imaris_dataset_grid(h5)
                dset = h5[grid[0][0]]
                source_shape = tuple(int(size) for size in dset.shape)
                zyx_shape = source_shape if len(source_shape) == 3 else (1, *source_shape)
                return FileInfo(
                    shape=(len(grid), len(grid[0]), *zyx_shape),
                    dtype=dset.dtype,
                    dim_order=_IMARIS_DIM_ORDER,
                    n_images=1,
                )

            paths, is_bdv = _require_dataset_paths(h5, file_path)
            root_attrs = _read_attrs(h5)
            largest = max(paths, key=lambda p: h5[p].size * h5[p].dtype.itemsize)
            dset = h5[largest]
            dim_order = _dim_order_for(dset, {**root_attrs, **_read_attrs(dset)}, is_bdv)
            return FileInfo(
                shape=tuple(int(s) for s in dset.shape),
                dtype=dset.dtype,
                dim_order=dim_order,
                n_images=len(paths),
            )

    def load(self, file_path: Path) -> Record:
        with _open_h5(file_path) as h5:
            if _is_imaris_file(h5):
                return _build_imaris_record(file_path, h5)
            paths, is_bdv = _require_dataset_paths(h5, file_path)
            return _build_record(file_path, h5, paths[0], is_bdv, _read_attrs(h5))

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        with _open_h5(file_path) as h5:
            if _is_imaris_file(h5):
                if start <= 0 < stop:
                    yield "0", _build_imaris_record(file_path, h5)
                return
            paths, is_bdv = _require_dataset_paths(h5, file_path)
            root_attrs = _read_attrs(h5)
            for dataset_path in paths[start:stop]:
                try:
                    record = _build_record(file_path, h5, dataset_path, is_bdv, root_attrs)
                except Exception as exc:
                    logger.exception(
                        "H5Loader: failed to read dataset '%s' in '%s': %s",
                        dataset_path, file_path.name, exc,
                    )
                    continue
                yield _child_id(dataset_path, is_bdv), record
