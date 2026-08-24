"""Pixel-Patrol loader for HDF5 files.

An HDF5 file is a container, not a single image: it can hold any number of
arrays at arbitrary paths. Every image-like dataset in the file is therefore
treated as one sub-image, so a file holding N of them reports ``n_images=N``
and streams them through ``load_range()``.

Two layouts are recognised:

* **BigDataViewer/BDV** - ``tXXXXX/sYY/<level>/cells`` hierarchies. Only
  resolution level 0 is read (the coarser levels are mipmaps of the same
  image), axes are ZYX, and voxel sizes come from the sibling ``.xml``.
* **plain HDF5** - every numeric dataset with at least two dimensions. Axis
  letters are taken from whichever axis attribute the writer left behind
  (``dim_order``, vigra/ilastik ``axistags``, HDF5 ``DIMENSION_LABELS``,
  xarray ``_ARRAY_DIMENSIONS``) and inferred as trailing YX otherwise.

Pixel data is returned lazily: the pipeline calls ``load()`` once per memory
chunk and slices the result, so eagerly reading a dataset would re-read the
whole thing for every chunk. Laziness is provided by ``_H5DatasetProxy``
rather than a live ``h5py.Dataset`` because loaders are shipped to worker
processes and must hold no open file handles.
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
_NUMERIC_DTYPE_KINDS = "uifb"

# Axis attributes written by common HDF5 producers, in order of preference.
_AXIS_ATTR_KEYS = ("dim_order", "DIMENSION_LABELS", "_ARRAY_DIMENSIONS", "axes")

_BDV_TIMEPOINT_RE = re.compile(r"^t\d+$")
_BDV_SETUP_RE = re.compile(r"^s\d+$")
_BDV_FULL_RESOLUTION = "0"
_BDV_CELLS = "cells"
_BDV_DIM_ORDER = "ZYX"


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------


def _open_h5(file_path: Path) -> h5py.File:
    """Open a file read-only, disabling HDF5 file locking where h5py allows it.

    Locking fails outright on many network filesystems, and several worker
    processes read the same file concurrently.
    """
    try:
        return h5py.File(str(file_path), "r", locking=False)
    except TypeError:  # h5py too old to know the locking kwarg
        return h5py.File(str(file_path), "r")


# ---------------------------------------------------------------------------
# Lazy pixel access
# ---------------------------------------------------------------------------


class _H5DatasetProxy:
    """Picklable, lazily-opening stand-in for one HDF5 dataset.

    Exposes the ``shape``/``dtype``/``__getitem__`` surface dask needs while
    holding only a path and a dataset name, so it survives being pickled to a
    worker process. Each block read opens the file, slices, and closes again.
    """

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
    """Wrap a dataset in a dask array whose blocks re-open the file on demand."""
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


def _is_bdv_file(h5: h5py.File) -> bool:
    """BDV writes one group per timepoint (``t00000``) at the file root."""
    return any(_BDV_TIMEPOINT_RE.match(key) for key in h5.keys())


def _bdv_dataset_paths(h5: h5py.File) -> List[str]:
    """Full-resolution ``cells`` datasets, ordered by timepoint then setup."""
    paths: List[str] = []
    for timepoint in sorted(k for k in h5.keys() if _BDV_TIMEPOINT_RE.match(k)):
        for setup in sorted(k for k in h5[timepoint].keys() if _BDV_SETUP_RE.match(k)):
            path = f"{timepoint}/{setup}/{_BDV_FULL_RESOLUTION}/{_BDV_CELLS}"
            if path in h5:
                paths.append(path)
            else:
                logger.warning("H5Loader: BDV setup '%s/%s' has no level-0 cells", timepoint, setup)
    return paths


def _plain_dataset_paths(h5: h5py.File) -> List[str]:
    """Every image-like dataset in the file, in stable (path-sorted) order."""
    paths: List[str] = []
    h5.visititems(lambda name, obj: paths.append(name) if _is_image_dataset(obj) else None)
    return sorted(paths)


def _dataset_paths(h5: h5py.File) -> List[str]:
    """The sub-images of this file, in the order load_range() indexes them."""
    return _bdv_dataset_paths(h5) if _is_bdv_file(h5) else _plain_dataset_paths(h5)


def _child_id(dataset_path: str, is_bdv: bool) -> str:
    """Stable identifier of a sub-image within the file."""
    if is_bdv:
        timepoint, setup, *_ = dataset_path.split("/")
        return f"{timepoint}/{setup}"
    return dataset_path


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------


def _as_text(value: Any) -> Optional[str]:
    """Decode an HDF5 string attribute, which may arrive as bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
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
    """Attributes of a group or dataset as a flat, table-safe dict."""
    return {str(key): _jsonable(value) for key, value in obj.attrs.items()}


# ---------------------------------------------------------------------------
# Axis labels
# ---------------------------------------------------------------------------


def _as_axis_letters(value: Any, ndim: int) -> Optional[str]:
    """Axis labels as one uppercase letter per axis, or None if unusable."""
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
    """vigra/ilastik store axes as JSON: ``{"axes": [{"key": "z"}, ...]}``."""
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
    """Axis letters from whichever axis attribute the writer left behind."""
    from_axistags = _dim_order_from_axistags(attrs.get("axistags"), ndim)
    if from_axistags:
        return from_axistags
    for key in _AXIS_ATTR_KEYS:
        letters = _as_axis_letters(attrs.get(key), ndim)
        if letters:
            return letters
    return None


def _dim_order_for(dset: h5py.Dataset, attrs: Dict[str, Any], is_bdv: bool) -> str:
    """BDV cells are always ZYX; otherwise trust attributes, then infer."""
    if is_bdv:
        return _BDV_DIM_ORDER
    return _dim_order_from_attrs(attrs, dset.ndim) or infer_dim_order(dset.ndim)


# ---------------------------------------------------------------------------
# BDV XML sidecar
# ---------------------------------------------------------------------------


def _parse_bdv_voxel_size(voxel: ET.Element) -> Dict[str, Any]:
    """``<voxelSize><unit/><size>dx dy dz</size></voxelSize>`` -> pixel_size_*."""
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
    """One ``<ViewSetup>`` -> (setup id, metadata) or None if it has no id."""
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
    """Per-setup metadata from the sibling BDV XML, keyed by setup id.

    The XML is where BDV keeps voxel sizes and channel names; the HDF5 itself
    only has pixels. A missing XML is normal, not an error - the file just
    carries no physical sizes then.

    Cached per worker process so one XML is parsed once, not once per
    load_range() call.
    """
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
    """Timepoint/setup labels plus that setup's metadata from the XML."""
    timepoint, setup, *_ = dataset_path.split("/")
    meta: Dict[str, Any] = {"bdv_timepoint": timepoint, "bdv_setup": setup}
    setup_id = setup[1:]
    if setup_id.isdigit():
        meta.update(_load_bdv_setup_meta(file_path).get(int(setup_id), {}))
    return meta


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _extract_metadata(dset: h5py.Dataset, dim_order: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Shape/dtype/axis metadata for one dataset, plus its raw HDF5 attributes."""
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
    return meta


def _build_record(file_path: Path, h5: h5py.File, dataset_path: str, is_bdv: bool) -> Record:
    """Build a Record with lazily-read pixels for one dataset in the file."""
    dset = h5[dataset_path]
    attrs = {**_read_attrs(h5), **_read_attrs(dset)}
    dim_order = _dim_order_for(dset, attrs, is_bdv)
    meta = _extract_metadata(dset, dim_order, attrs)
    if is_bdv:
        meta.update(_bdv_meta(file_path, dataset_path))
    return record_from(_as_dask(file_path, dset), meta, kind="intensity")


def _require_dataset_paths(h5: h5py.File, file_path: Path) -> List[str]:
    paths = _dataset_paths(h5)
    if not paths:
        raise RuntimeError(f"H5Loader: no image-like dataset found in '{file_path}'")
    return paths


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class H5Loader:
    """Loader that produces one Record per image-like dataset in an HDF5 file."""

    NAME = "h5"
    DESCRIPTION = (
        "Loads HDF5 files, treating every image-like dataset in the file as a sub-image. "
        "Recognises BigDataViewer/BDV timepoint-setup hierarchies (level 0 only, voxel "
        "sizes from the sibling XML) and reads axis labels from the file's own attributes."
    )

    SUPPORTED_EXTENSIONS: Set[str] = {"h5", "hdf5"}

    # HDF5 is always a container: a file can hold many datasets, and its
    # gzip-compressed on-disk size badly understates the uncompressed array,
    # so read_header() must never be skipped for it.
    FOLDER_EXTENSIONS:    Set[str] = set()
    CONTAINER_EXTENSIONS: Set[str] = {"h5", "hdf5"}

    OUTPUT_SCHEMA: Dict[str, Any] = {
        **RASTER_IMAGE_LOADER_SCHEMA,
        "h5_dataset_path": str,
        "h5_attributes": dict,
        "bdv_timepoint": str,
        "bdv_setup": str,
        "bdv_setup_name": str,
        "pixel_size_unit": str,
    }

    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = {
        "h5_dataset_path": "Path of the dataset within the HDF5 file this row was read from.",
        "h5_attributes": "Raw key-value attributes of the dataset, merged over the file's root attributes.",
        "bdv_timepoint": "BigDataViewer timepoint group the sub-image belongs to (e.g. 't00000').",
        "bdv_setup": "BigDataViewer setup group the sub-image belongs to (e.g. 's00').",
        "bdv_setup_name": "Name of the BigDataViewer setup, from the sibling XML.",
        "pixel_size_unit": "Physical unit the pixel_size_* values are given in.",
    }

    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]] = list(RASTER_IMAGE_LOADER_SCHEMA_PATTERNS)

    def is_folder_supported(self, path: Path) -> bool:
        """HDF5 datasets are single files, never directories."""
        return False

    def read_header(self, file_path: Path) -> FileInfo:
        """Report the first dataset's shape/dtype and how many datasets there are."""
        with _open_h5(file_path) as h5:
            paths = _require_dataset_paths(h5, file_path)
            dset = h5[paths[0]]
            dim_order = _dim_order_for(dset, {**_read_attrs(h5), **_read_attrs(dset)}, _is_bdv_file(h5))
            return FileInfo(
                shape=tuple(int(s) for s in dset.shape),
                dtype=dset.dtype,
                dim_order=dim_order,
                n_images=len(paths),
            )

    def load(self, file_path: Path) -> Record:
        """Load the first dataset in the file as a Record (single-dataset case)."""
        with _open_h5(file_path) as h5:
            paths = _require_dataset_paths(h5, file_path)
            return _build_record(file_path, h5, paths[0], _is_bdv_file(h5))

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        """Yield (child_id, Record) for the datasets at indices [start, stop)."""
        with _open_h5(file_path) as h5:
            is_bdv = _is_bdv_file(h5)
            for dataset_path in _require_dataset_paths(h5, file_path)[start:stop]:
                try:
                    record = _build_record(file_path, h5, dataset_path, is_bdv)
                except Exception as exc:
                    logger.exception(
                        "H5Loader: failed to read dataset '%s' in '%s': %s",
                        dataset_path, file_path.name, exc,
                    )
                    continue
                yield _child_id(dataset_path, is_bdv), record
