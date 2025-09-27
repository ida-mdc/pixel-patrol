import logging
import string
from pathlib import Path
from typing import Any, Dict, Optional, Set, List, Mapping

import dask.array as da
import numpy as np
import zarr

from pixel_patrol_base.core.record import record_from
from pixel_patrol_loader_bio.plugins.loaders._utils import is_zarr_store

logger = logging.getLogger(__name__)


def _load_zarr_array(path: Path) -> Optional[da.Array]:
    try:
        # 1) Try as a direct array path
        return da.from_zarr(str(path))
    except Exception as e1:
        try:
            # 2) Try as a group with NGFF multiscales
            root = zarr.open(str(path), mode="r")
            candidates = []

            if isinstance(root, zarr.Group):
                attrs = dict(root.attrs)
                # NGFF: multiscales[0].datasets[*].path  (often "0")
                for d in attrs.get("multiscales", [{}])[0].get("datasets", []):
                    p = d.get("path")
                    if p:
                        candidates.append(p)

                # Common fallbacks
                candidates += ["0", "data"]

                # Single-array group: use that array’s name
                if not candidates:
                    arrays = list(root.arrays())
                    if len(arrays) == 1:
                        candidates.append(arrays[0][0])

                for comp in candidates:
                    try:
                        return da.from_zarr(str(path), component=comp)
                    except Exception:
                        pass

            # 4) Last resort: open with zarr and wrap with dask
            arr = zarr.open_array(str(path), mode="r")
            return da.from_array(arr, chunks=arr.chunks)
        except Exception as e2:
            logger.warning(
                f"Could not load '{path}' as a Zarr array (tried as array/group): {e1}; {e2}"
            )
            return None


def _infer_dim_order(n: int) -> str:
    """
    Infer a simple dim order assuming the last two dims are YX.
    Preceding dims are assigned A,B,C,... in order.
    """
    if n <= 2:
        return "YX"[-n:]  # n==0 -> "", n==1 -> "X", n==2 -> "YX"
    return string.ascii_uppercase[: n - 2] + "YX"


def _read_zarr_root_and_primary_attrs(path: Path) -> Dict[str, Any]:
    """
    Read merged attributes from the Zarr root AND the primary child array
    (first of "0" or "data" if present). Returns a flat dict.
    """
    attrs: Dict[str, Any] = {}
    try:
        root = zarr.open(str(path), mode="r")
        attrs.update(dict(getattr(root, "attrs", {}) or {}))
        try:
            if hasattr(root, "arrays"):
                for name, item in root.arrays():
                    if name in ("data", "0"):
                        attrs.update(dict(getattr(item, "attrs", {}) or {}))
                        break
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Could not read Zarr attributes from '{path}': {e}")
    return attrs


def _extract_ngff_dim_names(attrs: Mapping[str, Any], ndim: int) -> Optional[List[str]]:
    """
    Parse OME-NGFF axes from attrs -> return dim_names if available
    """
    if not isinstance(attrs, dict):
        return None

    ms = attrs.get("multiscales")
    if not (isinstance(ms, list) and ms):
        return None

    first = ms[0]
    if not isinstance(first, dict):
        return None

    axes = first.get("axes")
    if not isinstance(axes, list):
        return None

    if len(axes) != int(ndim):
        logger.warning("NGFF: axes length (%s) != array ndim (%s); ignoring axes", len(axes), ndim)
        return None

    names: List[str] = []
    for a in axes:
        if isinstance(a, str):
            n = a
        elif isinstance(a, dict) and isinstance(a.get("name"), str):
            n = a["name"]
        else:
            logger.warning("NGFF: malformed axis entry %r; ignoring axes", a)
            return None
        names.append(n)

    return names


def _extract_zarr_metadata(arr: da.Array, path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    ndim = int(getattr(arr, "ndim", len(getattr(arr, "shape", []) or [])))

    attrs = _read_zarr_root_and_primary_attrs(path)
    if attrs:
        meta["zarr_attributes"] = attrs

    dim_names = _extract_ngff_dim_names(attrs, ndim)

    if dim_names and all(isinstance(n, str) and len(n) == 1 for n in dim_names):
        dim_order = "".join(n.upper() for n in dim_names)
    else:
        dim_order = _infer_dim_order(ndim)

    if not dim_names:
        dim_names = [f"dim{c}" for c in dim_order]

    meta["dim_order"] = dim_order
    meta["dim_names"] = dim_names

    meta["shape"] = np.array(arr.shape, dtype=int)
    meta["dtype"] = str(arr.dtype)
    meta["ndim"] = arr.ndim
    meta["num_pixels"] = int(np.prod(arr.shape))
    chunks = getattr(arr, "chunksize", None)
    meta["chunks"] = chunks if chunks is not None else arr.chunks

    for i, ax in enumerate(dim_order):
        meta[f"{ax}_size"] = int(arr.shape[i])

    return meta


class ZarrLoader:
    """
    Loader that produces an Record from Zarr.
    Protocol: single `load()` returning an Record.
    """

    NAME = "zarr"

    SUPPORTED_EXTENSIONS: Set[str] = {"zarr"}

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "dim_order": str,
        "dim_names": list,
        "n_images": int,
        "channel_names": list,
        "dtype": str,
        "zarr_attributes": dict,
    }

    OUTPUT_SCHEMA_PATTERNS = [
        (r"^[A-Za-z]_size$", int),
    ]

    FOLDER_EXTENSIONS: Set[str] = {"zarr", "ome.zarr"}

    def is_folder_supported(self, path: Path) -> bool:
        return is_zarr_store(path)

    def load(self, source: str):
        path = Path(source)

        arr = _load_zarr_array(path)
        if arr is None:
            raise RuntimeError(f"Cannot read Zarr array at: {source}")

        meta = _extract_zarr_metadata(arr, path)
        return record_from(arr, meta, kind="intensity")
