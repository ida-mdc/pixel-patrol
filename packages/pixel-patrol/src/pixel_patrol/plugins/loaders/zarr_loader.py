import logging
import string
from pathlib import Path
from typing import Any, Dict, Optional, Set

import dask.array as da
import zarr

from pixel_patrol.plugins.loaders.bioio_loader import BioIoLoader
from pixel_patrol_base.core.artifact import artifact_from
from pixel_patrol.plugins.loaders._utils import is_zarr_store

logger = logging.getLogger(__name__)


def _load_zarr_array(path: Path) -> Optional[da.Array]:
    """Load a Zarr array as a Dask array, or return None on failure."""
    try:
        data = da.from_zarr(str(path))
        logger.info(f"Successfully loaded '{path}' as a Zarr array.")
        return data
    except Exception as e:
        logger.warning(f"Could not load '{path}' as a Zarr array: {e}")
        return None


def _infer_dim_order(shape: tuple[int, ...]) -> str:
    """
    Infer a simple dim order assuming the last two dims are YX.
    Preceding dims are assigned A,B,C,... in order.
    """
    n = len(shape)
    if n <= 2:
        return ("YX"[-n:])  # n==0 -> "", n==1 -> "X", n==2 -> "YX"
    return string.ascii_uppercase[: n - 2] + "YX"


def _extract_zarr_metadata(arr: da.Array, path: Path) -> Dict[str, Any]:
    """
    Build a flat metadata dict from a zarr-backed dask array and its container.
    """
    meta: Dict[str, Any] = {}
    meta["shape"] = tuple(int(s) for s in arr.shape)
    meta["dtype"] = str(arr.dtype)
    meta["n_dimensions"] = arr.ndim

    # chunksize may be missing for irregular chunks; fall back to .chunks
    chunks = getattr(arr, "chunksize", None)
    meta["chunks"] = chunks if chunks is not None else arr.chunks

    # Infer dim order and per-dimension sizes
    dim_order = _infer_dim_order(arr.shape)
    meta["dim_order"] = dim_order
    for i, name in enumerate(dim_order):
        meta[f"{name}_size"] = int(arr.shape[i])

    # Zarr generally represents a single "image"
    meta["n_images"] = 1
    meta["channel_names"] = []

    # Read zarr attributes (array or group)
    try:
        root = zarr.open(str(path), mode="r")
        if isinstance(root, zarr.Array):
            meta["zarr_attributes"] = dict(root.attrs)
        elif isinstance(root, zarr.Group):
            attrs = dict(root.attrs)
            # Common conventions: arrays named "data" or "0" (multiscales)
            for name, item in list(root.arrays()):
                if name in ("data", "0"):
                    attrs.update(dict(item.attrs))
                    break
            meta["zarr_attributes"] = attrs
    except Exception as e:
        logger.warning(f"Could not read Zarr attributes from '{path}': {e}")

    return meta


class ZarrLoader:
    """
    Loader that produces an Artifact from Zarr (or OME-Zarr via BioIO fallback).
    Protocol: single `load()` returning an Artifact.
    """

    NAME = "zarr"

    SUPPORTED_EXTENSIONS: Set[str] = {"zarr"}

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "dim_order": str,
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

        try:
            bio_art = BioIoLoader().load(str(path))
            return bio_art
        except Exception:
            # fall back to raw zarr loading
            pass

        arr = _load_zarr_array(path)
        if arr is None:
            raise RuntimeError(f"Cannot read Zarr array at: {source}")

        meta = _extract_zarr_metadata(arr, path)
        return artifact_from(arr, meta, kind="intensity")
