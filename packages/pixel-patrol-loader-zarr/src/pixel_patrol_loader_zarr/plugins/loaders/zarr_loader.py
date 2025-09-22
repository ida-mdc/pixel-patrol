import logging
import string
from pathlib import Path
from typing import Any, Dict, Optional, Set

import dask.array as da
import zarr

from pixel_patrol_base.core.artifact import artifact_from

logger = logging.getLogger(__name__)



def is_zarr_store(path: Path) -> bool:
    """
    Robustly checks if a given path is a Zarr store (v2 or v3).

    This function uses the zarr library to attempt opening the store, which
    correctly handles both Zarr v2 and v3 specifications.

    Args:
        path: The pathlib.Path object to check.

    Returns:
        True if the path is a valid Zarr store, False otherwise.
    """
    try:
        store_obj = zarr.open(store=str(path.absolute()), mode='r')

        if isinstance(store_obj, zarr.Group):
            # A group is "processable" if it has any custom attributes.
            # A generic container group will have empty attrs.
            return bool(store_obj.attrs)

        return True

    except Exception as e:
        # Catches any error, indicating it's not a valid or accessible Zarr store.
        return False


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


def _infer_dim_order(shape: tuple[int, ...]) -> str:
    """
    Infer a simple dim order assuming the last two dims are YX.
    Preceding dims are assigned A,B,C,... in order.
    """
    n = len(shape)
    if n <= 2:
        return "YX"[-n:]  # n==0 -> "", n==1 -> "X", n==2 -> "YX"
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

    meta["dim_names"] = [f"dim{i + 1}" for i in range(len(dim_order))]

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
        return artifact_from(arr, meta, kind="intensity")
