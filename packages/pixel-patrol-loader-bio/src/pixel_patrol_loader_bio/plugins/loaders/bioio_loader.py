import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import bioio_imageio
import numpy as np
import dask.array as da
import polars as pl
from bioio import BioImage
from bioio_base.exceptions import UnsupportedFileFormatError

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.record import record_from, Record
import zarr as _zarr


def is_zarr_store(path: Path) -> bool:
    try:
        store_obj = _zarr.open(store=str(path.absolute()), mode='r')
        if isinstance(store_obj, _zarr.Group):
            return bool(store_obj.attrs)
        return True
    except Exception:
        return False

logger = logging.getLogger(__name__)


def _extract_metadata(img: Any) -> Dict[str, Any]:
    """
    Extract metadata from a BioImage-like object into a flat dict.
    """
    metadata: Dict[str, Any] = {}

    # Dim order and per-dimension sizes (e.g., X_size, Y_size, Z_size, C_size, T_size)
    dim_order = getattr(getattr(img, 'dims', None), 'order', '')
    metadata["dim_order"] = dim_order
    for letter in dim_order:
        dim_size= getattr(img.dims, letter, None)
        if not dim_size:
            dim_size = 1
        metadata[f"{letter}_size"] = int(dim_size)

    dim_names = getattr(getattr(img, 'dims', None), 'names', None)
    if isinstance(dim_names, (list, tuple)) and all(isinstance(x, str) for x in dim_names):
        metadata["dim_names"] = list(dim_names)

    metadata["n_images"] = len(img.scenes) if hasattr(img, "scenes") else 1

    if hasattr(img, "physical_pixel_sizes"):
        for ax in ("X", "Y", "Z", "T"):
            val = getattr(img.physical_pixel_sizes, ax, None)
            if val is not None:
                metadata[f"pixel_size_{ax}"] = val

    if hasattr(img, "channel_names"):
        metadata["channel_names"] = [str(c) for c in img.channel_names]

    if hasattr(img, "dtype"):
        metadata["dtype"] = str(img.dtype)

    if hasattr(img, "shape"):
        metadata["shape"] = np.array(img.shape)
        metadata["ndim"] = len(img.shape)
        metadata["num_pixels"] = math.prod(img.shape)

    return metadata


def normalize_metadata(metadata):
    dim_order = metadata["dim_order"]
    keep = [i for i, s in enumerate(metadata["shape"]) if s != 1]
    metadata["shape"] = [metadata["shape"][i] for i in keep]
    metadata["ndim"] = len(metadata["shape"])
    metadata["dim_order"] = "".join(dim_order[i] for i in keep)
    if "dim_names" in metadata:
        metadata["dim_names"] = [metadata["dim_names"][i] for i in keep]
    for ax in list(dim_order):
        if metadata.get(f"{ax}_size", None) == 1:
            metadata.pop(f"{ax}_size", None)

    return metadata


def _load_bioio_image(file_path: Path) -> Optional[BioImage]:
    """
    Try BioImage, then fall back to imageio reader; return None if both fail.
    """
    try:
        return BioImage(file_path)
    except UnsupportedFileFormatError:
        try:
            return BioImage(file_path, reader=bioio_imageio.Reader)
        except Exception as e:
            logger.warning(f"Could not load '{file_path}' with BioImage (imageio fallback): {e}")
            return None
    except Exception as e:
        logger.warning(f"Could not load '{file_path}' with BioImage: {e}")
        return None

class BioIoLoader:
    """
    Loader that produces an record from BioIO/BioImage.
    Protocol: single `load()` method returning an Record.
    """

    NAME = "bioio"

    SUPPORTED_EXTENSIONS: Set[str] = {"czi", "tif", "tiff", "ome.tif", "nd2", "lif", "jpg", "jpeg", "png", "bmp", "ome.zarr", "zarr"}

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "dim_order": str,
        "dim_names": list,
        "n_images": int,
        "num_pixels": int,
        "shape": pl.Array,       # or use `list` if you prefer to avoid polars types here
        "ndim": int,
        "channel_names": list,   # could be list[str]
        "dtype": str,
    }

    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]] = [
        (r"^pixel_size_[A-Za-z]$", float),
        (r"^[A-Za-z]_size$", int),
    ]

    FOLDER_EXTENSIONS:    Set[str] = {"zarr", "ome.zarr"}
    CONTAINER_EXTENSIONS: Set[str] = {"czi", "nd2", "lif", "tif", "tiff"}

    def is_folder_supported(self, path: Path) -> bool:
        return is_zarr_store(path)

    def read_header(self, file_path: Path) -> FileInfo:
        """Read file header; return shape/dtype/dim_order of the first scene plus total scene count."""
        img = _load_bioio_image(file_path)
        if img is None:
            raise UnsupportedFileFormatError(self.NAME, path=str(file_path))
        n_images = len(img.scenes) if hasattr(img, "scenes") else 1
        meta = _extract_metadata(img)
        meta = normalize_metadata(meta)
        shape = tuple(int(x) for x in meta["shape"])
        dim_order = tuple(meta["dim_order"])
        dtype = np.dtype(meta.get("dtype", "float32"))
        return FileInfo(shape=shape, dtype=dtype, dim_order=dim_order, n_images=n_images)

    def load(self, file_path: Path) -> Record:
        """Load a single-image (or first-scene) file; return a Record."""
        img = _load_bioio_image(file_path)
        if img is None:
            raise UnsupportedFileFormatError(self.NAME, path=str(file_path))
        return self._build_record(img)

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        """Yield (scene_name, Record) for scenes [start, stop) in a multi-scene file."""
        img = _load_bioio_image(file_path)
        if img is None:
            raise UnsupportedFileFormatError(self.NAME, path=str(file_path))
        scenes = list(img.scenes) if hasattr(img, "scenes") else [None]
        for scene in scenes[start:stop]:
            if scene is not None:
                img.set_scene(scene)
            yield str(scene) if scene is not None else "0", self._build_record(img)

    @staticmethod
    def _build_record(img: BioImage) -> Record:
        """Extract metadata, squeeze singleton dims, and build a Record."""
        if hasattr(img, "set_resolution_level"):
            img.set_resolution_level(0)
        meta = _extract_metadata(img)
        meta = normalize_metadata(meta)
        data = da.squeeze(img.dask_data)
        return record_from(data, meta, kind="intensity")
