import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import bioio_imageio
import numpy as np
import polars as pl
from bioio import BioImage
from bioio_base.exceptions import UnsupportedFileFormatError

from pixel_patrol_base.core.record import record_from
from pixel_patrol_loader_bio.plugins.loaders._utils import is_zarr_store

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
            metadata[f"pixel_size_{ax}"] = getattr(img.physical_pixel_sizes, ax, None)

    if hasattr(img, "channel_names"):
        metadata["channel_names"] = [str(c) for c in img.channel_names]

    if hasattr(img, "dtype"):
        metadata["dtype"] = str(img.dtype)

    if hasattr(img, "shape"):
        metadata["shape"] = np.array(img.shape)
        metadata["ndim"] = len(img.shape)
        metadata["num_pixels"] = math.prod(img.shape)

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

    SUPPORTED_EXTENSIONS: Set[str] = {"czi", "tif", "tiff", "ome.tif", "nd2", "lif", "jpg", "jpeg", "png", "bmp", "ome.zarr"}

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

    FOLDER_EXTENSIONS: Set[str] = {"zarr", "ome.zarr"}

    def is_folder_supported(self, path: Path) -> bool:
        return is_zarr_store(path)

    def load(self, source: str):
        img = _load_bioio_image(Path(source))
        if img is None:
            raise UnsupportedFileFormatError(self.NAME, path=source)

        meta = _extract_metadata(img)
        # dask-backed array; Record encapsulates axes/capabilities from meta["dim_order"]
        return record_from(img.dask_data, meta, kind="intensity")
