import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import bioio_imageio
import bioio_ome_tiff
import bioio_tifffile
import numpy as np
import polars as pl
import tifffile
from bioio import BioImage
from bioio.ome_utils import generate_ome_channel_id
from bioio_base.dimensions import DimensionNames, Dimensions
from bioio_base.exceptions import UnsupportedFileFormatError
from bioio_base.reader import Reader

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.loader_schema import (
    RASTER_IMAGE_LOADER_SCHEMA,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
)
from pixel_patrol_base.core.record import record_from, Record
from pixel_patrol_loader_bio.plugins.loaders._utils import is_zarr_store

logger = logging.getLogger(__name__)


def _extract_metadata(img: Any, data: Any) -> Dict[str, Any]:
    """
    Extract metadata from a bioio reader into a flat dict; dims and shape come from its xarray `data`.
    """
    metadata: Dict[str, Any] = {}

    # Dim order and per-dimension sizes (e.g., size_X, size_Y, size_Z, size_C, size_T)
    dims = Dimensions(dims="".join(data.dims), shape=data.shape)
    dim_order = dims.order
    metadata["dim_order"] = dim_order
    for letter in dim_order:
        dim_size= getattr(dims, letter, None)
        if not dim_size:
            dim_size = 1
        metadata[f"size_{letter}"] = int(dim_size)

    metadata["dim_names"] = list(data.dims)

    if hasattr(img, "physical_pixel_sizes"):
        for ax in ("X", "Y", "Z", "T"):
            val = getattr(img.physical_pixel_sizes, ax, None)
            if val is not None:
                metadata[f"pixel_size_{ax}"] = val

    if hasattr(img, "channel_names"):
        # Readers without channel names get the id BioImage would give them.
        channel_names = img.channel_names or [generate_ome_channel_id(image_id=img.current_scene, channel_id=0)]
        metadata["channel_names"] = [str(c) for c in channel_names]

    if hasattr(img, "dtype"):
        metadata["dtype"] = str(img.dtype)

    metadata["shape"] = np.array(data.shape)
    metadata["ndim"] = len(data.shape)
    metadata["num_pixels"] = math.prod(data.shape)

    return metadata


_TIFF_EXTENSIONS = {".tif", ".tiff"}


def _is_ome_tiff(file_path: Path) -> bool:
    try:
        with tifffile.TiffFile(file_path) as tif:
            return tif.is_ome
    except Exception:
        return False


def _load_bioio_image(file_path: Path) -> Optional[Reader]:
    """
    Open the reader BioImage would pick, then fall back to imageio reader; return None if both fail.
    """
    try:
        file_path = Path(file_path)
        if file_path.suffix.lower() in _TIFF_EXTENSIONS:
            reader = bioio_ome_tiff.Reader if _is_ome_tiff(file_path) else bioio_tifffile.Reader
            return reader(file_path)
        return BioImage.determine_plugin(file_path).metadata.get_reader()(file_path)
    except UnsupportedFileFormatError:
        try:
            return bioio_imageio.Reader(file_path)
        except Exception as e:
            logger.warning(f"Could not load '{file_path}' with bioio (imageio fallback): {e}")
            return None
    except Exception as e:
        logger.warning(f"Could not load '{file_path}' with bioio: {e}")
        return None


def _scene_xarray(img: Reader) -> Any:
    """
    The current scene as an xarray in the reader's dims, with mosaic tiles (M) stitched.

    BioImage would reshape it to TCZYX and silently keep only index 0 of any
    other dimension, e.g. the views (V) and illuminations (I) of a light-sheet CZI.
    """
    if DimensionNames.MosaicTile in img.dims.order:
        return img.mosaic_xarray_dask_data
    return img.xarray_dask_data

class BioIoLoader:
    """
    Loader that produces an record from a BioIO reader.
    Protocol: single `load()` method returning an Record.
    """

    NAME = "bioio"
    DESCRIPTION = "Opens a wide range of microscopy and standard image formats via BioIO, extracting pixel data and image metadata (dimensions, channels, pixel sizes)."

    SUPPORTED_EXTENSIONS: Set[str] = {"czi", "tif", "tiff", "ome.tif", "nd2", "lif", "jpg", "jpeg", "png", "bmp", "ome.zarr", "zarr"}

    OUTPUT_SCHEMA: Dict[str, Any] = dict(RASTER_IMAGE_LOADER_SCHEMA)
    OUTPUT_SCHEMA_PATTERNS: List[tuple[str, Any]] = list(RASTER_IMAGE_LOADER_SCHEMA_PATTERNS)

    FOLDER_EXTENSIONS:    Set[str] = {"zarr", "ome.zarr"}
    CONTAINER_EXTENSIONS: Set[str] = {"czi", "nd2", "lif", "tif", "tiff"}

    def is_folder_supported(self, path: Path) -> bool:
        return is_zarr_store(path)

    def read_header(self, file_path: Path) -> FileInfo:
        """Read file header; return shape/dtype/dim_order of the largest of the first few scenes, plus total scene count."""
        img = _load_bioio_image(file_path)
        if img is None:
            raise UnsupportedFileFormatError(self.NAME, path=str(file_path))
        scenes = list(img.scenes) if hasattr(img, "scenes") else [None]
        n_images = len(scenes)
        best_nbytes = -1
        shape = dtype = dim_order = None
        for scene in scenes[:min(3, n_images)]:
            if scene is not None:
                img.set_scene(scene)
            meta = _extract_metadata(img, _scene_xarray(img))
            candidate_shape = tuple(int(x) for x in meta["shape"])
            candidate_dtype = np.dtype(meta.get("dtype", "float32"))
            nbytes = int(np.prod(candidate_shape)) * candidate_dtype.itemsize
            if nbytes > best_nbytes:
                best_nbytes = nbytes
                shape, dtype, dim_order = candidate_shape, candidate_dtype, meta["dim_order"]
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
            child_id = str(scene) if scene is not None else "0"
            try:
                if scene is not None:
                    img.set_scene(scene)
                yield child_id, self._build_record(img)
            except Exception as e:
                logger.warning("BioIoLoader: failed to read scene %s in '%s': %s", child_id, file_path, e)
                yield child_id, None

    @staticmethod
    def _build_record(img: Reader) -> Record:
        """Extract metadata and build a Record."""
        if hasattr(img, "set_resolution_level"):
            img.set_resolution_level(0)
        data = _scene_xarray(img)
        meta = _extract_metadata(img, data)
        return record_from(data.data, meta, kind="intensity")
