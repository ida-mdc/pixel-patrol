import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import bioio_imageio
import bioio_ome_tiff
import bioio_tifffile
import fsspec
import numpy as np
import dask.array as da
import polars as pl
import tifffile
from bioio import BioImage
from bioio_base.exceptions import UnsupportedFileFormatError

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.loader_schema import (
    RASTER_IMAGE_LOADER_SCHEMA,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
)
from pixel_patrol_base.core.record import record_from, Record
from pixel_patrol_base.core.uri import is_remote_uri
from pixel_patrol_loader_bio.plugins.loaders._utils import is_zarr_store

logger = logging.getLogger(__name__)


def _extract_metadata(img: Any) -> Dict[str, Any]:
    """
    Extract metadata from a BioImage-like object into a flat dict.
    """
    metadata: Dict[str, Any] = {}

    # Dim order and per-dimension sizes (e.g., size_X, size_Y, size_Z, size_C, size_T)
    dim_order = getattr(getattr(img, 'dims', None), 'order', '')
    metadata["dim_order"] = dim_order
    for letter in dim_order:
        dim_size= getattr(img.dims, letter, None)
        if not dim_size:
            dim_size = 1
        metadata[f"size_{letter}"] = int(dim_size)

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
        if metadata.get(f"size_{ax}", None) == 1:
            metadata.pop(f"size_{ax}", None)

    return metadata


_TIFF_EXTENSIONS = {".tif", ".tiff"}
# Cloud object stores are read anonymously (public buckets); http(s) take no anon flag.
_ANON_SCHEMES = ("s3://", "gs://", "gcs://", "az://", "abfs://")


def _is_ome_tiff(file_path) -> bool:
    """Whether a TIFF is an OME-TIFF. Reads the header over fsspec for remote URIs
    so remote OME-TIFFs are detected (and routed to the OME reader) too."""
    try:
        if is_remote_uri(str(file_path)):
            src = str(file_path)
            fs_kwargs = {"anon": True} if src.startswith(_ANON_SCHEMES) else {}
            with fsspec.open(src, **fs_kwargs) as handle, tifffile.TiffFile(handle) as tif:
                return tif.is_ome
        with tifffile.TiffFile(file_path) as tif:
            return tif.is_ome
    except Exception:
        return False


def _open_target(file_path) -> Tuple[Any, Dict[str, Any]]:
    """Return (source, extra BioImage kwargs) for a local or remote path.

    Remote URIs are passed to bioio as strings with fsspec kwargs (anonymous for
    public cloud buckets), so the file is streamed via fsspec. Local paths become
    Path objects with no extra kwargs - exactly as before.
    """
    if is_remote_uri(str(file_path)):
        source = str(file_path)
        fs_kwargs = {"anon": True} if source.startswith(_ANON_SCHEMES) else {}
        return source, {"fs_kwargs": fs_kwargs}
    return Path(file_path), {}


def _load_bioio_image(file_path) -> Optional[BioImage]:
    """
    Try BioImage, then fall back to imageio reader; return None if both fail.
    Handles local paths and remote object-store URIs (s3://, gs://, https://, ...).
    """
    source, open_kwargs = _open_target(file_path)
    try:
        if str(source).lower().endswith(tuple(_TIFF_EXTENSIONS)):
            reader = bioio_ome_tiff.Reader if _is_ome_tiff(source) else bioio_tifffile.Reader
            return BioImage(source, reader=reader, **open_kwargs)
        return BioImage(source, **open_kwargs)
    except UnsupportedFileFormatError:
        try:
            return BioImage(source, reader=bioio_imageio.Reader, **open_kwargs)
        except Exception as e:
            logger.warning(f"Could not load '{source}' with BioImage (imageio fallback): {e}")
            return None
    except Exception as e:
        logger.warning(f"Could not load '{source}' with BioImage: {e}")
        return None

class BioIoLoader:
    """
    Loader that produces an record from BioIO/BioImage.
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
