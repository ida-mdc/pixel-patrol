import logging
import math
from pathlib import Path
from typing import Any, Tuple, Dict, Callable, List, Optional

import bioio_imageio
import dask.array as da
import numpy as np
import polars as pl
from bioio import BioImage
from bioio_base.exceptions import UnsupportedFileFormatError

from pixel_patrol_base.core.loader_interface import PixelPatrolLoader

logger = logging.getLogger(__name__)

def _mapping_for_bioimage_metadata_by_column_name() -> Dict[str, Callable[[BioImage], Dict]]:
    """
    Maps requested metadata fields to functions that extract them from a BioImage object.
    These functions return a dictionary with the extracted key-value pair.
    """
    return {
        "n_images": lambda img: {"n_images": len(img.scenes) if hasattr(img, 'scenes') else 1},
        "pixel_size_X": lambda img: {"pixel_size_X": getattr(img.physical_pixel_sizes, 'X', None)},
        "pixel_size_Y": lambda img: {"pixel_size_Y": getattr(img.physical_pixel_sizes, 'Y', None)},
        "pixel_size_Z": lambda img: {"pixel_size_Z": getattr(img.physical_pixel_sizes, 'Z', None)},
        "pixel_size_T": lambda img: {"pixel_size_T": getattr(img.physical_pixel_sizes, 'T', None)},
        "channel_names": lambda img: {"channel_names": img.channel_names},
        "dim_order": lambda img: {"dim_order": img.dims.order},
        "dtype": lambda img: {"dtype": str(img.dtype)},
        # "ome_metadata": lambda img: {"ome_metadata": img.ome_metadata},
    }

def _extract_metadata_from_mapping(img: Any, mapping: Dict[str, Any]):
    """
    Extracts metadata using a given mapping, handling individual column failures.
    """
    metadata: dict[str, Any] = {}
    metadata["dim_order"] = img.dims.order
    for letter in metadata["dim_order"]:
        dim_val = getattr(img.dims, letter, None)
        if not dim_val:
            dim_val = 1
        metadata[letter + "_size"] = int(dim_val)
    metadata["n_images"] = len(img.scenes) if hasattr(img, 'scenes') else 1
    if hasattr(img, 'physical_pixel_sizes'):
        metadata["pixel_size_X"] = getattr(img.physical_pixel_sizes, 'X', None)
        metadata["pixel_size_Y"] = getattr(img.physical_pixel_sizes, 'Y', None)
        metadata["pixel_size_Z"] = getattr(img.physical_pixel_sizes, 'Z', None)
        metadata["pixel_size_T"] = getattr(img.physical_pixel_sizes, 'T', None)
    if hasattr(img, 'channel_names'):
        metadata["channel_names"] = img.channel_names
    if hasattr(img, 'dtype'):
        metadata["dtype"] = str(img.dtype)
    if hasattr(img, 'shape'):
        metadata["shape"] = np.array(img.shape)
        metadata["ndim"] = len(img.shape)
        metadata["num_pixels"] = math.prod(img.shape)
    # "ome_metadata": lambda img: {"ome_metadata": img.ome_metadata},
    return metadata


def _load_bioio_image(file_path: Path) -> Optional[BioImage]:
    """Helper to load an image, returning the image object and its type."""
    try:
        img = BioImage(file_path)

    except UnsupportedFileFormatError:
        try:
            img = BioImage(file_path, reader=bioio_imageio.Reader)
        except Exception as e:
            logger.warning(f"Could not load '{file_path}' with BioImage: {e}")
            return None
    except Exception as e:
        logger.warning(f"Could not load '{file_path}' with BioImage: {e}")
        return None

    logger.info(f"Successfully loaded '{file_path}' with BioImage.")
    return img


class BioIoLoader(PixelPatrolLoader):

    @staticmethod
    def id() -> str:
        return "bioio"

    def read_metadata(self, path: Path) -> Optional[dict]:
        img = _load_bioio_image(path)
        if img:
            bioimage_mapping = _mapping_for_bioimage_metadata_by_column_name()
            return _extract_metadata_from_mapping(img, bioimage_mapping)
        else:
            return None

    def read_metadata_and_data(self, path: Path) -> Tuple[Optional[dict], Optional[da.array]]:
        img = _load_bioio_image(path)
        if img:
            bioimage_mapping = _mapping_for_bioimage_metadata_by_column_name()
            props = _extract_metadata_from_mapping(img, bioimage_mapping)
            return props, img.dask_data
        else:
            return None, None

    def get_specification(self) -> Dict[str, Any]:
        """
        Defines the expected Polars data types for the output of this processor.
        """
        return {
            'dim_order': str,
            'n_images': int,
            'num_pixels': int,
            'shape': pl.Array,
            'ndim': int,
            'channel_names': list,
            'dtype': str,
        }

    def get_dynamic_specification_patterns(self) -> List[Tuple[str, Any]]:
        return [
            (r'^pixel_size_[a-zA-Z]$', float),
            (r'^[a-zA-Z]_size$', int),
        ]