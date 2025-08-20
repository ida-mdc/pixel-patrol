import logging
import string
from pathlib import Path
from typing import Any, Tuple, Dict, List, Optional

import dask.array as da
import zarr

from pixel_patrol_base.core.loader_interface import PixelPatrolLoader
from pixel_patrol_base.core.loaders.bioio_loader import BioIoLoader

logger = logging.getLogger(__name__)


class ZarrLoader(PixelPatrolLoader):
    """
    A PixelPatrolLoader implementation for loading N-dimensional Zarr arrays.
    It assumes the last dimension is 'x' and the second to last is 'z'.
    """

    @staticmethod
    def id() -> str:
        return "zarr"

    def _load_zarr_array(self, path: Path) -> Optional[da.Array]:
        """Helper to load a Zarr array as a Dask array."""
        try:
            # Use dask.array.from_zarr to get a Dask array directly
            data = da.from_zarr(str(path))
            logger.info(f"Successfully loaded '{path}' as a Zarr array.")
            return data
        except Exception as e:
            logger.warning(f"Could not load '{path}' as a Zarr array: {e}")
            return None

    def read_metadata(self, path: Path) -> Dict[str, Any]:
        """
        Reads metadata from a Zarr array.
        Assumes the last dimension is 'x' and the second to last is 'z'.
        """

        try:
            # try bioio first in case of ome zarr
            metadata = BioIoLoader().read_metadata(path)
            logger.error("metadata from bioio: " + metadata)
            if metadata is {}:
                metadata, zarr_array = self._load_parse_zarr(path)
                return metadata
        except Exception:
            logger.error("exception in bioio")
            metadata, zarr_array = self._load_parse_zarr(path)
            return metadata

    def _load_parse_zarr(self, path: Path):
        zarr_array = self._load_zarr_array(path)
        if zarr_array is None:
            logger.error("cant read zarr array")
            return {}

        metadata: Dict[str, Any] = {
            "shape": zarr_array.shape,
            "dtype": str(zarr_array.dtype),
            "n_dimensions": zarr_array.ndim,
            "chunks": zarr_array.chunksize,
            "dim_order": string.ascii_uppercase[:len(zarr_array.shape) - 2] + "YX"
        }

        # Basic array properties

        # Inferring dim_order based on the 'x' and 'z' assumption
        for i, name in enumerate(metadata["dim_order"]):
            metadata[name + "_size"] = zarr_array.shape[i]

        # Zarr arrays don't have inherent concepts of 'scenes', 'channel_names', or 'physical_pixel_sizes'
        # unless stored as custom attributes. We can add a placeholder or rely on user-defined attributes.
        metadata["n_images"] = 1  # Zarr typically represents a single array/image
        metadata["channel_names"] = []  # No direct equivalent in Zarr by default

        # You can also include any user-defined attributes stored in the Zarr array
        try:
            root_group = zarr.open(str(path), mode='r')
            if isinstance(root_group, zarr.Array):
                metadata["zarr_attributes"] = dict(root_group.attrs)
            elif isinstance(root_group, zarr.Group):
                # If it's a group, you might need to specify which array to load,
                # or iterate through arrays in the group.
                # For simplicity, this example assumes a direct array at the path.
                # If your Zarrs are always groups, you'll need to extend this.
                metadata["zarr_attributes"] = dict(root_group.attrs)  # Group attributes
                for name, item in root_group.arrays():
                    if name == "data":  # Common convention for the actual data array
                        metadata["zarr_attributes"].update(dict(item.attrs))
                        break
        except Exception as e:
            logger.warning(f"Could not read Zarr attributes from '{path}': {e}")
        return metadata, zarr_array

    def read_metadata_and_data(self, path: Path) -> Tuple[Dict[str, Any], da.Array]:
        """
        Reads metadata and the Dask array data from a Zarr array.
        Assumes the last dimension is 'x' and the second to last is 'z'.
        """
        try:
            # try bioio first in case of ome zarr
            metadata = BioIoLoader().read_metadata(path)
            logger.error("metadata from bioio: " + metadata)
            if metadata is {}:
                return self._load_parse_zarr(path)
        except Exception:
            logger.error("exception in bioio")
            return self._load_parse_zarr(path)

    def get_specification(self) -> Dict[str, Any]:
        """
        Defines the expected Polars data types for the output of this processor.
        """
        return {
            'dim_order': str,
            'n_images': int,
            'channel_names': list,
            'dtype': str,
            'zarr_attributes': dict,
        }

    def get_dynamic_specification_patterns(self) -> List[Tuple[str, Any]]:
        return [
            (r'^[a-zA-Z]_size$', int)
        ]
