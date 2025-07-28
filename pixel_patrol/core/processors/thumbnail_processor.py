import logging
from typing import Dict, Any

import cv2
import dask.array as da
import numpy as np
import polars as pl

from pixel_patrol.config import SPRITE_SIZE
from pixel_patrol.core.processor_interface import PixelPatrolProcessor

logger = logging.getLogger(__name__)

# TODO: decide how we create thumbnail for images with S (probably call to gray) and C
def _generate_thumbnail(da_array: da.array, dim_order: str) -> np.array:
    """
    Internal function to generate a thumbnail (NumPy array) without direct dict wrapping.
    """
    if da_array is None or da_array.size == 0:
        return np.array([])

    arr_to_process = da_array.copy()

    # ✨ FIX: Cast boolean array to unsigned 8-bit integer type for math operations.
    if arr_to_process.dtype == bool:
        arr_to_process = arr_to_process.astype(np.uint8)

    current_dim_order = dim_order

    i = 0
    while arr_to_process.ndim > 2 and i < len(current_dim_order):
        dim = current_dim_order[i]
        if dim not in ["X", "Y", "C"]:  # Reduce non-spatial, non-channel dimensions
            center_index = arr_to_process.shape[i] // 2
            arr_to_process = da.take(arr_to_process, indices=center_index, axis=i)
            current_dim_order = current_dim_order.replace(dim, "")
        else:
            i += 1

    if arr_to_process.ndim > 2:
        logger.warning(f"Thumbnail: Array still multi-dimensional after reduction ({arr_to_process.ndim}D). "
                       f"Taking mean along remaining non-XY dimensions.")
        while arr_to_process.ndim > 2:
            arr_to_process = da.mean(arr_to_process, axis=0)

    min_val = da.min(arr_to_process)
    max_val = da.max(arr_to_process)
    logger.info(min_val)
    logger.info(max_val)

    # ✨ FIX: Handle constant arrays (e.g., all True or all False) correctly.
    if da.all(min_val == max_val).compute():
        # If array is constant, fill with black (0) or white (255).
        fill_value = 255 if float(max_val.compute()) > 0 else 0
        normalized_array = da.full_like(arr_to_process, fill_value=fill_value, dtype=da.uint8)
    else:
        # Perform normalization for arrays with varying values.
        normalized_array = (arr_to_process - min_val) / (max_val - min_val) * 255
        normalized_array = da.clip(normalized_array, 0, 255).astype(da.uint8)

    try:
        if normalized_array.ndim == 3 and normalized_array.shape[0] == 1:
            normalized_array = da.squeeze(normalized_array, axis=0)

        thumbnail = cv2.resize(
            normalized_array.compute(),
            (SPRITE_SIZE, SPRITE_SIZE),
            interpolation=cv2.INTER_LANCZOS4
        )

        return np.asarray(thumbnail).tolist()
    except TypeError as e:
        logger.error(
            f"Error converting array to PIL Image or resizing for thumbnail: {e}. Array shape: {normalized_array.shape}, dtype: {normalized_array.dtype}")
        return np.array([])


class ThumbnailProcessor(PixelPatrolProcessor):

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def description(self) -> str:
        return f"Extracts a square, grayscale thumbnail of size {SPRITE_SIZE}x{SPRITE_SIZE}."

    def process(self, data: da.array, dim_order: str) -> dict:
        return {'thumbnail': _generate_thumbnail(data, dim_order)}

    def get_specification(self) -> Dict[str, Any]:
        """
        Defines the expected Polars data types for the output of this processor.
        """
        return {
            'thumbnail': pl.Array
        }