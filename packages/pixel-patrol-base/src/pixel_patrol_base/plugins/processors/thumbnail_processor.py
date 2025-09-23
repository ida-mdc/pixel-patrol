import logging

import dask.array as da
import numpy as np
import polars as pl

from pixel_patrol_base.config import SPRITE_SIZE
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.contracts import ProcessResult
from pixel_patrol_base.core.specs import RecordSpec

logger = logging.getLogger(__name__)

# TODO: decide how we create thumbnail for images with S (probably call to gray) and C
def _generate_thumbnail(da_array: da.array, dim_order: str) -> np.ndarray:
    """
    Generate a square, grayscale thumbnail as a NumPy array.
    """
    if da_array is None or da_array.size == 0:
        return np.array([])

    arr_to_process = da_array.copy()

    # Cast boolean arrays so min/max/normalization behave
    if arr_to_process.dtype == bool:
        arr_to_process = arr_to_process.astype(np.uint8)

    current_dim_order = dim_order

    # Reduce non-spatial, non-channel dims by taking the center slice
    i = 0
    while arr_to_process.ndim > 2 and i < len(current_dim_order):
        dim = current_dim_order[i]
        if dim not in ["X", "Y", "C"]:
            center_index = arr_to_process.shape[i] // 2
            arr_to_process = da.take(arr_to_process, indices=center_index, axis=i)
            current_dim_order = current_dim_order.replace(dim, "")
        else:
            i += 1

    arr_to_process = da.squeeze(arr_to_process)

    # If still >2D, collapse remaining non-XY dims by mean
    if arr_to_process.ndim > 2:
        logger.warning(
            f"Thumbnail: Array still multi-dimensional after reduction ({arr_to_process.ndim}D). "
            f"Taking mean along remaining non-XY dimensions."
        )
        while arr_to_process.ndim > 2:
            arr_to_process = da.mean(arr_to_process, axis=0)

    min_val = da.min(arr_to_process)
    max_val = da.max(arr_to_process)

    # Handle constant arrays
    if da.all(min_val == max_val).compute():
        fill_value = 255 if float(max_val.compute()) > 0 else 0
        normalized_array = da.full_like(arr_to_process, fill_value=fill_value, dtype=da.uint8)
    else:
        normalized_array = (arr_to_process - min_val) / (max_val - min_val) * 255
        normalized_array = da.clip(normalized_array, 0, 255).astype(da.uint8)

    try:
        if normalized_array.ndim == 3 and normalized_array.shape[0] == 1:
            normalized_array = da.squeeze(normalized_array, axis=0)

        img = normalized_array.compute()
        h, w = img.shape[:2]

        if h == 0 or w == 0:
            return np.array([])
        row_idx = (np.linspace(0, h - 1, SPRITE_SIZE)).astype(np.int64)
        col_idx = (np.linspace(0, w - 1, SPRITE_SIZE)).astype(np.int64)
        return img[row_idx][:, col_idx]
    except TypeError as e:
        logger.error(
            f"Error resizing thumbnail: {e}. "
            f"Array shape: {normalized_array.shape}, dtype: {normalized_array.dtype}"
        )
        return np.array([])


class ThumbnailProcessor:
    NAME   = "thumbnail"
    INPUT  = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"

    OUTPUT_SCHEMA = {
        "thumbnail": pl.Array
    }

    def run(self, art: Record) -> ProcessResult:
        dim_order = art.dim_order
        return {"thumbnail": _generate_thumbnail(art.data, dim_order)}
