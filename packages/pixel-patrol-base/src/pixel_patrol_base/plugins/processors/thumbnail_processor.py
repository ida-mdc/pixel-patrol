import logging

import dask.array as da
import numpy as np

from pixel_patrol_base.config import SPRITE_SIZE
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.contracts import ProcessResult
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_base.core.feature_schema import validate_processor_output

logger = logging.getLogger(__name__)


def _get_color_dim(capabilities) -> str | None:
    """Return the color dimension letter encoded in an 'rgb:<dim>' capability, or None."""
    for cap in capabilities:
        if cap.startswith('rgb:'):
            return cap[4:]
    return None


def _reduce_to_spatial(arr, dim_order: str, keep_dims: set) -> tuple[da.Array, str]:
    """Reduce all dimensions not in keep_dims by taking the center slice."""
    current = dim_order
    i = 0
    while i < len(current):
        dim = current[i]
        if dim not in keep_dims:
            center = arr.shape[i] // 2
            arr = da.take(arr, indices=center, axis=i)
            current = current.replace(dim, '', 1)
        else:
            i += 1
    return arr, current


def _normalize(arr: da.Array) -> tuple[da.Array, float, float]:
    """
    Normalize to uint8 [0, 255].
    Lower bound = min(arr_min, 0) to keep zero as a fixed reference.
    Upper bound = arr_max.
    Returns (normalized_uint8, norm_min, norm_max).

    Uses ``nanmin`` / ``nanmax`` so NaN voxels do not poison the intensity range.
    If every value is NaN (or min/max are otherwise non-finite), returns a solid
    black image and ``(0.0, 0.0)`` for the norm metadata (no meaningful stretch).
    """
    mn, mx = da.compute(da.nanmin(arr), da.nanmax(arr))
    mn, mx = float(mn), float(mx)
    if not np.isfinite(mn) or not np.isfinite(mx):
        return da.full_like(arr, np.uint8(0), dtype=np.uint8), 0.0, 0.0
    lower = min(mn, 0.0)
    upper = mx
    if upper <= lower:
        fill = np.uint8(0 if upper <= 0 else 255)
        return da.full_like(arr, fill, dtype=np.uint8), lower, upper
    normalized = (arr.astype(np.float64) - lower) / (upper - lower) * 255.0
    return da.clip(normalized, 0, 255).astype(np.uint8), lower, upper


def _resize_and_pad(img: np.ndarray) -> np.ndarray:
    """
    Scale img to fit within SPRITE_SIZE × SPRITE_SIZE while preserving aspect ratio,
    then center-pad to exactly SPRITE_SIZE × SPRITE_SIZE.

    Downsampling / upsampling uses nearest-neighbor resampling.

    Returns an RGBA canvas (SPRITE_SIZE, SPRITE_SIZE, 4): padding pixels have alpha=0
    (transparent), content pixels have alpha=255 (opaque).

    img must be (H, W) or (H, W, C).
    """
    h, w = img.shape[:2]
    scale = min(SPRITE_SIZE / h, SPRITE_SIZE / w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    row_idx = np.linspace(0, h - 1, new_h).astype(np.int64)
    col_idx = np.linspace(0, w - 1, new_w).astype(np.int64)
    resized = img[row_idx][:, col_idx]

    # RGBA canvas: alpha=0 everywhere (transparent padding)
    canvas = np.zeros((SPRITE_SIZE, SPRITE_SIZE, 4), dtype=np.uint8)
    y_off = (SPRITE_SIZE - new_h) // 2
    x_off = (SPRITE_SIZE - new_w) // 2

    if resized.ndim == 2:
        # Grayscale → replicate to RGB
        canvas[y_off:y_off + new_h, x_off:x_off + new_w, 0] = resized
        canvas[y_off:y_off + new_h, x_off:x_off + new_w, 1] = resized
        canvas[y_off:y_off + new_h, x_off:x_off + new_w, 2] = resized
    else:
        canvas[y_off:y_off + new_h, x_off:x_off + new_w, :resized.shape[2]] = resized[:, :, :4]

    # Mark content region as fully opaque
    canvas[y_off:y_off + new_h, x_off:x_off + new_w, 3] = 255
    return canvas


def _generate_thumbnail(
    da_array: da.Array, dim_order: str, color_dim: str | None
) -> tuple[bytes, float, float, str] | None:
    """
    Generate a thumbnail as raw RGBA bytes (SPRITE_SIZE × SPRITE_SIZE × 4, uint8).

    All images are normalized: lower bound = min(data_min, 0), upper = data_max.

    Returns (raw_bytes, norm_min, norm_max, dtype_name), or None if generation fails.
    """
    if da_array is None or da_array.size == 0:
        return None

    dtype_name = str(da_array.dtype)
    arr = da_array.copy()
    if arr.dtype == bool:
        arr = arr.astype(np.float32)
    elif np.issubdtype(arr.dtype, np.floating):
        # replace NaNs with zeros to avoid RuntimeWarnings of dask
        arr = da.where(da.isnull(arr), 0.0, arr)

    is_rgb = color_dim is not None

    keep_dims = {'X', 'Y'}
    if is_rgb:
        keep_dims.add(color_dim)

    # For non-RGB, preserve C/S so they get meaned below rather than center-sliced
    reduce_keep_dims = keep_dims if is_rgb else keep_dims | {'C', 'S'}
    arr, current_dim_order = _reduce_to_spatial(arr, dim_order, reduce_keep_dims)

    # Collapse any leftover non-spatial dimensions by mean
    target_ndim = 3 if is_rgb else 2
    while arr.ndim > target_ndim:
        arr = da.mean(arr, axis=0)
        current_dim_order = current_dim_order[1:]

    normalized, norm_min, norm_max = _normalize(arr)
    img = normalized.compute()

    # Transpose to (H, W) or (H, W, C)
    dims = list(current_dim_order)
    if is_rgb and color_dim in dims:
        img = np.transpose(img, [dims.index('Y'), dims.index('X'), dims.index(color_dim)])
        if img.shape[2] == 4:
            img = img[:, :, :3]
    elif 'Y' in dims and 'X' in dims and img.ndim == 2:
        img = np.transpose(img, [dims.index('Y'), dims.index('X')])

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None

    try:
        canvas = _resize_and_pad(img)
        return canvas.tobytes(), norm_min, norm_max, dtype_name
    except Exception as e:
        logger.error(f"Error generating thumbnail: {e}. Shape: {img.shape}, dtype: {img.dtype}")
        return None


class ThumbnailProcessor:
    NAME   = "thumbnail"
    INPUT  = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"

    OUTPUT_SCHEMA = {
        "thumbnail":          bytes,
        "thumbnail_norm_min": float,
        "thumbnail_norm_max": float,
        "thumbnail_dtype":    str,
    }
    def run(self, art: Record):
        color_dim = _get_color_dim(art.capabilities)
        result_data = _generate_thumbnail(art.data, art.dim_order, color_dim)
        if result_data is None:
            return [{"obs_level": 0, **{x: None for x in self.OUTPUT_SCHEMA}}]
        return [{"obs_level": 0, **{x: result_data[i] for i, x in enumerate(self.OUTPUT_SCHEMA)}}]
