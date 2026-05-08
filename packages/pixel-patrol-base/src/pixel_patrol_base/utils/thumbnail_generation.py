"""Dask-backed thumbnail generation for huge images (fixed-size RGBA sprite bytes)."""

from __future__ import annotations

import logging
import os

import dask.array as da
import numpy as np

from pixel_patrol_base.config import SPRITE_SIZE

logger = logging.getLogger(__name__)

MAX_THUMBNAIL_CHUNK_PIXELS = int(
    os.environ.get("PIXEL_PATROL_MAX_THUMBNAIL_CHUNK_PIXELS", str(1024 * 1024))
)


def _reduce_to_spatial(arr: da.Array, dim_order: str, keep_dims: set[str]) -> tuple[da.Array, str]:
    """Reduce all dimensions not in ``keep_dims`` by taking the center slice."""
    current = dim_order
    i = 0
    while i < len(current):
        dim = current[i]
        if dim not in keep_dims:
            center = arr.shape[i] // 2
            arr = da.take(arr, indices=center, axis=i)
            current = current.replace(dim, "", 1)
        else:
            i += 1
    return arr, current


def _normalize(arr: da.Array) -> tuple[da.Array, float, float]:
    """
    Normalize to uint8 ``[0, 255]``.
    Lower bound = ``min(arr_min, 0)`` to keep zero as a fixed reference.
    Upper bound = ``arr_max``.
    Returns ``(normalized_uint8, norm_min, norm_max)``.
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


def _downsample_before_compute(arr: da.Array, dim_order: str) -> da.Array:
    """Reduce XY resolution lazily so final thumbnail compute stays bounded."""
    if "Y" not in dim_order or "X" not in dim_order:
        return arr
    y_axis = dim_order.index("Y")
    x_axis = dim_order.index("X")
    arr = _ensure_safe_xy_chunking(arr, y_axis, x_axis)
    y = int(arr.shape[y_axis]) if arr.shape[y_axis] is not None else SPRITE_SIZE
    x = int(arr.shape[x_axis]) if arr.shape[x_axis] is not None else SPRITE_SIZE
    fy = max(1, y // (SPRITE_SIZE * 2))
    fx = max(1, x // (SPRITE_SIZE * 2))
    if fy == 1 and fx == 1:
        return arr
    slices = [slice(None)] * arr.ndim
    slices[y_axis] = slice(0, None, fy)
    slices[x_axis] = slice(0, None, fx)
    return arr[tuple(slices)]


def _ensure_safe_xy_chunking(arr: da.Array, y_axis: int, x_axis: int) -> da.Array:
    """Prevent thumbnail path from pulling one giant XY chunk into memory."""
    if not arr.chunks or not arr.chunks[y_axis] or not arr.chunks[x_axis]:
        return arr
    max_y = max(int(c) for c in arr.chunks[y_axis])
    max_x = max(int(c) for c in arr.chunks[x_axis])
    if max_y * max_x <= MAX_THUMBNAIL_CHUNK_PIXELS:
        return arr

    target = int(np.sqrt(MAX_THUMBNAIL_CHUNK_PIXELS))
    target = max(256, target)
    chunk_spec = list(arr.chunksize)
    chunk_spec[y_axis] = min(int(arr.shape[y_axis]), target)
    chunk_spec[x_axis] = min(int(arr.shape[x_axis]), target)
    return arr.rechunk(tuple(chunk_spec))


def _resize_and_pad(img: np.ndarray) -> np.ndarray:
    """
    Scale ``img`` to fit within ``SPRITE_SIZE`` while preserving aspect ratio,
    then center-pad to exactly ``SPRITE_SIZE`` × ``SPRITE_SIZE``.
    """
    h, w = img.shape[:2]
    scale = min(SPRITE_SIZE / h, SPRITE_SIZE / w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    row_idx = np.linspace(0, h - 1, new_h).astype(np.int64)
    col_idx = np.linspace(0, w - 1, new_w).astype(np.int64)
    resized = img[row_idx][:, col_idx]

    canvas = np.zeros((SPRITE_SIZE, SPRITE_SIZE, 4), dtype=np.uint8)
    y_off = (SPRITE_SIZE - new_h) // 2
    x_off = (SPRITE_SIZE - new_w) // 2

    if resized.ndim == 2:
        canvas[y_off : y_off + new_h, x_off : x_off + new_w, 0] = resized
        canvas[y_off : y_off + new_h, x_off : x_off + new_w, 1] = resized
        canvas[y_off : y_off + new_h, x_off : x_off + new_w, 2] = resized
    else:
        canvas[y_off : y_off + new_h, x_off : x_off + new_w, : resized.shape[2]] = resized[:, :, :4]

    canvas[y_off : y_off + new_h, x_off : x_off + new_w, 3] = 255
    return canvas


def generate_thumbnail_rgba(
    da_array: da.Array,
    dim_order: str,
    color_dim: str | None,
) -> tuple[bytes, float, float, str] | None:
    """
    Generate a thumbnail as raw RGBA bytes (``SPRITE_SIZE`` × ``SPRITE_SIZE`` × 4, uint8).

    Returns ``(raw_bytes, norm_min, norm_max, dtype_name)``, or ``None`` if generation fails.
    """
    if da_array is None or da_array.size == 0:
        return None

    dtype_name = str(da_array.dtype)
    arr = da_array.copy()
    if arr.dtype == bool:
        arr = arr.astype(np.float32)
    elif np.issubdtype(arr.dtype, np.floating):
        arr = da.where(da.isnull(arr), 0.0, arr)

    is_rgb = color_dim is not None

    keep_dims = {"X", "Y"}
    if is_rgb:
        keep_dims.add(color_dim)

    reduce_keep_dims = keep_dims if is_rgb else keep_dims | {"C", "S"}
    arr, current_dim_order = _reduce_to_spatial(arr, dim_order, reduce_keep_dims)

    target_ndim = 3 if is_rgb else 2
    while arr.ndim > target_ndim:
        arr = da.mean(arr, axis=0)
        current_dim_order = current_dim_order[1:]

    arr = _downsample_before_compute(arr, current_dim_order)
    normalized, norm_min, norm_max = _normalize(arr)
    img = normalized.compute()

    dims = list(current_dim_order)
    if is_rgb and color_dim in dims:
        img = np.transpose(img, [dims.index("Y"), dims.index("X"), dims.index(color_dim)])
        if img.shape[2] == 4:
            img = img[:, :, :3]
    elif "Y" in dims and "X" in dims and img.ndim == 2:
        img = np.transpose(img, [dims.index("Y"), dims.index("X")])

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None

    try:
        canvas = _resize_and_pad(img)
        return canvas.tobytes(), norm_min, norm_max, dtype_name
    except Exception as e:
        logger.error(
            "Error generating thumbnail: %s. Shape: %s, dtype: %s",
            e,
            getattr(img, "shape", None),
            getattr(img, "dtype", None),
        )
        return None


def compute_thumbnail_fields(
    arr: "da.Array",
    dim_order: str,
    capabilities=None,
) -> dict:
    """Compute thumbnail and return the four output column values as a dict.

    ``capabilities`` is the Record's capability set (used to detect RGB dims).
    Returns keys: thumbnail, thumbnail_norm_min, thumbnail_norm_max, thumbnail_dtype.
    """
    color_dim = _get_color_dim(capabilities or set())
    dim_order  = dim_order.upper()
    result     = generate_thumbnail_rgba(arr, dim_order, color_dim)
    if result is None:
        return {
            "thumbnail":          None,
            "thumbnail_norm_min": None,
            "thumbnail_norm_max": None,
            "thumbnail_dtype":    None,
        }
    raw, norm_min, norm_max, dtype_name = result
    return {
        "thumbnail":          raw,
        "thumbnail_norm_min": norm_min,
        "thumbnail_norm_max": norm_max,
        "thumbnail_dtype":    dtype_name,
    }


def _get_color_dim(capabilities) -> "str | None":
    """Return the color dimension letter if an ``rgb:<letter>`` capability is present."""
    for cap in capabilities:
        if isinstance(cap, str) and cap.startswith("rgb:"):
            return cap[4:]
    return None
