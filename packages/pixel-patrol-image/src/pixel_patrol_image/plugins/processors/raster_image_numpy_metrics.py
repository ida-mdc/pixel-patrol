"""Image-specific metric kernels (Y, X at last two axes)."""

from typing import Dict, Optional, Tuple

import numpy as np

from pixel_patrol_base.plugins.processors.raster_processor import MetricContext


# Spatial (Y, X) axes within any (..., H, W) array.
_XY_AXES = (-2, -1)


def fold_to_chunks(
    arr: np.ndarray,
    chunk_sizes: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fold the last len(chunk_sizes) dimensions of arr into a regular grid.

    Output shape: (*leading_dims, n_0, ..., n_k, cs_0, ..., cs_k)
    """
    k = len(chunk_sizes)
    leading = arr.ndim - k
    chunk_sizes = tuple(min(cs, s) for cs, s in zip(chunk_sizes, arr.shape[leading:]))

    mask = np.ones(arr.shape, dtype=bool)
    pad = [(0, 0)] * leading + [
        (0, (cs - s % cs) % cs)
        for s, cs in zip(arr.shape[leading:], chunk_sizes)
    ]
    if any(p[1] for p in pad[leading:]):
        arr  = np.pad(arr,  pad, constant_values=0)
        mask = np.pad(mask, pad, constant_values=False)

    interleaved = [n for s, cs in zip(arr.shape[leading:], chunk_sizes) for n in (s // cs, cs)]
    arr  = arr.reshape(*arr.shape[:leading],  *interleaved)
    mask = mask.reshape(*mask.shape[:leading], *interleaved)

    base   = leading
    n_axes = [base + 2 * i     for i in range(k)]
    c_axes = [base + 2 * i + 1 for i in range(k)]
    perm   = list(range(leading)) + n_axes + c_axes
    return arr.transpose(perm), mask.transpose(perm)


def _nbr_stats(arr: np.ndarray):
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    nbr_sum    = np.zeros(arr[..., :-2, :-2].shape, dtype=np.float64)
    nbr_sq_sum = np.zeros_like(nbr_sum)
    h, w = arr.shape[-2], arr.shape[-1]
    for di in range(3):
        for dj in range(3):
            patch = arr[..., di:h - 2 + di, dj:w - 2 + dj]
            nbr_sum    += patch
            nbr_sq_sum += patch * patch
    local_mean = nbr_sum / 9.0
    local_std  = np.sqrt(np.maximum(nbr_sq_sum / 9.0 - local_mean ** 2, 0.0))
    return local_mean, local_std


def _nbr_stats_cached(arr: np.ndarray, cache: Optional[Dict]):
    if cache is not None and 'nbr' in cache:
        return cache['nbr']
    result = _nbr_stats(arr)
    if cache is not None:
        cache['nbr'] = result
    return result


def michelson_contrast(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES) -> np.ndarray:
    """Mean local range / spatial std over 3×3 windows."""
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    local_max = np.full_like(arr[..., :-2, :-2], np.nan)
    local_min = np.full_like(arr[..., :-2, :-2], np.nan)
    for di in range(3):
        for dj in range(3):
            patch = arr[..., di:h - 2 + di, dj:w - 2 + dj]
            local_max = np.fmax(local_max, patch)
            local_min = np.fmin(local_min, patch)
    mean_local_range = np.nanmean(local_max - local_min, axis=axes)
    with np.errstate(all="ignore"):
        spatial_std = np.nanstd(arr, axis=axes)
    result = np.full_like(spatial_std, np.nan)
    valid = spatial_std > 0
    result[valid] = mean_local_range[valid] / spatial_std[valid]
    return result


def mscn_variance(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
                  cache: Optional[Dict] = None) -> np.ndarray:
    """Variance of MSCN coefficients over 3×3 windows."""
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)
    local_mean, local_std = _nbr_stats_cached(arr, cache)
    with np.errstate(all="ignore"):
        arr_f = arr.astype(np.float32, copy=False)
        spatial_range = np.nanmax(arr_f, axis=axes, keepdims=True) - np.nanmin(arr_f, axis=axes, keepdims=True)
    C = np.maximum(spatial_range * 1e-3, 1e-10)
    center = arr[..., 1:-1, 1:-1].astype(np.float32, copy=False)
    c = (center - local_mean) / (local_std + C)
    all_nan = np.all(np.isnan(c), axis=axes)
    if not np.any(all_nan):
        return np.nanvar(c, axis=axes)
    c_safe = np.where(all_nan[..., np.newaxis, np.newaxis], 0.0, c)
    return np.where(all_nan, np.nan, np.nanvar(c_safe, axis=axes))


def local_std_ratio(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
                    cache: Optional[Dict] = None) -> np.ndarray:
    """Mean local 3×3 std / spatial std."""
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)
    _, loc_std = _nbr_stats_cached(arr, cache)
    all_nan = np.all(np.isnan(loc_std), axis=axes)
    if np.any(all_nan):
        loc_std_safe = np.where(all_nan[..., np.newaxis, np.newaxis], 0.0, loc_std)
        mean_local_std = np.where(all_nan, np.nan, np.nanmean(loc_std_safe, axis=axes))
    else:
        mean_local_std = np.nanmean(loc_std, axis=axes)
    with np.errstate(all="ignore"):
        spatial_std = np.nanstd(arr, axis=axes)
    result = np.full_like(spatial_std, np.nan)
    valid = spatial_std > 0
    result[valid] = mean_local_std[valid] / spatial_std[valid]
    return result


def calc_blocking(arr: np.ndarray) -> np.ndarray:
    """Average brightness jump across 8-pixel block boundaries."""
    lead = arr.shape[:-2]
    h, w = arr.shape[-2], arr.shape[-1]
    if h <= 8 or w <= 8:
        return np.full(lead, np.nan, dtype=np.float32)
    col_before = arr[..., :, 7::8].astype(np.float32, copy=False)
    col_after  = arr[..., :, 8::8].astype(np.float32, copy=False)
    row_before = arr[..., 7::8, :].astype(np.float32, copy=False)
    row_after  = arr[..., 8::8, :].astype(np.float32, copy=False)
    n_col = min(col_before.shape[-1], col_after.shape[-1])
    n_row = min(row_before.shape[-2], row_after.shape[-2])
    if n_col == 0 or n_row == 0:
        return np.full(lead, np.nan, dtype=np.float32)
    col_jumps = np.abs(col_before[..., :n_col] - col_after[..., :n_col])
    row_jumps = np.abs(row_before[..., :n_row, :] - row_after[..., :n_row, :])
    return (np.nanmean(col_jumps, axis=(-2, -1)) + np.nanmean(row_jumps, axis=(-2, -1))) / 2


def calc_ringing(arr: np.ndarray) -> np.ndarray:
    """Variance of high-pass (pixel minus 3×3 box average)."""
    arr_f = arr.astype(np.float32, copy=False)
    kernel_avg = (arr_f[..., :-2, :-2] + arr_f[..., :-2, 1:-1] + arr_f[..., :-2, 2:] +
                  arr_f[..., 1:-1, :-2] + arr_f[..., 1:-1, 1:-1] + arr_f[..., 1:-1, 2:] +
                  arr_f[..., 2:, :-2]  + arr_f[..., 2:, 1:-1]  + arr_f[..., 2:, 2:]) / 9.0
    return np.nanvar(arr_f[..., 1:-1, 1:-1] - kernel_avg, axis=(-2, -1))
