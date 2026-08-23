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


def _float32_cached(arr: np.ndarray, cache: Optional[Dict]) -> np.ndarray:
    """Convert arr to float32 once per chunk and reuse across every metric."""
    if np.issubdtype(arr.dtype, np.floating) and arr.dtype == np.float32:
        return arr
    if cache is not None and 'f32' in cache:
        return cache['f32']
    result = arr.astype(np.float32)
    if cache is not None:
        cache['f32'] = result
    return result


def _box3_mean(arr_f: np.ndarray) -> np.ndarray:
    """3x3 box-filter mean, valid convolution (output shrinks 2px per spatial dim).

    Tried a separable (horizontal-then-vertical) rewrite here, like the min/max
    trick in local_range_contrast_variability - measured *worse* peak memory in
    practice (an extra full-size intermediate stays live across the second pass),
    despite fewer additions. Kept as the flat 9-term sum; don't "optimize" this
    again without re-benchmarking.
    """
    return (arr_f[..., :-2, :-2] + arr_f[..., :-2, 1:-1] + arr_f[..., :-2, 2:] +
            arr_f[..., 1:-1, :-2] + arr_f[..., 1:-1, 1:-1] + arr_f[..., 1:-1, 2:] +
            arr_f[..., 2:, :-2]  + arr_f[..., 2:, 1:-1]  + arr_f[..., 2:, 2:]) / 9.0


def _nbr_stats(arr_f: np.ndarray):
    # Pure-numpy 3x3 box filter (valid, no padding) - local mean/std over the
    # interior (H-2, W-2). Matches the border-exclusion convention used by
    # every other 3x3 metric in this file (laplacian_variance, calc_blocking).
    local_mean = _box3_mean(arr_f)
    sq = arr_f ** 2
    mean_sq = _box3_mean(sq)
    del sq
    local_std = np.sqrt(np.maximum(mean_sq - local_mean ** 2, 0.0))
    return local_mean, local_std


def _nbr_stats_cached(arr: np.ndarray, cache: Optional[Dict]):
    if cache is not None and 'nbr' in cache:
        return cache['nbr']
    result = _nbr_stats(_float32_cached(arr, cache))
    if cache is not None:
        cache['nbr'] = result
    return result


def local_range_contrast_variability(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
                                     cache: Optional[Dict] = None) -> np.ndarray:
    """Mean local (max-min) range over 3x3 windows, divided by the image's spatial std.

    Not true Michelson contrast (that's (max-min)/(max+min) of the whole image) -
    this is a local-range-based variability measure instead.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)
    # Separable 3x3 min/max: a horizontal pass (window of 3 along X) followed by
    # a vertical pass (window of 3 along Y) is equivalent to a full 3x3 min/max,
    # since max/min are separable across independent axes. NaN-safe via fmax/fmin.
    # Deliberately NOT using the shared float32 cache here: min/max comparisons
    # don't need floating point, so this runs on the original (usually smaller)
    # dtype, and each intermediate is dropped as soon as it's consumed instead
    # of keeping all four alive at once (that held ~4x this array's size in
    # memory simultaneously and was the actual cost, not the float conversion).
    h_max = np.fmax(np.fmax(arr[..., :, :-2], arr[..., :, 1:-1]), arr[..., :, 2:])
    local_max = np.fmax(np.fmax(h_max[..., :-2, :], h_max[..., 1:-1, :]), h_max[..., 2:, :])
    del h_max
    h_min = np.fmin(np.fmin(arr[..., :, :-2], arr[..., :, 1:-1]), arr[..., :, 2:])
    local_min = np.fmin(np.fmin(h_min[..., :-2, :], h_min[..., 1:-1, :]), h_min[..., 2:, :])
    del h_min
    local_range = local_max.astype(np.float32) - local_min.astype(np.float32)
    del local_max, local_min
    mean_local_range = np.nanmean(local_range, axis=axes)
    with np.errstate(all="ignore"):
        spatial_std = np.nanstd(arr, axis=axes)
    result = np.full_like(spatial_std, np.nan)
    valid = spatial_std > 0
    result[valid] = mean_local_range[valid] / spatial_std[valid]
    return result


def local_texture_uniformity(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
                             cache: Optional[Dict] = None) -> np.ndarray:
    """Coefficient of variation of local 3x3 stds - how unevenly distributed texture is."""
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)
    _, loc_std = _nbr_stats_cached(arr, cache)
    all_nan = np.all(np.isnan(loc_std), axis=axes)
    if np.any(all_nan):
        loc_std_safe = np.where(all_nan[..., np.newaxis, np.newaxis], 0.0, loc_std)
        mean_local_std = np.where(all_nan, np.nan, np.nanmean(loc_std_safe, axis=axes))
        std_local_std  = np.where(all_nan, np.nan, np.nanstd(loc_std_safe, axis=axes))
    else:
        mean_local_std = np.nanmean(loc_std, axis=axes)
        std_local_std  = np.nanstd(loc_std, axis=axes)
    result = np.full_like(mean_local_std, np.nan)
    valid = mean_local_std > 0
    result[valid] = std_local_std[valid] / mean_local_std[valid]
    return result


def compression_blocking_score(arr: np.ndarray, cache: Optional[Dict] = None) -> np.ndarray:
    """Average brightness jump across 8-pixel block boundaries (JPEG-style blocking artifacts)."""
    lead = arr.shape[:-2]
    h, w = arr.shape[-2], arr.shape[-1]
    if h <= 8 or w <= 8:
        return np.full(lead, np.nan, dtype=np.float32)
    arr = _float32_cached(arr, cache)
    col_before = arr[..., :, 7::8]
    col_after  = arr[..., :, 8::8]
    row_before = arr[..., 7::8, :]
    row_after  = arr[..., 8::8, :]
    n_col = min(col_before.shape[-1], col_after.shape[-1])
    n_row = min(row_before.shape[-2], row_after.shape[-2])
    if n_col == 0 or n_row == 0:
        return np.full(lead, np.nan, dtype=np.float32)
    col_jumps = np.abs(col_before[..., :n_col] - col_after[..., :n_col])
    row_jumps = np.abs(row_before[..., :n_row, :] - row_after[..., :n_row, :])
    return (np.nanmean(col_jumps, axis=(-2, -1)) + np.nanmean(row_jumps, axis=(-2, -1))) / 2


def laplacian_variance(arr: np.ndarray, cache: Optional[Dict] = None) -> np.ndarray:
    """Variance of the discrete Laplacian - proxy for sharpness (higher = sharper)."""
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)
    arr_f = _float32_cached(arr, cache)
    lap = (arr_f[..., 1:-1, :-2] + arr_f[..., 1:-1, 2:] +
           arr_f[..., :-2, 1:-1] + arr_f[..., 2:, 1:-1] -
           4.0 * arr_f[..., 1:-1, 1:-1])
    return np.nanvar(lap, axis=(-2, -1))


def sobel_gradient_sharpness(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
                             cache: Optional[Dict] = None) -> np.ndarray:
    """Mean squared Sobel gradient magnitude - isotropic sharpness (Tenengrad), complements
    laplacian_variance's directionally-biased cross kernel."""
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)
    arr_f = _float32_cached(arr, cache)
    # Not cached: nothing else currently reuses gx/gy, and holding both plus
    # their squares alive at once (the old _sobel_gradients_cached approach)
    # was the single biggest memory cost in this file - free each gradient
    # before computing the next instead.
    gx = (arr_f[..., :-2, 2:] + 2 * arr_f[..., 1:-1, 2:] + arr_f[..., 2:, 2:] -
          arr_f[..., :-2, :-2] - 2 * arr_f[..., 1:-1, :-2] - arr_f[..., 2:, :-2])
    grad_sq = gx * gx
    del gx
    gy = (arr_f[..., 2:, :-2] + 2 * arr_f[..., 2:, 1:-1] + arr_f[..., 2:, 2:] -
          arr_f[..., :-2, :-2] - 2 * arr_f[..., :-2, 1:-1] - arr_f[..., :-2, 2:])
    grad_sq += gy * gy
    del gy
    return np.nanmean(grad_sq, axis=axes)


def estimated_noise_std(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
                        cache: Optional[Dict] = None) -> np.ndarray:
    """No-reference noise standard deviation estimate (Immerkaer 1996).

    Computed on raw pixel values by default - no Anscombe transform is applied.
    For photon-limited (shot-noise-dominated) modalities, this may under/overstate
    the true noise level; a variance-stabilizing transform could be applied
    upstream if needed, but isn't baked in here.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)
    arr_f = _float32_cached(arr, cache)
    # Fixed 3x3 kernel from the paper: [[1,-2,1],[-2,4,-2],[1,-2,1]].
    conv = (arr_f[..., :-2, :-2]  - 2 * arr_f[..., :-2, 1:-1]  + arr_f[..., :-2, 2:] -
            2 * arr_f[..., 1:-1, :-2] + 4 * arr_f[..., 1:-1, 1:-1] - 2 * arr_f[..., 1:-1, 2:] +
            arr_f[..., 2:, :-2]  - 2 * arr_f[..., 2:, 1:-1]  + arr_f[..., 2:, 2:])
    n = (h - 2) * (w - 2)
    scale = np.sqrt(np.pi / 2) / (6.0 * n)
    return scale * np.nansum(np.abs(conv), axis=axes)


def saturated_pixel_fraction(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
                             cache: Optional[Dict] = None) -> np.ndarray:
    """Fraction of pixels at the dtype's max representable value (overexposure/saturation).

    Only defined for integer dtypes (a fixed max representable value); returns
    NaN for floating-point data, which has no such bound. Uses the container
    dtype's max, not the sensor's true bit depth (e.g. 12-bit data stored as
    uint16 reads as if it were full-range uint16) - a known limitation.
    """
    if not np.issubdtype(arr.dtype, np.integer):
        return np.full(arr.shape[:-2], np.nan)
    max_val = np.iinfo(arr.dtype).max
    return np.mean(arr == max_val, axis=axes)


def min_value_pixel_fraction(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
                             cache: Optional[Dict] = None) -> np.ndarray:
    """Fraction of pixels at the dtype's min representable value (0 for unsigned types).

    Same caveats as saturated_pixel_fraction: only defined for integer dtypes
    (NaN for floating-point data), and uses the container dtype's min, not the
    sensor's true bit depth.
    """
    if not np.issubdtype(arr.dtype, np.integer):
        return np.full(arr.shape[:-2], np.nan)
    min_val = np.iinfo(arr.dtype).min
    return np.mean(arr == min_val, axis=axes)
