"""
NumPy backend for raster tiles: XY fold layout + per-metric tile kernels.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Downstream code emits float NaN into result columns that may have integer dtype.
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
    module="numpy",
)

from pixel_patrol_base.config import HISTOGRAM_BINS
from pixel_patrol_image.plugins.processors.raster_metric_definitions import enabled_ctx_tile_fields, MetricNames


def fold_to_tiles(
    block: np.ndarray, tile_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad a (..., H, W) block so H and W are multiples of tile_size, then reshape.

    Returns (tiled, mask) both of shape (n_planes, n_tiles_y, n_tiles_x, tile_size, tile_size).
    n_planes is the product of all leading dimensions.  mask is False on padding pixels.
    Padding pixels are filled with 0; the mask is the authority on validity.
    """
    h, w = block.shape[-2:]
    mask = np.ones(block.shape, dtype=bool)
    ts = min(tile_size, max(h, w))
    py, px = (ts - h % ts) % ts, (ts - w % ts) % ts
    if py or px:
        pad = [(0, 0)] * (block.ndim - 2) + [(0, py), (0, px)]
        block = np.pad(block, pad, constant_values=0)
        mask = np.pad(mask, pad, constant_values=False)
    n_ty = block.shape[-2] // ts
    n_tx = block.shape[-1] // ts
    tiled = block.reshape(-1, n_ty, ts, n_tx, ts).swapaxes(-3, -2)
    mask_tiled = mask.reshape(-1, n_ty, ts, n_tx, ts).swapaxes(-3, -2)
    return tiled, mask_tiled


def _nbr_stats(arr: np.ndarray):
    """Compute 3×3 neighbourhood mean and std — shared between mscn and local_std_ratio.

    Requires float input to avoid integer overflow in the squared-sum accumulator.
    """
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    nbr_sum = np.zeros(arr[..., :-2, :-2].shape, dtype=np.float64)
    nbr_sq_sum = np.zeros_like(nbr_sum)
    h, w = arr.shape[-2], arr.shape[-1]
    for di in range(3):
        for dj in range(3):
            patch = arr[..., di:h - 2 + di, dj:w - 2 + dj]
            nbr_sum += patch
            nbr_sq_sum += patch * patch
    local_mean = nbr_sum / 9.0
    local_std = np.sqrt(np.maximum(nbr_sq_sum / 9.0 - local_mean ** 2, 0.0))
    return local_mean, local_std


class NumpyRasterBackend:
    """Turn image chunks into a square tile grid and measure statistics on each tile."""

    def __init__(self, arr: np.ndarray, tile_size, s_min, s_max, enabled_metrics, dim_names):
        self.arr = arr
        self.tile_size = tile_size
        self.s_min = s_min
        self.s_max = s_max
        self.enabled_metrics = enabled_metrics
        self.dim_names = dim_names

    @staticmethod
    def compute_tile_metric(
        metric_name: str,
        arr: np.ndarray,
        mask: np.ndarray,
        axes: Tuple[int, int],
        ctx_list: List[Dict[str, float]],
        nbr_cache: Optional[Dict] = None,
    ) -> Optional[np.ndarray]:
        """Calculate one requested statistic across each tile.

        For extended (border) tiles arr is float32 with NaN at padding pixels.
        For interior tiles arr may be any dtype; nan* functions handle integer input
        correctly, and metric functions that need float convert locally.
        """
        match metric_name:
            case MetricNames.MIN_INTENSITY: return np.nanmin(arr, axis=axes)
            case MetricNames.MAX_INTENSITY: return np.nanmax(arr, axis=axes)
            case MetricNames.MEAN_INTENSITY: return np.nanmean(arr, axis=axes)
            case MetricNames.STD_INTENSITY: return np.nanstd(arr, axis=axes)
            case MetricNames.FINITE_PIXEL_COUNT:
                if np.issubdtype(arr.dtype, np.floating):
                    return np.sum(mask & ~np.isnan(arr), axis=axes)
                return mask.sum(axis=axes)
            case MetricNames.MICHELSON_CONTRAST: return michelson_contrast_tile(arr, axes)
            case MetricNames.MSCN_VARIANCE: return mscn_variance_tile(arr, axes, nbr_cache)
            case MetricNames.LOCAL_STD_RATIO: return local_std_ratio_tile(arr, axes, nbr_cache)
            case MetricNames.HISTOGRAM_NAN_COUNT:
                if np.issubdtype(arr.dtype, np.floating):
                    return np.sum(np.isnan(arr) & mask, axis=axes)
                return np.zeros(arr.shape[:-2], dtype=np.uint64)
            case MetricNames.HISTOGRAM_COUNTS: return histogram_counts(arr, mask, ctx_list)
            case MetricNames.BLOCKING_INDEX: return calc_blocking(arr, mask, None)
            case MetricNames.RINGING_INDEX: return calc_ringing(arr, mask, None)
            case _: return None

    def fold_block(self, block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return fold_to_tiles(block, self.tile_size)

    def process(self) -> List[Dict[str, Any]]:
        """Produce tile rows, one plane at a time to bound peak RAM."""
        block = self.arr
        ns_shape = block.shape[:-2]
        nd_leading = block.ndim - 2
        ctx_fields = enabled_ctx_tile_fields(self.enabled_metrics)
        tile_axes = (-2, -1)
        ts = self.tile_size

        rows: List[Dict[str, Any]] = []
        for ns_local in (np.ndindex(ns_shape) if ns_shape else [()]):
            plane = block[ns_local][np.newaxis]      # (1, H, W)
            reshaped, mask = self.fold_block(plane)  # (1, n_ty, n_tx, ts, ts)

            # Only convert to float and mark padding as NaN for border tiles that were
            # extended to reach tile_size. Interior tiles keep their original dtype.
            if not mask.all():
                reshaped = reshaped.astype(np.float32, copy=False)
                reshaped[~mask] = np.nan

            hist_ctx = [{"min": float(self.s_min[ns_local]), "max": float(self.s_max[ns_local])}]

            nbr_cache: Dict = {}
            tensors: Dict[str, np.ndarray] = {}
            for spec in self.enabled_metrics:
                out = self.compute_tile_metric(spec.name, reshaped, mask, tile_axes, hist_ctx, nbr_cache)
                if out is not None:
                    tensors[spec.name] = out

            valid_mask = np.any(mask, axis=tile_axes)  # (1, n_ty, n_tx)
            for idx in np.ndindex(valid_mask.shape):
                if not valid_mask[idx]:
                    continue
                _, tile_yi, tile_xi = idx
                row: Dict[str, Any] = {"obs_level": block.ndim}
                for i in range(nd_leading):
                    row[self.dim_names[i]] = ns_local[i]
                row["dim_y"] = tile_yi * ts
                row["dim_x"] = tile_xi * ts
                for name, tensor in tensors.items():
                    val = tensor[idx]
                    row[name] = float(val) if np.isscalar(val) else val
                for field_name, ctx_key in ctx_fields:
                    row[field_name] = hist_ctx[0][ctx_key]
                rows.append(row)
        return rows


def histogram_counts(
    reshaped: np.ndarray,
    mask: np.ndarray,
    ctx_list: List[Dict[str, float]],
) -> np.ndarray:
    """Count pixels per brightness bucket on each tile, using each slice's own min/max as the range."""
    ty, tx = reshaped.shape[1], reshaped.shape[2]
    all_h: List[np.ndarray] = []
    for i in range(reshaped.shape[0]):
        s_min, s_max = ctx_list[i]["min"], ctx_list[i]["max"]
        if np.issubdtype(reshaped.dtype, np.floating):
            valid_mask = mask[i] & ~np.isnan(reshaped[i])
        else:
            valid_mask = mask[i]
        h = np.zeros((ty, tx, HISTOGRAM_BINS), dtype=np.int64)
        if s_min < s_max:
            pixels = reshaped[i].reshape(ty, tx, -1).astype(np.float32, copy=False)
            vmask  = valid_mask.reshape(ty, tx, -1)
            bin_width = (s_max - s_min) / HISTOGRAM_BINS
            for iy in range(ty):
                for ix in range(tx):
                    valid = pixels[iy, ix, vmask[iy, ix]]
                    bin_indices = np.clip(
                        ((valid - s_min) / bin_width).astype(np.int32), 0, HISTOGRAM_BINS - 1
                    )
                    h[iy, ix] = np.bincount(bin_indices, minlength=HISTOGRAM_BINS)
        else:
            h[:, :, 0] = np.sum(valid_mask, axis=(-2, -1))
        all_h.append(h)
    return np.array(all_h)


def calc_blocking(arr: np.ndarray, _mask=None, _ctx=None) -> np.ndarray:
    """Average brightness jump across 8-pixel block boundaries as a proxy for JPEG-style blocking artifacts."""
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


def calc_ringing(arr: np.ndarray, _mask=None, _ctx=None) -> np.ndarray:
    """Ringing proxy: variance of a simple high-pass (pixel minus 3×3 box average)."""
    arr_f = arr.astype(np.float32, copy=False)
    kernel_avg = (arr_f[..., :-2, :-2] + arr_f[..., :-2, 1:-1] + arr_f[..., :-2, 2:] +
                  arr_f[..., 1:-1, :-2] + arr_f[..., 1:-1, 1:-1] + arr_f[..., 1:-1, 2:] +
                  arr_f[..., 2:, :-2] + arr_f[..., 2:, 1:-1] + arr_f[..., 2:, 2:]) / 9.0
    high_freq = arr_f[..., 1:-1, 1:-1] - kernel_avg
    return np.nanvar(high_freq, axis=(-2, -1))


def michelson_contrast_tile(arr: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    """Mean local range normalised by tile std: mean(local_max − local_min) / tile_std."""
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
        tile_std = np.nanstd(arr, axis=axes)
    result = np.full_like(tile_std, np.nan)
    valid = tile_std > 0
    result[valid] = mean_local_range[valid] / tile_std[valid]
    return result


def mscn_variance_tile(arr: np.ndarray, axes: Tuple[int, int] = (-2, -1),
                       cache: Optional[Dict] = None) -> np.ndarray:
    """Variance of MSCN coefficients: (I − μ_local) / (σ_local + C) over 3×3 windows."""
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)

    local_mean, local_std = _nbr_stats_cached(arr, cache)

    with np.errstate(all="ignore"):
        arr_f = arr.astype(np.float32, copy=False)
        tile_range = np.nanmax(arr_f, axis=axes, keepdims=True) - np.nanmin(arr_f, axis=axes, keepdims=True)
    C = np.maximum(tile_range * 1e-3, 1e-10)
    center = arr[..., 1:-1, 1:-1].astype(np.float32, copy=False)
    c = (center - local_mean) / (local_std + C)
    all_nan = np.all(np.isnan(c), axis=axes)
    if not np.any(all_nan):
        return np.nanvar(c, axis=axes)
    c_safe = np.where(all_nan[..., np.newaxis, np.newaxis], 0.0, c)
    return np.where(all_nan, np.nan, np.nanvar(c_safe, axis=axes))


def local_std_ratio_tile(arr: np.ndarray, axes: Tuple[int, int] = (-2, -1),
                         cache: Optional[Dict] = None) -> np.ndarray:
    """Mean local 3×3 std / tile std — sharpness proxy."""
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)

    _, local_std = _nbr_stats_cached(arr, cache)

    all_nan = np.all(np.isnan(local_std), axis=axes)
    if np.any(all_nan):
        local_std_safe = np.where(all_nan[..., np.newaxis, np.newaxis], 0.0, local_std)
        mean_local_std = np.where(all_nan, np.nan, np.nanmean(local_std_safe, axis=axes))
    else:
        mean_local_std = np.nanmean(local_std, axis=axes)
    with np.errstate(all="ignore"):
        tile_std = np.nanstd(arr, axis=axes)
    result = np.full_like(tile_std, np.nan)
    valid = tile_std > 0
    result[valid] = mean_local_std[valid] / tile_std[valid]
    return result


def _nbr_stats_cached(arr: np.ndarray, cache: Optional[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    if cache is not None and 'nbr' in cache:
        return cache['nbr']
    result = _nbr_stats(arr)
    if cache is not None:
        cache['nbr'] = result
    return result
