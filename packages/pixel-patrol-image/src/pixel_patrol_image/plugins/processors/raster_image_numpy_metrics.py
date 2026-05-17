"""
NumPy backend for raster tiles: XY fold layout + per-metric tile kernels.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# NaN is used as a sentinel for padding pixels and uncomputable metrics.
# Filling NaN into integer metric arrays is intentional — downstream code uses
# nanmin/nanmax/nanmean and the mask to ignore these values.
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

    If tile_size >= max(H, W) the block is treated as a single tile (no XY breakdown).
    Set PIXEL_PATROL_STATS_TILE_SIZE to a large value to get one value per Z-plane only.
    """
    h, w = block.shape[-2:]
    mask = np.ones(block.shape, dtype=bool)
    # Cap tile_size to the image extent so an oversized tile_size produces a single
    # tile rather than padding the array to (n, tile_size, tile_size) which OOMs.
    ts = min(tile_size, max(h, w))
    py, px = (ts - h % ts) % ts, (ts - w % ts) % ts
    if py or px:
        if not np.issubdtype(block.dtype, np.floating):
            block = block.astype(np.float32, copy=False)
        pad = [(0, 0)] * (block.ndim - 2) + [(0, py), (0, px)]
        block = np.pad(block, pad, constant_values=np.nan)
        mask = np.pad(mask, pad, constant_values=False)
    n_ty = block.shape[-2] // ts
    n_tx = block.shape[-1] // ts
    tiled = block.reshape(-1, n_ty, ts, n_tx, ts).swapaxes(-3, -2)
    mask_tiled = mask.reshape(-1, n_ty, ts, n_tx, ts).swapaxes(-3, -2)
    return tiled, mask_tiled


def _nbr_stats(arr: np.ndarray):
    """Compute 3×3 neighbourhood mean and std — shared between mscn and local_std_ratio."""
    nbr_sum = np.zeros_like(arr[..., :-2, :-2])
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
        """Calculate one requested statistic across each tile, or return nothing if it is not supported."""
        match metric_name:
            case MetricNames.MIN_INTENSITY: return np.nanmin(arr, axis=axes)
            case MetricNames.MAX_INTENSITY: return np.nanmax(arr, axis=axes)
            case MetricNames.MEAN_INTENSITY: return np.nanmean(arr, axis=axes)
            case MetricNames.STD_INTENSITY: return np.nanstd(arr, axis=axes)
            case MetricNames.FINITE_PIXEL_COUNT: return np.sum(mask & ~np.isnan(arr), axis=axes)
            case MetricNames.MICHELSON_CONTRAST: return michelson_contrast_tile(arr, axes)
            case MetricNames.MSCN_VARIANCE: return mscn_variance_tile(arr, axes, nbr_cache)
            case MetricNames.LOCAL_STD_RATIO: return local_std_ratio_tile(arr, axes, nbr_cache)
            case MetricNames.HISTOGRAM_NAN_COUNT: return np.sum(np.isnan(arr) & mask, axis=axes)
            case MetricNames.HISTOGRAM_COUNTS: return histogram_counts(arr, mask, ctx_list)
            case MetricNames.BLOCKING_INDEX: return calc_blocking(arr, mask, None)
            case MetricNames.RINGING_INDEX: return calc_ringing(arr, mask, None)
            case _: return None

    def fold_block(self, block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pad and reshape a block into a tile grid; delegates to module-level fold_to_tiles."""
        return fold_to_tiles(block, self.tile_size)

    def process(self) -> List[Dict[str, Any]]:
        """Produce tile rows, one plane at a time to bound peak RAM.

        Each non-spatial plane (Z-slice, channel, …) is folded and measured
        independently so metric intermediates are proportional to one plane
        rather than the full Z-stack.  Neighbourhood stats shared between
        mscn_variance and local_std_ratio are computed once per plane.
        """
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

            hist_ctx = [{"min": float(self.s_min[ns_local]), "max": float(self.s_max[ns_local])}]

            # Cache 3×3 neighbourhood stats: mscn and local_std_ratio share the same
            # computation (nbr_sum, nbr_sq_sum → local_mean, local_std).
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
        valid_mask = mask[i] & ~np.isnan(reshaped[i])  # (ty, tx, ts, ts)
        h = np.zeros((ty, tx, HISTOGRAM_BINS), dtype=np.int64)
        if s_min < s_max:
            pixels = reshaped[i].reshape(ty, tx, -1)   # (ty, tx, npix)
            vmask  = valid_mask.reshape(ty, tx, -1)
            # np.bincount is ~2x faster than np.histogram for this; it needs integer bin
            # indices in [0, HISTOGRAM_BINS), so we floor-divide by the bin width manually.
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
    """Average brightness jump across 8-pixel block boundaries as a proxy for JPEG-style blocking artifacts.

    JPEG compression divides images into 8×8 pixel blocks independently.  Strong compression
    introduces visible seams at the block edges.  This metric compares each pixel just before
    a block boundary to the pixel just after it, in both the column (X) and row (Y) directions.
    """
    lead = arr.shape[:-2]
    h, w = arr.shape[-2], arr.shape[-1]
    if h <= 8 or w <= 8:
        dt = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float32
        return np.full(lead, np.nan, dtype=dt)

    # Column boundaries: pixel at position 7 (last of block) vs. pixel at position 8 (first of next block)
    col_before_boundary = arr[..., :, 7::8]
    col_after_boundary  = arr[..., :, 8::8]
    n_col_pairs = min(col_before_boundary.shape[-1], col_after_boundary.shape[-1])

    # Row boundaries: same idea in the Y direction
    row_before_boundary = arr[..., 7::8, :]
    row_after_boundary  = arr[..., 8::8, :]
    n_row_pairs = min(row_before_boundary.shape[-2], row_after_boundary.shape[-2])

    if n_col_pairs == 0 or n_row_pairs == 0:
        dt = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float32
        return np.full(lead, np.nan, dtype=dt)

    col_jumps = np.abs(col_before_boundary[..., :n_col_pairs] - col_after_boundary[..., :n_col_pairs])
    row_jumps = np.abs(row_before_boundary[..., :n_row_pairs, :] - row_after_boundary[..., :n_row_pairs, :])
    return (np.nanmean(col_jumps, axis=(-2, -1)) + np.nanmean(row_jumps, axis=(-2, -1))) / 2


def calc_ringing(arr: np.ndarray, _mask=None, _ctx=None) -> np.ndarray:
    """Ringing proxy: variance of a simple high-pass (pixel minus 3×3 box average)."""
    kernel_avg = (arr[..., :-2, :-2] + arr[..., :-2, 1:-1] + arr[..., :-2, 2:] +
                  arr[..., 1:-1, :-2] + arr[..., 1:-1, 1:-1] + arr[..., 1:-1, 2:] +
                  arr[..., 2:, :-2] + arr[..., 2:, 1:-1] + arr[..., 2:, 2:]) / 9.0
    high_freq = arr[..., 1:-1, 1:-1] - kernel_avg
    return np.nanvar(high_freq, axis=(-2, -1))


def michelson_contrast_tile(arr: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    """Mean local range normalised by tile std: mean(local_max − local_min) / tile_std.

    The numerator is a 3×3 neighbourhood range (a difference, so translation-invariant).
    The denominator is the tile's overall standard deviation (also translation-invariant).
    Both scale identically under multiplicative intensity changes, so the ratio is
    invariant to both additive offsets (camera background, autofluorescence) and
    multiplicative scaling (gain, fluorophore concentration).

    Returns NaN for tiles where the overall std is zero (perfectly uniform tiles).

    Ref: Peli, E. (1990). Contrast in complex images. J. Opt. Soc. Am. A, 7(10), 2032–2040.
    (Adapted: denominator replaced with tile std to achieve additive invariance.)
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)

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
    """Variance of MSCN coefficients: (I − μ_local) / (σ_local + C) over 3×3 windows.

    Ref: Mittal et al. (2012). IEEE Trans. Image Process., 21(12), 4695–4708.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)

    local_mean, local_std = _nbr_stats_cached(arr, cache)

    with np.errstate(all="ignore"):
        tile_range = np.nanmax(arr, axis=axes, keepdims=True) - np.nanmin(arr, axis=axes, keepdims=True)
    C = np.maximum(tile_range * 1e-3, 1e-10)
    center = arr[..., 1:-1, 1:-1]
    c = (center - local_mean) / (local_std + C)
    all_nan = np.all(np.isnan(c), axis=axes)
    if not np.any(all_nan):
        return np.nanvar(c, axis=axes)
    c_safe = np.where(all_nan[..., np.newaxis, np.newaxis], 0.0, c)
    return np.where(all_nan, np.nan, np.nanvar(c_safe, axis=axes))


def local_std_ratio_tile(arr: np.ndarray, axes: Tuple[int, int] = (-2, -1),
                         cache: Optional[Dict] = None) -> np.ndarray:
    """Mean local 3×3 std / tile std — sharpness proxy.

    Ref: Groen et al. (1985). Cytometry, 6(2), 81–91.
    """
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
    """Return (local_mean, local_std) from cache if available, else compute and store."""
    if cache is not None and 'nbr' in cache:
        return cache['nbr']
    result = _nbr_stats(arr)
    if cache is not None:
        cache['nbr'] = result
    return result
