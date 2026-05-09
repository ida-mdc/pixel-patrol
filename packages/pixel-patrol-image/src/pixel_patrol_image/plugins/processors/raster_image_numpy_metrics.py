"""
NumPy backend for raster tiles: XY fold layout + per-metric tile kernels.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pixel_patrol_base.config import HISTOGRAM_BINS
from pixel_patrol_image.plugins.processors.raster_metric_definitions import enabled_ctx_tile_fields, MetricNames


def fold_to_tiles(
    block: np.ndarray, tile_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad a (..., H, W) block so H and W are multiples of tile_size, then reshape.

    Returns (tiled, mask) both of shape (n_planes, n_tiles_y, n_tiles_x, tile_size, tile_size).
    n_planes is the product of all leading dimensions.  mask is False on padding pixels.
    """
    h, w = block.shape[-2:]
    mask = np.ones(block.shape, dtype=bool)
    ts = tile_size
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


class NumpyRasterBackend:
    """Turn lazy image chunks into a square tile grid and measure statistics on each tile."""

    def __init__(self, dask_arr, tile_size, s_min, s_max, enabled_metrics, dim_names):
        """Store the chunked image, tile width and height, value ranges per slice, metrics to run, and dimension labels."""
        self.dask_arr = dask_arr
        self.tile_size = tile_size
        self.dim_starts = [np.cumsum((0,) + c[:-1]) for c in self.dask_arr.chunks]
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
    ) -> Optional[np.ndarray]:
        """Calculate one requested statistic across each tile, or return nothing if it is not supported."""
        match metric_name:
            case MetricNames.MIN_INTENSITY: return np.nanmin(arr, axis=axes)
            case MetricNames.MAX_INTENSITY: return np.nanmax(arr, axis=axes)
            case MetricNames.MEAN_INTENSITY: return np.nanmean(arr, axis=axes)
            case MetricNames.STD_INTENSITY: return np.nanstd(arr, axis=axes)
            case MetricNames.FINITE_PIXEL_COUNT: return np.sum(mask & ~np.isnan(arr), axis=axes)
            case MetricNames.MICHELSON_CONTRAST: return michelson_contrast_tile(arr, axes)
            case MetricNames.MSCN_VARIANCE: return mscn_variance_tile(arr, axes)
            case MetricNames.LOCAL_STD_RATIO: return local_std_ratio_tile(arr, axes)
            case MetricNames.TENENGRAD: return tenengrad_tile(arr, axes) / _tile_std_sq(arr, axes)
            case MetricNames.LAPLACIAN_VARIANCE: return laplacian_variance_tile(arr, axes) / _tile_std_sq(arr, axes)
            case MetricNames.BRENNER: return brenner_tile(arr, axes) / _tile_std_sq(arr, axes)
            case MetricNames.HISTOGRAM_NAN_COUNT: return np.sum(np.isnan(arr) & mask, axis=axes)
            case MetricNames.HISTOGRAM_COUNTS: return histogram_counts(arr, mask, ctx_list)
            case MetricNames.BLOCKING_INDEX: return calc_blocking(arr, mask, None)
            case MetricNames.RINGING_INDEX: return calc_ringing(arr, mask, None)
            case _: return None

    def fold_block(self, block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pad and reshape a block into a tile grid; delegates to module-level fold_to_tiles."""
        return fold_to_tiles(block, self.tile_size)

    def process(self, b_idx: Tuple[int, ...], precomputed: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Load one chunk from the lazy array and produce results for every tile that overlaps real pixels."""
        block = precomputed if precomputed is not None else self.dask_arr.blocks[b_idx].compute()
        block_origin = self._block_origin(b_idx)
        ns_shape = block.shape[:-2]

        reshaped, mask = self.fold_block(block)
        histogram_ctx = self._histogram_ctx_per_plane(ns_shape, block_origin)
        tile_axes = (-2, -1)
        tensors = self._evaluate_tile_metrics(reshaped, mask, tile_axes, histogram_ctx)

        ctx_fields = enabled_ctx_tile_fields(self.enabled_metrics)
        valid_mask = np.any(mask, axis=tile_axes)

        rows: List[Dict[str, Any]] = []
        for idx in np.ndindex(valid_mask.shape):
            if not valid_mask[idx]:
                continue
            rows.append(
                self._tile_row(idx, ns_shape, block_origin, tensors, histogram_ctx, ctx_fields)
            )
        return rows

    def _block_origin(self, b_idx: Tuple[int, ...]) -> List[int]:
        """Pixel offsets telling where this chunk begins inside the full image."""
        return [int(self.dim_starts[ax][b_idx[ax]]) for ax in range(self.dask_arr.ndim)]

    def _histogram_ctx_per_plane(
        self, ns_shape: Tuple[int, ...], block_origin: List[int]
    ) -> List[Dict[str, float]]:
        """For each slice in this chunk, the darkest and brightest values used when binning histograms."""
        nd_leading = self.dask_arr.ndim - 2
        ctx_list: List[Dict[str, float]] = []
        for n_idx in np.ndindex(ns_shape):
            global_ns = tuple(block_origin[i] + n_idx[i] for i in range(nd_leading))
            ctx_list.append(
                {"min": float(self.s_min[global_ns]), "max": float(self.s_max[global_ns])}
            )
        return ctx_list

    def _evaluate_tile_metrics(
        self,
        reshaped: np.ndarray,
        mask: np.ndarray,
        axes: Tuple[int, int],
        histogram_ctx: List[Dict[str, float]],
    ) -> Dict[str, np.ndarray]:
        """Run every chosen metric over the tiled layout and collect results under metric names."""
        tensors: Dict[str, np.ndarray] = {}
        for spec in self.enabled_metrics:
            out = self.compute_tile_metric(spec.name, reshaped, mask, axes, histogram_ctx)
            if out is not None:
                tensors[spec.name] = out
        return tensors

    def _tile_row(
        self,
        tile_idx: Tuple[int, ...],
        ns_shape: Tuple[int, ...],
        block_origin: List[int],
        tensors: Dict[str, np.ndarray],
        histogram_ctx: List[Dict[str, float]],
        ctx_fields: Tuple[Tuple[str, str], ...],
    ) -> Dict[str, Any]:
        """Build one record listing tile position, measured numbers, and histogram range metadata."""
        plane_flat, tile_yi, tile_xi = tile_idx  # indices into (n_planes, n_tiles_y, n_tiles_x)
        nd_leading = self.dask_arr.ndim - 2
        ns_local = np.unravel_index(plane_flat, ns_shape) if ns_shape else ()

        row: Dict[str, Any] = {"obs_level": self.dask_arr.ndim}
        for i in range(nd_leading):
            row[self.dim_names[i]] = block_origin[i] + ns_local[i]

        ts = self.tile_size
        row["dim_y"] = block_origin[-2] + tile_yi * ts
        row["dim_x"] = block_origin[-1] + tile_xi * ts

        for name, tensor in tensors.items():
            val = tensor[tile_idx]
            row[name] = float(val) if np.isscalar(val) else val

        slice_ctx = histogram_ctx[plane_flat]
        for field_name, ctx_key in ctx_fields:
            row[field_name] = slice_ctx[ctx_key]

        return row


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


def _tile_std_sq(arr: np.ndarray, axes: Tuple[int, int]) -> np.ndarray:
    """Per-tile intensity variance, used to normalize gradient-based focus metrics.

    Gradient and Laplacian values scale with local intensity variation, so dividing
    by the tile variance (std²) gives dimensionless scores that are comparable across
    images regardless of exposure level or background offset.

    Using std² rather than mean² is essential for background-subtracted images: after
    subtraction the tile mean is near zero, so mean² collapses to near zero and produces
    huge outliers.  std² is unaffected by the DC offset and always reflects the true
    signal amplitude.

    Returns NaN for flat tiles (std == 0) — no gradients exist there, so the metric
    is undefined.  nanmean in aggregation ignores these tiles automatically.
    """
    std_sq = np.nanstd(arr, axis=axes) ** 2
    return np.where(std_sq > 0, std_sq, np.nan)


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
    return np.where(tile_std > 0, mean_local_range / tile_std, np.nan)


def mscn_variance_tile(arr: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    """Variance of Mean-Subtracted Contrast-Normalized (MSCN) coefficients over 3×3 windows.

    For each interior pixel:
        c(i,j) = (I(i,j) − μ_local(i,j)) / (σ_local(i,j) + C)
    where μ_local and σ_local are the 3×3 neighbourhood mean and std, and C is a
    stability constant (0.1 % of tile dynamic range).

    Local mean subtraction removes the DC offset (handles background-subtracted images);
    local std division removes the gain (handles absolute brightness variation).  The
    resulting variance is intensity-independent by construction.

    Ref: Mittal, A., Moorthy, A. K., & Bovik, A. C. (2012). No-reference image quality
    assessment in the spatial domain. IEEE Trans. Image Process., 21(12), 4695–4708.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)

    nbr_sum = np.zeros_like(arr[..., :-2, :-2])
    nbr_sq_sum = np.zeros_like(nbr_sum)
    for di in range(3):
        for dj in range(3):
            patch = arr[..., di:h - 2 + di, dj:w - 2 + dj]
            nbr_sum += patch
            nbr_sq_sum += patch * patch

    local_mean = nbr_sum / 9.0
    local_std = np.sqrt(np.maximum(nbr_sq_sum / 9.0 - local_mean ** 2, 0.0))

    with np.errstate(all="ignore"):
        tile_range = np.nanmax(arr, axis=axes, keepdims=True) - np.nanmin(arr, axis=axes, keepdims=True)
    C = np.maximum(tile_range * 1e-3, 1e-10)  # shape (..., 1, 1) broadcasts to center

    center = arr[..., 1:-1, 1:-1]
    c = (center - local_mean) / (local_std + C)
    return np.nanvar(c, axis=axes)


def local_std_ratio_tile(arr: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    """Mean local 3×3 std divided by tile std: mean(local_std) / tile_std.

    Measures how much of the tile's overall variation is concentrated in small local
    patches.  A sharp, focused tile with clear edges has high local std at each edge
    relative to the global spread → ratio approaches 1.  A smooth or out-of-focus tile
    has low local std everywhere even if the global std is non-zero → ratio is low.

    Both numerator and denominator are standard deviations (differences of squared
    values relative to local means), so both are translation-invariant and scale
    identically under multiplicative intensity changes.  The ratio is therefore
    invariant to both additive offsets and multiplicative scaling.

    Returns NaN for tiles where tile_std is zero (perfectly uniform tiles).

    Ref: Groen, F. C., Young, I. T., & Ligthart, G. (1985). A comparison of different
    focus functions for use in autofocus algorithms. Cytometry, 6(2), 81–91.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)

    nbr_sum = np.zeros_like(arr[..., :-2, :-2])
    nbr_sq_sum = np.zeros_like(nbr_sum)
    for di in range(3):
        for dj in range(3):
            patch = arr[..., di:h - 2 + di, dj:w - 2 + dj]
            nbr_sum += patch
            nbr_sq_sum += patch * patch

    local_mean = nbr_sum / 9.0
    local_std = np.sqrt(np.maximum(nbr_sq_sum / 9.0 - local_mean ** 2, 0.0))

    mean_local_std = np.nanmean(local_std, axis=axes)
    with np.errstate(all="ignore"):
        tile_std = np.nanstd(arr, axis=axes)
    return np.where(tile_std > 0, mean_local_std / tile_std, np.nan)


def tenengrad_tile(reshaped: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    """Mean squared Sobel gradient magnitude as a sharpness measure (higher = sharper edges).

    Uses the standard 3×3 Sobel operator, which combines a 2-pixel centered difference
    in the gradient direction with a [1, 2, 1] smoothing pass in the perpendicular direction.
    The smoothing suppresses pixel noise before measuring edge strength.

    Sobel X kernel (horizontal gradient):   Sobel Y kernel (vertical gradient):
        -1  0  +1                               -1  -2  -1
        -2  0  +2                                0   0   0
        -1  0  +1                               +1  +2  +1
    """
    Gx = (reshaped[..., :-2, 2:] - reshaped[..., :-2, :-2] +
          2 * reshaped[..., 1:-1, 2:] - 2 * reshaped[..., 1:-1, :-2] +
          reshaped[..., 2:, 2:] - reshaped[..., 2:, :-2])
    Gy = (reshaped[..., 2:, :-2] - reshaped[..., :-2, :-2] +
          2 * reshaped[..., 2:, 1:-1] - 2 * reshaped[..., :-2, 1:-1] +
          reshaped[..., 2:, 2:] - reshaped[..., :-2, 2:])
    return np.nanmean(Gx**2 + Gy**2, axis=axes)


def laplacian_variance_tile(reshaped: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    """Spread of a neighborhood contrast operator related to fine detail."""
    lap = (4 * reshaped[..., 1:-1, 1:-1]
           - reshaped[..., 1:-1, :-2] - reshaped[..., 1:-1, 2:]
           - reshaped[..., :-2, 1:-1] - reshaped[..., 2:, 1:-1])
    return np.nanvar(lap, axis=axes)


def brenner_tile(reshaped: np.ndarray, axes: Tuple[int, int] = (-2, -1)) -> np.ndarray:
    """Classic focus measure from squared brightness gaps two pixels apart in horizontal and vertical directions."""
    dx = reshaped[..., :, 2:] - reshaped[..., :, :-2]
    dy = reshaped[..., 2:, :] - reshaped[..., :-2, :]
    mx = np.nanmean(dx * dx, axis=axes)
    my = np.nanmean(dy * dy, axis=axes)
    return mx + my
