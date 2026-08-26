"""Image-specific metric kernels (Y, X at last two axes)."""

from typing import Dict, Optional, Tuple

import numpy as np

from pixel_patrol_base.plugins.processors.raster_processor import MetricContext


# Spatial (Y, X) axes within any (..., H, W) array.
_XY_AXES = (-2, -1)


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


def laplacian_variance(arr: np.ndarray, cache: Optional[Dict] = None) -> np.ndarray:
    """Variance of the discrete Laplacian -- high-frequency content measure.

    Low values reliably indicate blur or heavy compression. High values are
    ambiguous: genuine sharpness, noise, quantization artifacts, and
    transmission errors all produce high scores. Clipped pixels create
    artificial high-frequency edges at clip boundaries and inflate the score;
    check fraction_at_image_max and min_value_pixel_fraction when interpreting.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)
    arr_f = _float32_cached(arr, cache)
    lap = (arr_f[..., 1:-1, :-2] + arr_f[..., 1:-1, 2:] +
           arr_f[..., :-2, 1:-1] + arr_f[..., 2:, 1:-1] -
           4.0 * arr_f[..., 1:-1, 1:-1])
    return np.nanvar(lap, axis=(-2, -1))


def ringing_index(arr: np.ndarray, cache: Optional[Dict] = None) -> np.ndarray:
    """Variance of the residual after subtracting a 3×3 local mean.

    Captures oscillatory high-frequency content. Ringing artifacts near edges
    (from lossy compression or over-sharpening) and noise both inflate this.
    Complements laplacian_variance: both respond to noise and edges, but this
    tends to score higher on ringing (oscillations near edges) relative to blur.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)
    arr_f = _float32_cached(arr, cache)
    local_mean = (arr_f[..., :-2, :-2] + arr_f[..., :-2, 1:-1] + arr_f[..., :-2, 2:] +
                  arr_f[..., 1:-1, :-2] + arr_f[..., 1:-1, 1:-1] + arr_f[..., 1:-1, 2:] +
                  arr_f[..., 2:, :-2]  + arr_f[..., 2:, 1:-1]  + arr_f[..., 2:, 2:]) / 9.0
    return np.nanvar(arr_f[..., 1:-1, 1:-1] - local_mean, axis=(-2, -1))


def jpeg_block_ratio(arr: np.ndarray, cache: Optional[Dict] = None) -> np.ndarray:
    """Ratio of mean pixel-difference at 8-pixel block boundaries vs within-block differences.

    Values near 1.0: boundary jumps are similar to interior jumps -- no systematic blocking.
    Values above 1.0: boundaries are consistently sharper than interiors, consistent with
    JPEG-style 8×8 DCT block artifacts.

    Normalising by within-block contrast cancels image content, so natural brightness
    and texture variation do not inflate the result. Typical ranges: natural or lossless
    images 0.9-1.3; visibly JPEG-blocked images above 1.5. JPEG2000 wavelet artifacts
    and natural image edges do not produce the specific 8-pixel periodicity this detects.
    """
    lead = arr.shape[:-2]
    h, w = arr.shape[-2], arr.shape[-1]
    if h <= 16 or w <= 16:
        return np.full(lead, np.nan, dtype=np.float32)
    arr_f = _float32_cached(arr, cache)

    diff_x = np.abs(arr_f[..., :, 1:] - arr_f[..., :, :-1])   # (..., H, W-1)
    diff_y = np.abs(arr_f[..., 1:, :] - arr_f[..., :-1, :])   # (..., H-1, W)

    # Boundary positions: diff at index k crosses a block boundary when k % 8 == 7.
    bx_idx = np.where((np.arange(diff_x.shape[-1]) % 8) == 7)[0]
    ix_idx = np.where((np.arange(diff_x.shape[-1]) % 8) != 7)[0]
    by_idx = np.where((np.arange(diff_y.shape[-2]) % 8) == 7)[0]
    iy_idx = np.where((np.arange(diff_y.shape[-2]) % 8) != 7)[0]

    mean_bnd = (np.nanmean(diff_x[..., bx_idx],      axis=(-2, -1)) +
                np.nanmean(diff_y[..., by_idx, :],    axis=(-2, -1))) / 2
    mean_int = (np.nanmean(diff_x[..., ix_idx],      axis=(-2, -1)) +
                np.nanmean(diff_y[..., iy_idx, :],    axis=(-2, -1))) / 2

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(mean_int > 0, mean_bnd / mean_int, np.nan).astype(np.float32)


def estimated_noise_std(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
                        cache: Optional[Dict] = None) -> np.ndarray:
    """High-frequency variation estimate (Immerkaer 1996).

    Applies a fixed 3×3 filter that suppresses smooth gradients, leaving mainly
    noise and fine texture. The result is proportional to the standard deviation
    of additive Gaussian noise but responds to any high-frequency content,
    including natural image texture. Reliable as a comparative metric within a
    batch of similar images; the absolute value is not a standalone noise reading.
    Cannot detect sparse noise (impulse/salt-and-pepper).
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)
    arr_f = _float32_cached(arr, cache)
    conv = (arr_f[..., :-2, :-2]  - 2 * arr_f[..., :-2, 1:-1]  + arr_f[..., :-2, 2:] -
            2 * arr_f[..., 1:-1, :-2] + 4 * arr_f[..., 1:-1, 1:-1] - 2 * arr_f[..., 1:-1, 2:] +
            arr_f[..., 2:, :-2]  - 2 * arr_f[..., 2:, 1:-1]  + arr_f[..., 2:, 2:])
    # Divide by valid count, not total pixels: float arrays may have NaN regions.
    valid_n = np.sum(np.isfinite(conv), axis=axes)
    scale = np.sqrt(np.pi / 2) / 6.0
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(valid_n > 0, scale * np.nansum(np.abs(conv), axis=axes) / valid_n, np.nan)


def min_value_pixel_fraction(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
                             cache: Optional[Dict] = None) -> np.ndarray:
    """Fraction of pixels at the dtype's minimum representable value (0 for unsigned integers).

    Returns NaN for floating-point data (no fixed lower bound).
    In natural images this often means underexposure; in scientific data it may
    be expected background or missing values (e.g. ocean nodata in SRTM).
    """
    if not np.issubdtype(arr.dtype, np.integer):
        return np.full(arr.shape[:-2], np.nan)
    return np.mean(arr == np.iinfo(arr.dtype).min, axis=axes)


def fraction_at_image_max(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
                          cache: Optional[Dict] = None) -> np.ndarray:
    """Fraction of pixels at this image's own observed maximum value (all dtypes).

    Detects pixel-value clipping regardless of storage format or dtype ceiling.
    For integer types, values clipping at the dtype max (e.g. 255 for uint8,
    65535 for uint16) are caught here. Also catches sub-dtype clipping: 12-bit
    data stored in uint16 clips at 4095, which the dtype ceiling (65535) misses.
    For float32, where there is no fixed ceiling, this is the only available
    clipping proxy. NaN pixels are excluded from both numerator and denominator.
    """
    chunk_max = np.nanmax(arr, axis=axes, keepdims=True)
    at_max = (arr == chunk_max)  # NaN comparisons are False -- NaN pixels excluded from numerator
    if np.issubdtype(arr.dtype, np.floating):
        valid_n = np.sum(np.isfinite(arr), axis=axes)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(valid_n > 0, np.sum(at_max, axis=axes) / valid_n, np.nan).astype(np.float32)
    return np.mean(at_max, axis=axes).astype(np.float32)  # integers: no NaN possible
