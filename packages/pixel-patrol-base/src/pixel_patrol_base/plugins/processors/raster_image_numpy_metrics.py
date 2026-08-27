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
    check bright_clipping_fraction and dark_clipping_fraction when interpreting.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 3 or w < 3:
        return np.full(arr.shape[:-2], np.nan)
    arr_f = _float32_cached(arr, cache)
    lap = (arr_f[..., 1:-1, :-2] + arr_f[..., 1:-1, 2:] +
           arr_f[..., :-2, 1:-1] + arr_f[..., 2:, 1:-1] -
           4.0 * arr_f[..., 1:-1, 1:-1])
    return np.nanvar(lap, axis=(-2, -1))


def noise_mad(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
              cache: Optional[Dict] = None) -> np.ndarray:
    """Noise standard deviation via Haar wavelet MAD estimator (Donoho & Johnstone 1994).

    Computes the diagonal detail subband (HH) of a single-level 2-D Haar wavelet
    and returns median(|HH|) / 0.6745. The median discards the sparse large
    coefficients that represent genuine image structure, so the estimate reflects
    additive noise rather than texture -- unlike mean-based estimators such as
    Immerkaer (1996), which texture spikes dominate.

    For integer dtypes an Anscombe transform (2*sqrt(x + 3/8)) is applied before
    decomposition. This variance-stabilises Poisson noise (the dominant noise model
    in fluorescence microscopy and photon-limited scientific imaging) so the
    Gaussian MAD estimator remains valid. For float arrays no transform is applied.

    Returns NaN for spatial extents smaller than 4x4 or where fewer than 16
    finite HH coefficients exist (typically arising in heavily NaN-masked regions).
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 4 or w < 4:
        return np.full(arr.shape[:-2], np.nan)
    arr_f = _float32_cached(arr, cache)
    # Anscombe transform for integer dtypes: stabilises Poisson variance to ~1.
    if np.issubdtype(arr.dtype, np.integer):
        arr_work = 2.0 * np.sqrt(arr_f + 0.375)
    else:
        arr_work = arr_f
    # Trim to even spatial dims for non-overlapping 2×2 Haar blocks.
    he, we = (h // 2) * 2, (w // 2) * 2
    a = arr_work[..., :he, :we]
    # HH diagonal detail subband: A - B - C + D (top-left, top-right, bottom-left, bottom-right).
    HH = (a[..., 0::2, 0::2] - a[..., 0::2, 1::2]
          - a[..., 1::2, 0::2] + a[..., 1::2, 1::2]) / 4.0
    valid_n = np.sum(np.isfinite(HH), axis=(-2, -1))
    sigma = np.nanmedian(np.abs(HH), axis=(-2, -1)) / 0.6745
    return np.where(valid_n >= 16, sigma, np.nan).astype(np.float32)


def spectral_slope(arr: np.ndarray, cache: Optional[Dict] = None) -> np.ndarray:
    """Log-log slope of the radially averaged power spectrum.

    Fits a line to log(power) vs log(frequency) over the mid-frequency band
    (5-40% of Nyquist), where natural image content dominates and both the DC
    component and the high-frequency noise floor are excluded.

    Blur suppresses high-frequency energy and steepens the slope (more negative).
    Noise adds flat energy across all frequencies and flattens the slope (toward
    zero). This is the only metric in this set that maps blur and noise to
    *opposite directions* on a single axis -- laplacian_variance cannot make
    this distinction.

    A 2-D Hanning window is applied before the FFT to suppress spectral leakage
    from chunk boundaries, which would otherwise inflate low-frequency power and
    bias the slope estimate. NaN regions in float arrays are filled with the image
    mean before windowing to avoid corrupting the spectrum.

    Typical values for natural and scientific images: -2 to -4 (well-focused,
    low noise). Blurry images trend toward -4 or steeper. Noisy or
    transmission-error images trend toward -1 or flatter.

    Returns NaN for spatial extents smaller than 32x32 or where the fit is
    ill-conditioned (fewer than 10 valid frequency samples in the fit band).
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h < 32 or w < 32:
        return np.full(arr.shape[:-2], np.nan)
    arr_f = _float32_cached(arr, cache)
    # Fill NaN regions for float arrays so they don't corrupt the FFT.
    if np.issubdtype(arr.dtype, np.floating):
        arr_mean = np.nanmean(arr_f, axis=(-2, -1), keepdims=True)
        arr_work = np.where(np.isfinite(arr_f), arr_f, arr_mean)
    else:
        arr_work = arr_f
    # 2-D Hanning window to suppress spectral leakage at chunk boundaries.
    win = (np.hanning(h).astype(np.float32)[:, None]
           * np.hanning(w).astype(np.float32)[None, :])
    arr_windowed = arr_work * win  # broadcasts over leading dims
    # 2-D FFT → power spectrum.
    fft = np.fft.fft2(arr_windowed.astype(np.float32), axes=(-2, -1))
    power = (fft.real ** 2 + fft.imag ** 2)  # (..., H, W), avoids complex abs
    # Radial frequency for every FFT bin (cycles/pixel; Nyquist = 0.5).
    fy = np.fft.fftfreq(h).astype(np.float32)
    fx = np.fft.fftfreq(w).astype(np.float32)
    freq = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2).ravel()  # (H*W,)
    # Mid-frequency band: skip DC and very low freqs; stop before noise floor.
    f_min, f_max = 0.05, 0.40
    mask = (freq >= f_min) & (freq <= f_max)
    n_fit = mask.sum()
    if n_fit < 10:
        return np.full(arr.shape[:-2], np.nan)
    log_freq = np.log(freq[mask])                # (M,)
    # Fit log(power) = alpha * log(freq) + const for each leading-dim slice.
    power_flat = power.reshape(-1, h * w)         # (N_lead, H*W)
    result_flat = np.full(power_flat.shape[0], np.nan, dtype=np.float32)
    for i, p_row in enumerate(power_flat):
        p_band = p_row[mask]
        valid = (p_band > 0) & np.isfinite(p_band)
        if valid.sum() < 10:
            continue
        lf = log_freq[valid]
        lp = np.log(p_band[valid])
        # Closed-form OLS slope: avoids the overhead and import of lstsq.
        n = lf.size
        denom = n * float(np.dot(lf, lf)) - float(lf.sum()) ** 2
        if denom == 0.0:
            continue
        result_flat[i] = (n * float(np.dot(lf, lp)) - float(lf.sum()) * float(lp.sum())) / denom
    return result_flat.reshape(arr.shape[:-2])


def dark_clipping_fraction(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
                           cache: Optional[Dict] = None) -> np.ndarray:
    """Fraction of pixels at the dtype's minimum representable value (0 for unsigned integers).

    Detects underexposure, background/void regions, and nodata at the dark end.
    Returns NaN for floating-point arrays (no fixed lower bound; use
    bright_clipping_fraction to detect clipping relative to the observed peak).
    In natural images this often indicates underexposure; in scientific data it
    may be expected background or missing values (e.g. ocean nodata in elevation).
    """
    if not np.issubdtype(arr.dtype, np.integer):
        return np.full(arr.shape[:-2], np.nan)
    return np.mean(arr == np.iinfo(arr.dtype).min, axis=axes).astype(np.float32)


def bright_clipping_fraction(arr: np.ndarray, axes: Tuple[int, int] = _XY_AXES,
                             cache: Optional[Dict] = None) -> np.ndarray:
    """Fraction of pixels at this image's own observed maximum value (all dtypes, NaN excluded).

    Detects clipping at the bright end regardless of storage format or dtype ceiling.
    For integer types, catches both dtype-level saturation (e.g. 255 for uint8)
    and sub-dtype clipping (e.g. 12-bit data stored in uint16 clipping at 4095).
    For float arrays, where there is no fixed ceiling, this is the only available
    proxy for bright-end clipping. A high value indicates potential loss of
    information at the bright end of the signal.
    """
    chunk_max = np.nanmax(arr, axis=axes, keepdims=True)
    at_max = (arr == chunk_max)  # NaN comparisons are False -- NaN excluded from numerator
    if np.issubdtype(arr.dtype, np.floating):
        valid_n = np.sum(np.isfinite(arr), axis=axes)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(valid_n > 0, np.sum(at_max, axis=axes) / valid_n, np.nan).astype(np.float32)
    return np.mean(at_max, axis=axes).astype(np.float32)
