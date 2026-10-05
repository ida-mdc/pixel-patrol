"""scale_bar_score kernel (Y, X at last two axes). Standalone from raster_image_numpy_metrics.py; see scale_bar_processor.py for the processor."""

from typing import Dict, Optional, Tuple

import numpy as np

# Horizontal-only: a scale bar is a thin, flat-intensity run near the border, sandwiched
# between two patches of similar background (checked locally, not against the whole interior,
# so it doesn't fire on an off-centre or frame-filling subject).
_SB_BORDER_FRAC = 0.1        # fraction of H treated as "near the border"
_SB_MIN_BAR_FRAC = 0.06      # shortest run counted as a candidate bar, as a fraction of the image width
_SB_MAX_BAR_FRAC = 0.5       # longest run counted as a candidate bar
_SB_TOL_FRAC = 0.02          # intensity tolerance for "constant run", as a fraction of the dynamic range
_SB_MARGIN_FRAC = 0.02       # offset (as a fraction of H) from the candidate row to sample local background
_SB_MARGIN_MIN = 3           # floor on the margin above, for small images
_SB_MARGIN_WIN = 3           # number of rows averaged when sampling local background
_SB_CONTRAST_FRAC = 0.12     # required jump vs. local background, as a fraction of the dynamic range
_SB_BG_CONSISTENCY_FRAC = 0.08  # how closely the flanking background levels must agree
_SB_MIN_SIZE = 8             # minimum H and W to attempt detection


def _max_constant_run(arr: np.ndarray, tol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Longest run of near-constant values along the last axis: its length in pixels and its value."""
    w = arr.shape[-1]
    lead_shape = arr.shape[:-1]
    if w < 2:
        return np.ones(lead_shape, dtype=np.float32), arr[..., 0].astype(np.float32)
    close = np.abs(np.diff(arr, axis=-1)) <= tol
    step = close.astype(np.int32)
    cumsum = np.cumsum(step, axis=-1)
    reset_at_break = np.where(step == 0, cumsum, 0)
    baseline = np.maximum.accumulate(reset_at_break, axis=-1)
    run_lengths = cumsum - baseline
    end_idx = np.argmax(run_lengths, axis=-1)
    longest = np.take_along_axis(run_lengths, end_idx[..., None], axis=-1)[..., 0] + 1
    value = np.take_along_axis(arr, (end_idx + 1)[..., None], axis=-1)[..., 0]
    return longest.astype(np.float32), value.astype(np.float32)


def _local_background_level(row_level: np.ndarray, offset: int) -> np.ndarray:
    """Mean of _SB_MARGIN_WIN rows starting `offset` rows away from each row (positive = toward lower indices)."""
    h = row_level.shape[-1]
    positions = np.arange(h)
    total = np.zeros_like(row_level)
    step = 1 if offset > 0 else -1
    for k in range(_SB_MARGIN_WIN):
        idx = np.clip(positions - offset - step * k, 0, h - 1)
        total = total + row_level[..., idx]
    return total / _SB_MARGIN_WIN


def _edge_band_score(band: np.ndarray, tol: np.ndarray, rng_row: np.ndarray, border_n: int, margin: int,
                      full_width: int) -> np.ndarray:
    """scale_bar_score for one border, given a (..., N, W) band whose local row 0 is the true image edge."""
    n = band.shape[-2]
    has_nan = np.issubdtype(band.dtype, np.floating) and np.isnan(band).any()
    run_length, run_value = _max_constant_run(band, tol)
    bar_frac = run_length / full_width
    row_level = np.nanmedian(band, axis=-1) if has_nan else np.median(band, axis=-1)
    above_level = _local_background_level(row_level, margin)
    below_level = _local_background_level(row_level, -margin)
    # A second, further-out sample catches curved boundaries that locally mimic a bar but keep drifting.
    far = margin + _SB_MARGIN_WIN
    above_far = _local_background_level(row_level, far)
    below_far = _local_background_level(row_level, -far)
    contrast_above = np.abs(run_value - above_level)
    contrast_below = np.abs(run_value - below_level)
    bg_consistency = np.abs(above_level - below_level)
    bg_stability = np.maximum(np.abs(above_level - above_far), np.abs(below_level - below_far))

    is_border_row = np.zeros(n, dtype=bool)
    is_border_row[:border_n] = True

    qualifies = (is_border_row & (bar_frac >= _SB_MIN_BAR_FRAC) & (bar_frac <= _SB_MAX_BAR_FRAC)
                 & (contrast_above >= rng_row * _SB_CONTRAST_FRAC)
                 & (contrast_below >= rng_row * _SB_CONTRAST_FRAC)
                 & (bg_consistency <= rng_row * _SB_BG_CONSISTENCY_FRAC)
                 & (bg_stability <= rng_row * _SB_BG_CONSISTENCY_FRAC))
    return np.where(qualifies, bar_frac, 0.0).max(axis=-1)


def _edge_scores(arr_f: np.ndarray, tol: np.ndarray, rng_row: np.ndarray, axis_len: int, full_width: int) -> np.ndarray:
    """Max scale_bar_score over the top and bottom borders; only the outer band is ever touched or copied."""
    border_n = max(1, int(round(axis_len * _SB_BORDER_FRAC)))
    margin = max(_SB_MARGIN_MIN, int(round(axis_len * _SB_MARGIN_FRAC)))
    band_n = min(axis_len, border_n + margin + 2 * _SB_MARGIN_WIN)
    near_start = np.ascontiguousarray(arr_f[..., :band_n, :])
    near_end = np.ascontiguousarray(arr_f[..., axis_len - band_n:, :][..., ::-1, :])
    return np.maximum(_edge_band_score(near_start, tol, rng_row, border_n, margin, full_width),
                       _edge_band_score(near_end, tol, rng_row, border_n, margin, full_width))


def scale_bar_score(arr: np.ndarray, cache: Optional[Dict] = None) -> np.ndarray:
    """Heuristic confidence (0-1) that a burned-in horizontal scale bar sits near the top or bottom image border.

    Pure heuristic, no OCR: can miss low-contrast bars and can be fooled by a genuine thin
    high-contrast feature (e.g. a strand) that runs straight through the border region.
    Returns 0 when no qualifying run is found, NaN for images smaller than 8x8.
    """
    h, w = arr.shape[-2], arr.shape[-1]
    if h < _SB_MIN_SIZE or w < _SB_MIN_SIZE:
        return np.full(arr.shape[:-2], np.nan, dtype=np.float32)
    # Bar length is a fraction of the full image width, not just this chunk's width - matters if
    # X ever gets split across chunks (it isn't, by default; see RasterImageProcessor.run_chunk).
    full_width = cache.get("full_width", w) if cache else w
    arr_f = arr.astype(np.float32, copy=False)
    s_min = np.nanmin(arr_f, axis=(-2, -1), keepdims=True)
    s_max = np.nanmax(arr_f, axis=(-2, -1), keepdims=True)
    rng = s_max - s_min
    tol = rng * _SB_TOL_FRAC
    rng_row = np.squeeze(rng, axis=-1)

    score = _edge_scores(arr_f, tol, rng_row, h, full_width)
    return np.where(np.squeeze(rng, axis=(-2, -1)) > 0, score, 0.0).astype(np.float32)
