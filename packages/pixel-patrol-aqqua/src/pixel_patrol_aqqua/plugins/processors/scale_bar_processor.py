"""Heuristic scale-bar detection processor."""

import warnings
from typing import Dict, Optional

import numpy as np

from pixel_patrol_base.plugins.processors.raster_image_processor import RasterImageProcessor
from pixel_patrol_base.plugins.processors.raster_processor import MetricContext, RasterMetricSpec, _scalar_rows_agg

_SB_BORDER_FRAC = 0.1
_SB_MIN_BAR_FRAC = 0.06
_SB_MAX_BAR_FRAC = 0.95
_SB_TOL_FRAC = 0.02
_SB_MARGIN_FRAC = 0.02
_SB_MARGIN_MIN = 3
_SB_MARGIN_WIN = 3
_SB_CONTRAST_FRAC = 0.12
_SB_BG_CONSISTENCY_FRAC = 0.08
_SB_MIN_SIZE = 8
_SB_COL_TOL = 2
_SB_MIN_GROUP_ROWS = 2


def _find_best_run(row: np.ndarray, tol: float, background: float):
    n = len(row)
    if n == 0:
        return None
    best = None
    best_contrast = -1.0
    cur_start = 0
    for i in range(1, n + 1):
        if i == n or abs(row[i] - row[i - 1]) > tol:
            contrast = abs(row[cur_start] - background)
            if contrast > best_contrast:
                best_contrast = contrast
                best = (cur_start, i - cur_start, float(row[cur_start]))
            cur_start = i
    return best


def _band_score_2d(band: np.ndarray, tol: float, rng: float, border_n: int, margin: int, full_width: int) -> float:
    n = band.shape[0]
    band_median = float(np.median(band[:min(border_n, n)]))
    candidates = []
    for r in range(min(border_n, n)):
        result = _find_best_run(band[r], tol, band_median)
        if result is None:
            continue
        start, length, val = result
        frac = length / full_width
        if _SB_MIN_BAR_FRAC <= frac <= _SB_MAX_BAR_FRAC:
            candidates.append((r, start, start + length - 1, val, frac))
    if not candidates:
        return 0.0
    groups = [[candidates[0]]]
    for cand in candidates[1:]:
        prev = groups[-1][-1]
        if (cand[0] == prev[0] + 1
                and abs(cand[1] - prev[1]) <= _SB_COL_TOL
                and abs(cand[2] - prev[2]) <= _SB_COL_TOL):
            groups[-1].append(cand)
        else:
            groups.append([cand])
    best_score = 0.0
    for group in groups:
        if len(group) < _SB_MIN_GROUP_ROWS:
            continue
        row_min = group[0][0]
        row_max = group[-1][0]
        bar_val = float(np.mean([c[3] for c in group]))
        above_hi = max(0, row_min - margin)
        above_lo = max(0, above_hi - _SB_MARGIN_WIN + 1)
        below_lo = min(n - 1, row_max + margin)
        below_hi = min(n - 1, below_lo + _SB_MARGIN_WIN - 1)
        above_bg = float(np.median(band[above_lo:above_hi + 1]))
        below_bg = float(np.median(band[below_lo:below_hi + 1]))
        contrast_above = abs(bar_val - above_bg)
        contrast_below = abs(bar_val - below_bg)
        bg_consistency = abs(above_bg - below_bg)
        if (contrast_above >= rng * _SB_CONTRAST_FRAC
                and contrast_below >= rng * _SB_CONTRAST_FRAC
                and bg_consistency <= rng * _SB_BG_CONSISTENCY_FRAC):
            best_score = max(best_score, min(contrast_above, contrast_below) / rng)
    return best_score


def _score_2d(arr2d: np.ndarray, full_width: int) -> float:
    h, w = arr2d.shape
    s_min = float(np.nanmin(arr2d))
    s_max = float(np.nanmax(arr2d))
    rng = s_max - s_min
    if rng == 0:
        return 0.0
    tol = rng * _SB_TOL_FRAC
    border_n = max(1, int(round(h * _SB_BORDER_FRAC)))
    margin = max(_SB_MARGIN_MIN, int(round(h * _SB_MARGIN_FRAC)))
    band_n = min(h, border_n + margin + _SB_MARGIN_WIN)
    top_score = _band_score_2d(arr2d[:band_n], tol, rng, border_n, margin, full_width)
    bot_score = _band_score_2d(arr2d[h - band_n:][::-1], tol, rng, border_n, margin, full_width)
    return max(top_score, bot_score)


def scale_bar_score(arr: np.ndarray, cache: Optional[Dict] = None) -> np.ndarray:
    """Heuristic confidence (0-1) that a burned-in horizontal scale bar is near the top or bottom border."""
    h, w = arr.shape[-2], arr.shape[-1]
    if h < _SB_MIN_SIZE or w < _SB_MIN_SIZE:
        return np.full(arr.shape[:-2], np.nan, dtype=np.float32)
    full_width = cache.get("full_width", w) if cache else w
    arr_f = arr.astype(np.float32, copy=False)
    lead_shape = arr_f.shape[:-2]
    if len(lead_shape) == 0:
        return np.float32(_score_2d(arr_f, full_width))
    flat = arr_f.reshape(-1, h, w)
    scores = np.array([_score_2d(flat[i], full_width) for i in range(flat.shape[0])], dtype=np.float32)
    return scores.reshape(lead_shape)


def _compute(spec: RasterMetricSpec, arr: np.ndarray, ctx: MetricContext):
    with np.errstate(invalid='ignore', divide='ignore'), \
         warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
        match spec.name:
            case "scale_bar_score": return float(np.nanmax(scale_bar_score(arr, ctx.cache)))
            case _:                 return None


class ScaleBarProcessor(RasterImageProcessor):
    NAME        = "raster-scale-bar"
    DESCRIPTION = "Heuristically scores whether a burned-in horizontal scale bar sits near the top or bottom image border."
    COMPUTE     = staticmethod(_compute)
    METRICS = (
        RasterMetricSpec(name="scale_bar_score", data_type=np.float32, aggregate_rows=_scalar_rows_agg(np.nanmax),
                         description="Heuristic confidence (0-1) that a burned-in horizontal scale bar is present near the top or bottom image border. 0 means no qualifying bar was found. Not a calibrated probability."),
    )
    OUTPUT_SCHEMA = {m.name: m.data_type for m in METRICS}
    OUTPUT_SCHEMA_DESCRIPTIONS = {m.name: m.description for m in METRICS}
