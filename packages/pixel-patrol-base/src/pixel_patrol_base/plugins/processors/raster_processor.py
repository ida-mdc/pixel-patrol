"""
N-D raster metric processors: shared types, aggregation helpers,
and concrete processors that work on the full leaf chunk without spatial axis assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from pixel_patrol_base.config import HISTOGRAM_BINS
from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec


@dataclass
class MetricContext:
    """Per-chunk compute context shared across all metrics for one run_chunk call."""
    s_min: float
    s_max: float
    cache: Dict = field(default_factory=dict)


class MetricNames(StrEnum):
    MAX_INTENSITY       = "max_intensity"
    MIN_INTENSITY       = "min_intensity"
    MEAN_INTENSITY      = "mean_intensity"
    STD_INTENSITY       = "std_intensity"
    FINITE_PIXEL_COUNT  = "finite_pixel_count"
    HISTOGRAM_MIN       = "histogram_min"
    HISTOGRAM_MAX       = "histogram_max"
    HISTOGRAM_NAN_COUNT = "histogram_nan_count"
    HISTOGRAM_COUNTS    = "histogram_counts"


@dataclass(frozen=True, slots=True)
class RasterMetricSpec:
    """Declares one output column: its type, how to aggregate it across rows, and a human description."""
    name:           str
    data_type:      Any
    aggregate_rows: Callable[[RasterMetricSpec, List[Dict[str, Any]]], Any]
    description:    str = ""


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _scalar_rows_agg(fn):
    def agg(spec: RasterMetricSpec, rows: List[Dict]) -> Any:
        vals = [r[spec.name] for r in rows if spec.name in r]
        if not vals:
            return None
        arr = np.asarray(vals, dtype=float)
        return fn(arr) if np.any(np.isfinite(arr)) else np.nan
    return agg


def _pooled_std_agg(spec: RasterMetricSpec, rows: List[Dict]) -> Any:
    """Pooled std: σ = √(Σ nᵢ·(σᵢ² + (μᵢ − μ)²) / Σ nᵢ)."""
    ns, means, stds = [], [], []
    for r in rows:
        if spec.name not in r or MetricNames.MEAN_INTENSITY not in r:
            continue
        n = float(r.get(MetricNames.FINITE_PIXEL_COUNT, 0) or 0)
        if n <= 0:
            continue
        s, m = float(r[spec.name]), float(r[MetricNames.MEAN_INTENSITY])
        if np.isfinite(s) and np.isfinite(m):
            ns.append(n); means.append(m); stds.append(s)
    if not ns:
        return None
    ns, means, stds = np.array(ns), np.array(means), np.array(stds)
    total = ns.sum()
    mu = (ns * means).sum() / total
    return float(np.sqrt(max(0.0, (ns * (stds**2 + (means - mu)**2)).sum() / total)))


def _integer_sum_agg(spec: RasterMetricSpec, rows: List[Dict]) -> Any:
    """Integer-preserving sum - keeps count columns as int so polars schema stays consistent."""
    vals = [r[spec.name] for r in rows if spec.name in r and r[spec.name] is not None]
    if not vals:
        return None
    return int(sum(int(v) for v in vals))


def _weighted_mean_agg(spec: RasterMetricSpec, rows: List[Dict]) -> Any:
    """Pixel-count weighted mean."""
    num = den = 0.0
    for r in rows:
        if spec.name not in r:
            continue
        w = float(r.get(MetricNames.FINITE_PIXEL_COUNT, 0) or 0)
        if w <= 0:
            continue
        v = float(r[spec.name])
        if np.isfinite(v):
            num += v * w; den += w
    if den > 0:
        return num / den
    vals = [float(r[spec.name]) for r in rows if spec.name in r and np.isfinite(r[spec.name])]
    return float(np.nanmean(vals)) if vals else None


def _aggregate_histograms(rows: List[Dict]) -> Any:
    """Merge per-chunk histograms onto a shared range via bin-centre remapping."""
    h_mins   = [r[MetricNames.HISTOGRAM_MIN]    for r in rows if MetricNames.HISTOGRAM_MIN    in r]
    h_maxs   = [r[MetricNames.HISTOGRAM_MAX]    for r in rows if MetricNames.HISTOGRAM_MAX    in r]
    h_counts = [r[MetricNames.HISTOGRAM_COUNTS] for r in rows if MetricNames.HISTOGRAM_COUNTS in r]
    if not h_counts:
        return None
    g_min, g_max = np.nanmin(h_mins), np.nanmax(h_maxs)
    if g_min == g_max:
        res = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
        res[0] = sum(int(np.sum(c)) for c in h_counts)
        return res
    if all(np.isclose(m, g_min, rtol=1e-9, atol=1e-9) for m in h_mins) and \
       all(np.isclose(m, g_max, rtol=1e-9, atol=1e-9) for m in h_maxs):
        return np.sum(np.stack([np.asarray(c, dtype=np.int64) for c in h_counts], axis=0), axis=0)
    target_bins = np.linspace(g_min, g_max, HISTOGRAM_BINS + 1)
    combined = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    for counts, s_min, s_max in zip(h_counts, h_mins, h_maxs):
        if s_min == s_max:
            idx = np.clip(np.searchsorted(target_bins, s_min) - 1, 0, HISTOGRAM_BINS - 1)
            combined[idx] += int(np.sum(counts))
            continue
        src_centers = np.linspace(s_min, s_max, len(counts), endpoint=False) + (s_max - s_min) / len(counts) / 2
        tgt = np.clip(np.searchsorted(target_bins, src_centers) - 1, 0, HISTOGRAM_BINS - 1)
        np.add.at(combined, tgt, np.asarray(counts, dtype=np.int64))
    return combined


# ---------------------------------------------------------------------------
# NumPy n-D backend
# ---------------------------------------------------------------------------

def _hist_bounds(arr: np.ndarray, s_min: float, s_max: float) -> Tuple[float, float]:
    """Value range the histogram bins span. Integer data whose span is narrower than
    the (fixed) bin count is widened to exactly HISTOGRAM_BINS so every bin is one
    integer level wide - otherwise 256 bins over a handful of levels produce a sparse
    'comb' of empty bins. Float data (and integer data spanning ≥ HISTOGRAM_BINS
    levels, where bins are already ≥ 1 wide) keeps its true span.

    The widened window is anchored to the dtype's representable range: it slides left
    so the top edge never lands past dtype.max + 1, so e.g. any uint8 image bins over
    [0, 256) to match the data type rather than spilling outside it."""
    if np.issubdtype(arr.dtype, np.floating) or (s_max - s_min) >= HISTOGRAM_BINS:
        return s_min, s_max
    try:
        info = np.iinfo(arr.dtype)
    except ValueError:      # non-integer, non-float (e.g. bool): no representable range to anchor to
        return s_min, s_min + HISTOGRAM_BINS
    lo = min(s_min, float(info.max) + 1 - HISTOGRAM_BINS)   # slide left so the top edge stays in-range
    lo = max(lo, float(info.min))                           # ...but never below the dtype minimum
    return lo, lo + HISTOGRAM_BINS


def _histogram_counts(arr: np.ndarray, s_min: float, s_max: float) -> np.ndarray:
    B = HISTOGRAM_BINS
    flat = arr.ravel().astype(np.float32, copy=False)
    finite = flat[np.isfinite(flat)] if np.issubdtype(arr.dtype, np.floating) else flat
    h = np.zeros(B, dtype=np.int64)
    if s_min < s_max:
        bins = np.clip(np.floor((finite - s_min) / ((s_max - s_min) / B)).astype(np.int32), 0, B - 1)
        np.add.at(h, bins, 1)
    else:
        h[0] = len(finite)
    return h


def numpy_compute(spec: RasterMetricSpec, arr: np.ndarray, ctx: MetricContext):
    """NumPy backend: compute one n-D metric on the chunk."""
    match spec.name:
        case MetricNames.MIN_INTENSITY:      return float(np.nanmin(arr))
        case MetricNames.MAX_INTENSITY:      return float(np.nanmax(arr))
        case MetricNames.MEAN_INTENSITY:     return float(np.nanmean(arr))
        case MetricNames.STD_INTENSITY:      return float(np.nanstd(arr))
        case MetricNames.FINITE_PIXEL_COUNT: return int(np.sum(np.isfinite(arr)))
        case MetricNames.HISTOGRAM_MIN:      return float(_hist_bounds(arr, ctx.s_min, ctx.s_max)[0])
        case MetricNames.HISTOGRAM_MAX:      return float(_hist_bounds(arr, ctx.s_min, ctx.s_max)[1])
        case MetricNames.HISTOGRAM_NAN_COUNT:
            return int(np.sum(np.isnan(arr))) if np.issubdtype(arr.dtype, np.floating) else 0
        case MetricNames.HISTOGRAM_COUNTS:   return _histogram_counts(arr, *_hist_bounds(arr, ctx.s_min, ctx.s_max))
        case _:                              return None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class RasterProcessor:
    NAME:        str = ""
    DESCRIPTION: str = ""
    CHUNK_KIND  = ChunkKind.LEAF
    METRICS:    Tuple[RasterMetricSpec, ...] = ()
    INPUT       = RecordSpec(axes=set(), kinds={"intensity"})
    OUTPUT      = "features"
    OUTPUT_SCHEMA: Dict[str, Any] = {}
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = {}
    IS_MEMORY_HEAVY: bool = False

    def run_chunk(self, record: Record) -> Dict:
        chunk = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)
        ctx = MetricContext(s_min=float(np.nanmin(chunk)), s_max=float(np.nanmax(chunk)))
        return {
            spec.name: val
            for spec in self.METRICS
            if (val := numpy_compute(spec, chunk, ctx)) is not None
        }

    def get_aggregation(self, name: str):
        spec = next((s for s in self.METRICS if s.name == name), None)
        if spec is None:
            return None
        return lambda rows, g_dims: spec.aggregate_rows(spec, rows)


# ---------------------------------------------------------------------------
# Concrete processors
# ---------------------------------------------------------------------------

class BasicMetricsProcessor(RasterProcessor):
    NAME        = "raster-basic"
    DESCRIPTION = "Computes basic per-image intensity statistics (min, max, mean, std) and the finite-pixel count, aggregated across chunks and dimensions."
    METRICS = (
        RasterMetricSpec(name=MetricNames.MIN_INTENSITY,      data_type=np.float32, aggregate_rows=_scalar_rows_agg(np.nanmin),
                         description="Minimum pixel intensity over the covered extent (ignoring NaNs)."),
        RasterMetricSpec(name=MetricNames.MAX_INTENSITY,      data_type=np.float32, aggregate_rows=_scalar_rows_agg(np.nanmax),
                         description="Maximum pixel intensity over the covered extent (ignoring NaNs)."),
        RasterMetricSpec(name=MetricNames.MEAN_INTENSITY,     data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Pixel-count-weighted mean intensity over the covered extent."),
        RasterMetricSpec(name=MetricNames.STD_INTENSITY,      data_type=np.float32, aggregate_rows=_pooled_std_agg,
                         description="Pooled standard deviation of intensity over the covered extent."),
        RasterMetricSpec(name=MetricNames.FINITE_PIXEL_COUNT, data_type=np.uint64,  aggregate_rows=_integer_sum_agg,
                         description="Number of finite (non-NaN/Inf) pixels contributing to the statistics."),
    )
    OUTPUT_SCHEMA = {m.name: m.data_type for m in METRICS}
    OUTPUT_SCHEMA_DESCRIPTIONS = {m.name: m.description for m in METRICS}


class HistogramProcessor(RasterProcessor):
    NAME        = "raster-histogram"
    DESCRIPTION = "Computes a per-image intensity histogram, merging per-chunk histograms onto a shared value range."
    METRICS = (
        RasterMetricSpec(name=MetricNames.HISTOGRAM_MIN,       data_type=np.float32, aggregate_rows=_scalar_rows_agg(np.nanmin),
                         description="Lower bound of the histogram's value range."),
        RasterMetricSpec(name=MetricNames.HISTOGRAM_MAX,       data_type=np.float32, aggregate_rows=_scalar_rows_agg(np.nanmax),
                         description="Upper bound of the histogram's value range."),
        RasterMetricSpec(name=MetricNames.HISTOGRAM_NAN_COUNT, data_type=np.uint64,  aggregate_rows=_integer_sum_agg,
                         description="Number of NaN pixels excluded from the histogram."),
        RasterMetricSpec(name=MetricNames.HISTOGRAM_COUNTS,    data_type=np.ndarray, aggregate_rows=lambda spec, rows: _aggregate_histograms(rows),
                         description="Per-bin pixel counts over the histogram value range (fixed number of bins; integer data uses one-level-wide bins)."),
    )
    OUTPUT_SCHEMA = {m.name: m.data_type for m in METRICS}
    OUTPUT_SCHEMA_DESCRIPTIONS = {m.name: m.description for m in METRICS}
