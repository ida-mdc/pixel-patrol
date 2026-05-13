"""
Which raster metrics exist, how toggles turn groups on or off, and how tile rows roll up into summaries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Dict, FrozenSet, List, Tuple

import numpy as np

from pixel_patrol_base.config import HISTOGRAM_BINS

# Environment variable: whether to emit individual XY tile rows or only rolled-up summaries.
RASTER_TILE_ROWS_ENV_VAR = "PIXEL_PATROL_RASTER_XY_TILE_METRICS"

AGG_SKIP = object()

SPATIAL_AXIS_DIMS = frozenset({"dim_x", "dim_y"})


class MetricNames(StrEnum):
    """Stable names for columns produced by the raster tile processor."""
    MAX_INTENSITY = "max_intensity"
    MIN_INTENSITY = "min_intensity"
    MEAN_INTENSITY = "mean_intensity"
    STD_INTENSITY = "std_intensity"
    FINITE_PIXEL_COUNT = "finite_pixel_count"
    MICHELSON_CONTRAST = "michelson_contrast"
    MSCN_VARIANCE = "mscn_variance"
    LOCAL_STD_RATIO = "local_std_ratio"
    HISTOGRAM_MIN = "histogram_min"
    HISTOGRAM_MAX = "histogram_max"
    HISTOGRAM_NAN_COUNT = "histogram_nan_count"
    HISTOGRAM_COUNTS = "histogram_counts"
    BLOCKING_INDEX = "blocking_index"
    RINGING_INDEX = "ringing_index"


def _scalar_rows_agg(fn: Callable[[List[Any]], Any]) -> Callable[[RasterMetricSpec, List[Dict[str, Any]]], Any]:
    """Build a rollup rule that gathers one number per child row and folds them with the given reducer."""

    def agg(spec: RasterMetricSpec, rows: List[Dict[str, Any]]) -> Any:
        vals = [r[spec.name] for r in rows if spec.name in r]
        return fn(vals) if vals else AGG_SKIP

    return agg


def _histogram_counts_agg(spec: RasterMetricSpec, rows: List[Dict[str, Any]]) -> Any:
    """Combine brightness histograms from many tiles into one shared bucket layout."""
    if not any(spec.name in r for r in rows):
        return AGG_SKIP
    return aggregate_histograms(rows)


def _weighted_mean_intensity_agg(spec: "RasterMetricSpec", rows: List[Dict[str, Any]]) -> Any:
    """Average brightness across tiles by counting real pixels per tile so large tiles matter more."""
    num = 0.0
    den = 0.0
    for r in rows:
        if spec.name not in r:
            continue
        if MetricNames.FINITE_PIXEL_COUNT not in r:
            continue
        w = float(r[MetricNames.FINITE_PIXEL_COUNT])
        if w <= 0:
            continue
        fn = float(r[spec.name])
        if not np.isfinite(fn):
            continue
        num += fn * w
        den += w
    if den <= 0:
        vals = [float(r[spec.name]) for r in rows if spec.name in r and np.isfinite(r[spec.name])]
        return float(np.nanmean(vals)) if vals else AGG_SKIP
    return num / den


@dataclass(frozen=True, slots=True)
class RasterMetricEnvGroup:
    """Bundle of metric columns switched together by one environment flag."""

    env_var: str
    members: FrozenSet[str]
    default_enabled: bool

    def enabled(self) -> bool:
        """Whether this bundle is currently turned on (reads the environment, falls back to defaults)."""
        default = "1" if self.default_enabled else "0"
        return os.environ.get(self.env_var, default) == "1"


@dataclass(frozen=True, slots=True)
class RasterMetricSpec:
    """Describes one output column: whether it appears on spatial tile rows and how children combine."""

    name: str
    data_type: Any
    is_spatial: bool
    aggregate_rows: Callable[["RasterMetricSpec", List[Dict[str, Any]]], Any]
    skip_aggregate_if_group_dims_intersects: FrozenSet[str] = frozenset()


RASTER_METRIC_ENV_GROUPS: Tuple[RasterMetricEnvGroup, ...] = (
    RasterMetricEnvGroup(
        "PIXEL_PATROL_METRICS_COMPRESSION",
        frozenset(
            {
                MetricNames.BLOCKING_INDEX,
                MetricNames.RINGING_INDEX
            }
        ),
        default_enabled=False,
    ),
    RasterMetricEnvGroup(
        "PIXEL_PATROL_METRICS_BASIC",
        frozenset(
            {
                MetricNames.MIN_INTENSITY,
                MetricNames.MAX_INTENSITY,
                MetricNames.MEAN_INTENSITY,
                MetricNames.STD_INTENSITY,
                MetricNames.FINITE_PIXEL_COUNT,
            }
        ),
        default_enabled=True,
    ),
    RasterMetricEnvGroup(
        "PIXEL_PATROL_METRICS_QUALITY",
        frozenset(
            {
                MetricNames.MICHELSON_CONTRAST,
                MetricNames.MSCN_VARIANCE,
                MetricNames.LOCAL_STD_RATIO,
            }
        ),
        default_enabled=True,
    ),
    RasterMetricEnvGroup(
        "PIXEL_PATROL_METRICS_HISTOGRAM",
        frozenset(
            {
                MetricNames.HISTOGRAM_NAN_COUNT,
                MetricNames.HISTOGRAM_COUNTS,
                MetricNames.HISTOGRAM_MIN,
                MetricNames.HISTOGRAM_MAX,
            }
        ),
        default_enabled=True,
    ),
)


RASTER_METRIC_REGISTRY: Tuple[RasterMetricSpec, ...] = (
    RasterMetricSpec(
        name=MetricNames.MIN_INTENSITY,
        data_type=np.float32,
        is_spatial=True,
        aggregate_rows=_scalar_rows_agg(np.nanmin),
    ),
    RasterMetricSpec(
        name=MetricNames.MAX_INTENSITY,
        data_type=np.float32,
        is_spatial=True,
        aggregate_rows=_scalar_rows_agg(np.nanmax),
    ),
    RasterMetricSpec(
        name=MetricNames.MEAN_INTENSITY,
        data_type=np.float32,
        is_spatial=True,
        aggregate_rows=_weighted_mean_intensity_agg,
    ),
    RasterMetricSpec(
        name=MetricNames.STD_INTENSITY,
        data_type=np.float32,
        is_spatial=True,
        # Averaged across tiles, not the true pooled std — sufficient for QC comparison.
        aggregate_rows=_scalar_rows_agg(np.nanmean),
    ),
    RasterMetricSpec(
        name=MetricNames.FINITE_PIXEL_COUNT,
        data_type=np.uint64,
        is_spatial=True,
        aggregate_rows=_scalar_rows_agg(np.sum),
    ),
    RasterMetricSpec(
        name=MetricNames.MICHELSON_CONTRAST,
        data_type=np.float32,
        is_spatial=True,
        aggregate_rows=_scalar_rows_agg(np.nanmean),
    ),
    RasterMetricSpec(
        name=MetricNames.MSCN_VARIANCE,
        data_type=np.float32,
        is_spatial=True,
        aggregate_rows=_scalar_rows_agg(np.nanmean),
    ),
    RasterMetricSpec(
        name=MetricNames.LOCAL_STD_RATIO,
        data_type=np.float32,
        is_spatial=True,
        aggregate_rows=_scalar_rows_agg(np.nanmean),
    ),
    RasterMetricSpec(
        name=MetricNames.HISTOGRAM_MIN,
        data_type=np.float32,
        is_spatial=True,
        aggregate_rows=_scalar_rows_agg(np.nanmin),
    ),
    RasterMetricSpec(
        name=MetricNames.HISTOGRAM_MAX,
        data_type=np.float32,
        is_spatial=True,
        aggregate_rows=_scalar_rows_agg(np.nanmax),
    ),
    RasterMetricSpec(
        name=MetricNames.HISTOGRAM_NAN_COUNT,
        data_type=np.uint64,
        is_spatial=True,
        aggregate_rows=_scalar_rows_agg(np.sum),
    ),
    RasterMetricSpec(
        name=MetricNames.HISTOGRAM_COUNTS,
        data_type=np.ndarray,
        is_spatial=False,
        skip_aggregate_if_group_dims_intersects=SPATIAL_AXIS_DIMS,
        aggregate_rows=_histogram_counts_agg,
    ),
    RasterMetricSpec(
        name=MetricNames.BLOCKING_INDEX,
        data_type=np.float32,
        is_spatial=True,
        aggregate_rows=_scalar_rows_agg(np.nanmean),
    ),
    RasterMetricSpec(
        name=MetricNames.RINGING_INDEX,
        data_type=np.float32,
        is_spatial=True,
        aggregate_rows=_scalar_rows_agg(np.nanmean),
    ),
)


# Histogram bounds copied onto each row come from slice context entries named ``min`` and ``max``.
CTX_TILE_FIELDS: Tuple[Tuple[str, str], ...] = (
    (MetricNames.HISTOGRAM_MIN, "min"),
    (MetricNames.HISTOGRAM_MAX, "max"),
)


def enabled_raster_metrics() -> FrozenSet[RasterMetricSpec]:
    """Return definitions for every statistic that environment switches currently turn on."""
    enabled: set[RasterMetricSpec] = set()
    for g in RASTER_METRIC_ENV_GROUPS:
        if g.enabled():
            enabled.update({m for m in RASTER_METRIC_REGISTRY if m.name in g.members})
    return frozenset(enabled)


def spatial_metric_keys_for_tile_rows() -> FrozenSet[str]:
    """Column keys allowed when emitting detailed XY tile records."""
    keys = {spec.name for spec in RASTER_METRIC_REGISTRY if spec.is_spatial}
    keys.update(f for f, _ in CTX_TILE_FIELDS)
    return frozenset(keys)


def enabled_ctx_tile_fields(enabled_metrics: FrozenSet[RasterMetricSpec]) -> Tuple[Tuple[str, str], ...]:
    """Which histogram boundary copies belong on tile rows given what else is enabled."""
    return tuple((fn, ck) for fn, ck in CTX_TILE_FIELDS if fn in [m.name for m in enabled_metrics])


def aggregate_metrics_for_group(
    rows: List[Dict[str, Any]],
    group_dim_names: Tuple[str, ...],
) -> Dict[str, Any]:
    """Turn many sibling tile summaries into one parent summary using each metric's rollup rule."""
    preserved_dims = frozenset(group_dim_names)
    out: Dict[str, Any] = {}
    for spec in RASTER_METRIC_REGISTRY:
        if preserved_dims & spec.skip_aggregate_if_group_dims_intersects:
            continue
        val = spec.aggregate_rows(spec, rows)
        if val is not AGG_SKIP:
            out[spec.name] = val
    return out

def aggregate_histograms(rows: List[Dict]) -> np.ndarray:
    """Merge per-tile histograms into one histogram covering the full value range across all tiles.

    Each tile was binned using its own slice's global min/max as the range.  When all tiles
    share the same range the bin boundaries align and counts can simply be summed.  When tiles
    have different ranges (e.g. different Z-planes with different dynamic ranges), each tile's
    bins are remapped onto a common target ladder by placing each source bin at its center value.
    This remapping is an approximation — it is accurate enough for visualisation but would not
    reproduce the histogram you would get by re-binning all raw pixels together.
    """
    if not rows:
        return np.zeros(HISTOGRAM_BINS)

    h_mins   = [r[MetricNames.HISTOGRAM_MIN]    for r in rows if MetricNames.HISTOGRAM_MIN    in r]
    h_maxs   = [r[MetricNames.HISTOGRAM_MAX]    for r in rows if MetricNames.HISTOGRAM_MAX    in r]
    h_counts = [r[MetricNames.HISTOGRAM_COUNTS] for r in rows if MetricNames.HISTOGRAM_COUNTS in r]

    if not h_counts:
        return np.zeros(HISTOGRAM_BINS)

    g_min, g_max = np.nanmin(h_mins), np.nanmax(h_maxs)

    if g_min == g_max:
        res = np.zeros(HISTOGRAM_BINS)
        res[0] = sum(np.sum(c) for c in h_counts)
        return res

    # Fast path: all tiles used the same brightness range, so their bin edges align exactly.
    if len(h_mins) == len(h_maxs) == len(h_counts) and all(
        np.isclose(m, g_min, rtol=0.0, atol=1e-9, equal_nan=True) for m in h_mins
    ) and all(np.isclose(m, g_max, rtol=0.0, atol=1e-9, equal_nan=True) for m in h_maxs):
        return np.sum(np.stack([np.asarray(c, dtype=np.float64) for c in h_counts], axis=0), axis=0)

    # Slow path: remap each tile's bins onto the shared target ladder via bin center values.
    target_bins = np.linspace(g_min, g_max, HISTOGRAM_BINS + 1)
    combined = np.zeros(HISTOGRAM_BINS)

    for counts, s_min, s_max in zip(h_counts, h_mins, h_maxs):
        if s_min == s_max:
            idx = np.clip(np.searchsorted(target_bins, s_min) - 1, 0, HISTOGRAM_BINS - 1)
            combined[idx] += np.sum(counts)
            continue

        bin_width  = (s_max - s_min) / len(counts)
        # Center value of each source bin (left edge + half width)
        src_centers = np.linspace(s_min, s_max, len(counts), endpoint=False) + bin_width / 2
        # Find which target bin each source bin center falls into
        target_indices = np.clip(np.searchsorted(target_bins, src_centers) - 1, 0, HISTOGRAM_BINS - 1)

        for src_bin, count in enumerate(counts):
            combined[target_indices[src_bin]] += count

    return combined
