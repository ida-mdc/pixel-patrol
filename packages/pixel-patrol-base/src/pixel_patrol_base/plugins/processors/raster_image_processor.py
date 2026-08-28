"""
Image-specific raster metric processors: base class and concrete processors
that require Y and X axes (transposed to last before kernel computation).
"""

import warnings
from typing import Any, Dict, Tuple

import numpy as np

from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_base.plugins.processors.raster_image_numpy_metrics import (
    MetricContext,
    _XY_AXES,
    bright_clipping_fraction,
    dark_clipping_fraction,
    laplacian_variance,
    spectral_slope,
)
from pixel_patrol_base.plugins.processors.raster_processor import (
    RasterMetricSpec,
    _weighted_mean_agg,
)


def numpy_image_compute(spec: RasterMetricSpec, arr: np.ndarray, ctx: MetricContext):
    """NumPy backend: compute one image metric on an (..., H, W) array.

    Metric functions reduce over the last two (spatial) axes, returning a value
    per non-spatial leading dim. nanmean collapses those to one scalar for the row.
    """
    with np.errstate(invalid='ignore', divide='ignore'), \
         warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
        warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
        warnings.filterwarnings('ignore', 'Degrees of freedom <= 0', RuntimeWarning)
        match spec.name:
            case "laplacian_variance":       return float(np.nanmean(laplacian_variance(arr, ctx.cache)))
            case "spectral_slope":           return float(np.nanmean(spectral_slope(arr, ctx.cache)))
            case "dark_clipping_fraction":   return float(np.nanmean(dark_clipping_fraction(arr, _XY_AXES, ctx.cache)))
            case "bright_clipping_fraction": return float(np.nanmean(bright_clipping_fraction(arr, _XY_AXES, ctx.cache)))
            case _:                          return None


class RasterImageProcessor:
    """Base class for image processors requiring Y and X at the last two axes."""

    NAME:        str = ""
    DESCRIPTION: str = ""
    CHUNK_KIND  = ChunkKind.LEAF
    METRICS:    Tuple[RasterMetricSpec, ...] = ()
    INPUT       = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT      = "features"
    OUTPUT_SCHEMA: Dict[str, Any] = {}
    IS_MEMORY_HEAVY: bool = False
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = {}

    def run_chunk(self, record: Record) -> Dict:
        chunk = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)
        dim_order_out = record.dim_order
        y_ax = dim_order_out.index("Y")
        x_ax = dim_order_out.index("X")
        if y_ax != len(dim_order_out) - 2 or x_ax != len(dim_order_out) - 1:
            other = [i for i in range(chunk.ndim) if i not in (y_ax, x_ax)]
            chunk = chunk.transpose(other + [y_ax, x_ax])
        ctx = MetricContext(s_min=float(np.nanmin(chunk)), s_max=float(np.nanmax(chunk)))
        return {
            spec.name: val
            for spec in self.METRICS
            if (val := numpy_image_compute(spec, chunk, ctx)) is not None
        }

    def get_aggregation(self, name: str):
        spec = next((s for s in self.METRICS if s.name == name), None)
        if spec is None:
            return None
        return lambda rows, g_dims: spec.aggregate_rows(spec, rows)


class QualityMetricsProcessor(RasterImageProcessor):
    NAME        = "raster-quality"
    DESCRIPTION = "Computes no-reference image quality metrics (sharpness, noise, spectral characteristics, clipping) over the 2D spatial extent of each image."
    IS_MEMORY_HEAVY = True  # measured ~21x peak memory vs. raw chunk bytes
    # Cosmetic order only - actual widget order is qualityMetricRank in plugin_violin.js
    # (kept in sync by hand); parquet columns get alphabetized upstream regardless.
    METRICS = (
        RasterMetricSpec(name="laplacian_variance", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="High-frequency content measure. Low values indicate blur or heavy compression. High values are ambiguous (noise, quantization artifacts, and genuine sharpness all produce high scores). Inflated by clipping boundaries."),
        RasterMetricSpec(name="spectral_slope", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Log-log slope of the radially averaged power spectrum (mid-frequency band). Blur steepens the slope (more negative); noise flattens it (toward zero). The only metric that maps blur and noise to opposite directions."),
        RasterMetricSpec(name="dark_clipping_fraction", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Fraction of pixels at the dtype's minimum representable value (integer types only; NaN for float). Indicates underexposure, background, or nodata."),
        RasterMetricSpec(name="bright_clipping_fraction", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Fraction of pixels at this image's own observed maximum value (all dtypes). Detects bright-end clipping regardless of storage format or bit depth."),
    )
    OUTPUT_SCHEMA = {m.name: m.data_type for m in METRICS}
    OUTPUT_SCHEMA_DESCRIPTIONS = {m.name: m.description for m in METRICS}
