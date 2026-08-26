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
    estimated_noise_std,
    fraction_at_image_max,
    jpeg_block_ratio,
    laplacian_variance,
    min_value_pixel_fraction,
    ringing_index,
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
            case "ringing_index":            return float(np.nanmean(ringing_index(arr, ctx.cache)))
            case "estimated_noise_std":      return float(np.nanmean(estimated_noise_std(arr, _XY_AXES, ctx.cache)))
            case "jpeg_block_ratio":         return float(np.nanmean(jpeg_block_ratio(arr, ctx.cache)))
            case "min_value_pixel_fraction": return float(np.nanmean(min_value_pixel_fraction(arr, _XY_AXES, ctx.cache)))
            case "fraction_at_image_max":    return float(np.nanmean(fraction_at_image_max(arr, _XY_AXES, ctx.cache)))
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
    DESCRIPTION = "Computes no-reference image quality metrics (sharpness, noise, compression artifacts, saturation) over the 2D spatial extent of each image."
    IS_MEMORY_HEAVY = True  # measured ~21x peak memory vs. raw chunk bytes
    # Cosmetic order only - actual widget order is qualityMetricRank in plugin_violin.js
    # (kept in sync by hand); parquet columns get alphabetized upstream regardless.
    METRICS = (
        RasterMetricSpec(name="laplacian_variance", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Sharpness/focus score. Saturated and min-value pixels are excluded before computation. Low values usually mean blur or defocus; high values can also result from artificial discontinuities."),
        RasterMetricSpec(name="ringing_index", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Variance of the residual after subtracting a 3x3 local mean. Elevated by ringing artifacts and noise."),
        RasterMetricSpec(name="estimated_noise_std", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="High-frequency variation estimate (Immerkaer 1996). Responds to noise and to natural texture -- the absolute value is only meaningful when comparing images of the same content type."),
        RasterMetricSpec(name="jpeg_block_ratio", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Ratio of mean pixel-difference at 8-pixel block boundaries vs within-block. Values near 1.0 are normal; values above ~1.5 suggest JPEG-style 8x8 blocking artifacts."),
        RasterMetricSpec(name="min_value_pixel_fraction", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Fraction of pixels at the dtype's minimum representable value (integer types only; NaN for float)."),
        RasterMetricSpec(name="fraction_at_image_max", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Fraction of pixels at this image's own observed maximum value (all dtypes). Detects clipping regardless of storage format or bit depth."),
    )
    OUTPUT_SCHEMA = {m.name: m.data_type for m in METRICS}
    OUTPUT_SCHEMA_DESCRIPTIONS = {m.name: m.description for m in METRICS}
