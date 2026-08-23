"""
Image-specific raster metric processors: base class and concrete processors
that require Y and X axes (transposed to last before kernel computation).
"""

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np

from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_base.plugins.processors.raster_image_numpy_metrics import (
    MetricContext,
    _XY_AXES,
    compression_blocking_score,
    estimated_noise_std,
    laplacian_variance,
    local_range_contrast_variability,
    local_texture_uniformity,
    min_value_pixel_fraction,
    saturated_pixel_fraction,
    sobel_gradient_sharpness,
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
        match spec.name:
            case "local_range_contrast_variability": return float(np.nanmean(local_range_contrast_variability(arr, _XY_AXES, ctx.cache)))
            case "local_texture_uniformity": return float(np.nanmean(local_texture_uniformity(arr, _XY_AXES, ctx.cache)))
            case "laplacian_variance":       return float(np.nanmean(laplacian_variance(arr, ctx.cache)))
            case "sobel_gradient_sharpness": return float(np.nanmean(sobel_gradient_sharpness(arr, _XY_AXES, ctx.cache)))
            case "estimated_noise_std":      return float(np.nanmean(estimated_noise_std(arr, _XY_AXES, ctx.cache)))
            case "saturated_pixel_fraction": return float(np.nanmean(saturated_pixel_fraction(arr, _XY_AXES, ctx.cache)))
            case "min_value_pixel_fraction": return float(np.nanmean(min_value_pixel_fraction(arr, _XY_AXES, ctx.cache)))
            case "compression_blocking_score": return float(np.nanmean(compression_blocking_score(arr, ctx.cache)))
            case _:                    return None


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
    DESCRIPTION = "Computes no-reference image quality metrics (sharpness, noise, contrast, texture, saturation, compression artifacts) over the 2D spatial extent of each image."
    IS_MEMORY_HEAVY = True  # measured ~21x peak memory vs. raw chunk bytes
    # Cosmetic order only - actual widget order is qualityMetricRank in plugin_violin.js
    # (kept in sync by hand); parquet columns get alphabetized upstream regardless.
    METRICS = (
        RasterMetricSpec(name="laplacian_variance", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Sharpness/focus score. Low values usually mean the image is blurry or out of focus."),
        RasterMetricSpec(name="sobel_gradient_sharpness", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="A second sharpness score. Use alongside laplacian_variance; when the two disagree it's usually about edge orientation in the image, not a measurement error."),
        RasterMetricSpec(name="estimated_noise_std", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Estimated noise level. Higher means grainier/noisier - check sensor gain, exposure time, or lighting."),
        RasterMetricSpec(name="saturated_pixel_fraction", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Fraction of pixels fully overexposed (blown out). Any real detail there is lost, not just dim."),
        RasterMetricSpec(name="min_value_pixel_fraction", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Fraction of pixels at the minimum representable value for the pixel type. In natural images this often means underexposure; in scientific data it may be expected background or missing values."),
        RasterMetricSpec(name="local_range_contrast_variability", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Local contrast score. Low values mean the image looks flat - check for underexposure, overexposure, or a genuinely low-contrast sample."),
        RasterMetricSpec(name="local_texture_uniformity", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="How evenly detail/texture is spread across the image. High values mean some regions are richly textured while others are flat."),
        RasterMetricSpec(name="compression_blocking_score", data_type=np.float32, aggregate_rows=_weighted_mean_agg,
                         description="Strength of JPEG-style blocky artifacts. Non-zero on data that should be lossless (TIFF etc.) usually means it was compressed somewhere along the way."),
    )
    OUTPUT_SCHEMA = {m.name: m.data_type for m in METRICS}
    OUTPUT_SCHEMA_DESCRIPTIONS = {m.name: m.description for m in METRICS}
