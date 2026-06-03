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
from pixel_patrol_image.plugins.processors.raster_image_numpy_metrics import (
    MetricContext,
    _XY_AXES,
    calc_blocking,
    calc_ringing,
    laplacian_variance,
    local_std_ratio,
    michelson_contrast,
    mscn_variance,
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
            case "michelson_contrast": return float(np.nanmean(michelson_contrast(arr, _XY_AXES)))
            case "mscn_variance":      return float(np.nanmean(mscn_variance(arr, _XY_AXES, ctx.cache)))
            case "local_std_ratio":    return float(np.nanmean(local_std_ratio(arr, _XY_AXES, ctx.cache)))
            case "laplacian_variance": return float(np.nanmean(laplacian_variance(arr)))
            case "blocking_index":     return float(np.nanmean(calc_blocking(arr)))
            case "ringing_index":      return float(np.nanmean(calc_ringing(arr)))
            case _:                    return None


class RasterImageProcessor:
    """Base class for image processors requiring Y and X at the last two axes."""

    NAME:       str = ""
    CHUNK_KIND  = ChunkKind.LEAF
    METRICS:    Tuple[RasterMetricSpec, ...] = ()
    INPUT       = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT      = "features"
    OUTPUT_SCHEMA: Dict[str, Any] = {}

    def run_chunk(self, record: Record) -> Dict:
        chunk = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)
        dim_order_out = list(record.dim_order)
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
    NAME    = "raster-quality"
    METRICS = (
        RasterMetricSpec(name="michelson_contrast", data_type=np.float32, aggregate_rows=_weighted_mean_agg),
        RasterMetricSpec(name="mscn_variance",      data_type=np.float32, aggregate_rows=_weighted_mean_agg),
        RasterMetricSpec(name="local_std_ratio",    data_type=np.float32, aggregate_rows=_weighted_mean_agg),
        RasterMetricSpec(name="laplacian_variance", data_type=np.float32, aggregate_rows=_weighted_mean_agg),
    )
    OUTPUT_SCHEMA = {m.name: m.data_type for m in METRICS}


class CompressionMetricsProcessor(RasterImageProcessor):
    NAME    = "raster-compression"
    METRICS = (
        RasterMetricSpec(name="blocking_index", data_type=np.float32, aggregate_rows=_weighted_mean_agg),
        RasterMetricSpec(name="ringing_index",  data_type=np.float32, aggregate_rows=_weighted_mean_agg),
    )
    OUTPUT_SCHEMA = {m.name: m.data_type for m in METRICS}
