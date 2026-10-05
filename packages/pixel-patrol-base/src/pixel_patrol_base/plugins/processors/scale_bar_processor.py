"""Scale-bar detection processor; metric kernel lives in scale_bar_numpy_metrics.py."""

import warnings

import numpy as np

from pixel_patrol_base.plugins.processors.raster_image_processor import RasterImageProcessor
from pixel_patrol_base.plugins.processors.raster_processor import MetricContext, RasterMetricSpec, _scalar_rows_agg
from pixel_patrol_base.plugins.processors.scale_bar_numpy_metrics import scale_bar_score


def _compute(spec: RasterMetricSpec, arr: np.ndarray, ctx: MetricContext):
    with np.errstate(invalid='ignore', divide='ignore'), \
         warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
        match spec.name:
            case "scale_bar_score": return float(np.nanmax(scale_bar_score(arr, ctx.cache)))
            case _:                 return None


class ScaleBarProcessor(RasterImageProcessor):
    NAME        = "raster-scale-bar"
    DESCRIPTION = "Heuristically scores whether a burned-in horizontal scale bar sits near the top or bottom image border, over the 2D spatial extent of each image."
    COMPUTE     = staticmethod(_compute)
    METRICS = (
        RasterMetricSpec(name="scale_bar_score", data_type=np.float32, aggregate_rows=_scalar_rows_agg(np.nanmax),
                         description="Heuristic confidence (0-1) that a burned-in horizontal scale bar is present near the top or bottom image border. 0 means no qualifying border run was found. Not a calibrated probability; use as a ranking/triage signal, not ground truth."),
    )
    OUTPUT_SCHEMA = {m.name: m.data_type for m in METRICS}
    OUTPUT_SCHEMA_DESCRIPTIONS = {m.name: m.description for m in METRICS}
