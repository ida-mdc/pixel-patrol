"""
Measure raster statistics from lazy arrays: pick chunks, tile them, then summarize at every useful grouping of dimensions.
"""

import logging
import os
import time
from typing import Any, Dict, List

import dask.array as da
import numpy as np

_log = logging.getLogger(__name__)

from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_image.plugins.processors.processor_block_utils import (
    RASTER_TILE_ROWS_ENV_VAR,
    accumulate_power_set,
    annotate_obs_shape,
    iter_blocks,
    rechunk_for_tiling,
)
from pixel_patrol_image.plugins.processors.raster_image_numpy_metrics import NumpyRasterBackend
from pixel_patrol_image.plugins.processors.raster_metric_definitions import (
    aggregate_metrics_for_group,
    enabled_raster_metrics,
    spatial_metric_keys_for_tile_rows,
)


class RasterImageDaskProcessor:
    """Entry point that reshapes an image record for tiling, runs tile math, and builds the rollup tree."""

    NAME = "raster-image"
    INPUT = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"

    OUTPUT_SCHEMA: Dict[str, Any] = {m.name: m.data_type for m in enabled_raster_metrics()}

    def run(self, art: Record) -> List[Dict]:
        """Compute metrics for the whole volume down through tiles; returns flat rows tagged by depth and dimensions."""
        logger_seconds = float(os.environ.get("PIXEL_PATROL_PROGRESS_LOG_SECONDS", "60"))
        t_run0 = time.perf_counter()

        arr = da.asarray(art.data)
        dim_order = [d.upper() for d in art.dim_order]
        y_ax, x_ax = dim_order.index("Y"), dim_order.index("X")
        arr = da.moveaxis(arr, [y_ax, x_ax], [-2, -1])
        dim_names = [f"dim_{d.lower()}" for d in dim_order if d not in ("Y", "X")] + ["dim_y", "dim_x"]

        if logger_seconds > 0:
            src = getattr(art, "source_path", None) or getattr(art, "path", None) or "unknown"
            _log.info("RasterImageDaskProcessor: begin record=%s shape=%s dtype=%s chunks=%s",
                      src, arr.shape, arr.dtype, arr.chunks)

        t0 = time.perf_counter()
        s_min, s_max = da.compute(arr.min(axis=(-2, -1)), arr.max(axis=(-2, -1)))
        if logger_seconds > 0:
            _log.info("RasterImageDaskProcessor: computed min/max in %.2fs", time.perf_counter() - t0)

        s_tile = int(os.environ.get("PIXEL_PATROL_STATS_TILE_SIZE", 256))
        arr = rechunk_for_tiling(arr, s_tile)

        array_backend = NumpyRasterBackend(arr, s_tile, s_min, s_max, enabled_raster_metrics(), dim_names)

        base_rows: List[Dict] = []
        for b_idx, block_np in iter_blocks(arr, arr.ndim - 2, t_run0, logger_seconds,
                                           desc="  Raster tiles"):
            base_rows.extend(array_backend.process(b_idx, precomputed=block_np))

        out = self._accumulate_power_set(base_rows, dim_names)
        annotate_obs_shape(out, arr, dim_names, dim_order, s_tile)
        if logger_seconds > 0:
            _log.info("RasterImageDaskProcessor: done (rows=%d, elapsed %.1fs)",
                      len(out), time.perf_counter() - t_run0)
        return out

    @staticmethod
    def _accumulate_power_set(rows: List[Dict], dim_names: List[str]) -> List[Dict]:
        enable_tile_rows = os.environ.get(RASTER_TILE_ROWS_ENV_VAR, "1") == "1"
        return accumulate_power_set(
            rows,
            dim_names,
            aggregate_fn=aggregate_metrics_for_group,
            enable_tile_rows=enable_tile_rows,
            leaf_keys=spatial_metric_keys_for_tile_rows(),
        )
