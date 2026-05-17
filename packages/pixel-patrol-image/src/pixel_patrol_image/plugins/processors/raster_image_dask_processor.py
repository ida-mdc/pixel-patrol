"""
Measure raster statistics from lazy arrays: divide into memory-sized chunks along Z/T,
tile each chunk in XY, summarise at every useful grouping of dimensions.
"""

import logging
import os
import time
from typing import Any, Dict, List

import dask.array as da
import numpy as np

from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_image.plugins.processors.processor_block_utils import (
    RASTER_TILE_ROWS_ENV_VAR,
    accumulate_power_set,
    annotate_obs_shape,
    raster_slicing_plan,
    slicing_plan,
)
from pixel_patrol_image.plugins.processors.raster_image_numpy_metrics import NumpyRasterBackend
from pixel_patrol_image.plugins.processors.raster_metric_definitions import (
    aggregate_metrics_for_group,
    enabled_raster_metrics,
    spatial_metric_keys_for_tile_rows,
)

_log = logging.getLogger(__name__)


def process_chunk(
    chunk: np.ndarray,
    origin: List[int],
    dim_names: List[str],
    tile_size: int,
    metrics,
    s_min: np.ndarray,
    s_max: np.ndarray,
) -> List[Dict[str, Any]]:
    """Run NumpyRasterBackend on one materialised numpy chunk.

    s_min/s_max are chunk-local per-plane arrays (shape = chunk.shape[:-2]).
    Non-spatial dim_* coordinates are shifted from chunk-local to global
    pixel space using `origin`.
    """
    backend = NumpyRasterBackend(chunk, tile_size, s_min, s_max, metrics, dim_names)
    rows = backend.process()

    # Apply origin offset to ALL dimensions (including Y/X) so Y-strip slicing works correctly.
    # When origin[i] == 0 (the common case for non-spatial dims and unsliced spatial dims)
    # this is a no-op; when Y is sliced the correct global coordinate is produced.
    for row in rows:
        for i, dn in enumerate(dim_names):
            if dn in row and row[dn] is not None:
                row[dn] = int(row[dn]) + origin[i]
    return rows


def collect_tile_rows(
    arr,                   # lazy dask array with Y, X last
    dim_order_out: List[str],
    tile_size: int,
    target_mb: float,
) -> List[Dict[str, Any]]:
    """Load slices and compute tile-level metrics. No accumulation — call accumulate_raster_tile_rows after."""
    ns_dims = [d for d in dim_order_out if d not in ('Y', 'X')]
    dim_names = [f'dim_{d.lower()}' for d in ns_dims] + ['dim_y', 'dim_x']
    dim_string = ''.join(dim_order_out)
    metrics = enabled_raster_metrics()
    sp_axes = tuple(range(arr.ndim - 2, arr.ndim))
    base_rows: List[Dict] = []
    for slc in raster_slicing_plan(arr.shape, dim_string, arr.dtype, target_mb):
        chunk = arr[slc].compute()
        origin = [s.start or 0 if isinstance(s, slice) else int(s) for s in slc]
        s_min = chunk.min(axis=sp_axes) if chunk.ndim > 2 else np.array(chunk.min())
        s_max = chunk.max(axis=sp_axes) if chunk.ndim > 2 else np.array(chunk.max())
        base_rows.extend(process_chunk(chunk, origin, dim_names, tile_size, metrics, s_min, s_max))
    return base_rows


def accumulate_raster_tile_rows(
    tile_rows: List[Dict[str, Any]],
    full_shape: tuple,
    dim_order_out: List[str],
) -> List[Dict[str, Any]]:
    """Roll up raw tile rows (from one or many slices) into the full power-set summary.

    Called once per file after all slices are collected, not per-slice.
    full_shape must be the complete array shape (after YX move-to-last), used by
    annotate_obs_shape to compute correct per-dimension sizes.
    """
    ns_dims = [d for d in dim_order_out if d not in ('Y', 'X')]
    dim_names = [f'dim_{d.lower()}' for d in ns_dims] + ['dim_y', 'dim_x']
    tile_size = int(os.environ.get('PIXEL_PATROL_STATS_TILE_SIZE', '256'))
    enable_tile_rows = os.environ.get(RASTER_TILE_ROWS_ENV_VAR, '1') == '1'
    out = accumulate_power_set(
        tile_rows, dim_names,
        aggregate_fn=aggregate_metrics_for_group,
        enable_tile_rows=enable_tile_rows,
        leaf_keys=spatial_metric_keys_for_tile_rows(),
    )
    dummy = da.empty(full_shape, dtype=np.float32)
    annotate_obs_shape(out, dummy, dim_names, dim_order_out, tile_size)
    return out


class RasterImageDaskProcessor:
    """Compute per-tile raster quality metrics and roll them up into a summary tree."""

    NAME = "raster-image"
    SLICE_SAFE = True   # can process individual Z/T slices; accumulation happens outside
    INPUT = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"
    OUTPUT_SCHEMA: Dict[str, Any] = {m.name: m.data_type for m in enabled_raster_metrics()}

    def run(self, art: Record) -> List[Dict]:
        t0 = time.perf_counter()
        src = getattr(art, 'source_path', None) or getattr(art, 'path', None) or 'unknown'
        arr, dim_order_out = self._open_array(art)
        _log.debug('begin  record=%s  shape=%s  dtype=%s', src, arr.shape, arr.dtype)
        tile_size = int(os.environ.get('PIXEL_PATROL_STATS_TILE_SIZE', '256'))
        target_mb = float(os.environ.get('PIXEL_PATROL_MAX_BLOCK_MB', '1024'))
        base_rows = collect_tile_rows(arr, dim_order_out, tile_size, target_mb)
        out = accumulate_raster_tile_rows(base_rows, tuple(arr.shape), dim_order_out)
        _log.debug('done   record=%s  rows=%d  elapsed=%.1fs', src, len(out), time.perf_counter() - t0)
        return out

    def run_slice(self, chunk: np.ndarray, origin: List[int], dim_order_out: List[str]) -> List[Dict]:
        """Process one tile chunk. Returns raw tile rows — call accumulate_slice_rows after all slices."""
        ns_dims = [d for d in dim_order_out if d not in ('Y', 'X')]
        dim_names = [f'dim_{d.lower()}' for d in ns_dims] + ['dim_y', 'dim_x']
        tile_size = int(os.environ.get('PIXEL_PATROL_STATS_TILE_SIZE', '256'))
        sp_axes = tuple(range(chunk.ndim - 2, chunk.ndim))
        s_min = chunk.min(axis=sp_axes) if chunk.ndim > 2 else np.array(chunk.min())
        s_max = chunk.max(axis=sp_axes) if chunk.ndim > 2 else np.array(chunk.max())
        return process_chunk(chunk, origin, dim_names, tile_size, enabled_raster_metrics(), s_min, s_max)

    @staticmethod
    def accumulate_slice_rows(
        tile_rows: List[Dict], full_shape: tuple, dim_order_out: List[str]
    ) -> List[Dict]:
        return accumulate_raster_tile_rows(tile_rows, full_shape, dim_order_out)

    @staticmethod
    def _open_array(art: Record):
        """Reorder axes so Y, X are last. Returns (lazy_dask_array, dim_order_out)."""
        arr = da.asarray(art.data)
        dim_order = [d.upper() for d in art.dim_order]
        y_ax, x_ax = dim_order.index('Y'), dim_order.index('X')
        arr = da.moveaxis(arr, [y_ax, x_ax], [-2, -1])
        ns_dims = [d for d in dim_order if d not in ('Y', 'X')]
        return arr, ns_dims + ['Y', 'X']
