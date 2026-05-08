"""
Measure raster statistics from lazy arrays: pick chunks, tile them, then summarize at every useful grouping of dimensions.
"""

import logging
import math
import os
import time
from itertools import combinations
from typing import Any, Dict, Generator, List, Tuple

import dask.array as da
import numpy as np

_log = logging.getLogger(__name__)
_BAR_MIN_BLOCKS = 10  # show a progress bar only when the run has enough blocks to justify it

from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_image.plugins.processors.raster_image_numpy_metrics import NumpyRasterBackend
from pixel_patrol_image.plugins.processors.raster_metric_definitions import (
    RASTER_TILE_ROWS_ENV_VAR,
    aggregate_metrics_for_group,
    spatial_metric_keys_for_tile_rows, enabled_raster_metrics,
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
        # Keep Y and X as the last two axes so every metric can reduce over axis=(-2, -1).
        arr = da.moveaxis(arr, [y_ax, x_ax], [-2, -1])
        # dim_names mirrors the axis order: non-spatial dims first, then dim_y, dim_x.
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

        # Control XY rechunking: fewer/larger XY blocks dramatically reduces Python overhead,
        # because the NumPy backend currently `.compute()`s one Dask block at a time.
        #
        # If set, PIXEL_PATROL_STATS_XY_CHUNK forces both Y and X chunk sizes (rounded up to tile size).
        # Example: 16384 or 32768.
        forced_xy = os.environ.get("PIXEL_PATROL_STATS_XY_CHUNK")
        if forced_xy:
            forced = max(1, int(forced_xy))
            xy_chunk = max(s_tile, math.ceil(forced / s_tile) * s_tile)
        else:
            auto_chunks = da.core.normalize_chunks("auto", shape=arr.shape[-2:], dtype=arr.dtype)
            xy_chunk = max(s_tile, math.ceil(auto_chunks[0][0] / s_tile) * s_tile)

        # Rechunk strategy for non-spatial axes (Z, C, …):
        # Never go below the source chunk size — rechunking smaller forces dask to reload
        # the entire source chunk for every sub-slice (e.g. 32× overhead for a Z-chunk of 32).
        # Cap at 64 so a file with huge source chunks doesn't produce unmanageable blocks.
        _NS_MAX_CHUNK = 64
        ns_chunks = {ax: min(max(arr.chunks[ax][0], 1), _NS_MAX_CHUNK)
                     for ax in range(arr.ndim - 2)}
        ns_block_elements = math.prod(v for v in ns_chunks.values()) if ns_chunks else 1

        # Cap xy_chunk so a single block stays within ~1 GB.  Without this, a file with
        # many non-spatial slices per block (e.g. Z-chunk=32, C=3) can produce blocks of
        # 10+ GB when the image is large, causing OOM before any tile is processed.
        _MAX_BLOCK_BYTES = int(os.environ.get("PIXEL_PATROL_MAX_BLOCK_MB", "1024")) * 1024**2
        max_xy_side = int(math.sqrt(_MAX_BLOCK_BYTES / (ns_block_elements * arr.dtype.itemsize)))
        max_xy_side = max(s_tile, (max_xy_side // s_tile) * s_tile)
        xy_chunk = min(xy_chunk, max_xy_side)

        arr = arr.rechunk({**ns_chunks, arr.ndim - 2: xy_chunk, arr.ndim - 1: xy_chunk})

        array_backend = NumpyRasterBackend(arr, s_tile, s_min, s_max, enabled_raster_metrics(), dim_names)

        base_rows: List[Dict] = []
        for b_idx, block_np in self._iter_blocks(arr, arr.ndim - 2, t_run0, logger_seconds):
            base_rows.extend(array_backend.process(b_idx, precomputed=block_np))

        out = self._accumulate_power_set(base_rows, dim_names)
        self._annotate_obs_shape(out, arr, dim_names, dim_order, s_tile)
        if logger_seconds > 0:
            _log.info("RasterImageDaskProcessor: done (rows=%d, elapsed %.1fs)",
                      len(out), time.perf_counter() - t_run0)
        return out

    @staticmethod
    def _iter_blocks(
        arr,
        ns_ndim: int,
        t_run0: float,
        logger_seconds: float,
    ) -> Generator[Tuple[tuple, np.ndarray], None, None]:
        """Yield (block_index, numpy_array) for every rechunked block.

        All XY blocks for one non-spatial slice are computed together so dask can
        deduplicate shared source-chunk reads and amortize scheduler overhead.
        Block memory is already bounded by the xy_chunk cap in the caller.
        """
        total = int(np.prod(arr.numblocks)) if getattr(arr, "numblocks", None) else 0
        ns_numblocks = arr.numblocks[:ns_ndim]
        all_xy_bidx = list(np.ndindex(*arr.numblocks[ns_ndim:]))

        pbar = None
        if total >= _BAR_MIN_BLOCKS:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(total=total, desc="  Raster tiles", unit="blk",
                         leave=False, smoothing=0.1)

        last_log = time.perf_counter()
        t_start = time.perf_counter()
        done = 0

        try:
            for ns_b_idx in np.ndindex(*ns_numblocks):
                dask_blocks = [arr.blocks[ns_b_idx + xy] for xy in all_xy_bidx]
                computed = da.compute(*dask_blocks)
                for xy_b_idx, block_np in zip(all_xy_bidx, computed):
                    yield ns_b_idx + xy_b_idx, block_np
                    done += 1
                    if pbar is not None:
                        pbar.update(1)
                    if logger_seconds > 0 and (time.perf_counter() - last_log) >= logger_seconds:
                        elapsed = time.perf_counter() - t_run0
                        rate = done / max(time.perf_counter() - t_start, 1e-6)
                        eta_str = (f", ETA ~{(total - done) / rate / 60:.1f} min"
                                   if done < total else "")
                        _log.info("RasterImageDaskProcessor: processed blocks %d/%d (elapsed %.1fs%s)",
                                  done, total, elapsed, eta_str)
                        last_log = time.perf_counter()
        finally:
            if pbar is not None:
                pbar.close()

    @staticmethod
    def _annotate_obs_shape(
        rows: List[Dict],
        arr,
        dim_names: List[str],
        dim_order: List[str],
        tile_size: int,
    ) -> None:
        """Fix shape, ndim, num_pixels, and {axis}_size for every non-global row.

        The file-level metadata from rcd.meta carries the full-image values into every
        row.  At the Z-plane or tile level those values are wrong: a per-Z row in a
        (40, 53638, 62366) image should report shape=[53638, 62366], ndim=2, not the
        whole-image values.

        Rules per dimension:
          - Collapsed (dim_value is None): covers the full extent → include in shape.
          - Spatial tile (dim_y / dim_x with a coordinate): covers one tile → include
            as min(tile_size, remaining_pixels) so edge tiles are exact.
          - Non-spatial slice (e.g. dim_z=5): resolved to one frame → {axis}_size=1,
            excluded from shape (the coordinate is already recorded in dim_z).
        """
        dim_letter = {f"dim_{d.lower()}": d for d in dim_order}

        for row in rows:
            if row.get("obs_level") == 0:
                continue  # global row: rcd.meta already has the correct full-image values

            row_shape = []
            size_updates: Dict[str, int] = {}

            for i, dim_name in enumerate(dim_names):
                full_extent = arr.shape[i]
                dim_val = row.get(dim_name)
                letter = dim_letter.get(dim_name, "")

                if dim_val is None:
                    # Not stratified: this row aggregates over the full extent of this dim.
                    row_shape.append(full_extent)
                    if letter:
                        size_updates[f"{letter}_size"] = full_extent
                elif dim_name in ("dim_y", "dim_x"):
                    # Spatial tile coordinate: the tile covers up to tile_size pixels,
                    # but may be shorter at the image edge.
                    tile_extent = max(1, min(tile_size, full_extent - int(dim_val)))
                    row_shape.append(tile_extent)
                    if letter:
                        size_updates[f"{letter}_size"] = tile_extent
                else:
                    # Non-spatial dimension resolved to a single frame: not a free
                    # dimension of this observation, so excluded from shape.
                    if letter:
                        size_updates[f"{letter}_size"] = 1

            row["shape"] = row_shape
            row["ndim"] = len(row_shape)
            row["num_pixels"] = int(np.prod(row_shape)) if row_shape else 0
            row.update(size_updates)

    @staticmethod
    def _accumulate_power_set(rows: List[Dict], dim_names: List[str]) -> List[Dict]:
        """Aggregate tile rows into every useful combination of dimensions.

        For an image with dimensions [C, Z, Y, X] this produces:
          - one global row (all tiles merged),
          - per-C, per-Z, per-Y, per-X rows,
          - per-(C,Z), per-(C,Y), … rows,
          - and optionally the individual tile rows.
        Each output row carries an obs_level equal to the number of dimensions it is
        stratified by (0 = global, ndim = individual tile).
        """
        all_res: List[Dict] = []

        spatial_tile_keys = spatial_metric_keys_for_tile_rows()
        enable_tile_metrics = os.environ.get(RASTER_TILE_ROWS_ENV_VAR, "1") == "1"
        ndim = len(dim_names)
        dim_y_key, dim_x_key = "dim_y", "dim_x"

        ny_tile = len({r[dim_y_key] for r in rows if dim_y_key in r}) if rows else 0
        nx_tile = len({r[dim_x_key] for r in rows if dim_x_key in r}) if rows else 0
        # Single tile along an axis → SQL-null on leaf rows; rollup skip logic uses the same counts.
        for r in rows:
            if ny_tile <= 1:
                r[dim_y_key] = None
            if nx_tile <= 1:
                r[dim_x_key] = None

        if enable_tile_metrics:
            for r in rows:
                all_res.append(
                    {
                        k: v
                        for k, v in r.items()
                        if k in spatial_tile_keys or k.startswith("dim_") or k == "obs_level"
                    }
                )

        # depth = number of dimensions to group by = obs_level of the produced rows.
        # depth 0 → one global row; depth ndim-1 → one row per unique coordinate combination.
        for depth in range(ndim - 1, -1, -1):
            if not enable_tile_metrics and depth > 0:
                continue

            for g_dims in combinations(dim_names, depth):
                has_y = dim_y_key in g_dims
                has_x = dim_x_key in g_dims
                gd_set = frozenset(g_dims)
                full_minus_y = frozenset(dim_names) - {dim_y_key}
                full_minus_x = frozenset(dim_names) - {dim_x_key}

                # No stratification along Y (or X): strips that only fix dim_y (or dim_x) add no degrees of freedom.
                if has_y and not has_x and ny_tile <= 1:
                    continue
                if has_x and not has_y and nx_tile <= 1:
                    continue
                # Normalized tiles already use null on a degenerate axis; omit the rollup that repeats the same row:
                # grouping by every coord except dim_y while ny_tile==1 merges only along that single Y row.
                if has_x and not has_y and ny_tile <= 1 and gd_set == full_minus_y:
                    continue
                if has_y and not has_x and nx_tile <= 1 and gd_set == full_minus_x:
                    continue

                groups: Dict[tuple, List[Dict]] = {}
                for r in rows:
                    groups.setdefault(tuple(r[d] for d in g_dims), []).append(r)

                for key, g_rows in groups.items():
                    agg: Dict = {"obs_level": depth}
                    for i, d in enumerate(g_dims):
                        agg[d] = key[i]

                    agg.update(aggregate_metrics_for_group(g_rows, g_dims))
                    all_res.append(agg)

        return all_res
