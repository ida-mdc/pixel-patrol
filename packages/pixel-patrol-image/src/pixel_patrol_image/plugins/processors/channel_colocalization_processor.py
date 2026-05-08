"""
Per-tile colocalization metrics between every channel pair, with the same power-set rollup
tree as the raster processor.  C is consumed by the pairwise comparison; rows are stratified
by all other dimensions (Z, T, …) and by spatial tile position (Y, X).
"""

import logging
import os
import time
from itertools import combinations
from typing import Any, Dict, List

import dask.array as da
import numpy as np

from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_image.plugins.processors.channel_pair_numpy_metrics import (
    joint_stats_tile,
    pearson_r_from_stats,
    ssim_from_stats,
)
from pixel_patrol_image.plugins.processors.processor_block_utils import (
    RASTER_TILE_ROWS_ENV_VAR,
    accumulate_power_set,
    annotate_obs_shape,
    iter_blocks,
    rechunk_for_tiling,
)
from pixel_patrol_image.plugins.processors.raster_image_numpy_metrics import fold_to_tiles

_log = logging.getLogger(__name__)

_ARRAY_METRICS = ("coloc_pearson_r", "coloc_ssim", "coloc_ssim_luminance",
                  "coloc_ssim_contrast", "coloc_ssim_structure")


class ChannelColocalizationProcessor:
    """Pearson r and SSIM between every channel pair, rolled up into the full obs_level tree.

    Requires a C dimension with at least 2 channels.  C is consumed by the pairwise
    comparison; output rows are stratified by all remaining dimensions (Z, T, …) and
    spatial tile coordinates (Y, X) — the same structure as RasterImageDaskProcessor.

    Each row carries array-valued columns with one float per channel pair, in
    combinations(range(C), 2) order: (0,1), (0,2), …, (C-2, C-1).
    """

    NAME = "channel-colocalization"
    INPUT = RecordSpec(axes={"X", "Y", "C"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"
    OUTPUT_SCHEMA: Dict[str, Any] = {
        "coloc_n_channels": int,
        **{m: np.ndarray for m in _ARRAY_METRICS},
    }
    OUTPUT_SCHEMA_PATTERNS: List = []

    def run(self, art: Record) -> List[Dict]:
        arr = da.asarray(art.data)
        dim_order = [d.upper() for d in art.dim_order]

        if "C" not in dim_order:
            return []

        n_c = int(arr.shape[dim_order.index("C")])
        if n_c < 2:
            return []

        logger_seconds = float(os.environ.get("PIXEL_PATROL_PROGRESS_LOG_SECONDS", "60"))
        t_run0 = time.perf_counter()

        y_ax, x_ax = dim_order.index("Y"), dim_order.index("X")
        arr = da.moveaxis(arr, [y_ax, x_ax], [-2, -1])
        new_order = [d for d in dim_order if d not in ("Y", "X")] + ["Y", "X"]

        # Move C to axis 0 so every block_np[ci] gives channel ci without extra indexing.
        c_ax = new_order.index("C")
        arr = da.moveaxis(arr, c_ax, 0)
        # axis layout after both moveaxis calls: (C, [other non-spatial...], Y, X)
        ordered = ["C"] + [d for d in new_order if d != "C"]

        # dim_names for output rows: all non-C non-spatial dims + spatial (no dim_c)
        dim_names = [f"dim_{d.lower()}" for d in ordered if d not in ("C", "Y", "X")] + ["dim_y", "dim_x"]
        dim_order_no_c = [d for d in ordered if d != "C"]

        s_tile = int(os.environ.get("PIXEL_PATROL_STATS_TILE_SIZE", "256"))

        # Pre-rechunk C to full width so rechunk_for_tiling's memory cap accounts for n_c.
        arr = arr.rechunk({0: n_c})
        arr = rechunk_for_tiling(arr, s_tile)

        pair_list = list(combinations(range(n_c), 2))
        dim_starts = [np.cumsum((0,) + c[:-1]) for c in arr.chunks]

        base_rows: List[Dict] = []
        for b_idx, block_np in iter_blocks(arr, arr.ndim - 2, t_run0, logger_seconds,
                                           desc="  Coloc tiles"):
            base_rows.extend(
                _process_coloc_block(block_np, b_idx, n_c, dim_starts, dim_names, s_tile, pair_list)
            )

        def _aggregate(rows, g_dims):
            out: Dict[str, Any] = {"coloc_n_channels": n_c}
            for metric in _ARRAY_METRICS:
                vals = [r[metric] for r in rows if metric in r]
                if vals:
                    out[metric] = np.nanmean(np.stack(vals, axis=0), axis=0).astype(np.float32)
            return out

        enable_tile_rows = os.environ.get(RASTER_TILE_ROWS_ENV_VAR, "1") == "1"
        leaf_keys = frozenset(_ARRAY_METRICS) | {"coloc_n_channels"}
        out = accumulate_power_set(base_rows, dim_names, _aggregate, enable_tile_rows, leaf_keys)
        # Pass arr[0] (single-channel shape) so annotate_obs_shape sees (Z, Y, X) not (C, Z, Y, X).
        annotate_obs_shape(out, arr[0], dim_names, dim_order_no_c, s_tile)
        return out


def _process_coloc_block(
    block_np: np.ndarray,
    b_idx: tuple,
    n_c: int,
    dim_starts: list,
    dim_names: list,
    s_tile: int,
    pair_list: list,
) -> List[Dict]:
    """Compute colocalization metrics for every tile in one block; return per-tile rows.

    block_np has shape (n_c, [ns_dims...], Y_block, X_block) with C at axis 0.
    Each channel is folded once; all pairs share those folds — no repeated I/O.
    """
    axes = (-2, -1)
    # Non-C non-spatial leading shape: block_np.shape[1:-2]
    ns_shape = block_np.shape[1:-2]

    # Fold each channel into sub-tiles once.
    tiled_channels = []
    mask_ref = None
    for ci in range(n_c):
        c_block = block_np[ci].astype(np.float64)          # (ns..., Y_block, X_block)
        tiled, mask = fold_to_tiles(c_block, s_tile)       # (n_planes, n_ty, n_tx, ts, ts)
        tiled_channels.append(tiled)
        if mask_ref is None:
            mask_ref = mask

    valid = np.any(mask_ref, axis=axes)                     # (n_planes, n_ty, n_tx)

    # Compute pair stats for all (plane, ty, tx) in one vectorised pass — no re-read.
    stats_per_pair = []
    for ci, cj in pair_list:
        stats = joint_stats_tile(tiled_channels[ci], tiled_channels[cj], axes)
        r_val = pearson_r_from_stats(stats)
        with np.errstate(all="ignore"):
            c_max = np.maximum(np.nanmax(tiled_channels[ci], axis=axes),
                               np.nanmax(tiled_channels[cj], axis=axes))
            c_min = np.minimum(np.nanmin(tiled_channels[ci], axis=axes),
                               np.nanmin(tiled_channels[cj], axis=axes))
        lum, con, struct, ssim_val = ssim_from_stats(stats, c_max - c_min)
        stats_per_pair.append({
            "coloc_pearson_r":      r_val,
            "coloc_ssim":           ssim_val,
            "coloc_ssim_luminance": lum,
            "coloc_ssim_contrast":  con,
            "coloc_ssim_structure": struct,
        })

    # Block origins: dim_starts[0] is C (always 0), [1...-2] are ns dims, [-2:] are Y/X.
    ns_origins = [int(dim_starts[1 + k][b_idx[1 + k]]) for k in range(len(ns_shape))]
    y_origin = int(dim_starts[-2][b_idx[-2]])
    x_origin = int(dim_starts[-1][b_idx[-1]])

    obs_level = len(dim_names)
    ns_dim_names = dim_names[:-2]   # e.g. ["dim_z"] for CZYX
    ts = s_tile
    rows = []

    for idx in np.ndindex(valid.shape):     # (plane_flat, tile_yi, tile_xi)
        if not valid[idx]:
            continue
        plane_flat, tile_yi, tile_xi = idx[0], idx[1], idx[2]
        ns_local = np.unravel_index(plane_flat, ns_shape) if ns_shape else ()

        row: Dict[str, Any] = {"obs_level": obs_level, "coloc_n_channels": n_c}
        for k, (ns_l, dname) in enumerate(zip(ns_local, ns_dim_names)):
            row[dname] = ns_origins[k] + int(ns_l)
        row["dim_y"] = y_origin + tile_yi * ts
        row["dim_x"] = x_origin + tile_xi * ts

        for metric in _ARRAY_METRICS:
            row[metric] = np.array(
                [float(ps[metric][idx]) for ps in stats_per_pair],
                dtype=np.float32,
            )

        rows.append(row)

    return rows
