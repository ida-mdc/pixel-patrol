"""Shared block iteration and rollup utilities for image processor plugins."""

import logging
import math
import os
import time
from itertools import combinations
from typing import Any, Callable, Dict, FrozenSet, Generator, List, Tuple

import dask.array as da
import numpy as np

_log = logging.getLogger(__name__)
_BAR_MIN_BLOCKS = 10

RASTER_TILE_ROWS_ENV_VAR = "PIXEL_PATROL_RASTER_XY_TILE_METRICS"
ITER_Y_ROWS_ENV_VAR = "PIXEL_PATROL_ITER_Y_ROWS"


def rechunk_for_tiling(arr: da.Array, s_tile: int) -> da.Array:
    """Rechunk arr so XY dimensions are aligned to a tile-multiple chunk size within memory budget.

    Non-spatial axes are left at their source chunk sizes (never smaller, capped at 64
    to avoid unmanageable blocks for deep-Z acquisitions).
    PIXEL_PATROL_STATS_XY_CHUNK overrides the auto XY chunk.
    PIXEL_PATROL_MAX_BLOCK_MB (default 1024) caps the per-block memory footprint.
    """
    forced_xy = os.environ.get("PIXEL_PATROL_STATS_XY_CHUNK")
    if forced_xy:
        forced = max(1, int(forced_xy))
        xy_chunk = max(s_tile, math.ceil(forced / s_tile) * s_tile)
    else:
        auto_chunks = da.core.normalize_chunks("auto", shape=arr.shape[-2:], dtype=arr.dtype)
        xy_chunk = max(s_tile, math.ceil(auto_chunks[0][0] / s_tile) * s_tile)

    _NS_MAX_CHUNK = 64
    ns_chunks = {ax: min(max(arr.chunks[ax][0], 1), _NS_MAX_CHUNK)
                 for ax in range(arr.ndim - 2)}
    ns_block_elements = math.prod(v for v in ns_chunks.values()) if ns_chunks else 1

    _MAX_BLOCK_BYTES = int(os.environ.get("PIXEL_PATROL_MAX_BLOCK_MB", "1024")) * 1024**2
    max_xy_side = int(math.sqrt(_MAX_BLOCK_BYTES / (ns_block_elements * arr.dtype.itemsize)))
    max_xy_side = max(s_tile, (max_xy_side // s_tile) * s_tile)
    xy_chunk = min(xy_chunk, max_xy_side)

    return arr.rechunk({**ns_chunks, arr.ndim - 2: xy_chunk, arr.ndim - 1: xy_chunk})


def iter_blocks(
    arr: da.Array,
    ns_ndim: int,
    t_run0: float,
    logger_seconds: float,
    desc: str = "  Tiles",
) -> Generator[Tuple[tuple, np.ndarray], None, None]:
    """Yield (block_index, numpy_array) for every rechunked block.

    All X-blocks within one Y-row are computed together so dask can deduplicate
    shared source-chunk reads within that row and amortise scheduler overhead.
    Processing one Y-row at a time caps peak memory at n_x_blocks × block_size
    rather than n_x_blocks × n_y_blocks × block_size, which matters when many
    XY chunks cover a large spatial extent.
    Block memory is bounded by the xy_chunk cap applied in rechunk_for_tiling.

    Set PIXEL_PATROL_ITER_Y_ROWS (default 1) to compute multiple Y-rows per
    ``da.compute`` call — trades higher peak RAM for fewer passes over large
    source chunks on disk (often 2–4 is enough).
    """
    total = int(np.prod(arr.numblocks)) if getattr(arr, "numblocks", None) else 0
    ns_numblocks = arr.numblocks[:ns_ndim]
    # Y dimension is at axis ns_ndim; X dimension is the last axis.
    all_y_bidx = list(np.ndindex(*arr.numblocks[ns_ndim:-1]))
    all_x_bidx = list(np.ndindex(*arr.numblocks[-1:]))
    y_rows_batch = max(1, int(os.environ.get(ITER_Y_ROWS_ENV_VAR, "1")))

    pbar = None
    if total >= _BAR_MIN_BLOCKS:
        from tqdm import tqdm as _tqdm
        pbar = _tqdm(total=total, desc=desc, unit="blk", leave=False, smoothing=0.1)

    last_log = time.perf_counter()
    t_start = time.perf_counter()
    done = 0

    try:
        for ns_b_idx in np.ndindex(*ns_numblocks):
            for y_start in range(0, len(all_y_bidx), y_rows_batch):
                y_slice = all_y_bidx[y_start : y_start + y_rows_batch]
                dask_blocks = [
                    arr.blocks[ns_b_idx + y_b_idx + x_b_idx]
                    for y_b_idx in y_slice
                    for x_b_idx in all_x_bidx
                ]
                computed = da.compute(*dask_blocks)
                off = 0
                for y_b_idx in y_slice:
                    for x_b_idx in all_x_bidx:
                        block_np = computed[off]
                        off += 1
                        yield ns_b_idx + y_b_idx + x_b_idx, block_np
                        done += 1
                        if pbar is not None:
                            pbar.update(1)
                        if logger_seconds > 0 and (time.perf_counter() - last_log) >= logger_seconds:
                            elapsed = time.perf_counter() - t_run0
                            rate = done / max(time.perf_counter() - t_start, 1e-6)
                            eta_str = (f", ETA ~{(total - done) / rate / 60:.1f} min"
                                       if done < total else "")
                            _log.info("iter_blocks: processed %d/%d (elapsed %.1fs%s)",
                                      done, total, elapsed, eta_str)
                            last_log = time.perf_counter()
    finally:
        if pbar is not None:
            pbar.close()


def accumulate_power_set(
    rows: List[Dict],
    dim_names: List[str],
    aggregate_fn: Callable[[List[Dict], Tuple[str, ...]], Dict[str, Any]],
    enable_tile_rows: bool,
    leaf_keys: FrozenSet[str],
) -> List[Dict]:
    """Aggregate tile rows into every useful grouping of dimensions.

    Caller provides aggregate_fn(rows, group_dims) → dict of aggregated metrics.
    Degenerate spatial axes (single tile) are collapsed to None before grouping.
    """
    all_res: List[Dict] = []
    ndim = len(dim_names)
    dim_y_key, dim_x_key = "dim_y", "dim_x"

    ny_tile = len({r[dim_y_key] for r in rows if dim_y_key in r}) if rows else 0
    nx_tile = len({r[dim_x_key] for r in rows if dim_x_key in r}) if rows else 0
    for r in rows:
        if ny_tile <= 1:
            r[dim_y_key] = None
        if nx_tile <= 1:
            r[dim_x_key] = None

    if enable_tile_rows:
        for r in rows:
            all_res.append(
                {k: v for k, v in r.items()
                 if k in leaf_keys or k.startswith("dim_") or k == "obs_level"}
            )

    for depth in range(ndim - 1, -1, -1):
        if not enable_tile_rows and depth > 0:
            continue

        for g_dims in combinations(dim_names, depth):
            has_y = dim_y_key in g_dims
            has_x = dim_x_key in g_dims
            gd_set = frozenset(g_dims)
            full_minus_y = frozenset(dim_names) - {dim_y_key}
            full_minus_x = frozenset(dim_names) - {dim_x_key}

            if has_y and not has_x and ny_tile <= 1:
                continue
            if has_x and not has_y and nx_tile <= 1:
                continue
            if has_x and not has_y and ny_tile <= 1 and gd_set == full_minus_y:
                continue
            if has_y and not has_x and nx_tile <= 1 and gd_set == full_minus_x:
                continue

            groups: Dict[tuple, List[Dict]] = {}
            for r in rows:
                groups.setdefault(tuple(r.get(d) for d in g_dims), []).append(r)

            for key, g_rows in groups.items():
                agg: Dict = {"obs_level": depth}
                for i, d in enumerate(g_dims):
                    agg[d] = key[i]
                agg.update(aggregate_fn(g_rows, g_dims))
                all_res.append(agg)

    return all_res


def annotate_obs_shape(
    rows: List[Dict],
    arr: da.Array,
    dim_names: List[str],
    dim_order: List[str],
    tile_size: int,
) -> None:
    """Fix shape, ndim, num_pixels, and {axis}_size for every non-global row (mutates in place)."""
    dim_letter = {f"dim_{d.lower()}": d for d in dim_order}

    for row in rows:
        if row.get("obs_level") == 0:
            continue

        row_shape = []
        size_updates: Dict[str, int] = {}

        for i, dim_name in enumerate(dim_names):
            full_extent = arr.shape[i]
            dim_val = row.get(dim_name)
            letter = dim_letter.get(dim_name, "")

            if dim_val is None:
                row_shape.append(full_extent)
                if letter:
                    size_updates[f"{letter}_size"] = full_extent
            elif dim_name in ("dim_y", "dim_x"):
                tile_extent = max(1, min(tile_size, full_extent - int(dim_val)))
                row_shape.append(tile_extent)
                if letter:
                    size_updates[f"{letter}_size"] = tile_extent
            else:
                if letter:
                    size_updates[f"{letter}_size"] = 1

        row["shape"] = row_shape
        row["ndim"] = len(row_shape)
        row["num_pixels"] = int(np.prod(row_shape)) if row_shape else 0
        row.update(size_updates)
