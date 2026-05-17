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

# Dimensions that must never be split across tasks (cross-channel math needs them whole).
ATOMIC_DIMS = frozenset({'C', 'S'})


def raster_slicing_plan(
    shape: tuple,
    dim_string: str,
    dtype: np.dtype,
    target_mb: float,
) -> List[Tuple[slice, ...]]:
    """Two-level slicing plan for raster-image: Z/T first, then Y (tile-aligned) if still too large.

    ATOMIC_DIMS (C, S) and X are never split so colocalization always gets all channels.
    Y is split as a secondary axis when a full C×X plane exceeds target_mb — the common
    case for very large XY microscopy images (e.g. 40C × 50000 × 40000).
    Y step is aligned to tile_size so tiles never cross strip boundaries.
    """
    tile_size = int(os.environ.get('PIXEL_PATROL_STATS_TILE_SIZE', '256'))
    dims = {name: {'idx': i, 'size': shape[i]} for i, name in enumerate(dim_string)}

    y_idx  = dims.get('Y', {}).get('idx')
    y_size = dims.get('Y', {}).get('size', 1) if y_idx is not None else 1

    # Bytes per one full C×Y×X plane (for Z/T step calculation and Y-split trigger)
    cx_px = 1
    for d, info in dims.items():
        if d in ATOMIC_DIMS or d == 'X':
            cx_px *= info['size']        # C × X (without Y)
    bytes_per_plane = cx_px * y_size * dtype.itemsize

    # Level 1: split along first non-atomic, non-spatial dim (Z or T)
    splittable = [d for d in dim_string if d not in ATOMIC_DIMS and d not in ('X', 'Y')]
    split_dim  = splittable[0] if splittable else None

    if split_dim:
        d_idx  = dims[split_dim]['idx']
        d_size = dims[split_dim]['size']
        z_step = max(1, int(target_mb * 1024 ** 2 // bytes_per_plane))
        z_starts = list(range(0, d_size, z_step))
    else:
        d_idx, d_size, z_step = None, 1, None
        z_starts = [None]

    # Level 2: split Y (tile-aligned) when a full plane exceeds target_mb
    bytes_per_tile_strip = cx_px * tile_size * dtype.itemsize
    if bytes_per_plane > target_mb * 1024 ** 2 and y_idx is not None:
        strips_per_task = max(1, int(target_mb * 1024 ** 2 // bytes_per_tile_strip))
        y_step = strips_per_task * tile_size
    else:
        y_step = y_size  # no Y splitting needed

    slices: List[Tuple[slice, ...]] = []
    for z_start in z_starts:
        for y_start in range(0, y_size, y_step):
            slc = [slice(None)] * len(shape)
            if d_idx is not None and z_start is not None:
                slc[d_idx] = slice(z_start, min(z_start + z_step, d_size))
            if y_step < y_size:
                slc[y_idx] = slice(y_start, min(y_start + y_step, y_size))
            slices.append(tuple(slc))

    return slices or [tuple(slice(None) for _ in shape)]


def slicing_plan(
    shape: tuple,
    dim_string: str,
    dtype: np.dtype,
    target_mb: float,
) -> List[Tuple[slice, ...]]:
    """Return slice tuples that divide an array into ~target_mb chunks.

    Atomic dims (C, S) and spatial dims (X, Y) are never split so each chunk
    always contains a complete channel/sample stack and full XY planes.
    The first remaining dim (typically Z or T) is divided to hit target_mb.
    A single-element list with all-slice(None) is returned when no split is needed.
    """
    dims = {name: {'idx': i, 'size': shape[i]} for i, name in enumerate(dim_string)}
    unit_px = 1
    for d, info in dims.items():
        if d in ATOMIC_DIMS or d in ('X', 'Y'):
            unit_px *= info['size']
    bytes_per_unit = unit_px * dtype.itemsize

    splittable = [d for d in dim_string if d not in ATOMIC_DIMS and d not in ('X', 'Y')]
    split_dim = splittable[0] if splittable else None

    if not split_dim or bytes_per_unit == 0:
        return [tuple(slice(None) for _ in shape)]

    d_idx = dims[split_dim]['idx']
    d_size = dims[split_dim]['size']
    step = max(1, int(target_mb * 1024 ** 2 // bytes_per_unit))

    slices = []
    for start in range(0, d_size, step):
        end = min(start + step, d_size)
        slc = [slice(None)] * len(shape)
        slc[d_idx] = slice(start, end)
        slices.append(tuple(slc))
    return slices


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
