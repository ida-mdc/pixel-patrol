"""Thumbnail processor — SLICE_SAFE: generates one spatial-mosaic thumbnail per Record."""

from typing import Any, Dict, List

import dask.array as da
import numpy as np

from pixel_patrol_base.config import SPRITE_SIZE
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_base.utils.thumbnail_generation import (
    _normalize,
    _reduce_to_spatial,
    compute_thumbnail_fields,
)

# Max size of each patch stored per slice task (pixels per side).
# Small enough to keep network transfer cheap; accumulation upscales to SPRITE_SIZE.
_PATCH_MAX = 64


class ThumbnailProcessor:
    """Generates a fixed-size RGBA thumbnail for each raster image Record.

    GLOBAL_ONLY keeps the result in the global row (obs_level=0) when run via the
    normal per-file path (small files, batch tasks).

    SLICE_SAFE enables the HPC tiled path: each slice task computes a small patch
    for its spatial region; accumulate_slice_rows stitches them into a mosaic.
    """

    NAME        = "thumbnail"
    GLOBAL_ONLY = True    # for normal (non-HPC) batch path
    SLICE_SAFE  = True    # for HPC tiled path
    INPUT       = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT      = "features"

    OUTPUT_SCHEMA: Dict[str, Any] = {
        "thumbnail":          bytes,
        "thumbnail_norm_min": float,
        "thumbnail_norm_max": float,
        "thumbnail_dtype":    str,
    }

    def run(self, art: Record) -> Dict[str, Any]:
        arr = art.data if isinstance(art.data, da.Array) else da.from_array(np.asarray(art.data))
        return compute_thumbnail_fields(arr, art.dim_order, getattr(art, "capabilities", None))

    def run_slice(self, chunk: np.ndarray, origin: List[int], dim_order_out: List[str]) -> List[Dict]:
        """Generate a small thumbnail patch from one spatial chunk.

        Takes the center of each non-spatial, non-channel dim (Z, T…), then
        reduces channels to greyscale (mean) or keeps ≤4 for colour.
        Stores a downsampled patch + position; accumulate_slice_rows stitches all patches.
        """
        if 'Y' not in dim_order_out or 'X' not in dim_order_out:
            return []

        y_pos = dim_order_out.index('Y')
        x_pos = dim_order_out.index('X')
        dtype_name = str(chunk.dtype)

        dim_str = ''.join(dim_order_out)
        arr = da.from_array(chunk.astype(np.float32), chunks=chunk.shape)

        # Take center of all non-spatial, non-channel dims (Z, T, …)
        arr, reduced_order = _reduce_to_spatial(arr, dim_str, keep_dims={'X', 'Y', 'C', 'S'})

        # Reduce C: mean if >4 channels, squeeze if single channel
        if 'C' in reduced_order:
            c_ax = list(reduced_order).index('C')
            n_c = arr.shape[c_ax]
            if n_c > 4:
                arr = da.mean(arr, axis=c_ax)
                reduced_order = reduced_order.replace('C', '', 1)
            elif n_c == 1:
                arr = arr.squeeze(axis=c_ax)
                reduced_order = reduced_order.replace('C', '', 1)

        normalized, norm_min, norm_max = _normalize(arr)
        patch = normalized.compute()

        # Transpose to (Y, X) or (Y, X, C)
        dims = list(reduced_order)
        if patch.ndim == 2:
            patch = np.transpose(patch, [dims.index('Y'), dims.index('X')])
        else:
            # 3-D: find the non-Y, non-X axis
            other = next(i for i, d in enumerate(dims) if d not in ('Y', 'X'))
            patch = np.transpose(patch, [dims.index('Y'), dims.index('X'), other])

        # Aggressively downsample so network transfer stays cheap
        h, w = patch.shape[:2]
        if h > _PATCH_MAX or w > _PATCH_MAX:
            r_idx = np.round(np.linspace(0, h - 1, min(_PATCH_MAX, h))).astype(np.int64)
            c_idx = np.round(np.linspace(0, w - 1, min(_PATCH_MAX, w))).astype(np.int64)
            patch = patch[r_idx][:, c_idx]

        return [{
            '__thumbnail_patch__': patch,
            '__y_origin__':        origin[y_pos],
            '__x_origin__':        origin[x_pos],
            '__y_extent__':        chunk.shape[y_pos],
            '__x_extent__':        chunk.shape[x_pos],
            '__norm_min__':        norm_min,
            '__norm_max__':        norm_max,
            '__dtype__':           dtype_name,
        }]

    @staticmethod
    def accumulate_slice_rows(
        tile_rows: List[Dict], full_shape: tuple, dim_order_out: List[str]
    ) -> List[Dict]:
        """Stitch per-chunk patches into a single SPRITE_SIZE × SPRITE_SIZE RGBA thumbnail.

        Patches are placed on the canvas proportionally to their spatial position in the
        full image.  For multiple Z/T slices at the same XY position, the center patch
        (by list order) is used so the result matches the single-file center-slice behaviour.
        """
        patch_rows = [r for r in tile_rows if '__thumbnail_patch__' in r]
        if not patch_rows:
            return []

        y_full = full_shape[dim_order_out.index('Y')]
        x_full = full_shape[dim_order_out.index('X')]

        # Group by spatial position; pick center patch per position (≈ center Z/T)
        from collections import defaultdict
        by_pos: dict = defaultdict(list)
        for r in patch_rows:
            by_pos[(r['__y_origin__'], r['__x_origin__'])].append(r)

        canvas = np.zeros((SPRITE_SIZE, SPRITE_SIZE, 4), dtype=np.uint8)

        for pos, rows in by_pos.items():
            r = rows[len(rows) // 2]   # center of the list ≈ center Z/T slice
            patch = r['__thumbnail_patch__']
            y_off, x_off = r['__y_origin__'], r['__x_origin__']
            y_ext, x_ext = r['__y_extent__'], r['__x_extent__']

            # Canvas cell for this patch
            cy  = int(y_off / y_full * SPRITE_SIZE)
            cx  = int(x_off / x_full * SPRITE_SIZE)
            ch  = max(1, round(y_ext / y_full * SPRITE_SIZE))
            cw  = max(1, round(x_ext / x_full * SPRITE_SIZE))
            cy2 = min(cy + ch, SPRITE_SIZE)
            cx2 = min(cx + cw, SPRITE_SIZE)
            ah, aw = cy2 - cy, cx2 - cx
            if ah <= 0 or aw <= 0 or patch.size == 0:
                continue

            h, w = patch.shape[:2]
            r_idx = np.round(np.linspace(0, h - 1, ah)).astype(np.int64)
            c_idx = np.round(np.linspace(0, w - 1, aw)).astype(np.int64)
            small = patch[r_idx][:, c_idx]

            if small.ndim == 2:
                canvas[cy:cy2, cx:cx2, 0] = small
                canvas[cy:cy2, cx:cx2, 1] = small
                canvas[cy:cy2, cx:cx2, 2] = small
            else:
                n_c = min(small.shape[2], 3)
                canvas[cy:cy2, cx:cx2, :n_c] = small[:, :, :n_c]
            canvas[cy:cy2, cx:cx2, 3] = 255

        # Norm stats from the spatially central patch
        center_r = min(patch_rows, key=lambda r: (
            (r['__y_origin__'] - y_full / 2) ** 2 +
            (r['__x_origin__'] - x_full / 2) ** 2
        ))

        return [{
            'obs_level':           0,
            'thumbnail':           canvas.tobytes(),
            'thumbnail_norm_min':  center_r['__norm_min__'],
            'thumbnail_norm_max':  center_r['__norm_max__'],
            'thumbnail_dtype':     center_r['__dtype__'],
        }]
