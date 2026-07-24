"""Thumbnail processor - generates a spatial patch per spatial chunk, assembled by get_aggregation."""

import warnings
from collections import defaultdict
from typing import Any, Dict, List

import dask.array as da
import numpy as np

from pixel_patrol_base.config import SPRITE_SIZE
from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec

warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)

_PATCH_MAX = 64


def _get_color_dim(capabilities) -> str | None:
    """Return the color dimension letter encoded in an 'rgb:<dim>' capability, or None."""
    for cap in capabilities:
        if cap.startswith('rgb:'):
            return cap[4:]
    return None


def _reduce_to_spatial(arr, dim_order: str, keep_dims: set, origin: tuple, full_shape: tuple):
    """Reduce non-kept dims to the image's true center; (None, None) if this chunk misses it."""
    current = dim_order
    i = 0
    while i < len(current):
        dim = current[i]
        if dim not in keep_dims:
            full_i = dim_order.index(dim)
            local_center = full_shape[full_i] // 2 - origin[full_i]
            if not (0 <= local_center < arr.shape[i]):
                return None, None
            arr = da.take(arr, indices=local_center, axis=i)
            current = current.replace(dim, '', 1)
        else:
            i += 1
    return arr, current


def _assemble(rows: List[Dict]) -> Dict[str, Any]:
    """Assemble per-chunk patches into one SPRITE_SIZE × SPRITE_SIZE RGBA thumbnail.

    Spatial (Y/X) extent comes from the patches' own position + size. Normalization
    is applied globally across positions so tiles don't show seams at boundaries.
    """
    valid = [r for r in rows if "__thumbnail_patch__" in r]
    if not valid:
        return {}

    y_full = max(r["dim_y"] + r["size_Y"] for r in valid)
    x_full = max(r["dim_x"] + r["size_X"] for r in valid)

    scale  = min(SPRITE_SIZE / y_full, SPRITE_SIZE / x_full)
    h_used = max(1, round(y_full * scale))
    w_used = max(1, round(x_full * scale))
    y_pad  = (SPRITE_SIZE - h_used) // 2
    x_pad  = (SPRITE_SIZE - w_used) // 2

    by_pos: dict = defaultdict(list)
    for r in valid:
        by_pos[(r["dim_y"], r["dim_x"])].append(r)

    # run_chunk already emits only the true-center chunk per spatial position, so each group has one row
    selected = [group[0] for group in by_pos.values()]

    # Global min/max across all spatial positions for seamless normalization
    flat_vals = [r["__thumbnail_patch__"].ravel() for r in selected if r["__thumbnail_patch__"].size > 0]
    if flat_vals:
        all_vals = np.concatenate(flat_vals)
        mn = float(np.nanmin(all_vals))
        mx = float(np.nanmax(all_vals))
    else:
        mn, mx = 0.0, 0.0

    if np.isfinite(mn) and np.isfinite(mx):
        norm_min = min(mn, 0.0)
        norm_max = mx
    else:
        norm_min, norm_max = 0.0, 0.0

    canvas = np.zeros((SPRITE_SIZE, SPRITE_SIZE, 4), dtype=np.uint8)
    for r in selected:
        raw = r["__thumbnail_patch__"]
        y_off, x_off = r["dim_y"], r["dim_x"]
        y_ext, x_ext = r["size_Y"], r["size_X"]

        cy  = y_pad + round(y_off * scale)
        cy2 = min(y_pad + h_used, max(cy + 1, y_pad + round((y_off + y_ext) * scale)))
        cx  = x_pad + round(x_off * scale)
        cx2 = min(x_pad + w_used, max(cx + 1, x_pad + round((x_off + x_ext) * scale)))
        ah, aw = cy2 - cy, cx2 - cx
        if ah <= 0 or aw <= 0 or raw.size == 0:
            continue

        if norm_max <= norm_min:
            patch = np.full(raw.shape, np.uint8(0 if norm_max <= 0 else 255), dtype=np.uint8)
        else:
            patch = np.clip(
                (raw.astype(np.float64) - norm_min) / (norm_max - norm_min) * 255.0,
                0, 255,
            ).astype(np.uint8)

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

    center = min(selected, key=lambda r: (
        (r["dim_y"] - y_full / 2) ** 2 +
        (r["dim_x"] - x_full / 2) ** 2
    ))
    return {
        "thumbnail":          canvas.tobytes(),
        "thumbnail_norm_min": norm_min,
        "thumbnail_norm_max": norm_max,
        "thumbnail_dtype":    center["__dtype__"],
    }


class ThumbnailProcessor:
    """Generates a downsampled spatial patch from each memory chunk.

    run_chunk skips non-spatial chunks that don't own the image's true center
    (only one chunk per spatial position does); get_aggregation assembles the
    surviving patches into the final thumbnail.
    """

    NAME        = "thumbnail"
    DESCRIPTION = "Generates a small downsampled RGBA thumbnail of each image, assembled from per-chunk spatial patches."
    CHUNK_KIND  = ChunkKind.MEMORY
    INPUT       = RecordSpec(axes={"X", "Y"}, kinds={"intensity"})
    OUTPUT      = "features"
    OUTPUT_SCHEMA: Dict[str, Any] = {
        "thumbnail":          bytes,
        "thumbnail_norm_min": float,
        "thumbnail_norm_max": float,
        "thumbnail_dtype":    str,
    }
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = {
        "thumbnail":          "Raw RGBA bytes of the assembled thumbnail sprite (fixed sprite size).",
        "thumbnail_norm_min": "Lower intensity bound used to normalize the thumbnail.",
        "thumbnail_norm_max": "Upper intensity bound used to normalize the thumbnail.",
        "thumbnail_dtype":    "Original pixel dtype of the source image the thumbnail was built from.",
    }

    def get_aggregation(self, name: str):
        if name not in self.OUTPUT_SCHEMA:
            return None
        return lambda rows, g_dims: _assemble(rows).get(name)

    def run_chunk(self, record: Record) -> Dict:
        dim_order_out = record.dim_order
        if "Y" not in dim_order_out or "X" not in dim_order_out:
            return {}

        chunk: np.ndarray = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)
        color_dim = _get_color_dim(record.capabilities)

        y_ax = dim_order_out.index("Y")
        x_ax = dim_order_out.index("X")
        h, w = chunk.shape[y_ax], chunk.shape[x_ax]
        full_shape = record.meta.get("full_shape", record.data.shape)
        full_h, full_w = full_shape[y_ax], full_shape[x_ax]
        y_off, x_off = record.meta.get("dim_y", 0), record.meta.get("dim_x", 0)
        # Sized from the full image + this chunk's position, so per-chunk sizes telescope like _assemble's placement math.
        if full_h > _PATCH_MAX or full_w > _PATCH_MAX:
            scale = min(_PATCH_MAX / full_h, _PATCH_MAX / full_w)
            new_h = max(1, round((y_off + h) * scale) - round(y_off * scale))
            new_w = max(1, round((x_off + w) * scale) - round(x_off * scale))
            r_idx = np.round(np.linspace(0, h - 1, new_h)).astype(np.int64)
            c_idx = np.round(np.linspace(0, w - 1, new_w)).astype(np.int64)
            chunk = np.take(np.take(chunk, r_idx, axis=y_ax), c_idx, axis=x_ax)

        arr = da.from_array(chunk.astype(np.float32), chunks=chunk.shape)
        if np.issubdtype(chunk.dtype, np.floating):
            arr = da.where(da.isnull(arr), 0.0, arr)

        origin     = tuple(record.meta.get(f"dim_{d.lower()}", 0) for d in dim_order_out)
        keep_dims  = {"X", "Y", color_dim} if color_dim else {"X", "Y", "S"}
        arr, reduced_order = _reduce_to_spatial(arr, dim_order_out, keep_dims, origin, full_shape)
        if arr is None or arr.size == 0:
            return {}

        if color_dim and color_dim in reduced_order:
            c_ax = list(reduced_order).index(color_dim)
            if arr.shape[c_ax] == 1:
                arr = arr.squeeze(axis=c_ax)
                reduced_order = reduced_order.replace(color_dim, "", 1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in cast")
            patch = arr.compute(scheduler='synchronous')

        dims = list(reduced_order)
        if patch.ndim == 2:
            patch = np.transpose(patch, [dims.index("Y"), dims.index("X")])
        else:
            other = next(i for i, d in enumerate(dims) if d not in ("Y", "X"))
            patch = np.transpose(patch, [dims.index("Y"), dims.index("X"), other])

        h, w = patch.shape[:2]
        if h > _PATCH_MAX or w > _PATCH_MAX:
            scale = min(_PATCH_MAX / h, _PATCH_MAX / w)
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            r_idx = np.round(np.linspace(0, h - 1, new_h)).astype(np.int64)
            c_idx = np.round(np.linspace(0, w - 1, new_w)).astype(np.int64)
            patch = patch[r_idx][:, c_idx]

        return {
            "__thumbnail_patch__": patch,
            "__dtype__":           str(chunk.dtype),
        }
