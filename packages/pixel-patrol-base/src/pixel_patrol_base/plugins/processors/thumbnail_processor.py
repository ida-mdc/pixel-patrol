"""Thumbnail processor — generates one spatial patch per memory chunk, assembled by get_aggregation."""

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


def _reduce_to_spatial(arr, dim_order: str, keep_dims: set) -> tuple[da.Array, str]:
    """Reduce all dimensions not in keep_dims by taking the center slice."""
    current = dim_order
    i = 0
    while i < len(current):
        dim = current[i]
        if dim not in keep_dims:
            center = arr.shape[i] // 2
            arr = da.take(arr, indices=center, axis=i)
            current = current.replace(dim, '', 1)
        else:
            i += 1
    return arr, current


def _normalize(arr: da.Array) -> tuple[da.Array, float, float]:
    """
    Normalize to uint8 [0, 255].
    Lower bound = min(arr_min, 0) to keep zero as a fixed reference.
    Upper bound = arr_max.
    Returns (normalized_uint8, norm_min, norm_max).

    Uses ``nanmin`` / ``nanmax`` so NaN voxels do not poison the intensity range.
    If every value is NaN (or min/max are otherwise non-finite), returns a solid
    black image and ``(0.0, 0.0)`` for the norm metadata (no meaningful stretch).
    """
    mn, mx = da.compute(da.nanmin(arr), da.nanmax(arr), scheduler='synchronous')
    mn, mx = float(mn), float(mx)
    if not np.isfinite(mn) or not np.isfinite(mx):
        return da.full_like(arr, np.uint8(0), dtype=np.uint8), 0.0, 0.0
    lower = min(mn, 0.0)
    upper = mx
    if upper <= lower:
        fill = np.uint8(0 if upper <= 0 else 255)
        return da.full_like(arr, fill, dtype=np.uint8), lower, upper
    normalized = (arr.astype(np.float64) - lower) / (upper - lower) * 255.0
    return da.clip(normalized, 0, 255).astype(np.uint8), lower, upper


def _assemble(rows: List[Dict]) -> Dict[str, Any]:
    """Assemble per-chunk patches into one SPRITE_SIZE × SPRITE_SIZE RGBA thumbnail.

    Full image extent is derived from the patches' own position + size metadata,
    so no external full_shape is needed.
    """
    valid = [r for r in rows if "__thumbnail_patch__" in r]
    if not valid:
        return {}

    y_full = max(r["dim_y"] + r["Y_size"] for r in valid)
    x_full = max(r["dim_x"] + r["X_size"] for r in valid)

    scale  = min(SPRITE_SIZE / y_full, SPRITE_SIZE / x_full)
    h_used = max(1, round(y_full * scale))
    w_used = max(1, round(x_full * scale))
    y_pad  = (SPRITE_SIZE - h_used) // 2
    x_pad  = (SPRITE_SIZE - w_used) // 2

    by_pos: dict = defaultdict(list)
    for r in valid:
        by_pos[(r["dim_y"], r["dim_x"])].append(r)

    canvas = np.zeros((SPRITE_SIZE, SPRITE_SIZE, 4), dtype=np.uint8)
    for group in by_pos.values():
        r = group[len(group) // 2]
        patch = r["__thumbnail_patch__"]
        y_off, x_off = r["dim_y"], r["dim_x"]
        y_ext, x_ext = r["Y_size"], r["X_size"]

        cy  = y_pad + round(y_off * scale)
        cy2 = min(y_pad + h_used, max(cy + 1, y_pad + round((y_off + y_ext) * scale)))
        cx  = x_pad + round(x_off * scale)
        cx2 = min(x_pad + w_used, max(cx + 1, x_pad + round((x_off + x_ext) * scale)))
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

    center = min(valid, key=lambda r: (
        (r["dim_y"] - y_full / 2) ** 2 +
        (r["dim_x"] - x_full / 2) ** 2
    ))
    return {
        "thumbnail":          canvas.tobytes(),
        "thumbnail_norm_min": center["__norm_min__"],
        "thumbnail_norm_max": center["__norm_max__"],
        "thumbnail_dtype":    center["__dtype__"],
    }


class ThumbnailProcessor:
    """Generates a downsampled spatial patch from each memory chunk.

    CHUNK_KIND = MEMORY: the pipeline delivers memory-safe chunks.
    run_chunk returns a patch dict with position metadata; get_aggregation
    assembles all patches into the final thumbnail, deriving the full extent
    from the patches themselves.
    """

    NAME       = "thumbnail"
    CHUNK_KIND = ChunkKind.MEMORY
    INPUT      = RecordSpec(axes={"X", "Y"}, kinds={"intensity"})
    OUTPUT     = "features"
    OUTPUT_SCHEMA: Dict[str, Any] = {
        "thumbnail":          bytes,
        "thumbnail_norm_min": float,
        "thumbnail_norm_max": float,
        "thumbnail_dtype":    str,
    }

    def get_aggregation(self, name: str):
        if name not in self.OUTPUT_SCHEMA:
            return None
        return lambda rows, g_dims: _assemble(rows).get(name)

    def run_chunk(self, record: Record) -> Dict:
        dim_order_out = list(record.dim_order)
        if "Y" not in dim_order_out or "X" not in dim_order_out:
            return {}

        chunk: np.ndarray = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)
        color_dim = _get_color_dim(record.capabilities)
        dim_str = "".join(dim_order_out)

        y_ax = dim_str.index("Y")
        x_ax = dim_str.index("X")
        h, w = chunk.shape[y_ax], chunk.shape[x_ax]
        if h > _PATCH_MAX or w > _PATCH_MAX:
            scale = min(_PATCH_MAX / h, _PATCH_MAX / w)
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            r_idx = np.round(np.linspace(0, h - 1, new_h)).astype(np.int64)
            c_idx = np.round(np.linspace(0, w - 1, new_w)).astype(np.int64)
            chunk = np.take(np.take(chunk, r_idx, axis=y_ax), c_idx, axis=x_ax)

        arr = da.from_array(chunk.astype(np.float32), chunks=chunk.shape)
        if np.issubdtype(chunk.dtype, np.floating):
            arr = da.where(da.isnull(arr), 0.0, arr)

        keep_dims = {"X", "Y", color_dim} if color_dim else {"X", "Y", "S"}
        arr, reduced_order = _reduce_to_spatial(arr, dim_str, keep_dims=keep_dims)
        if arr.size == 0:
            return {}

        if color_dim and color_dim in reduced_order:
            c_ax = list(reduced_order).index(color_dim)
            if arr.shape[c_ax] == 1:
                arr = arr.squeeze(axis=c_ax)
                reduced_order = reduced_order.replace(color_dim, "", 1)

        normalized, norm_min, norm_max = _normalize(arr)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in cast")
            patch = normalized.compute(scheduler='synchronous')

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
            "__norm_min__":        norm_min,
            "__norm_max__":        norm_max,
            "__dtype__":           str(chunk.dtype),
        }
