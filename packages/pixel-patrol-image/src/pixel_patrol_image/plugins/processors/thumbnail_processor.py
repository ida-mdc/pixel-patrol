"""Standalone thumbnail processor — generates one thumbnail per Record."""

import dask.array as da
import numpy as np
from typing import Any, Dict

from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_base.utils.thumbnail_generation import compute_thumbnail_fields


class ThumbnailProcessor:
    """Generates a fixed-size RGBA thumbnail for each raster image Record.

    Marked GLOBAL_ONLY so the pipeline places its output only in the global row
    (obs_level=0) when combined with a long-format processor such as
    RasterImageDaskProcessor.
    """

    NAME        = "thumbnail"
    GLOBAL_ONLY = True
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
