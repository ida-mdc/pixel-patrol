import logging
from typing import Any, Dict

import numpy as np

from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_base.plugins.processors import rgb_colour_stats as colour_stats

logger = logging.getLogger(__name__)

SUMS_KEY = "_colour_sums"

DESCRIPTIONS: Dict[str, str] = {
    "rgb_luma_std": "Contrast: standard deviation of luma (0.299 R + 0.587 G + 0.114 B) over all pixels, in the units of the pixel type. Low values mean a flat, washed-out picture.",
    "rgb_saturation_mean": "Mean HSV saturation, (max - min) / max per pixel, over pixels brighter than 5% of the range. 0 is grey, 1 is a pure colour.",
    "rgb_hue_coherence": "How much the coloured pixels share one hue: the length of their average colour direction over the average colour strength. 1 means a single hue (a tint or one stain), near 0 means many hues that cancel out in the average.",
}


class RgbColourProcessor:
    """Contrast, saturation and hue agreement of an RGB image."""

    NAME        = "rgb-colour"
    DESCRIPTION = ("Measures the contrast and saturation of the pixels of RGB images and whether their colours share one hue, "
                   "which per-channel statistics alone cannot give.")
    CHUNK_KIND  = ChunkKind.MEMORY
    INPUT       = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"rgb"})
    OUTPUT      = "features"
    OUTPUT_SCHEMA: Dict[str, Any] = {name: float for name in DESCRIPTIONS}
    OUTPUT_SCHEMA_DESCRIPTIONS: Dict[str, str] = DESCRIPTIONS

    def run_chunk(self, record: Record) -> Dict:
        chunk = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)
        planes = colour_stats.colour_planes(chunk, record.dim_order, colour_axis(record))
        if planes is None or planes.size == 0:
            return {}
        try:
            ceiling = colour_stats.value_ceiling(chunk.dtype, chunk)
            return {SUMS_KEY: colour_stats.chunk_sums(planes, ceiling)}
        except Exception as exc:
            logger.warning("rgb-colour: %s", exc)
            return {}

    def get_aggregation(self, name: str):
        if name not in self.OUTPUT_SCHEMA:
            return None
        return lambda rows, _dims: colour_stats.combine([r[SUMS_KEY] for r in rows if SUMS_KEY in r]).get(name)


def colour_axis(record: Record) -> str:
    return next(c[4:] for c in record.capabilities if c.startswith("rgb:"))
