from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec


class GlowSpotterProcessor:
    """Counts the bioluminescent "glows" lighting up a deep-sea snapshot.

    Some sharks make their own light - in 2021, researchers confirmed the
    kitefin shark as the largest known glowing vertebrate, producing a soft
    blue-green bioluminescence in ocean layers sunlight never reaches
    (https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2021.633582/full).
    A pixel counts as part of a glow when it stands out clearly from the
    patch's overall brightness - brighter than its median by a healthy
    margin. Sunlit patches have almost none; the deeper and darker it gets,
    the more glows light up - exactly the way real bioluminescence
    concentrates in the dark.

    Note this processor never imports or refers to ``SharkCamLoader`` - it
    only declares the *kind of record* it needs (``intensity``, with X/Y
    axes), so it runs equally well on these synthetic patches or on anyone
    else's images. A loader and a processor are independent extension
    pieces; nothing requires you to ship both.
    """

    NAME       = "glow-spotter"
    CHUNK_KIND = ChunkKind.LEAF
    INPUT      = RecordSpec(axes={"X", "Y"}, kinds={"intensity"})
    OUTPUT     = "features"

    OUTPUT_SCHEMA          = {"glow_count": int}
    OUTPUT_SCHEMA_PATTERNS = []

    def run_chunk(self, record: Record) -> Dict:
        arr = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)
        arr = arr.astype(np.float32)

        threshold = np.median(arr) + 60.0
        glow_count = int(np.sum(arr > threshold))
        return {"glow_count": glow_count}

    def get_aggregation(self, col: str) -> Optional[callable]:
        if col != "glow_count":
            return None
        # Glows are independent per pixel, so chunk counts simply add up.
        return lambda rows, g_dims: sum(r["glow_count"] for r in rows)
