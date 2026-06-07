from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec


class StarSpotterProcessor:
    """Counts the bright "stars" sprinkled across a sky patch.

    A pixel counts as a star when it stands out clearly from the patch's
    overall brightness — brighter than its median by a healthy margin. Night
    patches are scattered with many such bright pixels, daytime patches with
    almost none, so the count tracks ``time_of_day`` exactly the way you'd
    expect by just looking up at the sky.
    """

    NAME       = "star-spotter"
    CHUNK_KIND = ChunkKind.MEMORY          # patches are tiny - one chunk per file
    INPUT      = RecordSpec(axes={"X", "Y"}, kinds={"intensity"})
    OUTPUT     = "features"

    OUTPUT_SCHEMA          = {"star_count": int}
    OUTPUT_SCHEMA_PATTERNS = []

    def run_chunk(self, record: Record) -> Dict:
        arr = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)
        arr = arr.astype(np.float32)

        threshold = np.median(arr) + 60.0
        star_count = int(np.sum(arr > threshold))
        return {"star_count": star_count}

    def get_aggregation(self, col: str) -> Optional[callable]:
        if col not in self.OUTPUT_SCHEMA:
            return None
        # One chunk per file — return the value from the single chunk row.
        return lambda rows, g_dims: rows[0][col] if rows else None
