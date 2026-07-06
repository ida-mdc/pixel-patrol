
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec


class NoDataCountProcessor:
    """Calculated statistics of the no data percentage per image.

    No data is either a defined NoData value or NaN.
    Adds "no data pixel count" and "nan data pixel count"
    Supports a "no data percentage per image"
    """

    NAME       = "nodata-statistics"
    CHUNK_KIND = ChunkKind.LEAF
    INPUT      = RecordSpec(axes={"X", "Y"}, kinds={"intensity"})
    OUTPUT     = "features"

    OUTPUT_SCHEMA          = {"nodata_count": int}
    OUTPUT_SCHEMA_PATTERNS = []

    def run_chunk(self, record: Record) -> Dict:
        arr = record.data.compute() if hasattr(record.data, "compute") else np.asarray(record.data)

        nodata_value = record.meta.get("nodata_value", None)
        if isinstance(nodata_value, float):
            if np.isnan(nodata_value):
                nodata_count = np.isnan(arr).sum()
            else:
                nodata_count = (arr == nodata_value).sum()
        elif nodata_value is not None:
            nodata_count = (arr == nodata_value).sum()
        else:
            nodata_count = 0

        return {"nodata_count": nodata_count}


    def get_aggregation(self, col: str) -> Optional[callable]:
        if col != "nodata_count":
            return None
        # pixel counts are independent per pixel, so chunk counts simply add up.
        return lambda rows, g_dims: sum(r["nodata_count"] for r in rows)
