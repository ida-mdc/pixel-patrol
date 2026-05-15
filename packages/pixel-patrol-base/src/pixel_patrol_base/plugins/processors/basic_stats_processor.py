import logging
from typing import Dict, Callable

import numpy as np

from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_base.utils.array_utils import calculate_np_array_stats

logger = logging.getLogger(__name__)

def _column_fn_registry() -> Dict[str, Dict[str, Callable]]:
    return {
        'mean_intensity': {'fn': lambda a: np.float32(np.nanmean(a)) if a.size else np.float32(np.nan), 'agg': np.nanmean},
        'std_intensity': {'fn': lambda a: np.float32(np.nanstd(a)) if a.size else np.float32(np.nan), 'agg': np.nanmean},
        'min_intensity': {'fn': lambda a: np.float32(np.nanmin(a)) if a.size else np.float32(np.nan), 'agg': np.nanmin},
        'max_intensity': {'fn': lambda a: np.float32(np.nanmax(a)) if a.size else np.float32(np.nan), 'agg': np.nanmax},
    }

class BasicStatsProcessor:
    NAME = "basic-stats"
    INPUT = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"

    OUTPUT_SCHEMA = {name: np.float32 for name in _column_fn_registry().keys()}

    DESCRIPTION = "Extracts basic image statistics such as mean, min, max."

    def run(self, art):
        return calculate_np_array_stats(art.data, art.dim_order, _column_fn_registry())