import logging
from typing import Dict, Callable

import dask.array as da
import numpy as np

from pixel_patrol_base.utils.array_utils import calculate_sliced_stats
from pixel_patrol_base.core.specs import RecordSpec

logger = logging.getLogger(__name__)

def _column_fn_registry() -> Dict[str, Dict[str, Callable]]:
    return {
        'mean_intensity': {'fn': lambda a: np.mean(a) if a.size else np.nan, 'agg': np.mean},
        'std_intensity': {'fn': lambda a: np.std(a) if a.size else np.nan, 'agg': np.mean},
        'min_intensity': {'fn': lambda a: np.min(a) if a.size else np.nan, 'agg': np.min},
        'max_intensity': {'fn': lambda a: np.max(a) if a.size else np.nan, 'agg': np.max},
    }

def calculate_np_array_stats(array: da.array, dim_order: str) -> dict[str, float]:
    registry = _column_fn_registry()
    all_metrics = {k: v['fn'] for k, v in registry.items()}
    all_aggregators = {k: v['agg'] for k, v in registry.items() if v['agg'] is not None}
    return calculate_sliced_stats(array, dim_order, all_metrics, all_aggregators)

class BasicStatsProcessor:
    NAME = "basic-stats"
    INPUT = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"

    OUTPUT_SCHEMA = {name: float for name in _column_fn_registry().keys()}
    OUTPUT_SCHEMA_PATTERNS = [(rf"^(?:{name})_[a-zA-Z]\d+(_[a-zA-Z]\d+)*$", float) for name in _column_fn_registry().keys()]

    DESCRIPTION = "Extracts basic image statistics such as mean, min, max."

    def run(self, art):
        dim_order = art.dim_order
        return calculate_np_array_stats(art.data, dim_order)
