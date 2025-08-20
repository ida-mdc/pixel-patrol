import logging
from typing import Dict, Callable, Any, Tuple, List

import dask.array as da
import numpy as np

from pixel_patrol_base.core.image_operations_and_metadata import calculate_sliced_stats
from pixel_patrol_base.core.processor_interface import PixelPatrolProcessor

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

class BasicStatsProcessor(PixelPatrolProcessor):

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def description(self) -> str:
        return "Extracts basic image statistics such as mean, min, max."

    def process(self, data: da.array, dim_order: str) -> dict:
        return calculate_np_array_stats(data, dim_order)

    def get_specification(self) -> Dict[str, Any]:
        res = {}
        for name in _column_fn_registry().keys():
            res[name] = float
        return res

    def get_dynamic_specification_patterns(self) -> List[Tuple[str, Any]]:
        res = []
        for name in _column_fn_registry().keys():
            res.append((rf"^(?:{name})_[a-zA-Z]\d+(_[a-zA-Z]\d+)*$", float))
        return res