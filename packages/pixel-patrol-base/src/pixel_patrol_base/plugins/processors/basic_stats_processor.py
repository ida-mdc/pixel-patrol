import logging
from typing import Dict, Callable

import numpy as np

from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_base.utils.array_utils import calculate_np_array_stats
from pixel_patrol_base.core.feature_schema import validate_processor_output

logger = logging.getLogger(__name__)

def _column_fn_registry() -> Dict[str, Dict[str, Callable]]:
    return {
        'mean_intensity': {'fn': lambda a: np.float32(np.mean(a)) if a.size else np.float32(np.nan), 'agg': np.mean},
        'std_intensity': {'fn': lambda a: np.float32(np.std(a)) if a.size else np.float32(np.nan), 'agg': np.mean},
        'min_intensity': {'fn': lambda a: np.float32(np.min(a)) if a.size else np.float32(np.nan), 'agg': np.min},
        'max_intensity': {'fn': lambda a: np.float32(np.max(a)) if a.size else np.float32(np.nan), 'agg': np.max},
    }

class BasicStatsProcessor:
    NAME = "basic-stats"
    INPUT = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"

    OUTPUT_SCHEMA = {name: np.float32 for name in _column_fn_registry().keys()}
    OUTPUT_SCHEMA_PATTERNS = [(rf"^(?:{name})_[a-zA-Z]\d+(_[a-zA-Z]\d+)*$", np.float32) for name in _column_fn_registry().keys()]

    DESCRIPTION = "Extracts basic image statistics such as mean, min, max."

    def run(self, art):
        dim_order = art.dim_order
        registry = _column_fn_registry()
        result = calculate_np_array_stats(art.data, dim_order, registry)
        return validate_processor_output(
            result,
            self.OUTPUT_SCHEMA,
            self.OUTPUT_SCHEMA_PATTERNS,
            processor_name=self.NAME
        )