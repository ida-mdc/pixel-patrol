import logging
from typing import Dict, Callable

import numpy as np

from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.utils.array_utils import calculate_np_array_stats
from pixel_patrol_base.core.feature_schema import validate_processor_output

logger = logging.getLogger(__name__)

def _column_fn_registry() -> Dict[str, Dict[str, Callable]]:
    return {
        'mean_intensity': {'fn': lambda a: np.float32(np.nanmean(a)) if a.size else np.float32(np.nan), 'agg': np.nanmean},
        'std_intensity': {'fn': lambda a: np.float32(np.nanstd(a)) if a.size else np.float32(np.nan), 'agg': np.nanmean},
        'min_intensity': {'fn': lambda a: np.float32(np.nanmin(a)) if a.size else np.float32(np.nan), 'agg': np.nanmin},
        'max_intensity': {'fn': lambda a: np.float32(np.nanmax(a)) if a.size else np.float32(np.nan), 'agg': np.nanmax},
        'nan_count': {'fn': lambda a: np.float32(np.sum(np.isnan(a))) if a.size else np.float32(np.nan),
                      'agg': np.nansum},
    }

class BasicStatsProcessor:
    NAME = "basic-stats"
    INPUT = RecordSpec(axes={"X", "Y"}, kinds={"intensity"}, capabilities={"spatial-2d"})
    OUTPUT = "features"

    OUTPUT_SCHEMA = {name: np.float32 for name in _column_fn_registry().keys()}
    OUTPUT_SCHEMA_PATTERNS = [(rf"^(?:{name})_[a-zA-Z]\d+(_[a-zA-Z]\d+)*$", np.float32) for name in _column_fn_registry().keys()]

    DESCRIPTION = "Extracts basic image statistics such as mean, min, max."

    def run(self, art: Record):
        dim_order = art.dim_order
        registry = _column_fn_registry()
        result = calculate_np_array_stats(art.data, dim_order, registry)
        return validate_processor_output(
            result,
            self.OUTPUT_SCHEMA,
            self.OUTPUT_SCHEMA_PATTERNS,
            processor_name=self.NAME
        )