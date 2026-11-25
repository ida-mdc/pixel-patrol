"""N-D label metric processor for integer label masks."""

from enum import StrEnum
from typing import Any, Dict, Tuple

import numpy as np

from pixel_patrol_base.core.contracts import ChunkKind
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_base.plugins.processors.raster_processor import (
    RasterMetricSpec,
)


class LabelMetricNames(StrEnum):
    LABEL_COUNT       = "label_count"
    LABEL_IDS         = "label_ids"
    LABEL_AREAS       = "label_areas"
    LABEL_MIN_SIZE    = "label_min_size"
    # In case of ties, argmin/argmax return the first occurrence.
    # Since label_ids is sorted, the smallest label ID is selected.
    # TODO: Consider supporting multiple IDs with equal extremal sizes.
    LABEL_MIN_SIZE_ID = "label_min_size_id"
    LABEL_MAX_SIZE    = "label_max_size"
    LABEL_MAX_SIZE_ID = "label_max_size_id"
    LABEL_MEAN_SIZE   = "label_mean_size"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _single_chunk_agg(spec: RasterMetricSpec, rows: list) -> Any:
    """Return the metric value for a single chunk."""
    if not rows:
        return None

    if len(rows) != 1:
        raise NotImplementedError(
            "Aggregation across multiple label chunks is not supported yet."
        )

    return rows[0].get(spec.name)


# ---------------------------------------------------------------------------
# NumPy n-D backend
# ---------------------------------------------------------------------------

def numpy_label_compute(arr: np.ndarray) -> Dict[str, Any]:
    """Compute basic size statistics for an integer label mask.

    All label IDs are treated equally, including zero.
    Sizes are measured in pixels or voxels.
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("Label masks must contain integer IDs.")

    if arr.size == 0:
        return {}

    label_ids, counts = np.unique(arr, return_counts=True)
    label_count = label_ids.size

    min_idx = np.argmin(counts)
    max_idx = np.argmax(counts)

    return {
        LabelMetricNames.LABEL_COUNT: int(label_count),
        LabelMetricNames.LABEL_IDS: label_ids,
        LabelMetricNames.LABEL_AREAS: counts,
        LabelMetricNames.LABEL_MIN_SIZE: int(counts[min_idx]),
        LabelMetricNames.LABEL_MIN_SIZE_ID: int(label_ids[min_idx]),
        LabelMetricNames.LABEL_MAX_SIZE: int(counts[max_idx]),
        LabelMetricNames.LABEL_MAX_SIZE_ID: int(label_ids[max_idx]),
        LabelMetricNames.LABEL_MEAN_SIZE: arr.size / label_count,
    }


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class LabelProcessor:
    NAME        = "label-basic"
    # DEFAULT_ENABLED = False # temp_ 
    DESCRIPTION = "Computes basic label counts and size statistics for integer label masks."
    CHUNK_KIND  = ChunkKind.LEAF
    INPUT       = RecordSpec(axes=set(), kinds={"intensity"})
    OUTPUT      = "features"

    METRICS: Tuple[RasterMetricSpec, ...] = (
        RasterMetricSpec(
            name=LabelMetricNames.LABEL_COUNT,
            data_type=np.uint64,
            aggregate_rows=_single_chunk_agg,
            description="Number of distinct label IDs.",
        ),
        RasterMetricSpec(
            name=LabelMetricNames.LABEL_IDS,
            data_type=np.ndarray,
            aggregate_rows=_single_chunk_agg,
            description="Sorted unique label IDs.",
        ),
        RasterMetricSpec(
            name=LabelMetricNames.LABEL_AREAS,
            data_type=np.ndarray,
            aggregate_rows=_single_chunk_agg,
            description="Number of pixels or voxels per label ID.",
        ),
        RasterMetricSpec(
            name=LabelMetricNames.LABEL_MIN_SIZE,
            data_type=np.uint64,
            aggregate_rows=_single_chunk_agg,
            description="Size of the smallest label.",
        ),
        RasterMetricSpec(
            name=LabelMetricNames.LABEL_MIN_SIZE_ID,
            data_type=np.int64,
            aggregate_rows=_single_chunk_agg,
            description="ID of the smallest label.",
        ),
        RasterMetricSpec(
            name=LabelMetricNames.LABEL_MAX_SIZE,
            data_type=np.uint64,
            aggregate_rows=_single_chunk_agg,
            description="Size of the largest label.",
        ),
        RasterMetricSpec(
            name=LabelMetricNames.LABEL_MAX_SIZE_ID,
            data_type=np.int64,
            aggregate_rows=_single_chunk_agg,
            description="ID of the largest label.",
        ),
        RasterMetricSpec(
            name=LabelMetricNames.LABEL_MEAN_SIZE,
            data_type=np.float64,
            aggregate_rows=_single_chunk_agg,
            description="Mean label size in pixels or voxels.",
        ),
    )

    OUTPUT_SCHEMA = {m.name: m.data_type for m in METRICS}
    OUTPUT_SCHEMA_DESCRIPTIONS = {
        m.name: m.description for m in METRICS
    }

    def run_chunk(self, record: Record) -> Dict:
        chunk = (
            record.data.compute()
            if hasattr(record.data, "compute")
            else np.asarray(record.data)
        )

        return numpy_label_compute(chunk)

    def get_aggregation(self, name: str):
        spec = next((s for s in self.METRICS if s.name == name), None)
        if spec is None:
            return None
        return lambda rows, g_dims: spec.aggregate_rows(spec, rows)