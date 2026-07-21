"""Unit tests for _run_record: full_shape stamping and image_meta isolation."""
from __future__ import annotations

import numpy as np

from pixel_patrol_base.core.processing import _run_record
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.record import record_from
from _processing_mocks import MockMemoryProcessor


def _record(shape=(6, 20, 20), dim_order="ZYX"):
    arr = np.zeros(shape, dtype="uint8")
    return record_from(arr, {"dim_order": dim_order}, kind="intensity")


def test_full_shape_reflects_the_unsliced_record_not_the_chunk():
    seen_meta = {}

    class SpyProcessor(MockMemoryProcessor):
        def run_chunk(self, record):
            seen_meta.update(record.meta)
            return {}

    record = _record(shape=(6, 20, 20))
    data   = record.data[0:2]  # a chunk covering only Z=[0,2) of the full Z=6
    result = _run_record(record, data, (0, 0, 0), file_index=0, child_id=None,
                         processors=[SpyProcessor("spy", {})], config=ProcessingConfig(),
                         file_path="mock")

    assert seen_meta["full_shape"] == (6, 20, 20)
    assert result.image_meta.get("full_shape") is None  # never leaks into obs rows
