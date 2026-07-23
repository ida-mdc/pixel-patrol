"""Unit tests for _run_record: metadata stamping and processor-failure isolation.

Uses REAL processor classes (not mocks) with run_chunk monkeypatched to raise, so the
failure-isolation tests exercise the actual per-processor try/except in
_process_memory_chunk (processing.py) and the actual _rollup emit-even-if-empty
behavior, not a hand-built stand-in.
"""
from __future__ import annotations

import numpy as np
import pytest

from pixel_patrol_base.core.processing import _run_record, _rollup
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.record import record_from
from pixel_patrol_base.plugins.processors.raster_processor import BasicMetricsProcessor, HistogramProcessor
from pixel_patrol_base.plugins.processors.thumbnail_processor import ThumbnailProcessor
from _processing_mocks import MockMemoryProcessor

PROCESSORS = [BasicMetricsProcessor(), HistogramProcessor(), ThumbnailProcessor()]


def _raise(self, record):
    raise RuntimeError(f"{type(self).__name__} exploded")


def _real_record(shape=(64, 64), dim_order="YX"):
    arr = np.random.randint(0, 255, size=shape, dtype=np.uint16)
    return record_from(arr, {"dim_order": dim_order}, kind="intensity")


def test_full_shape_reflects_the_unsliced_record_not_the_chunk():
    seen_meta = {}

    class SpyProcessor(MockMemoryProcessor):
        def run_chunk(self, record):
            seen_meta.update(record.meta)
            return {}

    record = _real_record(shape=(6, 20, 20), dim_order="ZYX")
    data   = record.data[0:2]  # a chunk covering only Z=[0,2) of the full Z=6
    result = _run_record(record, data, (0, 0, 0), file_index=0, child_id=None,
                         processors=[SpyProcessor("spy", {})], config=ProcessingConfig(),
                         file_path="mock")

    assert seen_meta["full_shape"] == (6, 20, 20)
    assert result.image_meta.get("full_shape") is None  # never leaks into obs rows


def test_processors_succeed_sanity_check():
    """Baseline: confirms the record/processors are wired correctly before we break them."""
    record = _real_record()
    result = _run_record(record, record.data, (0, 0), 0, None, PROCESSORS, ProcessingConfig(), "/data/x.tif")

    assert result.leaf_rows, "expected leaf rows when processors succeed"
    assert result.chunk_rows, "expected chunk rows (thumbnail) when processors succeed"


def test_all_processors_raising_still_produces_metadata_row(monkeypatch):
    monkeypatch.setattr(BasicMetricsProcessor, "run_chunk", _raise)
    monkeypatch.setattr(HistogramProcessor, "run_chunk", _raise)
    monkeypatch.setattr(ThumbnailProcessor, "run_chunk", _raise)

    record = _real_record()
    result = _run_record(record, record.data, (0, 0), 0, None, PROCESSORS, ProcessingConfig(), "/data/x.tif")

    assert result.leaf_rows == []
    assert result.chunk_rows == {}
    assert result.image_meta.get("size_Y") == 64
    assert result.image_meta.get("size_X") == 64
    assert result.image_meta.get("dtype") == "uint16"

    obs_rows = _rollup([result], PROCESSORS, ProcessingConfig().slice_size)
    assert len(obs_rows) == 1
    row = obs_rows[0]
    assert row["num_pixels"] == 0
    assert row["size_Y"] == 64 and row["size_X"] == 64
    for metric_col in ("mean_intensity", "min_intensity", "max_intensity", "thumbnail"):
        assert metric_col not in row


def test_one_processor_raising_leaves_the_others_output_intact(monkeypatch):
    monkeypatch.setattr(BasicMetricsProcessor, "run_chunk", _raise)

    record = _real_record()
    result = _run_record(record, record.data, (0, 0), 0, None, PROCESSORS, ProcessingConfig(), "/data/x.tif")

    assert result.leaf_rows, "HistogramProcessor should still have produced a leaf row"
    assert "histogram_min" in result.leaf_rows[0]
    assert "min_intensity" not in result.leaf_rows[0]  # the raising processor contributed nothing
    assert result.chunk_rows, "ThumbnailProcessor (memory-kind) is unaffected by the leaf-kind failure"

    obs_rows = _rollup([result], PROCESSORS, ProcessingConfig().slice_size)
    assert len(obs_rows) == 1
    assert obs_rows[0]["num_pixels"] == 4096
    assert "histogram_min" in obs_rows[0]
    assert "min_intensity" not in obs_rows[0]
