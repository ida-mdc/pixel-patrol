from __future__ import annotations

from pathlib import Path

import polars as pl

from pixel_patrol_base.core import processing
from pixel_patrol_base.core.project_settings import Settings


def test_iter_indexed_batches_respects_batch_size():
    """Ensure batch iteration yields expected batch sizes and ordering."""
    df = pl.DataFrame({
        "row_index": [0, 1, 2, 3, 4],
        "path": ["a", "b", "c", "d", "e"],
    })

    batches = list(processing._iter_indexed_batches(df, batch_size=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert batches[0][0].row_index == 0
    assert batches[0][1].row_index == 1
    assert batches[-1][0].path == "e"


def test_combine_batch_with_basic_preserves_basic_overlaps():
    """Verify overlapping columns are kept from the basic frame, not deep rows."""
    basic = pl.DataFrame({
        "row_index": [0, 1],
        "path": ["a", "b"],
        "width": [5, 6],
    })
    batch = [processing._IndexedPath(0, "a"), processing._IndexedPath(1, "b")]
    deep_rows = [
        {"row_index": 0, "width": 100, "height": 9},
        {"row_index": 1, "width": 200, "height": 10},
    ]

    combined = processing._combine_batch_with_basic(basic, batch, deep_rows)

    assert combined.sort("row_index")["width"].to_list() == [5, 6]
    assert combined.sort("row_index")["height"].to_list() == [9, 10]


def test_process_batch_in_worker_without_loader_returns_empty(monkeypatch):
    """Workers without a loader should skip processing and return no rows."""
    monkeypatch.setattr(processing, "_PROCESS_WORKER_CONTEXT", {})
    batch = [processing._IndexedPath(0, "a")]

    assert processing._process_batch_in_worker(batch) == []


def test_records_accumulator_skips_combine_on_low_memory(tmp_path, monkeypatch):
    """When memory is insufficient, finalize should return an empty frame and keep chunks."""
    flush_dir = tmp_path / "batches"
    flush_dir.mkdir()
    acc = processing._RecordsAccumulator(flush_every_n=1, flush_dir=flush_dir)

    acc.add_batch(pl.DataFrame({"row_index": [0], "path": ["a"]}))

    monkeypatch.setattr(processing, "_estimate_available_memory_bytes", lambda: 1)
    final_df = acc.finalize()

    assert final_df.is_empty()
    assert list(flush_dir.glob("records_batch_*.parquet"))


def test_resolve_flush_threshold_caps_to_half_dataset():
    """Flush threshold should be capped to half the dataset size."""
    settings = Settings(records_flush_every_n=100)

    assert processing._resolve_flush_threshold(10, settings) == 5


def test_resolve_batch_size_uses_total_rows_when_flush_disabled():
    """Batch size should be derived from total rows when flush is disabled."""
    assert processing._resolve_batch_size(worker_count=2, flush_threshold=0, total_rows=9) == 5
    assert processing._resolve_batch_size(worker_count=4, flush_threshold=-1, total_rows=3) == 1


def test_resolve_batch_size_scales_with_flush_threshold():
    """Batch size should scale by flush threshold and worker count."""
    assert processing._resolve_batch_size(worker_count=4, flush_threshold=10, total_rows=100) == 3
    assert processing._resolve_batch_size(worker_count=1, flush_threshold=7, total_rows=100) == 7


def test_cleanup_partial_chunks_dir_preserves_combined_file(tmp_path):
    """Cleanup should remove only partial chunks and keep combined parquet."""
    flush_dir = tmp_path / "batches"
    flush_dir.mkdir()
    combined = flush_dir / "records_df.parquet"
    combined.write_bytes(b"combined")
    partial = flush_dir / "records_batch_00000.parquet"
    partial.write_bytes(b"partial")

    processing._cleanup_partial_chunks_dir(flush_dir)

    assert combined.exists()
    assert not partial.exists()
    assert flush_dir.exists()


def test_records_accumulator_handles_mixed_schema_batches():
    """Accumulator should preserve columns across batches with different schemas."""
    acc = processing._RecordsAccumulator(flush_every_n=0, flush_dir=None)

    acc.add_batch(pl.DataFrame({"row_index": [0], "width": [10]}))
    acc.add_batch(pl.DataFrame({"row_index": [1], "height": [20]}))

    final_df = acc.finalize().sort("row_index")

    assert set(final_df.columns) == {"row_index", "width", "height"}
    assert final_df["width"].to_list() == [10, None]
    assert final_df["height"].to_list() == [None, 20]


def test_resolve_batch_size_handles_zero_worker_count():
    """Batch sizing should still produce a valid size with zero workers."""
    assert processing._resolve_batch_size(worker_count=0, flush_threshold=4, total_rows=10) == 4
