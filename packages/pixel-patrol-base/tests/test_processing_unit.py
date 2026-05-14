from __future__ import annotations

import polars as pl

from pixel_patrol_base.core import processing
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec


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


def test_combine_batch_with_basic_preserves_loader_overlaps():
    """Verify overlapping columns are kept from the loader (deep) frame, not basic rows."""
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

    # Loader metadata should take precedence
    assert combined.sort("row_index")["width"].to_list() == [100, 200]
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
    config = ProcessingConfig(records_flush_every_n=100)

    assert processing._resolve_flush_threshold(10, config) == 5


def test_resolve_flush_threshold_enforces_max_intermediate_flushes(caplog):
    """Very low flush thresholds that would cause > MAX_INTERMEDIATE_FLUSHES are adjusted."""
    # Very large dataset with tiny requested flush => would result in many flushes
    config = ProcessingConfig(records_flush_every_n=1)

    with caplog.at_level("WARNING"):
        threshold = processing._resolve_flush_threshold(1_000_000, config)

    # Should be adjusted to ceil(total_rows / MAX_INTERMEDIATE_FLUSHES) => 1000
    assert threshold == 1000
    assert "Flushing this often on your dataset would result in" in caplog.text


def test_resolve_batch_size_uses_total_rows_when_flush_disabled():
    """Batch size should be derived from total rows when flush is disabled."""
    assert processing._resolve_batch_size(worker_count=2, flush_threshold=0, total_rows=9) == 5
    assert processing._resolve_batch_size(worker_count=4, flush_threshold=-1, total_rows=3) == 1


def test_resolve_batch_size_scales_with_flush_threshold():
    """Batch size should scale by flush threshold and worker count."""
    assert processing._resolve_batch_size(worker_count=4, flush_threshold=10, total_rows=100) == 3
    assert processing._resolve_batch_size(worker_count=1, flush_threshold=7, total_rows=100) == 7


def test_cleanup_partial_chunks_dir_removes_combined_by_default(tmp_path):
    """Cleanup should remove partial chunks and combined parquet by default."""
    flush_dir = tmp_path / "batches"
    flush_dir.mkdir()
    combined = flush_dir / "records_df.parquet"
    combined.write_bytes(b"combined")
    partial = flush_dir / "records_batch_00000.parquet"
    partial.write_bytes(b"partial")

    processing._cleanup_partial_chunks_dir(flush_dir)

    assert not combined.exists()
    assert not partial.exists()
    assert not flush_dir.exists()


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


def test_resolve_worker_count_single_file_and_caps(monkeypatch):
    """Worker count should be 1 when only 1 file; and capped by total rows."""
    monkeypatch.setattr(processing.os, "cpu_count", lambda: 8)
    # No settings: should not exceed total rows
    assert processing._resolve_worker_count(ProcessingConfig(), total_rows=1) == 1
    assert processing._resolve_worker_count(ProcessingConfig(), total_rows=3) == 3
    # Without total_rows provided, should default to CPU count
    assert processing._resolve_worker_count(ProcessingConfig()) == 8


def test_resolve_worker_count_respects_settings_and_total(monkeypatch):
    """Requested workers are capped by CPU count and by total rows."""
    monkeypatch.setattr(processing.os, "cpu_count", lambda: 8)
    config = ProcessingConfig(processing_max_workers=10)
    assert processing._resolve_worker_count(config, total_rows=6) == 6
    config2 = ProcessingConfig(processing_max_workers=4)
    assert processing._resolve_worker_count(config2, total_rows=6) == 4


# ---------------------------------------------------------------------------
# _extract_record_properties — obs_level guarantee
# ---------------------------------------------------------------------------

def _make_record(meta=None):
    return Record(
        data=None,
        dim_order="YX",
        dim_names=["Y", "X"],
        kind="image/intensity",
        meta=meta or {"path": "a.png"},
        capabilities=set(),
    )


def test_extract_record_properties_obs_level_present_no_processors():
    """obs_level=0 is injected even when no processor runs."""
    rows = processing._extract_record_properties(_make_record(), [], show_progress=False)
    assert len(rows) == 1
    assert rows[0]["obs_level"] == 0


def test_extract_record_properties_obs_level_present_when_processor_omits_it():
    """obs_level defaults to 0 when a processor returns rows without obs_level."""
    class _NoObsProc:
        NAME = "no_obs"
        INPUT = RecordSpec()
        def run(self, rcd):
            return [{"mean": 1.0}]

    rows = processing._extract_record_properties(_make_record(), [_NoObsProc()], show_progress=False)
    assert all("obs_level" in r for r in rows)
    assert all(r["obs_level"] == 0 for r in rows)


def test_extract_record_properties_obs_level_from_processor_wins():
    """obs_level emitted by a processor overrides the default 0."""
    class _SlicedProc:
        NAME = "sliced"
        INPUT = RecordSpec()
        def run(self, rcd):
            return [{"obs_level": 1, "mean": 0.5}, {"obs_level": 0, "mean": 1.0}]

    rows = processing._extract_record_properties(_make_record(), [_SlicedProc()], show_progress=False)
    levels = {r["obs_level"] for r in rows}
    assert levels == {0, 1}


def test_extract_record_properties_obs_level_scalar_processor_fallback():
    """obs_level=0 is present when a scalar-return (legacy) processor doesn't set it."""
    class _ScalarProc:
        NAME = "scalar"
        INPUT = RecordSpec()
        def run(self, rcd):
            return {"extra": 42}

    rows = processing._extract_record_properties(_make_record(), [_ScalarProc()], show_progress=False)
    assert len(rows) == 1
    assert rows[0]["obs_level"] == 0
    assert rows[0]["extra"] == 42


# ---------------------------------------------------------------------------
# _merge_long_rows
# ---------------------------------------------------------------------------

def test_merge_long_rows_matching_coords_merges_fields():
    existing = [{"obs_level": 0, "mean": 1.0}, {"obs_level": 1, "dim_t": 0, "mean": 2.0}]
    incoming = [{"obs_level": 0, "std": 0.5}, {"obs_level": 1, "dim_t": 0, "std": 0.1}]
    result = processing._merge_long_rows(existing, incoming)
    assert len(result) == 2
    g = next(r for r in result if r["obs_level"] == 0)
    assert g["mean"] == 1.0 and g["std"] == 0.5
    s = next(r for r in result if r["obs_level"] == 1)
    assert s["mean"] == 2.0 and s["std"] == 0.1


def test_merge_long_rows_unmatched_incoming_appended():
    existing = [{"obs_level": 0, "mean": 1.0}]
    incoming = [{"obs_level": 1, "dim_t": 0, "mean": 2.0}]
    result = processing._merge_long_rows(existing, incoming)
    assert len(result) == 2


