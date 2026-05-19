from __future__ import annotations

import polars as pl

from pixel_patrol_base.core import processing
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec


def test_assign_files_to_workers_respects_batch_size():
    """Ensure batch iteration yields expected batch sizes and ordering."""
    df = pl.DataFrame({
        "row_index": [0, 1, 2, 3, 4],
        "path": ["a", "b", "c", "d", "e"],
    })

    batches = list(processing._assign_files_to_workers(df, n_files_per_worker=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert batches[0][0].row_index == 0
    assert batches[0][1].row_index == 1
    assert batches[-1][0].path == "e"


def test_combine_with_basic_preserves_loader_overlaps():
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

    combined = processing._combine_with_basic(basic, batch, deep_rows)

    # Loader metadata should take precedence
    assert combined.sort("row_index")["width"].to_list() == [100, 200]
    assert combined.sort("row_index")["height"].to_list() == [9, 10]


def test_process_worker_files_without_loader_returns_empty(monkeypatch):
    """Workers without a loader should skip processing and return no rows."""
    monkeypatch.setattr(processing, "_PROCESS_WORKER_CONTEXT", {})
    batch = [processing._IndexedPath(0, "a")]

    assert processing._process_worker_files(batch) == []


def test_records_accumulator_skips_combine_on_low_memory(tmp_path, monkeypatch):
    """When memory is insufficient, finalize should return an empty frame and keep chunks."""
    chunks_dir = tmp_path / "batches"
    chunks_dir.mkdir()
    acc = processing._RecordsAccumulator(chunk_every_n=1, chunks_dir=chunks_dir)

    acc.add(pl.DataFrame({"row_index": [0], "path": ["a"]}))

    monkeypatch.setattr(processing, "_estimate_available_memory_bytes", lambda: 1)
    final_df = acc.finalize()

    assert final_df.is_empty()
    assert list(chunks_dir.glob("chunk_*.parquet"))


def test_resolve_chunk_every_n_returns_config_value():
    """Flush threshold should be the exact value from config, not capped by file count."""
    config = ProcessingConfig(chunk_every_n=100)
    assert processing._resolve_chunk_every_n(10, config) == 100
    assert processing._resolve_chunk_every_n(1, config) == 100


def test_resolve_chunk_every_n_zero_files_disables_flush():
    """No files to process — flushing should be disabled."""
    config = ProcessingConfig(chunk_every_n=50)
    assert processing._resolve_chunk_every_n(0, config) == 0


def test_resolve_chunk_every_n_exceeds_record_count(tmp_path):
    """When chunk_every_n > actual records, no intermediate flush occurs and finalize still works."""
    chunks_dir = tmp_path / "batches"
    chunks_dir.mkdir()
    acc = processing._RecordsAccumulator(chunk_every_n=1000, chunks_dir=chunks_dir)

    # Add only 5 records — well below the threshold of 1000
    acc.add(pl.DataFrame({"row_index": list(range(5)), "path": [str(i) for i in range(5)]}))

    assert acc._active_df.height == 5
    assert not list(chunks_dir.glob("chunk_*.parquet")), "no intermediate flush expected"

    final_df = acc.finalize()
    assert final_df.height == 5


def test_resolve_n_files_per_worker_uses_total_rows_when_flush_disabled():
    """Batch size should be derived from total rows when flush is disabled."""
    assert processing._resolve_n_files_per_worker(worker_count=2, chunk_every_n=0, total_files=9) == 5
    assert processing._resolve_n_files_per_worker(worker_count=4, chunk_every_n=-1, total_files=3) == 1


def test_resolve_n_files_per_worker_scales_with_flush_threshold():
    """Batch size should scale by flush threshold and worker count."""
    assert processing._resolve_n_files_per_worker(worker_count=4, chunk_every_n=10, total_files=100) == 3
    assert processing._resolve_n_files_per_worker(worker_count=1, chunk_every_n=7, total_files=100) == 7


def test_cleanup_chunks_dir_removes_combined_by_default(tmp_path):
    """Cleanup should remove partial chunks and combined parquet by default."""
    chunks_dir = tmp_path / "batches"
    chunks_dir.mkdir()
    combined = chunks_dir / "records_df.parquet"
    combined.write_bytes(b"combined")
    partial = chunks_dir / "chunk_00000.parquet"
    partial.write_bytes(b"partial")

    processing._cleanup_chunks_dir(chunks_dir)

    assert not combined.exists()
    assert not partial.exists()
    assert not chunks_dir.exists()


def test_records_accumulator_handles_mixed_schema_batches():
    """Accumulator should preserve columns across batches with different schemas."""
    acc = processing._RecordsAccumulator(chunk_every_n=0, chunks_dir=None)

    acc.add(pl.DataFrame({"row_index": [0], "width": [10]}))
    acc.add(pl.DataFrame({"row_index": [1], "height": [20]}))

    final_df = acc.finalize().sort("row_index")

    assert set(final_df.columns) == {"row_index", "width", "height"}
    assert final_df["width"].to_list() == [10, None]
    assert final_df["height"].to_list() == [None, 20]


def test_resolve_n_files_per_worker_handles_zero_worker_count():
    """Batch sizing should still produce a valid size with zero workers."""
    assert processing._resolve_n_files_per_worker(worker_count=0, chunk_every_n=4, total_files=10) == 4


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


