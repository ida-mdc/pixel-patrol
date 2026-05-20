from __future__ import annotations

import polars as pl
import pytest

from pixel_patrol_base.core import processing
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.record import Record
from pixel_patrol_base.core.specs import RecordSpec


def test_assign_files_to_workers_yields_correct_groups():
    """Groups of files are yielded with expected sizes and ordering."""
    df = pl.DataFrame({
        "row_index": [0, 1, 2, 3, 4],
        "path": ["a", "b", "c", "d", "e"],
    })

    batches = list(processing._assign_files_to_workers(df, n_files_per_worker=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert batches[0][0].row_index == 0
    assert batches[0][1].row_index == 1
    assert batches[-1][0].path == "e"


def test_combine_all_chunks_loader_columns_take_precedence(tmp_path):
    """When a column exists in both basic and chunks, the chunk (loader) value wins."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()

    pl.DataFrame({
        "row_index": [0, 1],
        "width": [100, 200],
        "height": [9, 10],
    }).with_columns(pl.col("row_index").cast(pl.Int64)).write_parquet(
        chunks_dir / "chunk_000000_00000.parquet"
    )

    basic_df = pl.DataFrame({
        "row_index": [0, 1],
        "path": ["a", "b"],
        "width": [5, 6],  # should be overridden by chunk values
    }).with_columns(pl.col("row_index").cast(pl.Int64))

    result = processing._combine_all_chunks(chunks_dir, basic_df).sort("path")

    assert result["width"].to_list() == [100, 200]
    assert result["height"].to_list() == [9, 10]


def test_process_worker_files_without_loader_writes_no_chunks(tmp_path, monkeypatch):
    """Worker with no loader should write nothing."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    monkeypatch.setattr(processing, "_PROCESS_WORKER_CONTEXT", {
        "loader": None,
        "processors": [],
        "chunks_dir": chunks_dir,
        "chunk_every_n": 100,
    })

    processing._process_worker_files([processing._IndexedPath(0, "a")], worker_id=0)

    assert not list(chunks_dir.glob("chunk_*.parquet"))


def test_combine_all_chunks_skips_on_low_memory(tmp_path, monkeypatch):
    """When available memory is insufficient, returns empty and leaves chunk files intact."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    pl.DataFrame({"row_index": [0], "path": ["a"]}).with_columns(
        pl.col("row_index").cast(pl.Int64)
    ).write_parquet(chunks_dir / "chunk_000000_00000.parquet")

    basic_df = pl.DataFrame({"row_index": [0], "path": ["a"]}).with_columns(
        pl.col("row_index").cast(pl.Int64)
    )

    monkeypatch.setattr(processing, "_estimate_available_memory_bytes", lambda: 1)
    result = processing._combine_all_chunks(chunks_dir, basic_df)

    assert result.is_empty()
    assert list(chunks_dir.glob("chunk_*.parquet")), "chunk files must be preserved"


def test_process_files_writes_one_final_chunk_when_threshold_not_reached(tmp_path):
    """When chunk_every_n > rows produced, exactly one final chunk is still written."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    f = tmp_path / "data.lmdb"
    f.write_bytes(b"")

    class _FiveRecords:
        NAME = "five"
        def load_iter(self, source):
            for i in range(5):
                yield str(i), type("R", (), {"meta": {"path": source}})()

    processing._process_files(
        [processing._IndexedPath(0, str(f))],
        loader=_FiveRecords(),
        processors=[],
        chunks_dir=chunks_dir,
        chunk_every_n=1000,
        worker_id=0,
        show_progress=False,
    )

    chunks = list(chunks_dir.glob("chunk_*.parquet"))
    assert len(chunks) == 1
    assert pl.read_parquet(chunks[0]).height == 5


def test_process_files_flushes_mid_file_at_threshold(tmp_path):
    """When rows exceed chunk_every_n mid-file, multiple chunks are written."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    f = tmp_path / "data.lmdb"
    f.write_bytes(b"")

    class _TenRecords:
        NAME = "ten"
        def load_iter(self, source):
            for i in range(10):
                yield str(i), type("R", (), {"meta": {"path": source}})()

    processing._process_files(
        [processing._IndexedPath(0, str(f))],
        loader=_TenRecords(),
        processors=[],
        chunks_dir=chunks_dir,
        chunk_every_n=3,
        worker_id=0,
        show_progress=False,
    )

    chunks = list(chunks_dir.glob("chunk_*.parquet"))
    total_rows = sum(pl.read_parquet(c).height for c in chunks)
    assert len(chunks) > 1, "expected multiple chunk files"
    assert total_rows == 10


def test_combine_all_chunks_handles_mixed_schemas(tmp_path):
    """Chunks with different schemas combine with nulls where columns are absent."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    pl.DataFrame({"row_index": [0], "width": [10]}).with_columns(
        pl.col("row_index").cast(pl.Int64)
    ).write_parquet(chunks_dir / "chunk_000000_00000.parquet")
    pl.DataFrame({"row_index": [1], "height": [20]}).with_columns(
        pl.col("row_index").cast(pl.Int64)
    ).write_parquet(chunks_dir / "chunk_000001_00000.parquet")

    basic_df = pl.DataFrame({"row_index": [0, 1], "path": ["a", "b"]}).with_columns(
        pl.col("row_index").cast(pl.Int64)
    )
    result = processing._combine_all_chunks(chunks_dir, basic_df).sort("path")

    assert "width" in result.columns
    assert "height" in result.columns
    assert result["width"].to_list() == [10, None]
    assert result["height"].to_list() == [None, 20]


def test_combine_all_chunks_includes_failed_files_as_basic_only(tmp_path):
    """Files that produced no rows still appear in the result with basic info and null processed cols."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    pl.DataFrame({"row_index": [0], "width": [42]}).with_columns(
        pl.col("row_index").cast(pl.Int64)
    ).write_parquet(chunks_dir / "chunk_000000_00000.parquet")

    basic_df = pl.DataFrame({
        "row_index": [0, 1],
        "path": ["a", "b"],  # row_index=1 has no chunk → should appear with null width
    }).with_columns(pl.col("row_index").cast(pl.Int64))

    result = processing._combine_all_chunks(chunks_dir, basic_df).sort("path")

    assert result.height == 2
    assert result.filter(pl.col("path") == "a")["width"][0] == 42
    assert result.filter(pl.col("path") == "b")["width"][0] is None


def test_cleanup_chunks_dir_removes_combined_by_default(tmp_path):
    """Cleanup removes chunk files and combined parquet, then removes the directory."""
    chunks_dir = tmp_path / "batches"
    chunks_dir.mkdir()
    combined = chunks_dir / "records_df.parquet"
    combined.write_bytes(b"combined")
    partial = chunks_dir / "chunk_000000_00000.parquet"
    partial.write_bytes(b"partial")

    processing._cleanup_chunks_dir(chunks_dir)

    assert not combined.exists()
    assert not partial.exists()
    assert not chunks_dir.exists()


def test_resolve_worker_count_single_file_and_caps(monkeypatch):
    """Worker count is 1 for a single file and capped by total file count."""
    monkeypatch.setattr(processing.os, "cpu_count", lambda: 8)
    assert processing._resolve_worker_count(ProcessingConfig(), total_rows=1) == 1
    assert processing._resolve_worker_count(ProcessingConfig(), total_rows=3) == 3
    assert processing._resolve_worker_count(ProcessingConfig()) == 8


def test_resolve_worker_count_respects_settings_and_total(monkeypatch):
    """Requested workers are capped by CPU count and by total file count."""
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
    rows = processing._extract_record_properties(_make_record(), [], show_progress=False)
    assert len(rows) == 1
    assert rows[0]["obs_level"] == 0


def test_extract_record_properties_obs_level_present_when_processor_omits_it():
    class _NoObsProc:
        NAME = "no_obs"
        INPUT = RecordSpec()
        def run(self, rcd):
            return [{"mean": 1.0}]

    rows = processing._extract_record_properties(_make_record(), [_NoObsProc()], show_progress=False)
    assert all("obs_level" in r for r in rows)
    assert all(r["obs_level"] == 0 for r in rows)


def test_extract_record_properties_obs_level_from_processor_wins():
    class _SlicedProc:
        NAME = "sliced"
        INPUT = RecordSpec()
        def run(self, rcd):
            return [{"obs_level": 1, "mean": 0.5}, {"obs_level": 0, "mean": 1.0}]

    rows = processing._extract_record_properties(_make_record(), [_SlicedProc()], show_progress=False)
    levels = {r["obs_level"] for r in rows}
    assert levels == {0, 1}


def test_extract_record_properties_obs_level_scalar_processor_fallback():
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