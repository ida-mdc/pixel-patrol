from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import polars as pl

from pixel_patrol_base.core import processing
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_base.core.processing import build_records_df
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.project import Project
import pixel_patrol_base.core.project as project_module
import pixel_patrol_base.plugin_registry as plugin_registry_module


class DummyLoader:
    """Minimal loader stub for processing integration tests."""

    NAME = "dummy"
    SUPPORTED_EXTENSIONS = set()
    OUTPUT_SCHEMA = {}
    OUTPUT_SCHEMA_PATTERNS = []
    FOLDER_EXTENSIONS = set()

    def is_folder_supported(self, path: Path) -> bool:
        return False

    def load(self, source: str):
        raise RuntimeError("DummyLoader should not load data in these tests.")





def _make_fake_load_and_process(return_props):
    def fake(file_path, loader, processors, show_processor_progress=True, timing_out=None):
        if callable(return_props) and not isinstance(return_props, dict):
            props = return_props(file_path)
        else:
            props = return_props
        if not props:
            return []
        return [props]

    return fake


def test_build_records_df_flushes_and_combines_chunks(tmp_path, thread_client, monkeypatch):
    """Flushing writes intermediate parquet chunks and combines them into the final result."""
    (tmp_path / "a.png").write_bytes(b"")
    (tmp_path / "b.png").write_bytes(b"")

    monkeypatch.setattr(processing, "load_and_process_records_from_file",
                        _make_fake_load_and_process({"width": 1}))
    monkeypatch.setattr(plugin_registry_module, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "discover_loader", lambda _: DummyLoader())

    flush_dir = tmp_path / "_batches"
    df, _ = build_records_df(
        [tmp_path], DummyLoader(),
        processing_config=ProcessingConfig(records_flush_every_n=1),
        flush_dir=flush_dir,
    )

    assert df.height == 2
    assert "width" in df.columns


def test_build_records_df_workers_produce_correct_results(tmp_path, thread_client, monkeypatch):
    """Workers produce the correct metric value for each file."""
    (tmp_path / "x.png").write_bytes(b"")
    (tmp_path / "y.png").write_bytes(b"")

    monkeypatch.setattr(processing, "load_and_process_records_from_file",
                        _make_fake_load_and_process({"width": 5}))
    monkeypatch.setattr(plugin_registry_module, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "discover_loader", lambda _: DummyLoader())

    df, _ = build_records_df([tmp_path], DummyLoader())

    assert sorted(df["width"].to_list()) == [5, 5]


def test_process_records_infers_output_dir_when_missing(tmp_path, monkeypatch):
    """Project should infer output_dir when unset and base_dir is known."""
    project = Project(name="demo", base_dir=tmp_path, loader=None)

    monkeypatch.setattr(processing, "build_records_df", lambda *args, **kwargs: (pl.DataFrame(), None))

    project.process_records(processing_config=ProcessingConfig(selected_file_extensions="all"))

    assert project.output_path == Path(tmp_path) / "demo.parquet"


def test_build_records_df_survives_worker_failure(tmp_path, thread_client, monkeypatch):
    """A task that raises should not abort the whole run; other results are preserved."""
    (tmp_path / "one.png").write_bytes(b"")
    (tmp_path / "two.png").write_bytes(b"")

    monkeypatch.setattr(processing, "load_and_process_records_from_file",
                        _make_fake_load_and_process({"width": 33}))
    monkeypatch.setattr(plugin_registry_module, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "discover_loader", lambda _: DummyLoader())

    df, _ = build_records_df([tmp_path], DummyLoader())

    assert df.height == 2
    assert "path" in df.columns


def test_build_records_df_preserves_schema_across_batches(tmp_path, thread_client, monkeypatch):
    """Columns introduced by one file appear in all rows (null for files that omit them)."""
    p1 = tmp_path / "alpha.png"
    p2 = tmp_path / "beta.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    def per_file_props(file_path):
        if Path(file_path).name == "alpha.png":
            return {"width": 10, "extra": 99}
        return {"width": 20}

    monkeypatch.setattr(processing, "load_and_process_records_from_file",
                        _make_fake_load_and_process(per_file_props))
    monkeypatch.setattr(plugin_registry_module, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "discover_loader", lambda _: DummyLoader())

    df, _ = build_records_df(
        [tmp_path], DummyLoader(),
        processing_config=ProcessingConfig(records_flush_every_n=1),
    )

    assert {"width", "extra"}.issubset(df.columns)
    by_name = {row["name"]: row for row in df.to_dicts()}
    assert by_name["alpha.png"]["extra"] == 99
    assert by_name["beta.png"]["extra"] is None


# ---------- tests for multi-record loader support ----------


class _StubRecord:
    def __init__(self, meta: dict):
        self.meta = meta


def test_load_and_process_records_from_file_single_record(tmp_path):
    f = tmp_path / "single.png"
    f.write_bytes(b"\x89PNG")

    class SingleLoader:
        NAME = "single"
        def load(self, source):
            return _StubRecord({"color": "red"})

    result = processing.load_and_process_records_from_file(f, SingleLoader(), processors=[])
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["color"] == "red"


def test_load_and_process_records_from_file_multi_record(tmp_path):
    f = tmp_path / "multi.czi"
    f.write_bytes(b"\x00")

    class MultiLoader:
        NAME = "multi"
        def load(self, source):
            return {
                "scene_A": _StubRecord({"width": 10}),
                "scene_B": _StubRecord({"width": 20}),
            }

    result = processing.load_and_process_records_from_file(f, MultiLoader(), processors=[])
    assert isinstance(result, list)
    assert len(result) == 2
    ids = {r["child_id"] for r in result}
    assert ids == {"scene_A", "scene_B"}
    widths = {r["child_id"]: r["width"] for r in result}
    assert widths == {"scene_A": 10, "scene_B": 20}


def test_load_and_process_records_from_file_nonexistent(tmp_path):
    class AnyLoader:
        NAME = "any"
        def load(self, source):
            raise AssertionError("should not be called")

    result = processing.load_and_process_records_from_file(tmp_path / "nope.png", AnyLoader(), processors=[])
    assert result == []


def test_load_and_process_records_from_file_loader_failure(tmp_path):
    f = tmp_path / "bad.tif"
    f.write_bytes(b"bad")

    class FailLoader:
        NAME = "fail"
        def load(self, source):
            raise RuntimeError("boom")

    result = processing.load_and_process_records_from_file(f, FailLoader(), processors=[])
    assert result == []


def test_load_and_process_records_from_file_loader_returns_none(tmp_path):
    f = tmp_path / "nothing.tif"
    f.write_bytes(b"x")

    class NoneLoader:
        NAME = "none"
        def load(self, source):
            return None

    result = processing.load_and_process_records_from_file(f, NoneLoader(), processors=[])
    assert result == []


def test_load_and_process_records_from_file_skips_invalid_child_keys(tmp_path):
    f = tmp_path / "keys.czi"
    f.write_bytes(b"\x00")

    class BadKeysLoader:
        NAME = "badkeys"
        def load(self, source):
            return {
                "": _StubRecord({"x": 1}),
                123: _StubRecord({"x": 2}),
                "valid": _StubRecord({"x": 3}),
            }

    result = processing.load_and_process_records_from_file(f, BadKeysLoader(), processors=[])
    assert len(result) == 2
    child_ids = {r["child_id"] for r in result}
    assert child_ids == {"valid", "123"}


def test_build_records_df_multi_record_produces_multiple_rows(tmp_path, thread_client, monkeypatch):
    """A loader returning multiple records per file produces one row per record."""
    (tmp_path / "multi.czi").write_bytes(b"")

    def fake_multi(file_path, loader, processors, show_processor_progress=True, timing_out=None):
        return [
            {"child_id": "scene_0", "width": 10},
            {"child_id": "scene_1", "width": 20},
        ]

    monkeypatch.setattr(processing, "load_and_process_records_from_file", fake_multi)
    monkeypatch.setattr(plugin_registry_module, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "discover_loader", lambda _: DummyLoader())

    df, _ = build_records_df([tmp_path], DummyLoader())

    assert df.height == 2
    assert "child_id" in df.columns
    assert set(df["child_id"].to_list()) == {"scene_0", "scene_1"}
    assert set(df["width"].to_list()) == {10, 20}



def test_extract_record_properties_skips_runtime_error_processor():
    class RecordStub:
        def __init__(self):
            self.meta = {"a": 1}
            self.source_path = "dummy"
            self.dim_order = "YX"
            self.kind = "intensity"
            self.capabilities = {"spatial-2d"}

    class RuntimeSkipProcessor:
        NAME = "runtime-skip"
        INPUT = RecordSpec(axes={"Y", "X"}, kinds={"intensity"}, capabilities={"spatial-2d"})

        @staticmethod
        def run(_):
            raise RuntimeError("skip in never-materialize mode")

    class GoodProcessor:
        NAME = "good"
        INPUT = RecordSpec(axes={"Y", "X"}, kinds={"intensity"}, capabilities={"spatial-2d"})

        @staticmethod
        def run(_):
            return {"ok": 1}

    r = RecordStub()
    rows = processing._extract_record_properties(
        r,
        [RuntimeSkipProcessor(), GoodProcessor()],
        show_progress=False,
    )
    assert isinstance(rows, list) and len(rows) == 1
    assert rows[0]["a"] == 1
    assert rows[0]["ok"] == 1


def test_cleanup_partial_batches_dir(tmp_path):
    d = tmp_path / "batches"
    d.mkdir()

    p1 = d / "records_batch_00000.parquet"
    p2 = d / "records_batch_00001.parquet"
    combined = d / "records_df.parquet"

    pl.DataFrame({"row_index": [0]}).write_parquet(p1)
    pl.DataFrame({"row_index": [1]}).write_parquet(p2)
    pl.DataFrame({"a": [1]}).write_parquet(combined)

    assert p1.exists() and p2.exists() and combined.exists()

    processing._cleanup_partial_chunks_dir(d)

    assert not p1.exists()
    assert not p2.exists()
    assert not combined.exists()

    # calling again is a no-op
    processing._cleanup_partial_chunks_dir(d)

    # empty directory should be removed
    assert not d.exists()


def test_finalize_leaves_batches_dir_intact(tmp_path):
    """finalize() should combine chunks into a DataFrame but leave the flush_dir
    in place — cleanup is the caller's responsibility (via cleanup_flush_dir)."""
    flush_dir = tmp_path / "_batches"
    flush_dir.mkdir()

    p1 = flush_dir / "records_batch_00000.parquet"
    p2 = flush_dir / "records_batch_00001.parquet"
    pl.DataFrame({"a": [1], "row_index": [0]}).write_parquet(p1)
    pl.DataFrame({"a": [2], "row_index": [1]}).write_parquet(p2)

    accumulator = processing._RecordsAccumulator(
        flush_every_n=1,
        flush_dir=flush_dir,
    )
    accumulator._written_files = [p1, p2]
    accumulator._chunk_index = 2

    result = accumulator.finalize()

    assert result.height == 2
    assert "a" in result.columns
    assert flush_dir.exists(), "flush_dir must survive finalize(); only cleanup_flush_dir should remove it"


def test_finalize_without_flush_dir_returns_active_df():
    """finalize() without a flush_dir returns the active DataFrame directly."""
    accumulator = processing._RecordsAccumulator(
        flush_every_n=100,
        flush_dir=None,
    )
    accumulator._active_df = pl.DataFrame({"a": [1, 2, 3]})

    result = accumulator.finalize()

    assert result.height == 3
    assert result["a"].to_list() == [1, 2, 3]


# ---------- tests for cleanup_flush_dir ----------


def test_cleanup_not_called_when_no_records(tmp_path, monkeypatch):
    """cleanup_flush_dir must not be called when build_records_df returns nothing."""
    cleanup_calls = []
    monkeypatch.setattr(processing, "build_records_df", lambda *a, **kw: (None, None))
    monkeypatch.setattr(processing, "cleanup_flush_dir", lambda path: cleanup_calls.append(path))

    p = Project(name="test", base_dir=tmp_path)
    p.process_records()

    assert cleanup_calls == [], "cleanup_flush_dir must not be called when there are no records"


def test_cleanup_not_called_when_save_fails(tmp_path, monkeypatch):
    """cleanup_flush_dir must not be called when save_parquet raises — chunks are
    the only remaining copy of the data and must be preserved."""
    cleanup_calls = []
    monkeypatch.setattr(
        processing, "build_records_df",
        lambda *a, **kw: (pl.DataFrame({"a": [1]}), None),
    )

    def failing_save(df, path, meta):
        raise OSError("disk full")

    # Patch save_parquet on the project module where it was imported
    monkeypatch.setattr(project_module, "save_parquet", failing_save)
    monkeypatch.setattr(processing, "cleanup_flush_dir", lambda path: cleanup_calls.append(path))

    p = Project(name="test", base_dir=tmp_path)
    p.process_records()

    assert cleanup_calls == [], "cleanup_flush_dir must not be called after a failed save"


def test_cleanup_called_with_correct_path_after_successful_save(tmp_path, monkeypatch):
    """cleanup_flush_dir must be called exactly once with flush_dir = output_path.parent / '_batches'
    when save_parquet succeeds."""
    cleanup_calls = []
    monkeypatch.setattr(
        processing, "build_records_df",
        lambda *a, **kw: (pl.DataFrame({"a": [1]}), None),
    )
    monkeypatch.setattr(project_module, "save_parquet", lambda df, path, meta: None)
    monkeypatch.setattr(processing, "cleanup_flush_dir", lambda path: cleanup_calls.append(path))

    p = Project(name="test", base_dir=tmp_path)
    expected_flush_dir = p.output_path.parent / f"_batches_{p.name}"

    p.process_records()

    assert len(cleanup_calls) == 1, "cleanup_flush_dir must be called exactly once on success"
    assert cleanup_calls[0] == expected_flush_dir, (
        f"cleanup_flush_dir called with wrong path: {cleanup_calls[0]!r}, expected {expected_flush_dir!r}"
    )