from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import List, Optional

import polars as pl

from pixel_patrol_base.core import processing
from pixel_patrol_base.core.specs import RecordSpec
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.project import Project
import pixel_patrol_base.core.project as project_module
import pixel_patrol_base.plugin_registry as plugin_registry_module


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

class DummyLoader:
    NAME = "dummy"
    SUPPORTED_EXTENSIONS = set()
    OUTPUT_SCHEMA = {}
    OUTPUT_SCHEMA_PATTERNS = []
    FOLDER_EXTENSIONS = set()

    def is_folder_supported(self, path: Path) -> bool:
        return False

    def load(self, source: str):
        raise RuntimeError("DummyLoader.load should not be called in these tests.")


class FakeProcessPoolExecutor:
    """Synchronous in-process executor that exercises the ProcessPoolExecutor code path."""

    last_instance: Optional["FakeProcessPoolExecutor"] = None

    def __init__(self, max_workers: int, initializer=None, initargs=None, mp_context=None) -> None:
        self.max_workers = max_workers
        self.initializer = initializer
        self.initargs = initargs or ()
        self.initializer_called = False
        if self.initializer:
            self.initializer(*self.initargs)
            self.initializer_called = True
        FakeProcessPoolExecutor.last_instance = self

    def __enter__(self) -> "FakeProcessPoolExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


class FakeThreadPoolExecutor:
    """Synchronous in-process executor for the thread-pool fallback path."""

    last_instance: Optional["FakeThreadPoolExecutor"] = None

    def __init__(self, max_workers: int) -> None:
        self.max_workers = max_workers
        FakeThreadPoolExecutor.last_instance = self

    def __enter__(self) -> "FakeThreadPoolExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


class FakeProcessPoolExecutorWithFailure:
    """Executor whose first submitted task always fails; subsequent tasks succeed."""

    def __init__(self, max_workers: int, initializer=None, initargs=None, mp_context=None) -> None:
        self._initializer = initializer
        self._initargs = initargs or ()
        self._failed_once = False
        if self._initializer:
            self._initializer(*self._initargs)

    def __enter__(self) -> "FakeProcessPoolExecutorWithFailure":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def submit(self, fn, *args, **kwargs) -> Future:
        future: Future = Future()
        if not self._failed_once:
            self._failed_once = True
            future.set_exception(RuntimeError("worker crash"))
            return future
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


def _basic_df_for_paths(paths: List[Path]) -> pl.DataFrame:
    return (
        pl.DataFrame({
            "path": [str(p) for p in paths],
            "type": ["file"] * len(paths),
        })
        .with_row_index("row_index")
        .with_columns(pl.col("row_index").cast(pl.Int64))
    )


def _make_fake_iter_file_rows(return_props):
    """Return a fake _iter_file_rows generator that yields one row dict per file."""
    def fake_iter(file_path, loader, processors, show_progress):
        if callable(return_props) and not isinstance(return_props, dict):
            props = return_props(file_path)
        else:
            props = return_props
        if props:
            yield props
    return fake_iter


# ---------------------------------------------------------------------------
# _build_deep_record_df — integration
# ---------------------------------------------------------------------------

def test_build_deep_record_df_flushes_and_combines_chunks(tmp_path, monkeypatch):
    """Chunk files are written during processing and combined into a single result."""
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    config = ProcessingConfig(processing_max_workers=1, chunk_every_n=1)
    monkeypatch.setattr(processing, "_iter_file_rows", _make_fake_iter_file_rows({"width": 1}))
    monkeypatch.setattr(plugin_registry_module, "discover_processor_plugins", lambda: [])

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(
        basic_df, DummyLoader(), processing_config=config,
        chunks_dir=tmp_path / "_chunks",
    )

    assert df.height == 2
    assert "width" in df.columns


def test_build_deep_record_df_process_pool_path_uses_initializer(tmp_path, monkeypatch):
    """ProcessPoolExecutor branch: initializer is called and results are combined."""
    p1 = tmp_path / "x.png"
    p2 = tmp_path / "y.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    config = ProcessingConfig(processing_max_workers=2)
    monkeypatch.setattr(processing, "_iter_file_rows", _make_fake_iter_file_rows({"width": 5}))
    monkeypatch.setattr(plugin_registry_module, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "discover_loader", lambda loader_id: DummyLoader())
    monkeypatch.setattr(processing, "ProcessPoolExecutor", FakeProcessPoolExecutor)

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(
        basic_df, DummyLoader(), processing_config=config,
        chunks_dir=tmp_path / "_chunks",
    )

    assert FakeProcessPoolExecutor.last_instance is not None
    assert FakeProcessPoolExecutor.last_instance.max_workers == 2
    assert FakeProcessPoolExecutor.last_instance.initializer_called is True
    assert sorted(df["width"].to_list()) == [5, 5]


def test_build_deep_record_df_thread_fallback_on_process_pool_error(tmp_path, monkeypatch):
    """Falls back to ThreadPoolExecutor when ProcessPoolExecutor creation fails."""
    p1 = tmp_path / "m.png"
    p2 = tmp_path / "n.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    config = ProcessingConfig(processing_max_workers=2)

    class FailingProcessPoolExecutor:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("process pool unavailable")

    monkeypatch.setattr(processing, "_iter_file_rows", _make_fake_iter_file_rows({"width": 7}))
    monkeypatch.setattr(plugin_registry_module, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "ProcessPoolExecutor", FailingProcessPoolExecutor)
    monkeypatch.setattr(processing, "ThreadPoolExecutor", FakeThreadPoolExecutor)

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(
        basic_df, DummyLoader(), processing_config=config,
        chunks_dir=tmp_path / "_chunks",
    )

    assert FakeThreadPoolExecutor.last_instance is not None
    assert FakeThreadPoolExecutor.last_instance.max_workers == 2
    assert sorted(df["width"].to_list()) == [7, 7]


def test_process_records_infers_output_dir_when_missing(tmp_path, monkeypatch):
    """Project infers output_dir when unset and base_dir is known."""
    project = Project(name="demo", base_dir=tmp_path, loader=None)
    monkeypatch.setattr(processing, "build_records_df", lambda *args, **kwargs: pl.DataFrame())
    project.process_records(processing_config=ProcessingConfig(selected_file_extensions="all"))
    assert project.output_path == Path(tmp_path) / "demo.parquet"


def test_build_deep_record_df_survives_worker_failure(tmp_path, monkeypatch):
    """A failed worker batch does not abort the run; the failed file appears with basic info only."""
    p1 = tmp_path / "one.png"
    p2 = tmp_path / "two.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    config = ProcessingConfig(processing_max_workers=2)
    monkeypatch.setattr(processing, "_iter_file_rows", _make_fake_iter_file_rows({"width": 33}))
    monkeypatch.setattr(plugin_registry_module, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "discover_loader", lambda loader_id: DummyLoader())
    monkeypatch.setattr(processing, "ProcessPoolExecutor", FakeProcessPoolExecutorWithFailure)

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(
        basic_df, DummyLoader(), processing_config=config,
        chunks_dir=tmp_path / "_chunks",
    )

    assert df.height == 2
    assert "path" in df.columns


def test_build_deep_record_df_preserves_schema_across_chunks(tmp_path, monkeypatch):
    """Columns introduced in one chunk appear in the final df; absent rows are null."""
    p1 = tmp_path / "alpha.png"
    p2 = tmp_path / "beta.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    config = ProcessingConfig(processing_max_workers=1, chunk_every_n=1)

    def per_file_props(file_path):
        if Path(file_path).name == p1.name:
            return {"width": 10, "extra": 99}
        return {"width": 20}

    monkeypatch.setattr(processing, "_iter_file_rows", _make_fake_iter_file_rows(per_file_props))
    monkeypatch.setattr(plugin_registry_module, "discover_processor_plugins", lambda: [])

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(
        basic_df, DummyLoader(), processing_config=config,
        chunks_dir=tmp_path / "_chunks",
    )

    assert "width" in df.columns
    assert "extra" in df.columns
    alpha = df.filter(pl.col("path") == str(p1))
    beta = df.filter(pl.col("path") == str(p2))
    assert alpha["extra"][0] == 99
    assert beta["extra"][0] is None


def test_build_deep_record_df_multi_record_produces_multiple_rows(tmp_path, monkeypatch):
    """A file yielding multiple rows (sub-images) produces one output row per sub-image."""
    p1 = tmp_path / "multi.czi"
    p1.write_bytes(b"")

    def fake_multi(file_path, loader, processors, show_progress):
        yield {"child_id": "scene_0", "width": 10}
        yield {"child_id": "scene_1", "width": 20}

    monkeypatch.setattr(processing, "_iter_file_rows", fake_multi)
    monkeypatch.setattr(plugin_registry_module, "discover_processor_plugins", lambda: [])

    basic_df = _basic_df_for_paths([p1])
    df = processing._build_deep_record_df(
        basic_df, DummyLoader(), chunks_dir=tmp_path / "_chunks",
    )

    assert df.height == 2
    assert "child_id" in df.columns
    assert set(df["child_id"].to_list()) == {"scene_0", "scene_1"}
    assert set(df["width"].to_list()) == {10, 20}
    assert set(df["path"].to_list()) == {str(p1)}


# ---------------------------------------------------------------------------
# _iter_file_rows
# ---------------------------------------------------------------------------

class _StubRecord:
    def __init__(self, meta: dict):
        self.meta = meta


def test_iter_file_rows_single_record(tmp_path):
    f = tmp_path / "single.png"
    f.write_bytes(b"\x89PNG")

    class SingleLoader:
        NAME = "single"
        def load(self, source):
            return _StubRecord({"color": "red"})

    rows = list(processing._iter_file_rows(f, SingleLoader(), processors=[], show_progress=False))
    assert len(rows) == 1
    assert rows[0]["color"] == "red"


def test_iter_file_rows_multi_record(tmp_path):
    f = tmp_path / "multi.czi"
    f.write_bytes(b"\x00")

    class MultiLoader:
        NAME = "multi"
        def load(self, source):
            return {
                "scene_A": _StubRecord({"width": 10}),
                "scene_B": _StubRecord({"width": 20}),
            }

    rows = list(processing._iter_file_rows(f, MultiLoader(), processors=[], show_progress=False))
    assert len(rows) == 2
    ids = {r["child_id"] for r in rows}
    assert ids == {"scene_A", "scene_B"}
    widths = {r["child_id"]: r["width"] for r in rows}
    assert widths == {"scene_A": 10, "scene_B": 20}


def test_iter_file_rows_nonexistent(tmp_path):
    class AnyLoader:
        NAME = "any"
        def load(self, source):
            raise AssertionError("should not be called")

    rows = list(processing._iter_file_rows(tmp_path / "nope.png", AnyLoader(), [], False))
    assert rows == []


def test_iter_file_rows_loader_failure(tmp_path):
    f = tmp_path / "bad.tif"
    f.write_bytes(b"bad")

    class FailLoader:
        NAME = "fail"
        def load(self, source):
            raise RuntimeError("boom")

    rows = list(processing._iter_file_rows(f, FailLoader(), [], False))
    assert rows == []


def test_iter_file_rows_loader_returns_none(tmp_path):
    f = tmp_path / "nothing.tif"
    f.write_bytes(b"x")

    class NoneLoader:
        NAME = "none"
        def load(self, source):
            return None

    rows = list(processing._iter_file_rows(f, NoneLoader(), [], False))
    assert rows == []


def test_iter_file_rows_skips_invalid_child_keys(tmp_path):
    f = tmp_path / "keys.czi"
    f.write_bytes(b"\x00")

    class BadKeysLoader:
        NAME = "badkeys"
        def load(self, source):
            return {
                "": _StubRecord({"x": 1}),       # empty key → skipped
                123: _StubRecord({"x": 2}),       # int key → coerced to "123"
                "valid": _StubRecord({"x": 3}),
            }

    rows = list(processing._iter_file_rows(f, BadKeysLoader(), [], False))
    assert len(rows) == 2
    child_ids = {r["child_id"] for r in rows}
    assert child_ids == {"valid", "123"}


# ---------------------------------------------------------------------------
# _combine_all_chunks — leaves chunks_dir intact
# ---------------------------------------------------------------------------

def test_combine_all_chunks_leaves_chunks_dir_intact(tmp_path):
    """_combine_all_chunks reads chunk files but does not remove them.
    Cleanup is the caller's responsibility via cleanup_chunks_dir."""
    chunks_dir = tmp_path / "_chunks"
    chunks_dir.mkdir()

    basic_df = _basic_df_for_paths([tmp_path / "a.png", tmp_path / "b.png"])

    p1 = chunks_dir / "chunk_000000_00000.parquet"
    p2 = chunks_dir / "chunk_000001_00000.parquet"
    pl.DataFrame({"a": [1], "row_index": [0]}).with_columns(
        pl.col("row_index").cast(pl.Int64)
    ).write_parquet(p1)
    pl.DataFrame({"a": [2], "row_index": [1]}).with_columns(
        pl.col("row_index").cast(pl.Int64)
    ).write_parquet(p2)

    result = processing._combine_all_chunks(chunks_dir, basic_df)

    assert result.height == 2
    assert "a" in result.columns
    assert chunks_dir.exists(), "chunks_dir must survive; only cleanup_chunks_dir should remove it"
    assert p1.exists() and p2.exists()


# ---------------------------------------------------------------------------
# _extract_record_properties
# ---------------------------------------------------------------------------

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

    rows = processing._extract_record_properties(
        RecordStub(),
        [RuntimeSkipProcessor(), GoodProcessor()],
        show_progress=False,
    )
    assert isinstance(rows, list) and len(rows) == 1
    assert rows[0]["a"] == 1
    assert rows[0]["ok"] == 1


# ---------------------------------------------------------------------------
# _cleanup_chunks_dir
# ---------------------------------------------------------------------------

def test_cleanup_chunks_dir(tmp_path):
    d = tmp_path / "batches"
    d.mkdir()

    p1 = d / "chunk_000000_00000.parquet"
    p2 = d / "chunk_000001_00000.parquet"
    combined = d / "records_df.parquet"

    pl.DataFrame({"row_index": [0]}).write_parquet(p1)
    pl.DataFrame({"row_index": [1]}).write_parquet(p2)
    pl.DataFrame({"a": [1]}).write_parquet(combined)

    assert p1.exists() and p2.exists() and combined.exists()

    processing._cleanup_chunks_dir(d)

    assert not p1.exists()
    assert not p2.exists()
    assert not combined.exists()
    assert not d.exists()

    # calling again is a no-op
    processing._cleanup_chunks_dir(d)


# ---------------------------------------------------------------------------
# Project-level cleanup lifecycle
# ---------------------------------------------------------------------------

def test_cleanup_not_called_when_no_records(tmp_path, monkeypatch):
    """cleanup_chunks_dir must not be called when build_records_df returns nothing."""
    cleanup_calls = []
    monkeypatch.setattr(processing, "build_records_df", lambda *a, **kw: None)
    monkeypatch.setattr(processing, "cleanup_chunks_dir", lambda path: cleanup_calls.append(path))

    p = Project(name="test", base_dir=tmp_path)
    p.process_records()

    assert cleanup_calls == [], "cleanup_chunks_dir must not be called when there are no records"


def test_cleanup_not_called_when_save_fails(tmp_path, monkeypatch):
    """cleanup_chunks_dir must not be called when save_parquet raises — chunks are
    the only remaining copy of the data."""
    cleanup_calls = []
    monkeypatch.setattr(
        processing, "build_records_df",
        lambda *a, **kw: pl.DataFrame({"a": [1]}),
    )

    def failing_save(df, path, meta):
        raise OSError("disk full")

    monkeypatch.setattr(project_module, "save_parquet", failing_save)
    monkeypatch.setattr(processing, "cleanup_chunks_dir", lambda path: cleanup_calls.append(path))

    p = Project(name="test", base_dir=tmp_path)
    p.process_records()

    assert cleanup_calls == [], "cleanup_chunks_dir must not be called after a failed save"


def test_cleanup_called_with_correct_path_after_successful_save(tmp_path, monkeypatch):
    """cleanup_chunks_dir is called exactly once with the correct chunks_dir on success."""
    cleanup_calls = []
    monkeypatch.setattr(
        processing, "build_records_df",
        lambda *a, **kw: pl.DataFrame({"a": [1]}),
    )
    monkeypatch.setattr(project_module, "save_parquet", lambda df, path, meta: None)
    monkeypatch.setattr(processing, "cleanup_chunks_dir", lambda path: cleanup_calls.append(path))

    p = Project(name="test", base_dir=tmp_path)
    expected_chunks_dir = p.output_path.parent / f"_chunks_{p.name}"

    p.process_records()

    assert len(cleanup_calls) == 1, "cleanup_chunks_dir must be called exactly once on success"
    assert cleanup_calls[0] == expected_chunks_dir