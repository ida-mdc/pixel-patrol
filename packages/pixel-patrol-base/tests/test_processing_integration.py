from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import List, Optional

import polars as pl

from pixel_patrol_base.core import processing
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.project import Project
import pixel_patrol_base.core.project as project_module


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


class FakeProcessPoolExecutor:
    """Synchronous executor to exercise the ProcessPoolExecutor code path."""

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
    """Synchronous executor to exercise the ThreadPoolExecutor code path."""

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
    """Executor that fails for the first submitted batch and succeeds after."""

    def __init__(self, max_workers: int, initializer=None, initargs=None) -> None:
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
        pl.DataFrame({"path": [str(p) for p in paths]})
        .with_row_index("row_index")
        .with_columns(pl.col("row_index").cast(pl.Int64))
    )


def _make_fake_load_and_process(return_props):
    def fake(file_path, loader, processors, show_processor_progress=True):
        if callable(return_props) and not isinstance(return_props, dict):
            props = return_props(file_path)
        else:
            props = return_props
        if not props:
            return []
        return [props]

    return fake


def test_build_deep_record_df_flushes_and_combines_chunks(tmp_path, monkeypatch):
    """Ensure chunk flushing writes batches, combines them, and cleans up _batches dir."""
    output_path = tmp_path / "output.parquet"
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    config = ProcessingConfig(
        processing_max_workers=1,
        records_flush_every_n=1,
    )

    monkeypatch.setattr(
        processing, "load_and_process_records_from_file",
        _make_fake_load_and_process({"width": 1}),
    )
    monkeypatch.setattr(processing, "discover_processor_plugins", lambda: [])

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(basic_df, DummyLoader(), processing_config=config)

    assert df.height == 2
    assert "width" in df.columns


def test_build_deep_record_df_process_pool_path_uses_initializer(tmp_path, monkeypatch):
    """Exercise the ProcessPoolExecutor branch with a synchronous fake executor."""
    p1 = tmp_path / "x.png"
    p2 = tmp_path / "y.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    config = ProcessingConfig(processing_max_workers=2)

    monkeypatch.setattr(
        processing, "load_and_process_records_from_file",
        _make_fake_load_and_process({"width": 5}),
    )
    monkeypatch.setattr(processing, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "discover_loader", lambda loader_id: DummyLoader())
    monkeypatch.setattr(processing, "ProcessPoolExecutor", FakeProcessPoolExecutor)

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(basic_df, DummyLoader(), processing_config=config)

    assert FakeProcessPoolExecutor.last_instance is not None
    assert FakeProcessPoolExecutor.last_instance.max_workers == 2
    assert FakeProcessPoolExecutor.last_instance.initializer_called is True
    assert df["width"].to_list() == [5, 5]


def test_build_deep_record_df_thread_fallback_on_process_pool_error(tmp_path, monkeypatch):
    """Fallback to threads when process pool creation fails."""
    p1 = tmp_path / "m.png"
    p2 = tmp_path / "n.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    config = ProcessingConfig(processing_max_workers=2)

    class FailingProcessPoolExecutor:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("process pool unavailable")

    monkeypatch.setattr(
        processing, "load_and_process_records_from_file",
        _make_fake_load_and_process({"width": 7}),
    )
    monkeypatch.setattr(processing, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "ProcessPoolExecutor", FailingProcessPoolExecutor)
    monkeypatch.setattr(processing, "ThreadPoolExecutor", FakeThreadPoolExecutor)

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(basic_df, DummyLoader(), processing_config=config)

    assert FakeThreadPoolExecutor.last_instance is not None
    assert FakeThreadPoolExecutor.last_instance.max_workers == 2
    assert df["width"].to_list() == [7, 7]


def test_process_records_infers_output_dir_when_missing(tmp_path, monkeypatch):
    """Project should infer output_dir when unset and base_dir is known."""
    project = Project(name="demo", base_dir=tmp_path, loader=None)

    monkeypatch.setattr(processing, "build_records_df", lambda *args, **kwargs: pl.DataFrame())

    project.process_records(processing_config=ProcessingConfig(selected_file_extensions="all"))

    assert project.output_path == Path(tmp_path) / "demo.parquet"


def test_build_deep_record_df_survives_worker_failure(tmp_path, monkeypatch):
    """A failed worker batch should not abort the whole run."""
    p1 = tmp_path / "one.png"
    p2 = tmp_path / "two.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    config = ProcessingConfig(processing_max_workers=2)

    monkeypatch.setattr(
        processing, "load_and_process_records_from_file",
        _make_fake_load_and_process({"width": 33}),
    )
    monkeypatch.setattr(processing, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "discover_loader", lambda loader_id: DummyLoader())
    monkeypatch.setattr(processing, "ProcessPoolExecutor", FakeProcessPoolExecutorWithFailure)

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(basic_df, DummyLoader(), processing_config=config)

    assert df.height == 2
    assert "path" in df.columns


def test_build_deep_record_df_preserves_schema_across_batches(tmp_path, monkeypatch):
    """Schema should include columns introduced in earlier batches with nulls later."""
    p1 = tmp_path / "alpha.png"
    p2 = tmp_path / "beta.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    config = ProcessingConfig(
        processing_max_workers=1,
        records_flush_every_n=1,
    )

    def per_file_props(file_path):
        if Path(file_path).name == p1.name:
            return {"width": 10, "extra": 99}
        return {"width": 20}

    monkeypatch.setattr(
        processing, "load_and_process_records_from_file",
        _make_fake_load_and_process(per_file_props),
    )
    monkeypatch.setattr(processing, "discover_processor_plugins", lambda: [])

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(basic_df, DummyLoader(), processing_config=config)

    df = df.sort("row_index")
    assert set(df.columns) == {"row_index", "path", "width", "extra"}
    assert df["extra"].to_list() == [99, None]


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


def test_build_deep_record_df_multi_record_produces_multiple_rows(tmp_path, monkeypatch):
    p1 = tmp_path / "multi.czi"
    p1.write_bytes(b"")

    def fake_multi(file_path, loader, processors, show_processor_progress=True):
        return [
            {"child_id": "scene_0", "width": 10},
            {"child_id": "scene_1", "width": 20},
        ]

    monkeypatch.setattr(processing, "load_and_process_records_from_file", fake_multi)
    monkeypatch.setattr(processing, "discover_processor_plugins", lambda: [])

    basic_df = _basic_df_for_paths([p1])
    df = processing._build_deep_record_df(basic_df, DummyLoader())

    assert df.height == 2
    assert "child_id" in df.columns
    assert set(df["child_id"].to_list()) == {"scene_0", "scene_1"}
    assert set(df["width"].to_list()) == {10, 20}
    assert df["path"].to_list() == [str(p1), str(p1)]



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
    monkeypatch.setattr(processing, "build_records_df", lambda *a, **kw: None)
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
        lambda *a, **kw: pl.DataFrame({"a": [1]}),
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
        lambda *a, **kw: pl.DataFrame({"a": [1]}),
    )
    monkeypatch.setattr(project_module, "save_parquet", lambda df, path, meta: None)
    monkeypatch.setattr(processing, "cleanup_flush_dir", lambda path: cleanup_calls.append(path))

    p = Project(name="test", base_dir=tmp_path)
    expected_flush_dir = p.output_path.parent / "_batches"

    p.process_records()

    assert len(cleanup_calls) == 1, "cleanup_flush_dir must be called exactly once on success"
    assert cleanup_calls[0] == expected_flush_dir, (
        f"cleanup_flush_dir called with wrong path: {cleanup_calls[0]!r}, expected {expected_flush_dir!r}"
    )