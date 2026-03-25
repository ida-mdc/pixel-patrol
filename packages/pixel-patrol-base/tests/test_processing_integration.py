from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import List, Optional

import polars as pl

from pixel_patrol_base.core import processing
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.project import Project


class DummyLoader:
    """Minimal loader stub for processing integration tests."""

    NAME = "dummy"
    SUPPORTED_EXTENSIONS = set()
    OUTPUT_SCHEMA = {}
    OUTPUT_SCHEMA_PATTERNS = []
    FOLDER_EXTENSIONS = set()

    def is_folder_supported(self, path: Path) -> bool:
        """Return False for all folders in this stub."""
        return False

    def load(self, source: str):
        """This stub does not load actual data."""
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
        """Execute the callable immediately and return a completed Future."""
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
        """Execute the callable immediately and return a completed Future."""
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
        """Fail once to simulate worker crash, then return real results."""
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
    """Build a minimal basic DataFrame with row indices and paths."""
    return (
        pl.DataFrame({"path": [str(p) for p in paths]})
        .with_row_index("row_index")
        .with_columns(pl.col("row_index").cast(pl.Int64))
    )


def _make_fake_load_and_process(return_props):
    """Build a fake load_and_process_records_from_file returning a single-element list.

    ``return_props`` can be:
    - a dict: every call returns ``[return_props]``
    - a callable ``(Path) -> dict``: result is wrapped in a list per file
    """

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
    """Ensure chunk flushing writes a combined parquet and cleans partial chunks."""
    flush_dir = tmp_path / "batches"
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    config = ProcessingConfig(
        processing_max_workers=1,
        records_flush_every_n=1,
        records_flush_dir=flush_dir,
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
    assert (flush_dir / "records_df.parquet").exists()
    assert list(flush_dir.glob("records_batch_*.parquet")) == []


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

    def fake_get_all_record_properties(path, loader, processors=None, show_processor_progress=True):
        return {"width": 7}

    class FailingProcessPoolExecutor:
        """Executor that fails immediately to trigger fallback."""

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


def test_process_records_infers_flush_dir_when_missing(tmp_path, monkeypatch):
    """Project should infer records_flush_dir when unset and base_dir is known."""
    project = Project(name="demo", base_dir=tmp_path, loader=None)

    monkeypatch.setattr(processing, "build_records_df", lambda *args, **kwargs: pl.DataFrame())

    project.process_records(processing_config=ProcessingConfig(selected_file_extensions="all"))

    assert project.records_flush_dir == Path(tmp_path) / "demo_batches"


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
        records_flush_dir=None,
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
    """Minimal Record-like object for unit tests."""

    def __init__(self, meta: dict):
        self.meta = meta


def test_load_and_process_records_from_file_single_record(tmp_path):
    """Single-record loader returns a one-element list."""
    f = tmp_path / "single.png"
    f.write_bytes(b"\x89PNG")

    class SingleLoader:
        NAME = "single"

        def load(self, source):
            return _StubRecord({"color": "red"})

    result = processing.load_and_process_records_from_file(
        f, SingleLoader(), processors=[]
    )
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["color"] == "red"


def test_load_and_process_records_from_file_multi_record(tmp_path):
    """Multi-record loader returns one dict per child with child_id set."""
    f = tmp_path / "multi.czi"
    f.write_bytes(b"\x00")

    class MultiLoader:
        NAME = "multi"

        def load(self, source):
            return {
                "scene_A": _StubRecord({"width": 10}),
                "scene_B": _StubRecord({"width": 20}),
            }

    result = processing.load_and_process_records_from_file(
        f, MultiLoader(), processors=[]
    )
    assert isinstance(result, list)
    assert len(result) == 2
    ids = {r["child_id"] for r in result}
    assert ids == {"scene_A", "scene_B"}
    widths = {r["child_id"]: r["width"] for r in result}
    assert widths == {"scene_A": 10, "scene_B": 20}


def test_load_and_process_records_from_file_nonexistent(tmp_path):
    """Missing files return an empty list."""

    class AnyLoader:
        NAME = "any"

        def load(self, source):
            raise AssertionError("should not be called")

    result = processing.load_and_process_records_from_file(
        tmp_path / "nope.png", AnyLoader(), processors=[]
    )
    assert result == []


def test_load_and_process_records_from_file_loader_failure(tmp_path):
    """Loader exceptions return an empty list, not a crash."""
    f = tmp_path / "bad.tif"
    f.write_bytes(b"bad")

    class FailLoader:
        NAME = "fail"

        def load(self, source):
            raise RuntimeError("boom")

    result = processing.load_and_process_records_from_file(
        f, FailLoader(), processors=[]
    )
    assert result == []


def test_load_and_process_records_from_file_loader_returns_none(tmp_path):
    """Loader returning None yields an empty list."""
    f = tmp_path / "nothing.tif"
    f.write_bytes(b"x")

    class NoneLoader:
        NAME = "none"

        def load(self, source):
            return None

    result = processing.load_and_process_records_from_file(
        f, NoneLoader(), processors=[]
    )
    assert result == []


def test_load_and_process_records_from_file_skips_invalid_child_keys(tmp_path):
    """Invalid child keys (empty) are skipped; ints are accepted and normalized to strings."""
    f = tmp_path / "keys.czi"
    f.write_bytes(b"\x00")

    class BadKeysLoader:
        NAME = "badkeys"

        def load(self, source):
            return {
                "": _StubRecord({"x": 1}),          # empty key — skip
                123: _StubRecord({"x": 2}),          # int key — accept
                "valid": _StubRecord({"x": 3}),
            }

    result = processing.load_and_process_records_from_file(
        f, BadKeysLoader(), processors=[]
    )

    assert len(result) == 2
    child_ids = {r["child_id"] for r in result}
    assert child_ids == {"valid", "123"}


def test_build_deep_record_df_multi_record_produces_multiple_rows(tmp_path, monkeypatch):
    """A loader returning multiple records per file expands to multiple rows."""
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
    # Both rows share the same source path
    assert df["path"].to_list() == [str(p1), str(p1)]