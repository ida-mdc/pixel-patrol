from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from typing import List, Optional

import polars as pl

from pixel_patrol_base.core import processing
from pixel_patrol_base.core.project import Project
from pixel_patrol_base.core.project_settings import Settings


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


def test_build_deep_record_df_flushes_and_combines_chunks(tmp_path, monkeypatch):
    """Ensure chunk flushing writes a combined parquet and cleans partial chunks."""
    flush_dir = tmp_path / "batches"
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    settings = Settings(
        processing_max_workers=1,
        records_flush_every_n=1,
        records_flush_dir=flush_dir,
    )

    def fake_load_and_process_records_from_file(path, loader, processors=None, show_processor_progress=True):
        return [{"width": 1}]

    monkeypatch.setattr(processing, "load_and_process_records_from_file", fake_load_and_process_records_from_file)
    monkeypatch.setattr(processing, "discover_processor_plugins", lambda: [])

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(basic_df, DummyLoader(), settings=settings)

    assert df.height == 2
    assert "width" in df.columns
    assert (flush_dir / "records_df.parquet").exists()
    assert list(flush_dir.glob("records_batch_*.parquet")) == []


def test_build_deep_record_df_resumes_from_partial_chunks(tmp_path, monkeypatch):
    """Verify resume skips previously processed rows and preserves their data."""
    flush_dir = tmp_path / "batches"
    flush_dir.mkdir()
    p1 = tmp_path / "first.png"
    p2 = tmp_path / "second.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    partial_df = pl.DataFrame({
        "row_index": [0],
        "path": [str(p1)],
        "width": [11],
    })
    partial_df.write_parquet(flush_dir / "records_batch_00000.parquet")

    settings = Settings(
        processing_max_workers=1,
        records_flush_every_n=1,
        records_flush_dir=flush_dir,
        resume=True,
    )

    processed: List[str] = []

    def fake_load_and_process_records_from_file(path, loader, processors=None, show_processor_progress=True):
        processed.append(Path(path).name)
        return [{"width": 22}]

    monkeypatch.setattr(processing, "load_and_process_records_from_file", fake_load_and_process_records_from_file)
    monkeypatch.setattr(processing, "discover_processor_plugins", lambda: [])

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(basic_df, DummyLoader(), settings=settings)

    assert processed == [p2.name]
    df = df.sort("row_index")
    assert df["path"].to_list() == [str(p1), str(p2)]
    assert df["width"].to_list() == [11, 22]


def test_build_deep_record_df_process_pool_path_uses_initializer(tmp_path, monkeypatch):
    """Exercise the ProcessPoolExecutor branch with a synchronous fake executor."""
    p1 = tmp_path / "x.png"
    p2 = tmp_path / "y.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    settings = Settings(processing_max_workers=2)

    def fake_load_and_process_records_from_file(path, loader, processors=None, show_processor_progress=True):
        return [{"width": 5}]

    monkeypatch.setattr(processing, "load_and_process_records_from_file", fake_load_and_process_records_from_file)
    monkeypatch.setattr(processing, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "discover_loader", lambda loader_id: DummyLoader())
    monkeypatch.setattr(processing, "ProcessPoolExecutor", FakeProcessPoolExecutor)

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(basic_df, DummyLoader(), settings=settings)

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

    settings = Settings(processing_max_workers=2)

    def fake_load_and_process_records_from_file(path, loader, processors=None, show_processor_progress=True):
        return [{"width": 7}]

    class FailingProcessPoolExecutor:
        """Executor that fails immediately to trigger fallback."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("process pool unavailable")

    monkeypatch.setattr(processing, "load_and_process_records_from_file", fake_load_and_process_records_from_file)
    monkeypatch.setattr(processing, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "ProcessPoolExecutor", FailingProcessPoolExecutor)
    monkeypatch.setattr(processing, "ThreadPoolExecutor", FakeThreadPoolExecutor)

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(basic_df, DummyLoader(), settings=settings)

    assert FakeThreadPoolExecutor.last_instance is not None
    assert FakeThreadPoolExecutor.last_instance.max_workers == 2
    assert df["width"].to_list() == [7, 7]


def test_process_records_infers_flush_dir_when_missing(tmp_path, monkeypatch):
    """Project should infer records_flush_dir when unset and base_dir is known."""
    project = Project(name="demo", base_dir=tmp_path, loader=None)
    settings = Settings(selected_file_extensions={"png"})
    project.set_settings(settings)

    monkeypatch.setattr(processing, "build_records_df", lambda *args, **kwargs: pl.DataFrame())

    project.process_records()

    assert project.settings.records_flush_dir == Path(tmp_path) / "demo_batches"


def test_build_deep_record_df_ignores_corrupt_resume_chunk(tmp_path, monkeypatch):
    """Corrupt chunk files should be skipped and processing should continue."""
    flush_dir = tmp_path / "batches"
    flush_dir.mkdir()

    p1 = tmp_path / "first.png"
    p2 = tmp_path / "second.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    valid_df = pl.DataFrame({
        "row_index": [0],
        "path": [str(p1)],
        "width": [11],
    })
    valid_df.write_parquet(flush_dir / "records_batch_00000.parquet")
    (flush_dir / "records_batch_00001.parquet").write_bytes(b"corrupt")

    settings = Settings(
        processing_max_workers=1,
        records_flush_every_n=1,
        records_flush_dir=flush_dir,
        resume=True,
    )

    def fake_load_and_process_records_from_file(path, loader, processors=None, show_processor_progress=True):
        return [{"width": 22}]

    monkeypatch.setattr(processing, "load_and_process_records_from_file", fake_load_and_process_records_from_file)
    monkeypatch.setattr(processing, "discover_processor_plugins", lambda: [])

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(basic_df, DummyLoader(), settings=settings)

    df = df.sort("row_index")
    assert df["path"].to_list() == [str(p1), str(p2)]
    assert df["width"].to_list() == [11, 22]


def test_build_deep_record_df_survives_worker_failure(tmp_path, monkeypatch):
    """A failed worker batch should not abort the whole run."""
    p1 = tmp_path / "one.png"
    p2 = tmp_path / "two.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    settings = Settings(processing_max_workers=2)

    def fake_load_and_process_records_from_file(path, loader, processors=None, show_processor_progress=True):
        return {"width": 33}

    monkeypatch.setattr(processing, "load_and_process_records_from_file", fake_load_and_process_records_from_file)
    monkeypatch.setattr(processing, "discover_processor_plugins", lambda: [])
    monkeypatch.setattr(processing, "discover_loader", lambda loader_id: DummyLoader())
    monkeypatch.setattr(processing, "ProcessPoolExecutor", FakeProcessPoolExecutorWithFailure)

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(basic_df, DummyLoader(), settings=settings)

    assert df.height == 2
    assert "path" in df.columns


def test_build_deep_record_df_preserves_schema_across_batches(tmp_path, monkeypatch):
    """Schema should include columns introduced in earlier batches with nulls later."""
    p1 = tmp_path / "alpha.png"
    p2 = tmp_path / "beta.png"
    p1.write_bytes(b"")
    p2.write_bytes(b"")

    settings = Settings(
        processing_max_workers=1,
        records_flush_every_n=1,
        records_flush_dir=None,
    )

    def fake_load_and_process_records_from_file(path, loader, processors=None, show_processor_progress=True):
        if Path(path).name == p1.name:
            return [{"width": 10, "extra": 99}]
        return [{"width": 20}]

    monkeypatch.setattr(processing, "load_and_process_records_from_file", fake_load_and_process_records_from_file)
    monkeypatch.setattr(processing, "discover_processor_plugins", lambda: [])

    basic_df = _basic_df_for_paths([p1, p2])
    df = processing._build_deep_record_df(basic_df, DummyLoader(), settings=settings)

    df = df.sort("row_index")
    assert set(df.columns) == {"row_index", "path", "width", "extra"}
    assert df["extra"].to_list() == [99, None]
