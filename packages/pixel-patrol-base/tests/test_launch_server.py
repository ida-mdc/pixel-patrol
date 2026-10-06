"""
Tests for the JS/HTML processing launch server (launch_server.py).

Covers the parts that are meaningfully testable without a browser:
- input parsing helpers (the same kind of user-typed strings the JS form submits)
- the processing state machine (_start_processing / _run_processing), including
  validation, completion, error, and cancellation
- the HTTP API surface (_LaunchHandler routing, status/cancel/reports/query)
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from pixel_patrol_base import launch_server as ls
from pixel_patrol_base import report_library as library
from pixel_patrol_base.core.project_metadata import ProjectMetadata
from pixel_patrol_base.io.parquet_io import save_parquet


@pytest.fixture(autouse=True)
def _reset_global_state():
    """launch_server keeps module-level mutable state; reset it around each test."""
    saved_state = dict(ls._state)
    saved_warnings = list(ls._warning_queue)
    cancel_was_set = ls._cancel_event.is_set()

    yield

    ls._state.clear()
    ls._state.update(saved_state)
    ls._warning_queue.clear()
    ls._warning_queue.extend(saved_warnings)
    library._registry.clear()
    library._meta_cache.clear()
    ls._report_conns.clear()
    if cancel_was_set:
        ls._cancel_event.set()
    else:
        ls._cancel_event.clear()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def test_parse_csv():
    assert ls._parse_csv(None) == []
    assert ls._parse_csv("") == []
    assert ls._parse_csv("a, b ,c,") == ["a", "b", "c"]


def test_parse_slice_size():
    assert ls._parse_slice_size(None) is None
    assert ls._parse_slice_size("") is None
    assert ls._parse_slice_size("Z=1, C=2") == {"Z": 1, "C": 2}

    with pytest.raises(ValueError, match="DIM=SIZE"):
        ls._parse_slice_size("Z")

    with pytest.raises(ValueError, match="integer"):
        ls._parse_slice_size("Z=abc")


def test_parse_dims():
    assert ls._parse_dims(None) is None
    assert ls._parse_dims("") is None
    assert ls._parse_dims("z=0, t=1") == {"z": "0", "t": "1"}

    with pytest.raises(ValueError, match="key=value"):
        ls._parse_dims("z")


def test_parse_filter():
    assert ls._parse_filter(None, None, None) is None
    assert ls._parse_filter("col", "", "value") is None
    assert ls._parse_filter(" col ", "eq", " value ") == {"col": {"op": "eq", "value": "value"}}


# ---------------------------------------------------------------------------
# _start_processing validation
# ---------------------------------------------------------------------------

def test_start_processing_requires_base_dir():
    ls.update_state(status="idle", error=None)

    ls._start_processing({})

    state = ls.get_state()
    assert state["status"] == "error"
    assert "required" in state["error"]


def test_start_processing_defaults_output_into_reports_dir(tmp_path, monkeypatch):
    """Omitting output_path auto-names a parquet inside REPORTS_DIR."""
    monkeypatch.setattr(library, "REPORTS_DIR", tmp_path / "reports")
    # Don't actually launch the background processing thread.
    monkeypatch.setattr(ls.threading, "Thread", lambda *a, **k: SimpleNamespace(start=lambda: None))

    payload = {"base_directory": str(tmp_path)}
    ls.update_state(status="idle", error=None)
    ls._start_processing(payload)

    assert payload["output_path"].endswith(".parquet")
    assert str(tmp_path / "reports") in payload["output_path"]


def test_start_processing_rejects_bad_slice_size(tmp_path):
    ls.update_state(status="idle", error=None)

    ls._start_processing({
        "base_directory": str(tmp_path),
        "output_path": str(tmp_path / "out.parquet"),
        "slice_size": "Z=not-a-number",
    })

    state = ls.get_state()
    assert state["status"] == "error"
    assert "integer" in state["error"]


def test_start_processing_noop_while_running(tmp_path):
    ls.update_state(status="running", message="already going")

    ls._start_processing({
        "base_directory": str(tmp_path),
        "output_path": str(tmp_path / "out.parquet"),
    })

    state = ls.get_state()
    assert state["status"] == "running"
    assert state["message"] == "already going"


# ---------------------------------------------------------------------------
# _run_processing: completion / error / cancellation
# ---------------------------------------------------------------------------

def _fake_project(output_path):
    return SimpleNamespace(output_path=output_path)


def test_run_processing_completed(tmp_path, monkeypatch):
    output = tmp_path / "out.parquet"
    save_parquet(pl.DataFrame({"a": [1, 2]}), output, ProjectMetadata(project_name="p"))

    project = _fake_project(output)
    monkeypatch.setattr(ls.api, "create_project", lambda *a, **k: project)
    monkeypatch.setattr(ls.api, "add_paths", lambda *a, **k: None)

    captured_kwargs = {}

    def fake_process_files(proj, **kwargs):
        captured_kwargs.update(kwargs)
        kwargs["progress_callback"](1, 1)

    monkeypatch.setattr(ls.api, "process_files", fake_process_files)

    ls._run_processing({"base_directory": str(tmp_path), "output_path": str(output)}, None)

    state = ls.get_state()
    assert state["status"] == "completed"
    assert state["output_parquet"] == str(output)
    assert state["progress"] == 100
    # progress_callback was wired through to api.process_files
    assert "progress_callback" in captured_kwargs


def test_run_processing_missing_output_file_is_error(tmp_path, monkeypatch):
    missing_output = tmp_path / "missing.parquet"

    project = _fake_project(missing_output)
    monkeypatch.setattr(ls.api, "create_project", lambda *a, **k: project)
    monkeypatch.setattr(ls.api, "add_paths", lambda *a, **k: None)
    monkeypatch.setattr(ls.api, "process_files", lambda proj, **kwargs: None)

    ls._run_processing({"base_directory": str(tmp_path), "output_path": str(missing_output)}, None)

    state = ls.get_state()
    assert state["status"] == "error"
    assert "not found" in state["error"]


def test_run_processing_exception_is_error(tmp_path, monkeypatch):
    project = _fake_project(tmp_path / "out.parquet")
    monkeypatch.setattr(ls.api, "create_project", lambda *a, **k: project)
    monkeypatch.setattr(ls.api, "add_paths", lambda *a, **k: None)

    def boom(proj, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ls.api, "process_files", boom)

    ls._run_processing({"base_directory": str(tmp_path), "output_path": str(tmp_path / "out.parquet")}, None)

    state = ls.get_state()
    assert state["status"] == "error"
    assert "boom" in state["error"]


def test_run_processing_cancelled(tmp_path, monkeypatch):
    project = _fake_project(tmp_path / "out.parquet")
    monkeypatch.setattr(ls.api, "create_project", lambda *a, **k: project)
    monkeypatch.setattr(ls.api, "add_paths", lambda *a, **k: None)

    def fake_process_files(proj, **kwargs):
        ls._cancel_event.set()
        # First progress_callback after cancel must abort the run.
        kwargs["progress_callback"](1, 10)

    monkeypatch.setattr(ls.api, "process_files", fake_process_files)

    ls._run_processing({"base_directory": str(tmp_path), "output_path": str(tmp_path / "out.parquet")}, None)

    state = ls.get_state()
    assert state["status"] == "cancelled"


def test_start_processing_clears_stale_cancel_flag(tmp_path, monkeypatch):
    """A cancel from a previous run must not immediately abort the next one."""
    ls.update_state(status="idle", error=None)
    ls._cancel_event.set()

    project = _fake_project(tmp_path / "out.parquet")
    save_parquet(pl.DataFrame({"a": [1]}), project.output_path, ProjectMetadata(project_name="p"))
    monkeypatch.setattr(ls.api, "create_project", lambda *a, **k: project)
    monkeypatch.setattr(ls.api, "add_paths", lambda *a, **k: None)
    monkeypatch.setattr(ls.api, "process_files", lambda proj, **kwargs: kwargs["progress_callback"](1, 1))

    ls._start_processing({"base_directory": str(tmp_path), "output_path": str(project.output_path)})

    # Wait for the background thread to finish.
    for _ in range(100):
        if ls.get_state()["status"] != "running":
            break
        time.sleep(0.01)

    assert ls.get_state()["status"] == "completed"


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ls._LaunchHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


def _url(httpd, path):
    return f"http://127.0.0.1:{httpd.server_address[1]}{path}"


def _post_json(httpd, path, payload):
    req = urllib.request.Request(
        _url(httpd, path),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req)


def test_status_endpoint(server):
    ls.update_state(status="idle", error=None)

    with urllib.request.urlopen(_url(server, "/api/status")) as resp:
        data = json.loads(resp.read())

    assert data["status"] == "idle"
    assert data["warnings"] == []


def test_status_includes_console_output(server):
    import logging
    ls.clear_warnings()
    ls._install_warning_capture()
    logging.getLogger("pixel_patrol_base.test").info("hello console")

    with urllib.request.urlopen(_url(server, "/api/status")) as resp:
        data = json.loads(resp.read())

    assert any("hello console" in line for line in data["console"])


def test_browse_files_off_lists_only_folders(server, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.parquet").write_bytes(b"")

    with urllib.request.urlopen(_url(server, f"/api/browse?path={tmp_path}&files=0")) as resp:
        names = [e["name"] for e in json.loads(resp.read())["entries"]]

    assert names == ["sub"]


def test_loaders_and_processors_endpoints(server):
    with urllib.request.urlopen(_url(server, "/api/loaders")) as resp:
        loaders = json.loads(resp.read())
    assert isinstance(loaders, list) and loaders
    assert loaders[0]["value"] == ""  # "None (basic file info only)" is always first

    with urllib.request.urlopen(_url(server, "/api/processors")) as resp:
        processors = json.loads(resp.read())
    assert isinstance(processors, list)


def test_process_endpoint_validation_error(server):
    ls.update_state(status="idle", error=None)

    with _post_json(server, "/api/process", {}) as resp:
        data = json.loads(resp.read())

    assert data["status"] == "error"
    assert "required" in data["error"]


def test_cancel_endpoint_sets_event_only_when_running(server):
    ls.update_state(status="idle", error=None)
    with _post_json(server, "/api/cancel", {}) as resp:
        json.loads(resp.read())
    assert not ls._cancel_event.is_set()

    ls.update_state(status="running", error=None)
    with _post_json(server, "/api/cancel", {}) as resp:
        data = json.loads(resp.read())
    assert ls._cancel_event.is_set()
    assert data["status"] == "running"


def test_report_url_missing_output(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post_json(server, "/api/report-url", {"output_parquet": ""})
    assert excinfo.value.code == 400


def test_report_url_file_not_found(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post_json(server, "/api/report-url", {"output_parquet": "/no/such/report.parquet"})
    assert excinfo.value.code == 404


# ---------------------------------------------------------------------------
# Report file browser
# ---------------------------------------------------------------------------

def test_browse_lists_dirs_and_parquet_files(server, tmp_path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "report.parquet").touch()
    (tmp_path / "notes.txt").touch()

    with urllib.request.urlopen(_url(server, f"/api/browse?path={tmp_path}")) as resp:
        data = json.loads(resp.read())

    assert data["path"] == str(tmp_path)
    assert data["parent"] == str(tmp_path.parent)
    names = {(e["name"], e["is_dir"]) for e in data["entries"]}
    assert ("subdir", True) in names
    assert ("report.parquet", False) in names
    assert all(name != "notes.txt" for name, _ in names)


def test_browse_unknown_path_returns_404(server, tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(_url(server, f"/api/browse?path={missing}"))
    assert excinfo.value.code == 404


def test_browse_file_path_returns_404(server, tmp_path):
    file_path = tmp_path / "report.parquet"
    file_path.touch()
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(_url(server, f"/api/browse?path={file_path}"))
    assert excinfo.value.code == 404


def test_browse_no_path_defaults_to_home(server, monkeypatch, tmp_path):
    monkeypatch.setattr(ls.Path, "home", lambda: tmp_path)

    with urllib.request.urlopen(_url(server, "/api/browse")) as resp:
        data = json.loads(resp.read())

    assert data["path"] == str(tmp_path)


# ---------------------------------------------------------------------------
# Version check / self-update
# ---------------------------------------------------------------------------

def test_version_endpoint_unmanaged(server, monkeypatch):
    monkeypatch.setattr(ls, "_is_managed_install", lambda: False)
    monkeypatch.setattr(ls, "_latest_pixel_patrol_version", lambda: "999.0.0")

    with urllib.request.urlopen(_url(server, "/api/version")) as resp:
        data = json.loads(resp.read())

    assert data["managed"] is False
    assert data["latest"] == "999.0.0"
    assert data["update_available"] is True
    assert data["pypi_url"] == "https://pypi.org/project/pixel-patrol/"


def test_version_endpoint_up_to_date(server, monkeypatch):
    current = ls._installed_version()
    monkeypatch.setattr(ls, "_latest_pixel_patrol_version", lambda: current)

    with urllib.request.urlopen(_url(server, "/api/version")) as resp:
        data = json.loads(resp.read())

    assert data["current"] == current
    assert data["update_available"] is False


def test_version_endpoint_pypi_unreachable(server, monkeypatch):
    monkeypatch.setattr(ls, "_latest_pixel_patrol_version", lambda: None)

    with urllib.request.urlopen(_url(server, "/api/version")) as resp:
        data = json.loads(resp.read())

    assert data["latest"] is None
    assert data["update_available"] is False


def test_update_endpoint_rejects_unmanaged_install(server, monkeypatch):
    monkeypatch.setattr(ls, "_is_managed_install", lambda: False)

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post_json(server, "/api/update", {})
    assert excinfo.value.code == 400


def test_update_endpoint_runs_uv_upgrade(server, monkeypatch, tmp_path):
    monkeypatch.setattr(ls, "_is_managed_install", lambda: True)
    monkeypatch.setattr(ls, "_find_uv", lambda: Path("/fake/uv"))
    monkeypatch.setattr(ls, "_LAUNCHER_HOME", tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"loader_pkgs": ["pixel-patrol-demo"]}))

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ls.subprocess, "run", fake_run)

    with _post_json(server, "/api/update", {}) as resp:
        data = json.loads(resp.read())

    assert data == {"status": "ok"}
    assert captured["cmd"][0] == str(Path("/fake/uv"))
    assert "--upgrade" in captured["cmd"]
    assert "pixel-patrol" in captured["cmd"]
    assert "pixel-patrol-demo" in captured["cmd"]


def test_update_endpoint_no_uv_found(server, monkeypatch):
    monkeypatch.setattr(ls, "_is_managed_install", lambda: True)
    monkeypatch.setattr(ls, "_find_uv", lambda: None)

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post_json(server, "/api/update", {})
    assert excinfo.value.code == 500


def test_update_endpoint_uv_failure(server, monkeypatch, tmp_path):
    monkeypatch.setattr(ls, "_is_managed_install", lambda: True)
    monkeypatch.setattr(ls, "_find_uv", lambda: Path("/fake/uv"))
    monkeypatch.setattr(ls, "_LAUNCHER_HOME", tmp_path)

    def fake_run(cmd, **kwargs):
        raise ls.subprocess.CalledProcessError(1, cmd, stderr="boom")

    monkeypatch.setattr(ls.subprocess, "run", fake_run)

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post_json(server, "/api/update", {})
    assert excinfo.value.code == 500


def test_report_url_builds_query_string(server, tmp_path):
    output = tmp_path / "report.parquet"
    save_parquet(pl.DataFrame({"a": [1]}), output, ProjectMetadata(project_name="p"))

    payload = {
        "output_parquet": str(output),
        "group_by": "path",
        "filter_col": "file_extension",
        "filter_op": "eq",
        "filter_value": "tif",
        "dimensions": "z=0, t=1",
        "widgets_exclude": "histogram, summary",
        "is_show_significance": True,
        "palette": "viridis",
    }
    with _post_json(server, "/api/report-url", payload) as resp:
        data = json.loads(resp.read())

    url = data["url"]
    assert url.startswith("/report?path=")
    for fragment in ("group=path", "fc=file_extension", "fo=eq", "fv=tif",
                     "dims=z0.t1", "sig=1", "palette=viridis", "hidden=histogram.summary"):
        assert fragment in url


def test_report_url_no_extras_returns_bare_url(server, tmp_path):
    output = tmp_path / "report.parquet"
    save_parquet(pl.DataFrame({"a": [1]}), output, ProjectMetadata(project_name="p"))

    with _post_json(server, "/api/report-url", {"output_parquet": str(output)}) as resp:
        data = json.loads(resp.read())

    assert data["url"] == f"/report?path={urllib.parse.quote(str(output.resolve()))}"


def test_report_url_bad_dimensions(server, tmp_path):
    output = tmp_path / "report.parquet"
    save_parquet(pl.DataFrame({"a": [1]}), output, ProjectMetadata(project_name="p"))

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post_json(server, "/api/report-url", {"output_parquet": str(output), "dimensions": "bad"})
    assert excinfo.value.code == 400


# ---------------------------------------------------------------------------
# Report library: list / metadata / delete / query routing
# ---------------------------------------------------------------------------

def test_reports_endpoint_lists_parquet_with_metadata(server, tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(library, "REPORTS_DIR", reports_dir)
    output = reports_dir / "20260101_120000_demo.parquet"
    save_parquet(pl.DataFrame({"a": [1, 2, 3]}), output, ProjectMetadata(project_name="demo"))

    with urllib.request.urlopen(_url(server, "/api/reports")) as resp:
        data = json.loads(resp.read())

    assert data["reports_dir"] == str(reports_dir)
    assert len(data["reports"]) == 1
    report = data["reports"][0]
    assert report["filename"] == output.name
    assert report["project_name"] == "demo"
    assert report["n_files"] == 3
    assert report["id"]


def test_delete_report_removes_file(server, tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(library, "REPORTS_DIR", reports_dir)
    output = reports_dir / "demo.parquet"
    save_parquet(pl.DataFrame({"a": [1]}), output, ProjectMetadata(project_name="demo"))

    with _post_json(server, "/api/delete-report", {"path": str(output)}) as resp:
        data = json.loads(resp.read())

    assert data == {"status": "ok"}
    assert not output.exists()


def test_import_report_adds_external_to_index_and_lists_it(server, tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(library, "REPORTS_DIR", reports_dir)
    outside = tmp_path / "elsewhere" / "ext.parquet"
    outside.parent.mkdir()
    save_parquet(pl.DataFrame({"a": [1, 2]}), outside, ProjectMetadata(project_name="ext"))

    with _post_json(server, "/api/import-report", {"path": str(outside)}) as resp:
        assert json.loads(resp.read())["status"] == "ok"

    # Persisted to the index file and surfaced as an "imported" report.
    assert str(outside.resolve()) in library._load_index()
    with urllib.request.urlopen(_url(server, "/api/reports")) as resp:
        reports = json.loads(resp.read())["reports"]
    ext = [r for r in reports if r["path"] == str(outside.resolve())]
    assert len(ext) == 1
    assert ext[0]["source"] == "imported"
    assert ext[0]["exists"] is True


def test_import_report_missing_file_is_404(server, tmp_path, monkeypatch):
    monkeypatch.setattr(library, "REPORTS_DIR", tmp_path / "reports")
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post_json(server, "/api/import-report", {"path": str(tmp_path / "nope.parquet")})
    assert excinfo.value.code == 404


def test_delete_imported_report_keeps_file(server, tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(library, "REPORTS_DIR", reports_dir)
    outside = tmp_path / "elsewhere.parquet"
    save_parquet(pl.DataFrame({"a": [1]}), outside, ProjectMetadata(project_name="demo"))
    library._add_to_index(outside)

    with _post_json(server, "/api/delete-report", {"path": str(outside)}) as resp:
        assert json.loads(resp.read()) == {"status": "ok"}

    # Removed from the list but the external file is left untouched.
    assert str(outside.resolve()) not in library._load_index()
    assert outside.exists()


def test_missing_imported_report_listed_as_unavailable(server, tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    monkeypatch.setattr(library, "REPORTS_DIR", reports_dir)
    gone = tmp_path / "gone.parquet"
    save_parquet(pl.DataFrame({"a": [1]}), gone, ProjectMetadata(project_name="demo"))
    library._add_to_index(gone)
    gone.unlink()

    with urllib.request.urlopen(_url(server, "/api/reports")) as resp:
        reports = json.loads(resp.read())["reports"]
    entry = [r for r in reports if r["path"] == str(gone.resolve())]
    assert len(entry) == 1
    assert entry[0]["exists"] is False


def test_query_routes_to_report(server, tmp_path):
    import pyarrow.ipc as ipc

    output = tmp_path / "report.parquet"
    save_parquet(pl.DataFrame({"a": [1, 2, 3, 4]}), output, ProjectMetadata(project_name="p"))
    rid = library.register_report(output.resolve())
    # Open the DuckDB connection in the main thread first. Under pytest's
    # assertion-rewrite import hook, importing duckdb lazily inside the request
    # handler thread can raise "__import__ returned a result with an exception
    # set"; pre-warming avoids that test-only race (the server works in real use).
    ls._get_report_conn(output.resolve())

    req = urllib.request.Request(
        _url(server, f"/api/query?report={rid}"),
        data=json.dumps({"sql": "SELECT count(*) AS n FROM pp_data"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        table = ipc.open_stream(resp.read()).read_all()

    assert table.to_pydict() == {"n": [4]}


def test_query_unknown_report_is_404(server):
    req = urllib.request.Request(
        _url(server, "/api/query?report=deadbeef"),
        data=json.dumps({"sql": "SELECT 1"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req)
    assert excinfo.value.code == 404


def test_report_index_injects_config(server, tmp_path):
    try:
        ls.find_viewer_dist()
    except Exception:
        pytest.skip("viewer dist not built")

    output = tmp_path / "report.parquet"
    save_parquet(pl.DataFrame({"a": [1]}), output, ProjectMetadata(project_name="p"))
    rid = library.register_report(output.resolve())

    url = _url(server, f"/report?path={urllib.parse.quote(str(output.resolve()))}")
    with urllib.request.urlopen(url) as resp:
        body = resp.read().decode()

    assert "__PP_SERVER" in body
    assert rid in body


def test_static_paths_serve_index(server):
    for path in ("/", "/does/not/exist"):
        with urllib.request.urlopen(_url(server, path)) as resp:
            body = resp.read().decode()
        assert "<html" in body.lower()


def test_unknown_post_path_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post_json(server, "/api/does-not-exist", {})
    assert excinfo.value.code == 404
