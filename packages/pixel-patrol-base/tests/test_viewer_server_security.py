"""The standalone viewer server only answers same-origin local requests and confines its SQL."""

from __future__ import annotations

import threading
import urllib.error
import urllib.request
from pathlib import Path

import polars as pl
import pytest

from pixel_patrol_base import viewer_server
from pixel_patrol_base.core.project_metadata import ProjectMetadata
from pixel_patrol_base.io.parquet_io import save_parquet


@pytest.fixture
def report(tmp_path):
    path = tmp_path / "report.parquet"
    save_parquet(pl.DataFrame({"a": [1, 2]}), path, ProjectMetadata(project_name="p"))
    return path


@pytest.fixture
def viewer_url(report, tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html><head></head><body>viewer</body></html>")
    ready = threading.Event()
    result = {}

    def on_ready(_port, url):
        result["url"] = url
        ready.set()

    threading.Thread(
        target=viewer_server.serve_viewer,
        kwargs=dict(parquet_path=report, port=0, open_browser=False, dist_dir=dist, ready_callback=on_ready),
        daemon=True,
    ).start()
    assert ready.wait(10)
    return result["url"]


def test_local_request_is_served_without_cors(viewer_url):
    with urllib.request.urlopen(viewer_url) as resp:
        assert b"viewer" in resp.read()
        assert resp.headers.get("Access-Control-Allow-Origin") is None
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"


@pytest.mark.parametrize("headers", [{"Origin": "https://evil.example"}, {"Host": "evil.example"}])
def test_foreign_origin_or_host_is_rejected(viewer_url, headers):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(urllib.request.Request(viewer_url + "data.parquet", headers=headers))
    assert excinfo.value.code == 403


def test_sql_cannot_touch_other_files(report, tmp_path):
    conn, _meta = viewer_server._setup_duckdb(report)
    other = tmp_path / "secret.txt"
    other.write_text("secret")
    outside_out = tmp_path / "out.csv"

    assert conn.execute("SELECT count(*) FROM pp_data").fetchall() == [(2,)]
    for sql in (
        f"SELECT * FROM read_text('{other}')",
        f"COPY (SELECT 1) TO '{outside_out}'",
        "SET enable_external_access=true",
    ):
        with pytest.raises(Exception, match="Permission|Invalid Input"):
            conn.execute(sql)
    assert not outside_out.exists()


def test_sql_can_write_only_to_export_dir(report):
    conn, _meta = viewer_server._setup_duckdb(report)
    target = Path(viewer_server._export_dir()) / "export.parquet"

    conn.execute(f"COPY (SELECT 1) TO '{target}' (FORMAT parquet)")
    assert target.exists()
    target.unlink()


def test_failed_export_leaves_no_temp_file(viewer_url):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(viewer_url + "api/export-parquet?where=WHERE%20nonsense(")
    assert excinfo.value.code == 500
    assert list(Path(viewer_server._export_dir()).iterdir()) == []
