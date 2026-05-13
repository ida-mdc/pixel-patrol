"""
Browser-level smoke test for the static viewer.

Starts the viewer HTTP server in-process against a freshly processed Parquet file,
then uses Playwright to load the page and assert no JavaScript errors occur.

Skipped automatically when Playwright is not installed.
"""

from __future__ import annotations

import contextlib
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import tifffile

from pixel_patrol_base import api
from pixel_patrol_base.viewer_server import _setup_duckdb, _ViewerHandler, find_viewer_dist


@contextlib.contextmanager
def _running_viewer_server(parquet_path):
    """Start the static viewer server in-process and yield its URL."""
    dist_dir = find_viewer_dist()
    duck_conn, project_name, description = _setup_duckdb(parquet_path)
    query_lock = threading.Lock()

    handler = type(
        "_TestViewerHandler",
        (_ViewerHandler,),
        {
            "dist_dir": dist_dir,
            "parquet_path": parquet_path,
            "duck_conn": duck_conn,
            "query_lock": query_lock,
            "project_name": project_name,
            "description": description,
            "extension_dirs": [],
        },
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield url
    finally:
        server.shutdown()
        server.server_close()
        duck_conn.close()
        thread.join(timeout=5)


def test_static_viewer_has_no_js_errors_after_processing(tmp_path):
    """Open the real static viewer page and fail on JavaScript runtime errors."""
    pw = pytest.importorskip(
        "playwright.sync_api",
        reason="Install Playwright to run browser-level viewer integration checks",
    )

    images_dir = tmp_path / "images"
    images_dir.mkdir()
    tif_path = images_dir / "img_0000.tif"
    data = np.random.randint(0, 256, size=(1, 2, 1, 10, 10), dtype=np.uint8)
    tifffile.imwrite(str(tif_path), data, photometric="minisblack")

    project = api.create_project("proj", base_dir=images_dir, loader="bioio")
    api.process_files(project, selected_file_extensions={"tif"})
    parquet_path = project.output_path

    errors: list[str] = []
    with _running_viewer_server(parquet_path) as viewer_url, pw.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.on(
            "console",
            lambda msg: errors.append(f"console.{msg.type}: {msg.text}")
            if msg.type == "error"
            else None,
        )
        page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))

        page.goto(viewer_url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_timeout(2500)
        browser.close()

    assert not errors, "Static viewer reported JS errors:\n" + "\n".join(errors)