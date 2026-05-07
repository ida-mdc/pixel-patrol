"""
End-to-end check: after a normal process_records run, every applicable report widget's
main update method returns a value Dash can JSON-encode (same path as browser callbacks).

On success, prints a short summary to the real terminal (via ``capsys.disabled()``):
which widget ``NAME`` values were exercised, which were skipped for missing columns,
and which were skipped as TensorBoard-only.

Skips EmbeddingProjectorWidget (TensorBoard lifecycle / button-driven; not a table payload issue).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import tifffile
from dash._utils import to_json as dash_to_json

from pixel_patrol_base import api
from pixel_patrol_base.core.report_config import ReportConfig
from pixel_patrol_base.plugin_registry import discover_widget_plugins
from pixel_patrol_base.plugins.widgets.dataset_stats.dataset_histograms import DatasetHistogramWidget
from pixel_patrol_base.plugins.widgets.visualization.image_mosaic import ImageMosaicWidget
from pixel_patrol_base.report.dashboard_app import prepare_app, should_display_widget
from pixel_patrol_base.report.global_controls import compute_filtered_row_positions
from pixel_patrol_base.viewer_server import _setup_duckdb, _ViewerHandler, find_viewer_dist


@dataclass
class _SmokeCtx:
    global_config: dict
    subset: list[int] | None


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


def _encode_dash_output(value) -> None:
    if value is None:
        return
    dash_to_json(value)


def _widget_label(widget) -> str:
    return getattr(widget, "NAME", type(widget).__name__)


def _smoke_render_widget(widget, ctx: _SmokeCtx) -> None:
    if isinstance(widget, DatasetHistogramWidget):
        out = widget._update_plot(
            {}, "shape", [], None, ctx.subset, ctx.global_config
        )
    elif isinstance(widget, ImageMosaicWidget):
        out = widget._update_plot(
            {}, None, ctx.subset, ctx.global_config, "normalized"
        )
    elif hasattr(widget, "_update_plot"):
        out = widget._update_plot({}, ctx.subset, ctx.global_config)
    elif hasattr(widget, "_update_content"):
        out = widget._update_content({}, ctx.subset, ctx.global_config)
    else:
        raise AssertionError(
            f"Widget {type(widget).__name__} has no _update_plot / _update_content"
        )
    _encode_dash_output(out)


def test_report_widgets_dash_json_after_processing(tmp_path, capsys):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    tif_path = images_dir / "img_0000.tif"
    data = np.random.randint(0, 256, size=(1, 2, 1, 10, 10), dtype=np.uint8)
    tifffile.imwrite(str(tif_path), data, photometric="minisblack")

    project = api.create_project("proj", base_dir=images_dir, loader="bioio")
    api.process_files(project, selected_file_extensions={"tif"})

    df = project.records_df
    assert df is not None and not df.is_empty()

    gc_conf = ReportConfig().to_dict()
    subset = compute_filtered_row_positions(df, None)
    ctx = _SmokeCtx(global_config=gc_conf, subset=subset)

    tested_names: list[str] = []
    skipped_embedding: list[str] = []
    skipped_columns: list[str] = []

    for w in discover_widget_plugins():
        label = _widget_label(w)
        if not should_display_widget(w, df.columns):
            skipped_columns.append(label)
            continue
        w._df = df
        _smoke_render_widget(w, ctx)
        tested_names.append(label)

    assert tested_names, (
        "expected at least one widget to apply to the processed TIFF; "
        f"skipped (columns): {skipped_columns or '—'}; "
        f"skipped (embedding): {skipped_embedding or '—'}"
    )

    app = prepare_app(project, None)
    layout = app.layout() if callable(app.layout) else app.layout
    dash_to_json(layout)

    lines = [
        "",
        "[report-widgets-integration] Dash JSON smoke passed.",
        f"  Exercised ({len(tested_names)}): {', '.join(tested_names)}",
    ]
    if skipped_embedding:
        lines.append(
            f"  Skipped — TensorBoard / not table-driven: {', '.join(skipped_embedding)}"
        )
    if skipped_columns:
        lines.append(
            f"  Skipped — missing columns for this dataset: {', '.join(skipped_columns)}"
        )
    lines.append("")

    with capsys.disabled():
        print("\n".join(lines), flush=True)


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