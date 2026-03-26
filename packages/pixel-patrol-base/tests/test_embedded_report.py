"""
Tests for report dashboard components used by both standalone and subprocess-launched report.

The report is launched as a separate process from the processing dashboard (View Report button).
These tests verify _get_report_context, build_report_layout, and related logic.
"""
from pathlib import Path

import pytest

from pixel_patrol_base import api
from pixel_patrol_base.report.dashboard_app import _get_report_context, build_report_layout
from pixel_patrol_base.report import context as report_context
from pixel_patrol_base.report.global_controls import compute_filtered_row_positions
from pixel_patrol_base.report.constants import REPORT_ZIP_PATH_STORE_ID


def test_get_report_context_loads_from_zip_path(project_with_all_data, tmp_path):
    """
    When _current_report_project is None (simulating different request), passing zip_path
    to _get_report_context should load the project and return data.
    """
    export_path = tmp_path / "report_test.zip"
    api.export_project(project_with_all_data, export_path)
    assert export_path.exists()
    assert project_with_all_data.records_df is not None
    expected_rows = project_with_all_data.records_df.height

    # Simulate different request: clear module-level state
    report_context.clear_for_test()

    # Call with zip_path - should load from file, not from _current_report_project
    df, project, name = _get_report_context(str(export_path.resolve()))

    assert df is not None, "records_df should be loaded from zip"
    assert project is not None
    assert name == project_with_all_data.name
    assert df.height == expected_rows, "Should have same row count as exported project"


def test_embedded_report_filtered_indices_with_zip_path(project_with_all_data, tmp_path):
    """
    Simulates update_filtered_indices callback: when zip_path is in State,
    _get_report_context loads data and compute_filtered_row_positions works.
    With no filters, it returns None (meaning use all rows). With filters, returns indices.
    """
    export_path = tmp_path / "report_test.zip"
    api.export_project(project_with_all_data, export_path)

    report_context.clear_for_test()

    df, _, _ = _get_report_context(str(export_path.resolve()))
    assert df is not None
    assert not df.is_empty()

    # No filters => returns None (widgets interpret as "use full df")
    global_config = {"group_col": "imported_path_short", "filter": {}, "dimensions": {}}
    indices = compute_filtered_row_positions(df, global_config)
    assert indices is None  # Expected when no filters


def test_embedded_report_layout_includes_zip_path_store(project_with_all_data, tmp_path):
    """build_report_layout with zip_path should include REPORT_ZIP_PATH_STORE with that path."""
    from dash import Dash
    import dash_bootstrap_components as dbc

    export_path = tmp_path / "report_test.zip"
    api.export_project(project_with_all_data, export_path)
    project = api.import_project(export_path)

    app = Dash(__name__, suppress_callback_exceptions=True)
    layout = build_report_layout(app, project, zip_path=str(export_path.resolve()))

    # Find the store in the layout (layout is a tree of components)
    def find_store(component, store_id):
        if hasattr(component, "id") and getattr(component, "id", None) == store_id:
            return component
        if hasattr(component, "children"):
            children = component.children if isinstance(component.children, list) else [component.children]
            for child in children:
                if child is not None:
                    found = find_store(child, store_id)
                    if found is not None:
                        return found
        return None

    store = find_store(layout, REPORT_ZIP_PATH_STORE_ID)
    assert store is not None
    assert store.data == str(export_path.resolve())


def test_processing_app_serves_layout(project_with_all_data, tmp_path):
    """
    Launcher app serves 200. Report at /report?zip=... (single port).
    Dash returns the initial shell; content is rendered client-side.
    """
    from pixel_patrol_base.processing_dashboard import create_processing_app

    app = create_processing_app()
    client = app.server.test_client()

    response = client.get("/")
    assert response.status_code == 200


def test_report_route_serves_embedded_report(project_with_all_data, tmp_path):
    """GET /report?zip=... returns 200 and the display_page callback can build report layout."""
    from urllib.parse import quote
    from pixel_patrol_base.processing_dashboard import create_processing_app

    export_path = tmp_path / "report_test.zip"
    api.export_project(project_with_all_data, export_path)
    encoded = quote(str(export_path.resolve()))

    app = create_processing_app()
    client = app.server.test_client()

    response = client.get(f"/report?zip={encoded}")
    assert response.status_code == 200


def test_widget_callbacks_registered_at_app_startup():
    """
    Reproduces the bug: clicking "Open" on the launcher home page shows widget
    card skeletons but no Plotly plots.  Only a manual page reload (F5) fixes it.

    Root cause
    ----------
    Widget callbacks were registered lazily, inside the ``display_page`` Dash
    callback.  The browser fetches ``/_dash-dependencies`` **once** at app
    load; any callback registered afterwards is invisible to it.  When the
    user navigates via the SPA (no full reload), the stores appear with data
    but the browser's dependency graph has no widget callbacks to fire, so
    every widget div stays empty.  After a reload the server's Python module
    already has the callbacks registered, ``/_dash-dependencies`` returns them,
    and graphs appear.

    Fix
    ---
    ``create_processing_app`` now calls ``pre_register_widget_callbacks(app)``
    and ``register_report_callbacks(app, None)`` before serving any request,
    ensuring all callbacks are in ``/_dash-dependencies`` from the start.
    Widget callbacks fetch the dataframe from the report context (set by
    ``build_report_layout`` during navigation) rather than from ``self._df``,
    which allows pre-registration with ``df=None``.
    """
    from pixel_patrol_base.processing_dashboard import create_processing_app
    from pixel_patrol_base.plugins.widgets.file_stats.file_stats import FileStatisticsWidget
    from pixel_patrol_base.report import context as report_context

    report_context.clear_for_test()
    app = create_processing_app()

    # The FileStatisticsWidget output must be in the callback map before any
    # navigation happens.  If it isn't, the browser never fires it on first
    # SPA navigation and plots never appear.
    expected_output = f"{FileStatisticsWidget.CONTENT_ID}.children"
    registered = set(app.callback_map.keys())
    assert any(FileStatisticsWidget.CONTENT_ID in k for k in registered), (
        f"FileStatisticsWidget callback ({expected_output!r}) must be registered "
        f"at app startup, not lazily inside display_page.  "
        f"Registered callback outputs: {sorted(registered)}"
    )


def test_no_duplicate_location_in_embedded_report_page(project_with_all_data, tmp_path):
    """
    The embedded report page must NOT include a second dcc.Location with id='url'.
    A duplicate causes display_page to loop (inner Location fires -> display_page
    re-renders -> new inner Location fires -> ...) which prevents plots from rendering.
    """
    from dash import Dash
    from pixel_patrol_base.processing_dashboard import _build_page_content_for_url

    export_path = tmp_path / "no_dup_location.zip"
    api.export_project(project_with_all_data, export_path)

    app = Dash(__name__, suppress_callback_exceptions=True)

    page = _build_page_content_for_url("/report", f"?zip={export_path.resolve()}", app)

    def _collect_ids(component, found=None):
        if found is None:
            found = []
        cid = getattr(component, "id", None)
        if cid is not None:
            found.append(cid)
        children = getattr(component, "children", None) or []
        if not isinstance(children, list):
            children = [children]
        for child in children:
            if child is not None:
                _collect_ids(child, found)
        return found

    ids = _collect_ids(page)
    url_ids = [cid for cid in ids if cid == "url"]
    assert len(url_ids) == 0, (
        "Embedded report page must not include a dcc.Location(id='url') - "
        "the outer static layout already has one and duplicating it causes a "
        "callback loop that prevents Plotly graphs from rendering."
    )


def test_file_stats_widget_callback_returns_graphs(project_with_all_data):
    """
    Verify FileStatisticsWidget._update_plot returns at least one dcc.Graph
    for a valid project. This is the non-GUI equivalent of checking that
    plots actually appear (not just the layout skeleton).
    """
    from dash import dcc
    from pixel_patrol_base.plugins.widgets.file_stats.file_stats import FileStatisticsWidget
    from pixel_patrol_base.report.constants import (
        GC_GROUP_COL, GC_FILTER, GC_DIMENSIONS, GC_IS_SHOW_SIGNIFICANCE,
        DEFAULT_REPORT_GROUP_COL,
    )

    df = project_with_all_data.records_df
    assert df is not None and not df.is_empty(), "Fixture must provide non-empty records_df"

    # Ensure the widget's required columns are present
    required = {"name", "file_extension", "size_bytes", "modification_date"}
    missing = required - set(df.columns)
    assert not missing, f"records_df is missing columns needed by FileStatisticsWidget: {missing}"

    widget = FileStatisticsWidget()
    widget._df = df

    global_config = {
        GC_GROUP_COL: DEFAULT_REPORT_GROUP_COL,
        GC_FILTER: {},
        GC_DIMENSIONS: {},
        GC_IS_SHOW_SIGNIFICANCE: False,
    }

    result = widget._update_plot(
        color_map=None,
        subset_indices=None,
        global_config=global_config,
    )

    graphs = [c for c in result if isinstance(c, dcc.Graph)]
    assert graphs, (
        "FileStatisticsWidget._update_plot must return at least one dcc.Graph "
        "for a valid project. Got: " + repr([type(c).__name__ for c in result])
    )
