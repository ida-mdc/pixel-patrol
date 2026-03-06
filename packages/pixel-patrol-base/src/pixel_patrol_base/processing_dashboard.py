"""
Pixel Patrol Launcher - report-centric app for clusters.

Uses $HOME/pixel-patrol-reports as default storage. Shows a list of existing reports
(paths, subpaths, file filters) with minimal "Add report" form.
Reports load via direct navigation (no embedded routing that breaks on reload).
"""
import base64
import hashlib
import io
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, parse_qs, urlparse

import dash_bootstrap_components as dbc
import polars as pl
from dash import ALL, Dash, html, dcc, Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate

from pixel_patrol_base import api
from pixel_patrol_base.core.file_system import walk_filesystem
from pixel_patrol_base.core.project_settings import Settings

logger = logging.getLogger(__name__)

ASSETS_DIR = (Path(__file__).parent / "report" / "assets").resolve()

# Default reports directory
REPORTS_DIR = Path(os.environ.get("PIXEL_PATROL_REPORTS_DIR", str(Path.home() / "pixel-patrol-reports")))
INDEX_FILE = REPORTS_DIR / "reports_index.json"

# Report metadata schema
# { "zip_path": str, "base_dir": str, "subpaths": list[str], "loader": str, "file_extensions": str, "created_at": str }

_report_callbacks_registered = False
_processing_state = {
    "status": "idle",
    "progress": 0,
    "message": "",
    "error": None,
    "output_zip": None,
}
_processing_lock = threading.Lock()


def _ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR


def _load_index() -> List[Dict[str, Any]]:
    """Load reports index, newest first."""
    _ensure_reports_dir()
    if not INDEX_FILE.exists():
        return []
    try:
        with open(INDEX_FILE) as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else []
        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return entries
    except Exception as e:
        logger.warning("Could not load reports index: %s", e)
        return []


def _save_index(entries: List[Dict[str, Any]]) -> None:
    _ensure_reports_dir()
    with open(INDEX_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def _zip_hash(zip_path: str) -> str:
    """Stable short ID for a zip_path, used as delete-button component ID."""
    return hashlib.md5(zip_path.encode()).hexdigest()[:16]


def _format_size(n_bytes: int) -> str:
    if n_bytes <= 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.0f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _make_histogram_svg(counts: List[int], width: int = 200, height: int = 56) -> str:
    """Render a mean-intensity histogram as a compact inline SVG sparkline."""
    if not counts or len(counts) < 2:
        return ""
    max_v = max(counts) or 1
    n = len(counts)
    bar_w = width / n
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" rx="3" fill="#f1f3f5"/>',
    ]
    for i, v in enumerate(counts):
        h = max(1, int(v / max_v * (height - 3)))
        x = i * bar_w
        parts.append(
            f'<rect x="{x:.1f}" y="{height - h - 1}" '
            f'width="{bar_w + 0.4:.1f}" height="{h}" fill="#4c9be8" opacity="0.75"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _make_folder_bar(folder_counts: Dict[str, int]) -> html.Div:
    """CSS stacked proportional bar + folder legend showing per-subfolder file distribution."""
    if not folder_counts or len(folder_counts) < 2:
        return html.Div()
    total = sum(folder_counts.values()) or 1
    palette = ["#4c9be8", "#f5a623", "#7ed321", "#9b59b6", "#e74c3c", "#1abc9c", "#e67e22", "#2ecc71"]
    items = sorted(folder_counts.items(), key=lambda x: -x[1])[:8]

    segments = [
        html.Div(style={
            "width": f"{count / total * 100:.1f}%",
            "backgroundColor": palette[i % len(palette)],
            "height": "100%",
        }, title=f"{name}: {count} files")
        for i, (name, count) in enumerate(items)
    ]
    legend = [
        html.Span([
            html.Span(style={"display": "inline-block", "width": "8px", "height": "8px",
                             "backgroundColor": palette[i % len(palette)],
                             "borderRadius": "1px", "marginRight": "3px", "flexShrink": "0"}),
            Path(name).name or name,
            html.Span(f" {count}", style={"color": "#868e96"}),
        ], className="me-2 d-flex align-items-center",
           style={"fontSize": "11px", "color": "#495057"})
        for i, (name, count) in enumerate(items)
    ]
    return html.Div([
        html.Div(segments, style={
            "height": "6px", "display": "flex", "borderRadius": "3px",
            "overflow": "hidden", "backgroundColor": "#e9ecef", "marginBottom": "5px",
        }),
        html.Div(legend, style={"display": "flex", "flexWrap": "wrap"}),
    ], className="mb-2")


def _add_report_to_index(
    zip_path: str,
    base_dir: str,
    subpaths: List[str],
    loader: str,
    file_extensions: str,
    n_files: int = 0,
    total_size_bytes: int = 0,
    thumbnail_b64: Optional[str] = None,
    file_type_counts: Optional[Dict[str, int]] = None,
    folder_counts: Optional[Dict[str, int]] = None,
    mean_histogram: Optional[List[int]] = None,
    hist_min: Optional[int] = None,
    hist_max: Optional[int] = None,
) -> None:
    entry = {
        "zip_path": str(Path(zip_path).resolve()),
        "base_dir": str(Path(base_dir).resolve()),
        "subpaths": subpaths,
        "loader": loader or "",
        "file_extensions": file_extensions or "",
        "created_at": datetime.now().isoformat(),
        "n_files": n_files,
        "total_size_bytes": total_size_bytes,
        "thumbnail_b64": thumbnail_b64,
        "file_type_counts": file_type_counts or {},
        "folder_counts": folder_counts or {},
        "mean_histogram": mean_histogram or [],
        "hist_min": hist_min,
        "hist_max": hist_max,
    }
    entries = _load_index()
    entries.insert(0, entry)
    entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    _save_index(entries)


def _get_initial_path_from_request() -> Tuple[Optional[str], Optional[str]]:
    """
    Get pathname and search from the current request.
    Used to pre-render report layout on first load (avoids needing F5).
    Uses request path/args when direct, or Referer when layout is requested via _dash-layout.
    """
    try:
        from flask import request

        # Direct request (e.g. GET /report?zip=X) - e.g. initial page load
        path = (getattr(request, "path", None) or "").rstrip("/") or "/"
        if path == "/report":
            qs = getattr(request, "query_string", None)
            search = f"?{qs.decode()}" if qs else ""
            if search and "zip=" in search:
                return "/report", search

        # Layout requested via _dash-layout; Referer has the page URL
        referrer = getattr(request, "referrer", None) or ""
        if referrer and "/report" in referrer and "zip=" in referrer:
            parsed = urlparse(referrer)
            if parsed.path.rstrip("/") == "/report" and parsed.query:
                return "/report", f"?{parsed.query}"
    except Exception:
        pass
    return None, None


def _get_available_loaders() -> List[Dict[str, Any]]:
    from pixel_patrol_base.plugin_registry import discover_plugins_from_entrypoints
    loaders = [{"label": "None (basic file info only)", "value": "", "extensions": []}]
    for loader_class in discover_plugins_from_entrypoints("pixel_patrol.loader_plugins"):
        try:
            ext = getattr(loader_class, "SUPPORTED_EXTENSIONS", set())
            loaders.append({
                "label": loader_class.NAME,
                "value": loader_class.NAME,
                "extensions": sorted(list(ext)) if ext else [],
            })
        except Exception as e:
            logger.warning("Loader %s: %s", getattr(loader_class, "NAME", "?"), e)
    return loaders


def create_processing_app() -> Dash:
    """Create the report-centric launcher app."""
    external_stylesheets = [
        dbc.themes.BOOTSTRAP,
        "https://codepen.io/chriddyp/pen/bWLwgP.css",
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css",
    ]
    app = Dash(
        __name__,
        external_stylesheets=external_stylesheets,
        suppress_callback_exceptions=True,
        assets_folder=str(ASSETS_DIR),
    )

    # Callable layout: called fresh on every page request so that
    # reports-index-store is always up-to-date (fixes stale data on reload).
    # processing-status-banner lives here (outside page-content) so it is
    # always in the DOM and show_processing_banner can always update it.

    def serve_layout():
        footer = html.Footer(
            dbc.Container([
                html.Div([
                    # Left: logo + attribution
                    html.Div([
                        html.A(
                            html.Img(src=app.get_asset_url("Helmholtz-Imaging_Mark.png"),
                                     style={"height": "36px", "marginRight": "14px",
                                            "verticalAlign": "middle"}),
                            href="https://helmholtz-imaging.de", target="_blank",
                        ),
                        html.Div([
                            html.Span("Pixel Patrol is developed by ",
                                      style={"color": "#adb5bd"}),
                            html.A("Helmholtz Imaging", href="https://helmholtz-imaging.de",
                                   target="_blank",
                                   style={"color": "#dee2e6", "textDecoration": "none"}),
                            html.Span(".", style={"color": "#adb5bd"}),
                        ], style={"fontSize": "12px", "lineHeight": "1.4"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                    # Right: contact + GitHub
                    html.Div([
                        html.A("support@helmholtz-imaging.de",
                               href="mailto:support@helmholtz-imaging.de",
                               style={"color": "#adb5bd", "textDecoration": "none",
                                      "fontSize": "12px", "marginRight": "16px"}),
                        html.A([html.I(className="bi bi-github me-1"), "GitHub"],
                               href="https://github.com/ida-mdc/pixel-patrol",
                               target="_blank",
                               style={"color": "#adb5bd", "textDecoration": "none",
                                      "fontSize": "12px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                ], style={"display": "flex", "justifyContent": "space-between",
                          "alignItems": "center", "padding": "16px 0"}),
            ], fluid=True, style={"maxWidth": "1400px"}),
            style={"backgroundColor": "#212529", "marginTop": "auto"},
        )
        return html.Div([
            dcc.Location(id="url", refresh=False),
            html.Div(id="page-content"),
            footer,
            dcc.Store(id="reports-index-store", data=_load_index()),
            dcc.Store(id="processing-state-store", data=_get_processing_state()),
            dcc.Interval(id="progress-interval", interval=500, n_intervals=0, disabled=True),
        ], style={"minHeight": "100vh", "display": "flex", "flexDirection": "column"})

    app.layout = serve_layout

    _register_callbacks(app)

    # Pre-register ALL report + widget callbacks now so that they appear in
    # /_dash-dependencies (fetched once by the browser at startup).  Without
    # this, callbacks registered lazily inside display_page are invisible to
    # the browser's dependency graph, so plots never render on first SPA
    # navigation ("Open") – only after a manual page reload.
    from pixel_patrol_base.report.dashboard_app import (
        pre_register_widget_callbacks,
        register_report_callbacks,
    )
    register_report_callbacks(app, None)
    pre_register_widget_callbacks(app)
    global _report_callbacks_registered
    _report_callbacks_registered = True

    return app


def _get_processing_state() -> Dict[str, Any]:
    with _processing_lock:
        s = _processing_state.copy()
        s.pop("_run_meta", None)
        return {k: v for k, v in s.items() if isinstance(v, (str, int, float, bool, type(None), list, dict))}


def _build_page_content_for_url(
    pathname: str | None, search: str | None, app: Dash
) -> html.Div | dbc.Alert:
    """
    Build page content from pathname/search. Used for both initial layout
    (server-side from request) and display_page callback (client-side from Location).
    """
    if pathname and pathname.rstrip("/") == "/report" and search:
        params = parse_qs(search.lstrip("?"))
        zip_param = params.get("zip", [None])[0]
        if zip_param:
            try:
                zip_path = Path(unquote(zip_param))
                if not zip_path.is_absolute():
                    zip_path = zip_path.resolve()
                if not zip_path.exists():
                    return dbc.Alert(f"ZIP not found: {zip_path}", color="danger", className="m-4")

                # Use shared report context so we only import once per ZIP per process.
                from pixel_patrol_base.report import context as report_context
                df, project, _name = report_context.get_report_context(str(zip_path))
                if project is None or df is None:
                    return dbc.Alert(f"Failed to load report from ZIP: {zip_path}", color="danger", className="m-4")

                from pixel_patrol_base.report.dashboard_app import build_report_layout, register_report_callbacks

                global _report_callbacks_registered
                if not _report_callbacks_registered:
                    register_report_callbacks(app, project)
                    _report_callbacks_registered = True
                report_layout = build_report_layout(app, project, zip_path=str(zip_path))
                # The outer dcc.Location(id="url") in the static layout already
                # reflects the full URL (including ?zip=...). Adding a second
                # dcc.Location with the same id here would create a duplicate
                # that loops: display_page re-fires every time the inner
                # Location fires its initial value, preventing plots from ever
                # rendering. Report callbacks reference the outer "url" directly.
                return html.Div([
                    html.Div(
                        dcc.Link("← Back to Reports", href="/",
                                 style={"textDecoration": "none", "color": "#6c757d",
                                        "fontSize": "13px"}),
                        style={"padding": "8px 16px", "borderBottom": "1px solid #e9ecef"},
                    ),
                    report_layout,
                ])
            except Exception as e:
                logger.exception("Error loading report")
                return dbc.Alert(f"Failed to load report: {e}", color="danger", className="m-4")
    return _build_home_layout(app)


def _build_home_layout(app: Dash) -> html.Div:
    """Main view: list of reports + Add report button."""
    loaders = _get_available_loaders()
    _LEFT_W = "160px"
    return html.Div([
        dbc.Container([
            # ── Page header: image+button column left, title right ────────
            html.Div([
                html.Div([
                    html.Img(src=app.get_asset_url("prevalidation.png"),
                             style={"width": "100%", "height": "auto",
                                    "display": "block", "marginBottom": "8px"}),
                    dbc.Button(
                        [html.I(className="bi bi-plus-circle me-2"), "Add Report"],
                        id="add-report-btn", color="primary", size="lg",
                        className="w-100",
                    ),
                ], style={"width": _LEFT_W, "flexShrink": "0", "marginRight": "20px"}),
                html.Div([
                    html.H4("Pixel Patrol Reports", className="mb-0",
                            style={"fontWeight": "700", "color": "#212529"}),
                    dbc.Button(
                        [html.I(className="bi bi-arrow-clockwise me-1"), "Recalculate Overview"],
                        id="recalculate-btn", color="outline-secondary", size="sm",
                        className="mt-2",
                        title="Re-extract thumbnails, histograms and file stats from all ZIP files",
                    ),
                ], className="align-self-center"),
            ], className="d-flex align-items-start pt-3 mb-3"),
            # ── Processing / status banner ────────────────────────────────
            html.Div(id="processing-status-banner", className="mb-2"),
            # ── Reports list ──────────────────────────────────────────────
            html.Div(id="reports-list"),
            _build_add_report_modal(loaders),
        ], fluid=True, style={"maxWidth": "1400px"}, className="pb-4"),
    ])


def _build_reports_list(entries: List[Dict[str, Any]]) -> html.Div:
    """Render list of report cards."""
    if not entries:
        return html.Div([
            html.I(className="bi bi-folder2-open", style={"fontSize": "2.5rem", "color": "#ced4da"}),
            html.P("No reports yet. Click '+ Add Report' to create one.", className="text-muted mt-2"),
        ], className="text-center py-4")

    cards = []
    for e in entries:
        zip_path = e.get("zip_path", "")
        base_dir = e.get("base_dir", "")
        subpaths = e.get("subpaths", [])
        loader = e.get("loader", "")
        created = e.get("created_at", "")
        n_files = e.get("n_files") or 0
        total_size_bytes = e.get("total_size_bytes") or 0
        thumbnail_b64 = e.get("thumbnail_b64")
        file_type_counts: Dict[str, int] = e.get("file_type_counts") or {}
        folder_counts: Dict[str, int] = e.get("folder_counts") or {}
        mean_histogram: List[int] = e.get("mean_histogram") or []
        hist_min: Optional[int] = e.get("hist_min")
        hist_max: Optional[int] = e.get("hist_max")
        zip_exists = Path(zip_path).exists()

        # ── Date column (two rows: date / time) ───────────────────────────
        try:
            dt = datetime.fromisoformat(created)
            date_str = dt.strftime("%Y-%m-%d")
            time_str = dt.strftime("%H:%M")
        except Exception:
            date_str = created[:10] if len(created) >= 10 else (created or "")
            time_str = created[11:16] if len(created) >= 16 else ""

        col_date = html.Div([
            html.Div(date_str, style={"fontWeight": "600", "color": "#343a40", "whiteSpace": "nowrap"}),
            html.Div(time_str, style={"color": "#6c757d", "whiteSpace": "nowrap"}),
        ], style={"flexShrink": "0", "width": "90px", "paddingRight": "16px",
                  "alignSelf": "center"})

        # ── Col 1: full path + zip path + subpaths ────────────────────────
        col1_parts = [
            html.Div(
                [html.I(className="bi bi-folder2-open me-1"), base_dir],
                className="text-muted",
                style={"overflow": "hidden", "textOverflow": "ellipsis",
                       "whiteSpace": "nowrap"},
                id=created+"_base_dir"
            ),
            dbc.Tooltip(
                base_dir,
                target=created+"_base_dir",
            ),
            html.Div(
                [html.I(className="bi bi-archive me-1"), zip_path],
                className="text-muted",
                style={"overflow": "hidden", "textOverflow": "ellipsis",
                       "whiteSpace": "nowrap", "marginBottom": "2px"},
                id=created + "_zip"
            ),
            dbc.Tooltip(
                zip_path,
                target=created+"_zip",
            ),
        ]
        # Only show subpaths when the user explicitly configured comparison groups
        if subpaths:
            pills = [
                dbc.Badge(Path(sp).name or sp, color="secondary", className="me-1",
                          style={"fontWeight": "normal", "fontSize": "10px"})
                for sp in subpaths[:6]
            ]
            if len(subpaths) > 6:
                pills.append(dbc.Badge(f"+{len(subpaths)-6}", color="light",
                                       text_color="secondary", className="me-1",
                                       style={"fontWeight": "normal", "fontSize": "10px"}))
            col1_parts.append(html.Div(pills, className="d-flex flex-wrap mt-1"))
        col1 = html.Div(col1_parts, style={"flex": "2", "minWidth": "0", "paddingRight": "20px",
                                           "alignSelf": "center"})

        # ── Col 2a: file count + size ──────────────────────────────────────
        col2a_rows = []
        if n_files:
            col2a_rows.append(html.Div(
                [html.I(className="bi bi-files me-1"), f"{n_files:,} files"],
                style={"fontSize": "12px", "color": "#495057", "marginBottom": "3px",
                       "whiteSpace": "nowrap"},
            ))
        if total_size_bytes:
            col2a_rows.append(html.Div(
                [html.I(className="bi bi-hdd me-1"), _format_size(total_size_bytes)],
                className="text-muted", style={"fontSize": "11px", "whiteSpace": "nowrap"},
            ))
        col2a = html.Div(col2a_rows,
            style={"flexShrink": "0", "minWidth": "90px", "paddingRight": "20px",
                   "alignSelf": "center"})

        # ── Col 2b: loader + file types ────────────────────────────────────
        col2b_rows = []
        if loader:
            col2b_rows.append(html.Div(
                [html.I(className="bi bi-cpu me-1"), loader],
                style={"fontSize": "12px", "color": "#495057", "marginBottom": "3px",
                       "whiteSpace": "nowrap"},
            ))
        if file_type_counts:
            top = sorted(file_type_counts.items(), key=lambda x: -x[1])[:4]
            col2b_rows.append(html.Div(
                "  ".join(f".{ext} ×{cnt}" for ext, cnt in top),
                className="text-muted", style={"fontSize": "11px", "whiteSpace": "nowrap"},
            ))
        col2b = html.Div(col2b_rows,
            style={"flexShrink": "0", "minWidth": "90px", "paddingRight": "20px",
                   "alignSelf": "center"})

        # ── Col 3: intensity histogram (256 bins, full height) ────────────
        hist_svg = _make_histogram_svg(mean_histogram)
        if hist_svg:
            svg_b64 = base64.b64encode(hist_svg.encode()).decode()
            range_str = (f"{hist_min} – {hist_max}"
                         if hist_min is not None and hist_max is not None else "")
            col3 = html.Div([
                html.Div("intensity", style={"fontSize": "10px", "color": "#adb5bd",
                                             "marginBottom": "2px"}),
                html.Img(src=f"data:image/svg+xml;base64,{svg_b64}",
                         style={"width": "200px", "height": "56px", "display": "block"}),
                html.Div([
                    html.Span(str(hist_min) if hist_min is not None else "",
                              style={"fontSize": "10px", "color": "#adb5bd"}),
                    html.Span(str(hist_max) if hist_max is not None else "",
                              style={"fontSize": "10px", "color": "#adb5bd"}),
                ], style={"display": "flex", "justifyContent": "space-between",
                          "width": "200px", "marginTop": "1px"}),
            ], style={"flexShrink": "0", "paddingRight": "16px", "alignSelf": "center"})
        else:
            col3 = html.Div(style={"flexShrink": "0"})

        # ── Shared height for thumbnail + buttons ──────────────────────────
        _ROW_H = "64px"
        _ts = {"width": _ROW_H, "height": _ROW_H, "flexShrink": "0",
               "borderRadius": "6px", "overflow": "hidden", "border": "1px solid #dee2e6"}
        if thumbnail_b64:
            thumb_el = html.Img(
                src=f"data:image/jpeg;base64,{thumbnail_b64}",
                style={**_ts, "objectFit": "cover", "display": "block"},
            )
        else:
            thumb_el = html.Div(
                html.I(className="bi bi-grid-3x3-gap",
                       style={"fontSize": "1.6rem", "color": "#ced4da"}),
                style={**_ts, "display": "flex", "alignItems": "center",
                       "justifyContent": "center", "backgroundColor": "#f8f9fa"},
            )

        # ── Open (left) and Delete (right) ────────────────────────────────
        zh = _zip_hash(zip_path)
        _btn_style = {"height": _ROW_H, "width": "80px", "display": "flex",
                      "alignItems": "center", "justifyContent": "center"}
        open_btn = html.A(
            dbc.Button("Open", color="primary", disabled=not zip_exists,
                       style=_btn_style),
            href=f"/report?zip={quote(zip_path)}" if zip_exists else "#",
            style={"textDecoration": "none", "flexShrink": "0", "marginRight": "12px"},
        )
        delete_btn = dbc.Button(
            "Delete Report",
            id={"type": "delete-report-btn", "zip_hash": zh},
            color="danger", outline=True,
            style={**_btn_style, "whiteSpace": "normal", "lineHeight": "1.2",
                   "textAlign": "center", "fontSize": "13px"},
            title="Delete report ZIP from disk",
            className="ms-2", n_clicks=0,
        )

        card = dbc.Card(
            dbc.CardBody(
                html.Div([
                    open_btn,
                    html.Div(thumb_el, style={"flexShrink": "0", "marginRight": "14px",
                                              "alignSelf": "center"}),
                    col_date,
                    col1,
                    col2a,
                    col2b,
                    col3,
                    delete_btn,
                ], style={"display": "flex", "alignItems": "center"}),
                style={"padding": "10px 14px"},
            ),
            className="mb-2",
            style={"opacity": "0.6" if not zip_exists else "1"},
        )
        cards.append(card)

    return html.Div(cards)


def _build_add_report_modal(loaders: List[Dict]) -> html.Div:
    base_placeholder = os.environ.get("PIXEL_PATROL_DEFAULT_BASE_DIR", "/path/to/dataset")
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Add Report")),
        dbc.ModalBody([
            dbc.Label("Base Directory *", html_for="add-base-dir"),
            dbc.Input(id="add-base-dir", type="text", placeholder=base_placeholder, className="mb-3"),
            dbc.Label("Loader", html_for="add-loader"),
            dbc.Select(
                id="add-loader",
                options=[{"label": l["label"], "value": l["value"]} for l in loaders],
                value=loaders[1]["value"] if len(loaders) > 1 else "",
                className="mb-3",
            ),
            dbc.Label("Paths (subfolders)", html_for="add-paths"),
            dbc.Input(id="add-paths", type="text", placeholder="path1, path2... (leave empty for base dir)", className="mb-3"),
            dbc.Label("File Extensions", html_for="add-extensions"),
            dbc.Input(id="add-extensions", type="text", placeholder="Leave empty for all", className="mb-3"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="add-cancel-btn", color="secondary"),
            dbc.Button("Run Processing", id="add-run-btn", color="primary"),
        ]),
    ], id="add-report-modal", is_open=False)


def _register_callbacks(app: Dash):
    loaders = _get_available_loaders()

    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
        Input("url", "search"),
    )
    def display_page(pathname: str | None, search: str | None):
        return _build_page_content_for_url(pathname, search, app)

    @app.callback(
        Output("reports-list", "children"),
        Input("reports-index-store", "data"),
    )
    def render_reports_list(entries: List[Dict] | None):
        return _build_reports_list(entries or [])

    @app.callback(
        Output("reports-index-store", "data", allow_duplicate=True),
        Input({"type": "delete-report-btn", "zip_hash": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def delete_report(n_clicks_list):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        clicks = n_clicks_list if isinstance(n_clicks_list, list) else [n_clicks_list]
        if not any(clicks):
            raise PreventUpdate
        triggered_id = json.loads(ctx.triggered[0]["prop_id"].rsplit(".", 1)[0])
        target_hash = triggered_id["zip_hash"]
        entries = _load_index()
        to_delete = [e for e in entries if _zip_hash(e.get("zip_path", "")) == target_hash]
        new_entries = [e for e in entries if _zip_hash(e.get("zip_path", "")) != target_hash]
        if len(new_entries) == len(entries):
            raise PreventUpdate
        for e in to_delete:
            zip_path = Path(e.get("zip_path", ""))
            try:
                if zip_path.exists():
                    zip_path.unlink()
            except Exception as exc:
                logger.warning("Could not delete ZIP %s: %s", zip_path, exc)
        _save_index(new_entries)
        return new_entries

    @app.callback(
        Output("add-report-modal", "is_open"),
        Output("progress-interval", "disabled"),
        Output("processing-status-banner", "children", allow_duplicate=True),
        Input("add-report-btn", "n_clicks"),
        Input("add-cancel-btn", "n_clicks"),
        Input("add-run-btn", "n_clicks"),
        State("add-report-modal", "is_open"),
        State("add-base-dir", "value"),
        State("add-loader", "value"),
        State("add-paths", "value"),
        State("add-extensions", "value"),
        prevent_initial_call=True,
    )
    def toggle_modal_and_start_run(n_add, n_cancel, n_run, is_open, base_dir, loader, paths, extensions):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        # When page-content is dynamically replaced (e.g. display_page fires),
        # Dash re-mounts all buttons with n_clicks=0 and may fire this callback
        # even though prevent_initial_call=True only blocks the very first app
        # load.  Guard against that by ignoring triggers with a falsy value.
        if not ctx.triggered[0]["value"]:
            raise PreventUpdate
        tid = ctx.triggered[0]["prop_id"].split(".")[0]

        if tid == "add-report-btn":
            return True, True, no_update

        if tid == "add-cancel-btn":
            return False, True, no_update

        if tid == "add-run-btn" and n_run:
            if not base_dir or not base_dir.strip():
                return True, True, no_update
            base_dir = base_dir.strip()
            path_list = [p.strip() for p in (paths or "").split(",") if p.strip()]
            ext_set = {e.strip().lstrip(".") for e in (extensions or "").split(",") if e.strip()} if extensions else "all"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(base_dir).name or "project"
            output_zip = REPORTS_DIR / f"{timestamp}_{base_name}.zip"
            _run_processing_async(base_dir, str(output_zip), path_list, loader or None, ext_set)
            # Directly paint the spinner banner so it appears immediately,
            # before the interval has a chance to poll (avoids the gap where
            # processing completes before the first interval fire).
            spinner = dbc.Alert([
                html.Strong("Processing in progress", className="me-2"),
                dbc.Spinner(size="sm", color="light", spinner_class_name="me-2"),
                "Starting...",
                dbc.Progress(value=5, striped=True, animated=True, className="mt-2", style={"height": "8px"}),
            ], color="info", className="mb-0")
            return False, False, spinner

        raise PreventUpdate

    @app.callback(
        Output("processing-state-store", "data"),
        Input("progress-interval", "n_intervals"),
    )
    def poll_processing_state(_n):
        return _get_processing_state()

    @app.callback(
        Output("processing-status-banner", "children"),
        Input("processing-state-store", "data"),
    )
    def show_processing_banner(state: Dict):
        if not state or state.get("status") == "idle":
            return html.Div()
        if state.get("status") == "running":
            return dbc.Alert([
                html.Strong("Processing in progress", className="me-2"),
                dbc.Spinner(size="sm", color="light", spinner_class_name="me-2"),
                state.get("message", "Processing..."),
                dbc.Progress(value=state.get("progress", 0), striped=True, animated=True,
                             className="mt-2", style={"height": "8px"}),
            ], color="info", className="mb-0")
        if state.get("status") == "completed":
            extra = [] if state.get("output_zip") == "__rebuild__" else [
                " ", html.A("Open report",
                            href=f"/report?zip={quote(state.get('output_zip', ''))}",
                            className="ms-2"),
            ]
            return dbc.Alert([
                html.I(className="bi-check-circle me-2"),
                state.get("message", "Done."),
                *extra,
            ], color="success", className="mb-0")
        if state.get("status") == "error":
            return dbc.Alert(
                [html.I(className="bi-x-circle me-2"), state.get("error", "Error")],
                color="danger", className="mb-0")
        return html.Div()

    _last_refreshed_zip: list = [None]  # mutable to allow closure

    @app.callback(
        Output("reports-index-store", "data", allow_duplicate=True),
        Output("progress-interval", "disabled", allow_duplicate=True),
        Input("processing-state-store", "data"),
        prevent_initial_call=True,
    )
    def on_processing_complete(state: Dict):
        if state and state.get("status") == "completed" and state.get("output_zip"):
            zip_path = state["output_zip"]
            if _last_refreshed_zip[0] == zip_path:
                raise PreventUpdate
            _last_refreshed_zip[0] = zip_path
            # Index was already written by the background thread; just reload it.
            return _load_index(), True
        if state and state.get("status") == "error":
            return no_update, True
        raise PreventUpdate

    @app.callback(
        Output("processing-status-banner", "children", allow_duplicate=True),
        Output("progress-interval", "disabled", allow_duplicate=True),
        Input("recalculate-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def start_recalculate(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        _rebuild_metadata_async()
        spinner = dbc.Container(dbc.Alert([
            html.Strong("Recalculating overview", className="me-2"),
            dbc.Spinner(size="sm", color="light", spinner_class_name="me-2"),
            "Re-importing ZIP files...",
        ], color="info", className="mb-0"), fluid=True, className="px-4 pt-2")
        return spinner, False


def _extract_metadata_from_records(rdf, base_directory: str) -> Dict[str, Any]:
    """Extract index card metadata from a records_df. Returns partial entry dict."""
    import numpy as np
    from PIL import Image as _PILImage

    n_files = 0
    total_size_bytes = 0
    thumbnail_b64 = None
    file_type_counts: Dict[str, int] = {}
    folder_counts: Dict[str, int] = {}
    mean_histogram: List[int] = []
    hist_min: Optional[int] = None
    hist_max: Optional[int] = None

    if rdf is None or rdf.is_empty():
        return {}

    n_files = rdf.height

    if "size_bytes" in rdf.columns:
        total_size_bytes = int(rdf["size_bytes"].cast(pl.Int64).drop_nulls().sum() or 0)

    if "file_extension" in rdf.columns:
        ct = rdf.group_by("file_extension").len().sort("len", descending=True)
        file_type_counts = dict(zip(ct["file_extension"].to_list(), ct["len"].to_list()))

    if "parent" in rdf.columns:
        base_p = Path(base_directory).resolve()
        grp = rdf.group_by("parent").len().sort("len", descending=True)
        for row in grp.iter_rows():
            folder_path, count = row[0], row[1]
            try:
                rel = str(Path(folder_path).relative_to(base_p))
            except ValueError:
                rel = Path(folder_path).name or str(folder_path)
            folder_counts[rel] = count

    if "histogram_counts" in rdf.columns:
        valid_h = rdf.filter(pl.col("histogram_counts").is_not_null())
        if valid_h.height > 0:
            hist_rows = valid_h["histogram_counts"].to_list()
            if hist_rows and len(hist_rows[0]) == 256:
                stacked = np.array([list(r) for r in hist_rows], dtype=float)
                mean256 = stacked.mean(axis=0)
                mean_histogram = [int(v) for v in mean256]
                nonzero = np.nonzero(mean256 > 0.01)[0]
                if len(nonzero):
                    hist_min = int(nonzero[0])
                    hist_max = int(nonzero[-1])

    if "thumbnail" in rdf.columns:
        valid_t = rdf.filter(pl.col("thumbnail").is_not_null())
        for idx in range(min(3, valid_t.height)):
            try:
                raw = valid_t["thumbnail"][idx]
                arr = np.asarray(raw).squeeze()
                if arr.ndim == 1:
                    side = int(np.sqrt(arr.size))
                    if side * side != arr.size:
                        continue
                    arr = arr.reshape((side, side))
                if arr.ndim == 2:
                    img = _PILImage.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
                elif arr.ndim == 3:
                    img = _PILImage.fromarray(arr.astype(np.uint8)).convert("RGB")
                else:
                    continue
                img = img.resize((96, 96), _PILImage.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=72)
                thumbnail_b64 = base64.b64encode(buf.getvalue()).decode()
                break
            except Exception:
                continue

    return {
        "n_files": n_files,
        "total_size_bytes": total_size_bytes,
        "thumbnail_b64": thumbnail_b64,
        "file_type_counts": file_type_counts,
        "folder_counts": folder_counts,
        "mean_histogram": mean_histogram,
        "hist_min": hist_min,
        "hist_max": hist_max,
    }


def _rebuild_metadata_async() -> None:
    """Re-import every ZIP in the reports dir, add missing ones, and refresh all card metadata."""
    with _processing_lock:
        _processing_state.update(
            status="running", progress=5,
            message="Scanning for reports...",
            error=None, output_zip=None,
        )

    def _run():
        try:
            entries = _load_index()
            known_paths = {e.get("zip_path", "") for e in entries}

            # Discover ZIPs in REPORTS_DIR that are not yet in the index
            new_zips = [
                p for p in REPORTS_DIR.glob("*.zip")
                if str(p.resolve()) not in known_paths
            ]
            for zip_path in new_zips:
                entries.append({
                    "zip_path": str(zip_path.resolve()),
                    "base_dir": str(zip_path.parent),
                    "subpaths": [],
                    "loader": "",
                    "file_extensions": "",
                    "created_at": datetime.fromtimestamp(zip_path.stat().st_mtime).isoformat(),
                })

            total = len(entries)
            for i, entry in enumerate(entries):
                zip_path = entry.get("zip_path", "")
                base_dir = entry.get("base_dir", "")
                if not Path(zip_path).exists():
                    continue
                try:
                    project = api.import_project(Path(zip_path))
                    meta = _extract_metadata_from_records(project.records_df, base_dir)
                    entry.update(meta)
                except Exception as _e:
                    logger.warning("Could not rebuild metadata for %s: %s", zip_path, _e)
                pct = 5 + int((i + 1) / max(total, 1) * 90)
                with _processing_lock:
                    _processing_state.update(
                        progress=pct,
                        message=f"Processing {i + 1}/{total}...",
                    )
            entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
            _save_index(entries)
            n_new = len(new_zips)
            msg = f"Done. {n_new} new report{'s' if n_new != 1 else ''} added." if n_new else "Overview metadata recalculated."
            with _processing_lock:
                _processing_state.update(
                    status="completed", progress=100,
                    message=msg,
                    output_zip="__rebuild__",
                )
        except Exception as e:
            logger.exception("Rebuild metadata failed")
            with _processing_lock:
                _processing_state.update(status="error", error=str(e))

    threading.Thread(target=_run, daemon=True).start()


def _run_processing_async(
    base_directory: str,
    output_zip: str,
    path_list: List[str],
    loader: Optional[str],
    file_extensions: str | set,
):
    """Run processing in background thread."""
    ext_for_settings = file_extensions if isinstance(file_extensions, str) else ",".join(sorted(file_extensions))

    # Update state synchronously so the interval picks up "running" on its
    # first fire instead of reading the old "idle" and resetting the banner.
    with _processing_lock:
        _processing_state.update(status="running", progress=5, message="Starting...", error=None, output_zip=None)

    def _run():
        try:
            base_dir = Path(base_directory).resolve()
            if not base_dir.exists():
                with _processing_lock:
                    _processing_state.update(status="error", error=f"Base directory does not exist: {base_directory}")
                return

            project_name = base_dir.name or "project"
            project = api.create_project(project_name, str(base_dir), loader=loader)

            if path_list:
                api.add_paths(project, path_list)
            else:
                api.add_paths(project, base_dir)

            settings = Settings(cmap="rainbow", n_example_files=9, selected_file_extensions=file_extensions, pixel_patrol_flavor="")
            api.set_settings(project, settings)

            with _processing_lock:
                _processing_state.update(progress=10, message="Processing files...")

            basic_df = walk_filesystem(
                project.get_paths() or [project.get_base_dir()],
                loader=project.get_loader(),
                accepted_extensions=file_extensions if isinstance(file_extensions, set) else "all",
            )
            total_files = basic_df.filter(pl.col("type") == "file").height if basic_df is not None and not basic_df.is_empty() else 0

            def progress_cb(current: int, total: int, cf: Path):
                pct = 10 + int((current / total) * 80) if total > 0 else 10
                with _processing_lock:
                    _processing_state.update(progress=pct, message=f"Processing {current}/{total}...")

            api.process_files(project, progress_callback=progress_cb)

            output_path = Path(output_zip).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            api.export_project(project, output_path)

            # Extract rich metadata for the index card.
            try:
                meta = _extract_metadata_from_records(project.records_df, base_directory)
            except Exception as _stats_err:
                logger.debug("Could not extract index metadata: %s", _stats_err)
                meta = {}

            # Write index entry from the thread so it's always persisted,
            # regardless of callback timing on the Dash side.
            _add_report_to_index(
                zip_path=str(output_path),
                base_dir=base_directory,
                subpaths=path_list,
                loader=loader or "",
                file_extensions=ext_for_settings,
                **meta,
            )

            with _processing_lock:
                _processing_state.update(
                    status="completed",
                    progress=100,
                    message=f"Exported to {output_path}",
                    output_zip=str(output_path),
                )
        except Exception as e:
            logger.exception("Processing failed")
            with _processing_lock:
                _processing_state.update(status="error", error=str(e))

    threading.Thread(target=_run, daemon=True).start()
