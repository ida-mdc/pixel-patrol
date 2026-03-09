"""Single-file inspection Dash app.

Shows a compact, informative report for a single image file.

Called by ``pixel-patrol inspect``.
"""
from __future__ import annotations

import base64
import io
import re
import threading
import time
from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
import numpy as np
import polars as pl
from dash import Dash, Input, Output, dcc, html

from pixel_patrol_base.report.constants import (
    FILTERED_INDICES_STORE_ID,
    GLOBAL_CONFIG_STORE_ID,
    GC_DIMENSIONS,
    GC_FILTER,
    GC_GROUP_COL,
    GC_IS_SHOW_SIGNIFICANCE,
    NO_GROUPING_COL,
    NO_GROUPING_LABEL,
)

ASSETS_DIR = (Path(__file__).parent / "assets").resolve()

# Must match pixel_patrol_base.config.SPRITE_SIZE
_THUMBNAIL_SIZE = 64


_CHANNEL_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756",
    "#B07AA1", "#FF9DA6", "#9D755D", "#BAB0AC",
]

_QUALITY_METRIC_LABELS = {
    "laplacian_variance": "Laplacian variance",
    "tenengrad":          "Tenengrad",
    "brenner":            "Brenner",
    "noise_std":          "Noise (std)",
    "blocking_records":   "Blocking",
    "ringing_records":    "Ringing",
}

_QUALITY_METRIC_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#B07AA1", "#9D755D",
]


# ── Public API ──────────────────────────────────────────────────────────────────

def create_inspect_app(path: Path, loader: Any, processors: list) -> Dash:
    """Create a Dash app that processes *path* in the background and shows results.

    The window opens immediately with a loading screen; once processing finishes
    the report is rendered without requiring a page reload.
    """
    import datetime
    from pixel_patrol_base.core.processing import load_and_process_records_from_file

    filename = path.name
    _state: dict[str, Any] = {
        "status":  "loading",   # "loading" | "done" | "error"
        "message": f"Processing {filename}…",
        "df":      None,
        "layout":  None,        # pre-built report layout (avoids blocking Flask)
        "error":   None,
    }

    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css",
        ],
        assets_folder=str(ASSETS_DIR),
        suppress_callback_exceptions=True,
    )

    # No watchdog needed: pywebview owns the lifecycle. When the window is
    # closed webview.start() returns, the daemon Flask thread dies, and the
    # process exits. In browser mode (remote host) the user terminates with
    # Ctrl-C, same as `pixel-patrol launch`.

    # ── Background processing ───────────────────────────────────────────────
    def _process() -> None:
        try:
            records = load_and_process_records_from_file(path, loader, processors)
            if not records:
                _state["error"]  = f"No data could be read from '{filename}'."
                _state["status"] = "error"
                return

            stat   = path.stat()
            size   = stat.st_size
            fs_mdate = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
            for r in records:
                r.setdefault("path",           str(path))
                r.setdefault("name",           filename)
                r.setdefault("file_extension", path.suffix.lstrip("."))
                r.setdefault("size",           size)
                r.setdefault("size_readable",  _fmt_bytes(size))
                if not r.get("modification_date"):
                    r["modification_date"] = fs_mdate

            df = pl.DataFrame(records, nan_to_null=True,
                              strict=False, infer_schema_length=None)
            _state["df"]      = df
            _state["message"] = "Building report…"
            try:
                _state["layout"] = _report_layout(app, filename, df)
            except Exception:
                import traceback
                _state["error"]  = traceback.format_exc()
                _state["status"] = "error"
                return
            _state["status"] = "done"
        except Exception:
            import traceback
            _state["error"]  = traceback.format_exc()
            _state["status"] = "error"

    threading.Thread(target=_process, daemon=True).start()

    # ── Shell layout (always present so interval IDs are stable) ────────────
    app.layout = html.Div([
        dcc.Interval(id="_poll-interval", interval=500, n_intervals=0,
                     max_intervals=-1),
        html.Div(id="_root"),
    ])

    # ── Polling callback ────────────────────────────────────────────────────
    @app.callback(
        Output("_root", "children"),
        Output("_poll-interval", "max_intervals"),
        Input("_poll-interval", "n_intervals"),
    )
    def _poll(n: int):
        if _state["status"] == "error":
            return _error_layout(app, filename, _state["error"]), 0
        if _state["status"] == "done":
            return _state["layout"], 0
        return _loading_layout(app, filename, _state["message"]), -1

    return app


def _loading_layout(app: Dash, filename: str, message: str) -> html.Div:
    return html.Div([
        _navbar(app, filename),
        html.Div(
            html.Div([
                dbc.Spinner(color="primary", size="lg"),
                html.P(message, className="mt-3 text-muted",
                       style={"fontSize": "0.9rem"}),
            ], className="d-flex flex-column align-items-center justify-content-center",
               style={"gap": "8px"}),
            style={"flex": "1", "display": "flex",
                   "alignItems": "center", "justifyContent": "center"},
        ),
        _footer(app),
    ], style={"minHeight": "100vh", "display": "flex", "flexDirection": "column"})


def _error_layout(app: Dash, filename: str, error: str) -> html.Div:
    return html.Div([
        _navbar(app, filename),
        html.Div(
            dbc.Alert([
                html.H5("Could not process file", className="alert-heading"),
                html.P(error, className="mb-0 font-monospace",
                       style={"fontSize": "0.85rem", "whiteSpace": "pre-wrap"}),
            ], color="danger", style={"maxWidth": "600px"}),
            style={"flex": "1", "display": "flex",
                   "alignItems": "center", "justifyContent": "center",
                   "padding": "32px"},
        ),
        _footer(app),
    ], style={"minHeight": "100vh", "display": "flex", "flexDirection": "column"})


def _report_layout(app: Dash, filename: str, df: pl.DataFrame) -> html.Div:
    initial_global_config = {
        GC_GROUP_COL:            NO_GROUPING_COL,
        GC_FILTER:               {},
        GC_DIMENSIONS:           {},
        GC_IS_SHOW_SIGNIFICANCE: False,
    }
    row = df.row(0, named=True)
    return html.Div([
        dcc.Store(id=GLOBAL_CONFIG_STORE_ID,    data=initial_global_config),
        dcc.Store(id=FILTERED_INDICES_STORE_ID, data=list(range(len(df)))),
        dcc.Store(id="color-map-store",         data={NO_GROUPING_LABEL: "#4C78A8"}),

        _navbar(app, filename),

        dbc.Container(
            [
                html.Div(className="my-3"),

                # Row 1: file info (md=7) | intensity summary (md=5)
                dbc.Row([
                    dbc.Col(_file_info_card(row), md=7, className="d-flex"),
                    dbc.Col(_right_summary_panel(row), md=5, className="d-flex"),
                ], className="g-3"),

                html.Div(className="my-3"),

                # Row 2: histograms (md=7) | quality metrics (md=5)
                *_histogram_quality_row(row),

                # Full-width: intensity across slices
                *_optional([_sliced_intensity_card(row)]),

                # Full-width: quality across slices
                *_optional([_sliced_quality_card(row)]),
            ],
            fluid=True,
            style={"maxWidth": "1200px", "margin": "0 auto",
                   "paddingBottom": "32px", "flex": "1"},
        ),

        _footer(app),
    ], style={"minHeight": "100vh", "display": "flex", "flexDirection": "column"})


# ── Navbar ──────────────────────────────────────────────────────────────────────

def _navbar(app: Dash, filename: str) -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(
                        src=app.get_asset_url("icon.png"),
                        style={"height": "28px", "marginRight": "8px"},
                    )),
                    dbc.Col(dbc.NavbarBrand(
                        "Pixel Patrol", className="ms-1 fs-5 mb-0",
                        style={"color": "#212529"},
                    )),
                ], align="center", className="g-0"),
                style={"textDecoration": "none"},
            ),
            dbc.Nav(
                html.Span(filename, style={"color": "#6c757d", "fontSize": "0.9rem"}),
                className="ms-auto", navbar=True,
            ),
        ], fluid=True),
        color="white", dark=False, className="mb-0",
        style={"borderBottom": "1px solid #dee2e6"},
    )


# ── File info card (thumbnail inline) ──────────────────────────────────────────

def _file_info_card(row: dict[str, Any]) -> dbc.Card:
    return dbc.Card(dbc.CardBody(
        dbc.Row([
            dbc.Col(_thumbnail_widget(row), width="auto",
                    className="d-flex align-items-start pe-3"),
            dbc.Col(_metadata_table(row), width=True),
        ], className="g-0"),
    ), className="h-100 w-100")


def _thumbnail_widget(row: dict[str, Any]) -> html.Div:
    thumbnail = row.get("thumbnail")
    src = None
    if thumbnail is not None:
        try:
            arr = np.array(thumbnail, dtype=np.uint8).reshape(
                _THUMBNAIL_SIZE, _THUMBNAIL_SIZE
            )
            buf = io.BytesIO()
            from PIL import Image
            Image.fromarray(arr, mode="L").save(buf, format="PNG")
            src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            src = None

    shared = {"width": f"{_THUMBNAIL_SIZE}px", "height": f"{_THUMBNAIL_SIZE}px",
               "borderRadius": "4px", "border": "1px solid #dee2e6"}
    if src:
        inner = html.Img(src=src, style={**shared, "imageRendering": "pixelated",
                                          "display": "block"})
    else:
        inner = html.Div(
            html.I(className="bi bi-image text-muted", style={"fontSize": "2.5rem"}),
            style={**shared, "display": "flex", "alignItems": "center",
                   "justifyContent": "center", "background": "#f8f9fa"},
        )
    return html.Div(inner)


def _metadata_table(row: dict[str, Any]) -> html.Div:
    # Build axis sizes from the actual dim_order so all dimensions are shown.
    dim_order_str = row.get("dim_order") or ""
    axis_sizes = {
        ax: row.get(f"{ax}_size")
        for ax in dim_order_str
        if row.get(f"{ax}_size") is not None
    }
    if not axis_sizes:
        # Fallback: scan for any *_size columns
        axis_sizes = {
            col.removesuffix("_size").upper(): val
            for col, val in row.items()
            if col.endswith("_size") and val is not None and len(col) <= 7
        }
    s   = {"fontSize": "0.84rem", "paddingTop": "3px", "paddingBottom": "3px"}
    lbl = {**s, "color": "#6c757d", "width": "130px", "fontWeight": "normal"}

    def _tr(label, content):
        return html.Tr([html.Th(label, style=lbl), html.Td(content, style=s)])

    rows = [_tr("Filename", html.Strong(row.get("name", "–")))]
    rows.append(_tr("Format",    (row.get("file_extension") or "–").upper()))
    size = row.get("size")
    rows.append(_tr("File size", _fmt_bytes(size) if size else "–"))

    if axis_sizes:
        dim_pills = [
            dbc.Badge(f"{ax}: {int(sz)}",
                      color="primary" if ax in ("X", "Y") else "secondary",
                      className="me-1", style={"fontSize": "0.75rem"})
            for ax, sz in axis_sizes.items()
        ]
        rows.append(_tr("Dimensions", html.Div(dim_pills, className="d-flex flex-wrap gap-1")))
    elif row.get("shape"):
        rows.append(_tr("Shape", " × ".join(str(v) for v in row["shape"])))

    dim_order = row.get("dim_order") or ""
    if dim_order:
        rows.append(_tr("Axis order", html.Code(dim_order, style={"fontSize": "0.8rem"})))
    dtype = row.get("dtype")
    if dtype:
        rows.append(_tr("Data type", html.Code(str(dtype), style={"fontSize": "0.8rem"})))
    mod = row.get("modification_date")
    if mod:
        rows.append(_tr("Modified", str(mod)[:19].replace("T", " ")))

    return html.Div([
        _section_label("File Information"),
        html.Table(html.Tbody(rows), className="table table-sm table-borderless mb-0"),
    ])


# ── Right summary panel: intensity + quality stacked ───────────────────────────

def _right_summary_panel(row: dict[str, Any]) -> html.Div:
    card = _intensity_summary_card(row)
    return html.Div(card, className="h-100 w-100") if card is not None else html.Div()


# ── Intensity summary — metrics as rows, channels as columns ───────────────────

_INTENSITY_METRICS = [
    ("mean_intensity", "Mean",  np.mean),
    ("std_intensity",  "Std",   np.mean),
    ("min_intensity",  "Min",   np.min),
    ("max_intensity",  "Max",   np.max),
]


def _parse_sliced_cols(row: dict, metrics: list[str]) -> list[dict]:
    """Return parsed info for every sliced metric column in *row*.

    Uses the same ``parse_metric_dimension_column`` logic as the dataset widgets
    so behaviour is fully dimension-agnostic.

    Each entry: {"col": str, "metric": str, "dims": {dim_letter: int}, "val": float}
    """
    from pixel_patrol_base.report.data_utils import parse_metric_dimension_column  # lazy — avoids circular import
    result = []
    for col, val in row.items():
        if val is None:
            continue
        parsed = parse_metric_dimension_column(col, metrics)
        if parsed is None or not parsed[1]:   # no dims → scalar aggregate, skip
            continue
        metric, dims = parsed
        try:
            result.append({"col": col, "metric": metric, "dims": dims, "val": float(val)})
        except (TypeError, ValueError):
            pass
    return result


def _free_dims(parsed: list[dict], min_slices: int = 2) -> list[str]:
    """Return dim letters that appear as the *only* dim in ≥ min_slices 1-dim cols."""
    counts: dict[str, set] = {}
    for p in parsed:
        if len(p["dims"]) == 1:
            d, idx = next(iter(p["dims"].items()))
            counts.setdefault(d, set()).add(idx)
    return sorted(d for d, idxs in counts.items() if len(idxs) >= min_slices)


def _channel_stats(row: dict[str, Any]) -> list[dict]:
    """Collect per-channel intensity summary using dimension-agnostic column parsing."""
    channel_names: list = row.get("channel_names") or []

    def _ch_label(ci: int) -> str:
        return channel_names[ci] if ci < len(channel_names) else f"C{ci}"

    metrics = [name for name, _, _ in _INTENSITY_METRICS]
    parsed = _parse_sliced_cols(row, metrics)

    # Find 1-dim "c" columns (per-channel aggregates)
    c_cols: dict[int, dict[str, float]] = {}
    for p in parsed:
        if list(p["dims"].keys()) == ["c"]:
            ci = p["dims"]["c"]
            c_cols.setdefault(ci, {})[p["metric"]] = p["val"]

    if c_cols:
        results = []
        for ci in sorted(c_cols):
            entry = {"channel": ci, "label": _ch_label(ci)}
            for metric, label, _ in _INTENSITY_METRICS:
                entry[label] = c_cols[ci].get(metric)
            results.append(entry)
        return results

    # Fallback: use the scalar aggregate (no channel dim)
    overall: dict[str, float] = {}
    for col, val in row.items():
        if val is None:
            continue
        for metric, _, _ in _INTENSITY_METRICS:
            if col == metric:
                try:
                    overall[metric] = float(val)
                except (TypeError, ValueError):
                    pass
    if overall:
        entry = {"channel": "overall", "label": "Overall"}
        for _, label, _ in _INTENSITY_METRICS:
            pass  # labels not used here
        for metric, label, _ in _INTENSITY_METRICS:
            entry[label] = overall.get(metric)
        return [entry]

    return []


def _intensity_summary_card(row: dict[str, Any]) -> dbc.Card | None:
    channels = _channel_stats(row)
    if not channels:
        return None

    stat_labels = [label for _, label, _ in _INTENSITY_METRICS]
    s    = {"fontSize": "0.83rem", "paddingTop": "2px", "paddingBottom": "2px"}
    lbl  = {**s, "color": "#6c757d", "fontWeight": "normal"}
    mono = {**s, "fontFamily": "monospace", "textAlign": "right"}
    multi = len(channels) > 1

    def _ch_color(ch: dict) -> str:
        idx = ch["channel"] if isinstance(ch["channel"], int) else 0
        return _CHANNEL_COLORS[idx % len(_CHANNEL_COLORS)]

    header_cells = [html.Th("", style=lbl)]
    for ch in channels:
        style = {**s, "textAlign": "right",
                 "color": _ch_color(ch) if multi else "#212529"}
        header_cells.append(html.Th(ch["label"], style=style))

    body_rows = []
    for ml in stat_labels:
        cells = [html.Td(ml, style=lbl)]
        for ch in channels:
            cells.append(html.Td(_fmt_float(ch.get(ml)), style=mono))
        body_rows.append(html.Tr(cells))

    return dbc.Card(dbc.CardBody([
        _section_label("Intensity Summary"),
        html.Table(
            [html.Thead(html.Tr(header_cells)), html.Tbody(body_rows)],
            className="table table-sm table-hover mb-0",
        ),
    ]), className="h-100 w-100")


# ── Quality metrics summary ──────────────────────────────────────────────────────

def _quality_metrics_card(row: dict[str, Any]) -> dbc.Card | None:
    rows = []
    s    = {"fontSize": "0.83rem", "paddingTop": "2px", "paddingBottom": "2px"}
    lbl  = {**s, "color": "#6c757d"}
    mono = {**s, "fontFamily": "monospace", "textAlign": "right"}

    for base, display in _QUALITY_METRIC_LABELS.items():
        vals = [
            float(v) for col, v in row.items()
            if (col == base or re.match(rf"^{base}_", col)) and v is not None
        ]
        if not vals:
            continue
        rows.append(html.Tr([
            html.Td(display, style=lbl),
            html.Td(_fmt_float(float(np.mean(vals))), style=mono),
        ]))

    if not rows:
        return None

    return dbc.Card(dbc.CardBody([
        _section_label("Quality Metrics"),
        html.Table(html.Tbody(rows), className="table table-sm mb-0"),
    ]), className="h-100 w-100")


# ── Row 2: histograms (md=7) | quality metrics (md=5) ─────────────────────────

def _histogram_quality_row(row: dict[str, Any]) -> list:
    hist  = _histograms_card(row)
    qual  = _quality_metrics_card(row)
    if hist and qual:
        return [dbc.Row([
            dbc.Col(hist, md=7, className="d-flex"),
            dbc.Col(qual, md=5, className="d-flex"),
        ], className="g-3")]
    if hist:
        return [hist]
    if qual:
        return [qual]
    return []


def _histograms_card(row: dict[str, Any]) -> dbc.Card | None:
    import plotly.graph_objects as go

    channel_names: list = row.get("channel_names") or []

    def _ch_label(ci: int) -> str:
        return channel_names[ci] if ci < len(channel_names) else f"C{ci}"

    def _bar_fig(counts, hist_min, hist_max, color: str, title: str):
        if counts is None or hist_min is None or hist_max is None:
            return None
        c = np.array(counts, dtype=np.float64)
        if c.sum() == 0:
            return None
        bins = np.linspace(float(hist_min), float(hist_max), len(c))
        fig = go.Figure(go.Bar(x=bins.tolist(), y=c.tolist(),
                               marker_color=color, marker_line_width=0))
        fig.update_layout(
            title=dict(text=title, font=dict(size=12), x=0.5),
            margin=dict(l=30, r=10, t=32, b=30),
            height=190, bargap=0,
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False,
                       range=[float(hist_min), float(hist_max)]),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        )
        return fig

    graphs = []
    ch_indices = sorted(
        int(m.group(1))
        for col in row
        if (m := re.fullmatch(r"histogram_counts_c(\d+)", col))
    )
    if ch_indices:
        for ci in ch_indices:
            fig = _bar_fig(
                row.get(f"histogram_counts_c{ci}"),
                row.get(f"histogram_min_c{ci}"),
                row.get(f"histogram_max_c{ci}"),
                color=_CHANNEL_COLORS[ci % len(_CHANNEL_COLORS)],
                title=_ch_label(ci),
            )
            if fig:
                graphs.append(dbc.Col(
                    dcc.Graph(figure=fig, config={"displayModeBar": False}),
                    width=12, md=6,
                ))
    else:
        counts_col = next(
            (col for col in row
             if re.match(r"^histogram_counts(_z\d+)?$", col) and row[col] is not None),
            None,
        )
        if counts_col:
            suffix = counts_col[len("histogram_counts"):]
            fig = _bar_fig(
                row.get(counts_col),
                row.get(f"histogram_min{suffix}"),
                row.get(f"histogram_max{suffix}"),
                color=_CHANNEL_COLORS[0], title="Intensity",
            )
            if fig:
                graphs.append(dbc.Col(
                    dcc.Graph(figure=fig, config={"displayModeBar": False}),
                    width=12,
                ))

    if not graphs:
        return None

    return dbc.Card(dbc.CardBody([
        _section_label("Pixel Value Histograms"),
        dbc.Row(graphs, className="g-3"),
    ]), className="h-100 w-100")


# ── Sliced intensity (mean ± std band + min/max outer band) ────────────────────

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _build_intensity_fig(
    parsed: list[dict],
    free_dim: str,
    channel_names: list,
) -> "go.Figure | None":
    """Build a mean ± std / min–max band figure for *free_dim*."""
    import plotly.graph_objects as go

    def _ch_label(ci: int) -> str:
        return channel_names[ci] if ci < len(channel_names) else f"C{ci}"

    # Prefer 2-dim (c, free_dim) columns for per-channel traces; fall back to 1-dim.
    two_dim = [p for p in parsed if set(p["dims"].keys()) == {"c", free_dim}]
    one_dim = [p for p in parsed if list(p["dims"].keys()) == [free_dim]]

    def _build_series(entries: list[dict]) -> dict[str, dict[str, dict[int, float]]]:
        """Returns {ch_key: {metric: {slice_idx: val}}}."""
        out: dict[str, dict[str, dict[int, float]]] = {}
        for p in entries:
            ch_key = f"c{p['dims']['c']}" if "c" in p["dims"] else ""
            idx = p["dims"][free_dim]
            out.setdefault(ch_key, {}).setdefault(p["metric"], {})[idx] = p["val"]
        return out

    series = _build_series(two_dim) if two_dim else _build_series(one_dim)
    if not series:
        return None

    fig = go.Figure()
    multi = len(series) > 1

    for ch_key in sorted(series):
        ci = int(ch_key[1:]) if ch_key else 0
        color = _CHANNEL_COLORS[ci % len(_CHANNEL_COLORS)]
        name  = _ch_label(ci) if ch_key else "Mean"
        m_by_idx = series[ch_key].get("mean_intensity", {})
        if not m_by_idx:
            continue
        xs    = sorted(m_by_idx)
        means = [m_by_idx[x] for x in xs]

        def _aligned(metric: str) -> list[float] | None:
            d = series[ch_key].get(metric, {})
            vals = [d.get(x) for x in xs]
            return vals if all(v is not None for v in vals) else None

        mins = _aligned("min_intensity")
        maxs = _aligned("max_intensity")
        stds = _aligned("std_intensity")

        if mins and maxs:
            fig.add_trace(go.Scatter(
                x=xs + xs[::-1], y=maxs + mins[::-1],
                fill="toself", fillcolor=_hex_to_rgba(color, 0.12),
                line=dict(width=0), showlegend=False, hoverinfo="skip",
                legendgroup=name,
            ))
        if stds:
            upper = [m + s for m, s in zip(means, stds)]
            lower = [m - s for m, s in zip(means, stds)]
            fig.add_trace(go.Scatter(
                x=xs + xs[::-1], y=upper + lower[::-1],
                fill="toself", fillcolor=_hex_to_rgba(color, 0.28),
                line=dict(width=0), showlegend=False, hoverinfo="skip",
                legendgroup=name,
            ))
        fig.add_trace(go.Scatter(
            x=xs, y=means, mode="lines+markers", name=name,
            line=dict(color=color, width=2), marker=dict(size=4),
            showlegend=multi, legendgroup=name,
        ))

    if not fig.data:
        return None

    fig.update_layout(
        **_slice_chart_layout(free_dim.upper(), "Intensity"),
        showlegend=multi,
        annotations=[dict(
            text="Bands: ±std (inner) · min–max (outer)",
            xref="paper", yref="paper", x=1, y=1.02,
            xanchor="right", yanchor="bottom", showarrow=False,
            font=dict(size=10, color="#adb5bd"),
        )],
    )
    return fig


def _sliced_intensity_card(row: dict[str, Any]) -> dbc.Card | None:
    intensity_metrics = ["mean_intensity", "std_intensity", "min_intensity", "max_intensity"]
    parsed = _parse_sliced_cols(row, intensity_metrics)
    dims   = _free_dims(parsed)
    if not dims:
        return None

    channel_names: list = row.get("channel_names") or []
    graphs = []
    for d in dims:
        fig = _build_intensity_fig(parsed, d, channel_names)
        if fig:
            graphs.append(dbc.Col(
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                width=12,
            ))

    if not graphs:
        return None

    return dbc.Card(dbc.CardBody([
        _section_label("Intensity across slices"),
        dbc.Row(graphs, className="g-3"),
    ]))


# ── Sliced quality metrics (one mini-chart per metric per free dim) ─────────────


def _sliced_quality_card(row: dict[str, Any]) -> dbc.Card | None:
    import plotly.graph_objects as go

    channel_names: list = row.get("channel_names") or []

    def _ch_label(ci: int) -> str:
        return channel_names[ci] if ci < len(channel_names) else f"C{ci}"

    parsed = _parse_sliced_cols(row, list(_QUALITY_METRIC_LABELS.keys()))
    dims   = _free_dims(parsed)
    if not dims:
        return None

    graphs = []
    for (base, display), q_color in zip(_QUALITY_METRIC_LABELS.items(), _QUALITY_METRIC_COLORS):
        metric_parsed = [p for p in parsed if p["metric"] == base]
        if not metric_parsed:
            continue
        for free_dim in dims:
            # prefer 2-dim (c, free_dim); fallback to 1-dim (free_dim)
            entries = [p for p in metric_parsed if set(p["dims"].keys()) == {"c", free_dim}] \
                   or [p for p in metric_parsed if list(p["dims"].keys()) == [free_dim]]
            if not entries:
                continue

            # group by channel
            by_ch: dict[str, dict[int, float]] = {}
            for p in entries:
                ch_key = f"c{p['dims']['c']}" if "c" in p["dims"] else ""
                by_ch.setdefault(ch_key, {})[p["dims"][free_dim]] = p["val"]

            if not by_ch:
                continue

            fig = go.Figure()
            multi = len(by_ch) > 1
            for ch_key in sorted(by_ch):
                ci = int(ch_key[1:]) if ch_key else 0
                color = _CHANNEL_COLORS[ci % len(_CHANNEL_COLORS)] if multi else q_color
                name  = _ch_label(ci) if ch_key else display
                pts   = sorted(by_ch[ch_key].items())
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in pts], y=[p[1] for p in pts],
                    mode="lines+markers", name=name, showlegend=multi,
                    line=dict(color=color, width=2), marker=dict(size=4),
                ))

            title = f"{display} / {free_dim.upper()}" if len(dims) > 1 else display
            fig.update_layout(
                title=dict(text=title, font=dict(size=11), x=0.5),
                **_slice_chart_layout(free_dim.upper(), "", height=170,
                                      margin=dict(l=40, r=10, t=30, b=45)),
                showlegend=multi,
            )
            graphs.append(dbc.Col(
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                width=12, md=4,
            ))

    if not graphs:
        return None

    return dbc.Card(dbc.CardBody([
        _section_label("Quality metrics across slices"),
        dbc.Row(graphs, className="g-3"),
    ]))


# ── Footer ──────────────────────────────────────────────────────────────────────

def _footer(app: Dash) -> html.Footer:
    return html.Footer(
        dbc.Container(
            dbc.Row([
                dbc.Col(
                    html.A(
                        html.Img(src=app.get_asset_url("Helmholtz-Imaging_Mark.png"),
                                 style={"height": "28px",
                                        "filter": "brightness(0) invert(1)"}),
                        href="https://helmholtz-imaging.de", target="_blank",
                    ),
                    width="auto", className="d-flex align-items-center",
                ),
                dbc.Col(
                    html.A("Helmholtz Imaging", href="https://helmholtz-imaging.de",
                           target="_blank",
                           style={"color": "rgba(255,255,255,0.7)",
                                  "textDecoration": "none"}),
                    width="auto", className="d-flex align-items-center",
                ),
            ], align="center", className="g-2"),
            fluid=True,
        ),
        style={"backgroundColor": "#212529", "padding": "14px 24px", "marginTop": "32px"},
    )


# ── Shared helpers ───────────────────────────────────────────────────────────────

def _slice_chart_layout(x_title: str, y_title: str, height: int = 230,
                        margin: dict | None = None) -> dict:
    """Common layout kwargs for slice line charts."""
    return dict(
        height=height,
        margin=margin or dict(l=50, r=10, t=10, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            title=x_title, showgrid=False, zeroline=False,
            type="linear", tickangle=-45, automargin=True,
        ),
        yaxis=dict(
            title=y_title, showgrid=True, gridcolor="#f0f0f0",
            zeroline=False, automargin=True,
        ),
    )


def _section_label(text: str) -> html.Div:
    return html.Div(text, className="mb-2 fw-semibold text-secondary",
                    style={"fontSize": "0.7rem", "letterSpacing": "0.06em",
                           "textTransform": "uppercase"})


def _optional(components: list) -> list:
    filtered = [c for c in components if c is not None]
    if not filtered:
        return []
    return [html.Div(className="my-3"), *filtered]


# ── Utilities ────────────────────────────────────────────────────────────────────

def _fmt_float(v: Any) -> str:
    if v is None:
        return "–"
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
