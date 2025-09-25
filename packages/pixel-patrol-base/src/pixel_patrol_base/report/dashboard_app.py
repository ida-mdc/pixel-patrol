import re
from pathlib import Path
from typing import List, Dict, Sequence

import dash_bootstrap_components as dbc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import polars as pl
from dash import Dash, html, dcc
from dash.dependencies import Input, Output

import plotly.io as pio

from pixel_patrol_base.core.contracts import PixelPatrolWidget
from pixel_patrol_base.core.project import Project
from pixel_patrol_base.plugin_registry import discover_widget_plugins
from pixel_patrol_base.report.widget import organize_widgets_by_tab


def load_and_concat_parquets(paths: List[str]) -> pl.DataFrame:
    """Read parquet files/dirs and concatenate into one DataFrame."""
    dfs = []
    for base_str in paths:
        base = Path(base_str)
        files = sorted(base.rglob("*.parquet")) if base.is_dir() else []
        if base.is_file() and base.suffix == ".parquet":
            files = [base]
        for file in files:
            dfs.append(pl.read_parquet(file))
    return pl.concat(dfs, how="diagonal", rechunk=True) if dfs else pl.DataFrame()


def create_app(project: Project) -> Dash:
    return _create_app(
        project.records_df,
        project.get_settings().cmap,
        pixel_patrol_flavor=project.get_settings().pixel_patrol_flavor,
    )


def _create_app(
    df: pl.DataFrame,
    default_palette_name: str = "tab10",
    pixel_patrol_flavor: str = "",
) -> Dash:
    """Instantiate Dash app, register callbacks, and assign layout."""
    df = df.with_row_index(name="unique_id")
    external_stylesheets = [dbc.themes.BOOTSTRAP, "https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

    pio.templates.default = "plotly"

    # Discover widget instances (new or legacy)
    group_widgets: List[PixelPatrolWidget] = discover_widget_plugins()

    # ✅ prefer new API; fallback to legacy register_callbacks for older widgets
    for w in group_widgets:
        if hasattr(w, "register"):
            w.register(app, df)
        elif hasattr(w, "register_callbacks"):
            w.register_callbacks(app, df)

    def serve_layout_closure() -> html.Div:
        DEFAULT_WIDGET_WIDTH = 12

        # --- Header ---
        header_row = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Img(
                                src=app.get_asset_url("prevalidation.png"),
                                style={"height": "110px", "marginRight": "15px"},
                            ),
                            html.H1("Pixel Patrol", className="m-0"),
                            html.Span(
                                pixel_patrol_flavor,
                                style={
                                    "color": "#d9534f",
                                    "fontSize": "2rem",
                                    "fontWeight": "bold",
                                    "fontFamily": "cursive",
                                    "transform": "rotate(-6deg)",
                                    "marginLeft": "15px",
                                    "marginTop": "10px",
                                },
                            ),
                            dbc.Col(html.Div(), width="auto"),
                        ],
                        className="d-flex align-items-center",
                    ),
                    width=True,
                ),
                dbc.Col(
                    html.Div(
                        [
                            "This is a prototype. Data may be incomplete or inaccurate.",
                            html.Br(),
                            "Use for experimental purposes only.",
                        ],
                        style={"color": "#d9534f", "textAlign": "right"},
                    ),
                    width="auto",
                ),
            ],
            align="center",
            className="my-3",
        )

        settings_controls = dbc.Row(
            [
                dbc.Col(html.Div(), width=True),
                dbc.Col(
                    html.Div(
                        [
                            html.Label("Color Palette:", className="me-2"),
                            dcc.Dropdown(
                                id="palette-selector",
                                options=[{"label": name, "value": name} for name in sorted(plt.colormaps())],
                                value=default_palette_name,
                                clearable=False,
                                style={"width": "200px"},
                            ),
                        ],
                        className="d-flex align-items-center justify-content-end",
                    ),
                    width="auto",
                ),
            ]
        )

        # --- Group Widget Layout Generation ---
        group_widget_content = []
        tabbed_group_widgets = organize_widgets_by_tab(group_widgets)  # assumes it looks at NAME/TAB

        for group_name, ws in tabbed_group_widgets.items():
            group_widget_content.append(dbc.Row(dbc.Col(html.H3(group_name, className="my-3 text-primary"))))
            current_group_cols = []
            current_row_width = 0

            for w in ws:
                if should_display_widget(w, df.columns):
                    widget_width = getattr(w, "width", DEFAULT_WIDGET_WIDTH)

                    # wrap to next row if needed
                    if current_row_width + widget_width > 12:
                        group_widget_content.append(dbc.Row(current_group_cols, className="g-4 p-3"))
                        current_group_cols, current_row_width = [], 0

                    # Title + body
                    title = getattr(w, "NAME", w.__class__.__name__)
                    current_group_cols.append(dbc.Row(dbc.Col(html.H4(title, className="my-3 text-primary"))))
                    current_group_cols.append(
                        dbc.Col(html.Div(w.layout()), width=widget_width, className="mb-3")
                    )
                    current_row_width += widget_width

            if current_group_cols:
                group_widget_content.append(dbc.Row(current_group_cols, className="g-4 p-3"))

        # --- Data Stores ---
        stores = html.Div(
            [
                dcc.Store(id="color-map-store"),
                dcc.Store(id="tb-process-store-tensorboard-embedding-projector", data={}),
            ]
        )

        # --- Final Layout Assembly ---
        return html.Div(
            dbc.Container(
                [header_row, stores, settings_controls, html.Hr(), *group_widget_content],
                fluid=True,
                style={"maxWidth": "1200px", "margin": "0 auto"},
            )
        )

    app.layout = serve_layout_closure

    @app.callback(Output("color-map-store", "data"), Input("palette-selector", "value"))
    def update_color_map(palette: str) -> Dict[str, str]:
        folders = df.select(pl.col("imported_path_short")).unique().to_series().to_list()
        cmap = cm.get_cmap(palette, len(folders))
        return {
            f: f"#{int(cmap(i)[0] * 255):02x}{int(cmap(i)[1] * 255):02x}{int(cmap(i)[2] * 255):02x}"
            for i, f in enumerate(folders)
        }

    return app


def should_display_widget(widget: PixelPatrolWidget, available_columns: Sequence[str]) -> bool:
    """
    Return True if the widget can render with the given columns.

    - REQUIRES: exact column names that must all be present
    - REQUIRES_PATTERNS: regex patterns; each must match at least one column
    """
    cols = set(available_columns)
    name = getattr(widget, "NAME", widget.__class__.__name__)

    requires = set(getattr(widget, "REQUIRES", set()) or set())
    patterns = list(getattr(widget, "REQUIRES_PATTERNS", []) or [])

    # 1) Exact column requirements
    missing = sorted(c for c in requires if c not in cols)
    if missing:
        print(f"DEBUG: Hiding widget '{name}' — missing columns: {missing}")
        return False

    # 2) Pattern requirements (accept str or compiled regex)
    for pat in patterns:
        pattern_str = getattr(pat, "pattern", pat)  # support compiled or plain string
        if not any(re.search(pattern_str, c) for c in cols):
            print(f"DEBUG: Hiding widget '{name}' — no column matches pattern: {pattern_str!r}")
            return False

    return True
