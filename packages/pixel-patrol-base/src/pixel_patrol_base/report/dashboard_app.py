import re
from typing import List, Dict, Sequence

import dash_bootstrap_components as dbc
import matplotlib.cm as cm
import polars as pl
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.io as pio

from pixel_patrol_base.core.contracts import PixelPatrolWidget
from pixel_patrol_base.core.project import Project
from pixel_patrol_base.plugin_registry import discover_widget_plugins
from pixel_patrol_base.report.widget import organize_widgets_by_tab
from pixel_patrol_base.report.global_controls import (
    build_sidebar,
    apply_global_config,
    PALETTE_SELECTOR_ID,
    GLOBAL_CONFIG_STORE_ID,
    GLOBAL_GROUPBY_COLS_ID,
    GLOBAL_FILTER_COLUMN_ID,
    GLOBAL_FILTER_TEXT_ID,
    GLOBAL_APPLY_BUTTON_ID,
    EXPORT_CSV_BUTTON_ID,
    EXPORT_PROJECT_BUTTON_ID,
    EXPORT_CSV_DOWNLOAD_ID,
    EXPORT_PROJECT_DOWNLOAD_ID,
)

from pathlib import Path  # add if missing

DEFAULT_WIDGET_WIDTH = 12

ASSETS_DIR = (Path(__file__).parent / "assets").resolve()

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
        df=project.records_df,
        default_palette_name=project.get_settings().cmap,
        pixel_patrol_flavor=project.get_settings().pixel_patrol_flavor,
        project_name=project.name,
        project=project,
    )


def _create_app(
    df: pl.DataFrame,
    default_palette_name: str,
    pixel_patrol_flavor: str,
    project_name: str,
    project: Project,
):
    """Instantiate Dash app, register callbacks, and assign layout."""
    df = df.with_row_index(name="unique_id")
    external_stylesheets = [dbc.themes.BOOTSTRAP,
                            "https://codepen.io/chriddyp/pen/bWLwgP.css",
                            "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css",
                            ]
    #app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

    app = Dash(
        __name__,
        external_stylesheets=external_stylesheets,
        suppress_callback_exceptions=True,
        assets_folder=str(ASSETS_DIR),
    )

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

        # --- Header (inside main content) ---
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

        # --- Sidebar with global controls (built once, outside content container) ---
        sidebar_controls, global_control_stores, extra_components = build_sidebar(
            df, default_palette_name
        )

        # --- Group Widget Layout Generation (content only) ---
        group_widget_content = []
        tabbed_group_widgets = organize_widgets_by_tab(group_widgets)

        for group_name, ws in tabbed_group_widgets.items():
            group_widget_content.append(
                dbc.Row(
                    dbc.Col(html.H3(group_name, className="my-3 text-primary"))
                )
            )
            current_group_cols = []
            current_row_width = 0

            for widget in ws:
                if should_display_widget(widget, df.columns):
                    widget_width = getattr(widget, "width", DEFAULT_WIDGET_WIDTH)

                    # wrap to next row if needed
                    if current_row_width + widget_width > 12:
                        group_widget_content.append(
                            dbc.Row(current_group_cols, className="g-4 p-3")
                        )
                        current_group_cols, current_row_width = [], 0

                    # widget body
                    current_group_cols.append(
                        dbc.Col(
                            html.Div(widget.layout()),
                            width=widget_width,
                            className="mb-3",
                        )
                    )
                    current_row_width += widget_width

            if current_group_cols:
                group_widget_content.append(
                    dbc.Row(current_group_cols, className="g-4 p-3")
                )

        # --- Data Stores ---
        stores = html.Div(
            [
                dcc.Store(id="color-map-store"),
                *global_control_stores,
                dcc.Store(
                    id="tb-process-store-tensorboard-embedding-projector",
                    data={},
                ),
                *extra_components,
            ]
        )

        # --- Main content container (header + widgets, centered with maxWidth) ---
        content_container = dbc.Container(
            [header_row, html.Hr(), *group_widget_content],
            fluid=True,
            style={"maxWidth": "1200px", "margin": "0 auto"},
        )

        # --- Overall layout: sidebar outside margins + main content ---
        layout_row = dbc.Row(
            [
                # Sidebar column: fixed-ish width, outside main container margins
                dbc.Col(
                    sidebar_controls,
                    width="auto",
                    style={
                        "minWidth": "280px",
                        "maxWidth": "320px",
                        "padding": "20px",
                    },
                ),
                # Main app content column: takes remaining width
                dbc.Col(
                    content_container,
                    width=True,
                ),
            ],
            className="gx-0",  # no horizontal gutter between sidebar and content
        )

        return html.Div(
            [
                stores,
                layout_row,
            ]
        )

    app.layout = serve_layout_closure

    @app.callback(
        Output("color-map-store", "data"),
        # Swapped Input/State: Updates only when Config Store updates (via Apply button)
        Input(GLOBAL_CONFIG_STORE_ID, "data"),
        # Palette is now a State, so selecting it doesn't trigger immediate updates
        State(PALETTE_SELECTOR_ID, "value"),
    )
    def update_color_map(global_config: Dict, palette: str) -> Dict[str, str]:
        df_processed, group_col = apply_global_config(df, global_config)

        if group_col is None or group_col not in df_processed.columns or df_processed.is_empty():
            return {}

        groups = (
            df_processed
            .select(pl.col(group_col).unique())
            .to_series()
            .to_list()
        )

        if not groups:
            return {}

        cmap = cm.get_cmap(palette, len(groups))

        color_map: Dict[str, str] = {}
        for i, group in enumerate(groups):
            r, g_chan, b, _ = cmap(i)
            color_map[str(group)] = f"#{int(r * 255):02x}{int(g_chan * 255):02x}{int(b * 255):02x}"
        return color_map


    @app.callback(
        Output(GLOBAL_CONFIG_STORE_ID, "data"),
        Input(GLOBAL_APPLY_BUTTON_ID, "n_clicks"),
        State(GLOBAL_GROUPBY_COLS_ID, "value"),
        State(GLOBAL_FILTER_COLUMN_ID, "value"),
        State(GLOBAL_FILTER_TEXT_ID, "value"),
        prevent_initial_call=False,
    )
    def update_global_config(
        _n_clicks: int,
        group_col,
        filter_col,
        filter_text,
    ) -> Dict:
        # Single grouping column; default to report_group if nothing chosen
        if not group_col:
            group_cols = ["report_group"]
        else:
            group_cols = [group_col]

        filters: Dict[str, List[str]] = {}
        if filter_col and filter_text:
            allowed_vals = [v.strip() for v in filter_text.split(",") if v.strip()]
            if allowed_vals:
                filters[filter_col] = allowed_vals

        return {"group_cols": group_cols, "filters": filters}

    @app.callback(
        Output(EXPORT_CSV_DOWNLOAD_ID, "data"),
        Input(EXPORT_CSV_BUTTON_ID, "n_clicks"),
        State(GLOBAL_CONFIG_STORE_ID, "data"),
        prevent_initial_call=True,
    )
    def export_current_table_as_csv(n_clicks: int, global_config: Dict):
        if not n_clicks:
            raise PreventUpdate

        df_filtered, group_col = apply_global_config(df, global_config)

        # add a human-readable group label column for clarity
        df_export = df_filtered.with_columns(
            pl.col(group_col).alias("group_label")
        )

        csv_str = df_export.to_pandas().to_csv(index=False)
        filename = f"{project_name}_filtered_table.csv"

        return dcc.send_string(csv_str, filename)

    @app.callback(
        Output(EXPORT_PROJECT_DOWNLOAD_ID, "data"),
        Input(EXPORT_PROJECT_BUTTON_ID, "n_clicks"),
        State(GLOBAL_CONFIG_STORE_ID, "data"),
        prevent_initial_call=True,
    )
    def export_filtered_project(n_clicks: int, global_config: Dict):
        if not n_clicks:
            raise PreventUpdate

        import tempfile
        import copy
        from pathlib import Path
        from pixel_patrol_base import api as pp_api

        # 1. Apply filters
        df_filtered, _ = apply_global_config(df, global_config)

        # 2. Clone the project and override records_df
        new_project = copy.copy(project)
        new_project.records_df = df_filtered

        # 3. Export using PixelPatrol's real export_project API
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / f"{project_name}_filtered.zip"

            # This creates ONE zip file (not a directory)
            pp_api.export_project(new_project, zip_path)

            # 4. Read the zip file into memory
            with open(zip_path, "rb") as f:
                zip_bytes = f.read()

        # 5. Return it to browser
        return dcc.send_bytes(zip_bytes, f"{project_name}_filtered.ppproj.zip")

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
