import re
from typing import List, Dict, Sequence, Union, Optional
from pathlib import Path
import dash_bootstrap_components as dbc
import matplotlib.cm as cm
import polars as pl
from dash import Dash, html, dcc, ALL, callback_context
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.exceptions import PreventUpdate
from datetime import datetime
import plotly.io as pio
import logging

from pixel_patrol_base.core.contracts import PixelPatrolWidget
from pixel_patrol_base.core.project import Project
from pixel_patrol_base.core.project_metadata import ProjectMetadata
from pixel_patrol_base.core.report_config import ReportConfig
from pixel_patrol_base.io.parquet_io import to_parquet_bytes, resolve_report_source
from pixel_patrol_base.plugin_registry import discover_widget_plugins
from pixel_patrol_base.report.widget import organize_widgets_by_tab
from pixel_patrol_base.report.global_controls import (build_sidebar,
                                                      apply_global_row_filters_and_grouping,
                                                      compute_filtered_row_positions,
                                                      prepare_widget_data,
                                                      validate_report_config,)
from pixel_patrol_base.report.constants import (
    PALETTE_SELECTOR_ID,
    GLOBAL_CONFIG_STORE_ID,
    FILTERED_INDICES_STORE_ID,
    GLOBAL_GROUPBY_COLS_ID,
    GLOBAL_FILTER_COLUMN_ID,
    GLOBAL_FILTER_OP_ID,
    GLOBAL_FILTER_TEXT_ID,
    GLOBAL_DIM_FILTER_TYPE,
    GLOBAL_SHOW_SIGNIFICANCE_ID,
    GLOBAL_APPLY_BUTTON_ID,
    GLOBAL_RESET_BUTTON_ID,
    EXPORT_CSV_BUTTON_ID,
    EXPORT_PROJECT_BUTTON_ID,
    EXPORT_CSV_DOWNLOAD_ID,
    EXPORT_PROJECT_DOWNLOAD_ID,
    SAVE_SNAPSHOT_BUTTON_ID,
    SAVE_SNAPSHOT_DOWNLOAD_ID,
    DEFAULT_REPORT_GROUP_COL,
    DEFAULT_CMAP,
    DEFAULT_WIDGET_WIDTH,
)

logger = logging.getLogger(__name__)

ASSETS_DIR = (Path(__file__).parent / "assets").resolve()

EXTERNAL_STYLESHEETS = [dbc.themes.BOOTSTRAP,
                        "https://codepen.io/chriddyp/pen/bWLwgP.css",
                        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css",
                        ]

def prepare_app(
    source: Union[Project, Path],
    report_config: Optional[ReportConfig],
) -> Dash:
    """Resolves data source and config, then creates the Dash app."""
    records_df, metadata = resolve_report_source(source)

    if records_df is None:
        logger.error("Cannot generate report: no data available (records_df is None).")
        raise ValueError("No data available. Check that files exist and match the loader's supported extensions.")
    if records_df.is_empty():
        logger.error("Cannot generate report: dataset is empty.")
        raise ValueError("Dataset is empty. Files were found but contained no valid records.")

    validated_config = validate_report_config(records_df, report_config)
    return _create_app(
        df=records_df,
        metadata=metadata,
        report_config=validated_config,
    )


def _create_app(
        df: pl.DataFrame,
        metadata: ProjectMetadata,
        report_config: ReportConfig | None = None,
):
    """Instantiate Dash app, register callbacks, and assign layout."""

    project_name = metadata.project_name
    pixel_patrol_flavor = metadata.flavor
    default_palette_name = report_config.cmap if report_config else DEFAULT_CMAP

    app = Dash(
        __name__,
        external_stylesheets=EXTERNAL_STYLESHEETS,
        suppress_callback_exceptions=True,
        assets_folder=str(ASSETS_DIR),
    )

    pio.templates.default = "plotly"

    # Discover and filter widgets based on report_config
    group_widgets: List[PixelPatrolWidget] = _filter_widgets(discover_widget_plugins(), report_config)

    for w in group_widgets:
        if hasattr(w, "register"):
            w.register(app, df)
        elif hasattr(w, "register_callbacks"):
            w.register_callbacks(app, df)

    def serve_layout_closure() -> html.Div:

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
            df, default_palette_name, initial_report_config=report_config
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
                dcc.Store(id=FILTERED_INDICES_STORE_ID, data=None),
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

        return html.Div([stores, layout_row])

    app.layout = serve_layout_closure

    @app.callback(
        Output(FILTERED_INDICES_STORE_ID, "data"),
        Input(GLOBAL_CONFIG_STORE_ID, "data"),
    )
    def update_filtered_indices(global_config_dict: Dict) -> List[int]:
        """
        Compute filtered row POSITIONS once per global-config change.
        Widgets reuse these positions for fast positional indexing (df[positions]).
        """
        report_cfg = ReportConfig.from_dict(global_config_dict) if global_config_dict else None
        return compute_filtered_row_positions(df, report_cfg)


    @app.callback(
        Output("color-map-store", "data"),
        Input(FILTERED_INDICES_STORE_ID, "data"),
        State(GLOBAL_CONFIG_STORE_ID, "data"),
        State(PALETTE_SELECTOR_ID, "value"),
    )
    def update_color_map(
            subset_positions: List[int] | None,
            global_config_dict: Dict,
            palette: str,
    ) -> Dict[str, str]:
        """
        Build color map based on the *already filtered* subset and current grouping.
        """
        report_cfg = ReportConfig.from_dict(global_config_dict) if global_config_dict else None
        df_processed, group_col, _resolved, _warning, order = prepare_widget_data(
            df,
            subset_positions,
            report_cfg,
            metric_base=None,
        )

        if not group_col or group_col not in df_processed.columns or df_processed.is_empty():
            return {}

        groups = (
            df_processed
            .select(pl.col(group_col).unique().drop_nulls().sort())
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
        Input(GLOBAL_RESET_BUTTON_ID, "n_clicks"),
        State(GLOBAL_SHOW_SIGNIFICANCE_ID, "value"),
        State(GLOBAL_GROUPBY_COLS_ID, "value"),
        State(GLOBAL_FILTER_COLUMN_ID, "value"),
        State(GLOBAL_FILTER_OP_ID, "value"),
        State(GLOBAL_FILTER_TEXT_ID, "value"),
        State({"type": GLOBAL_DIM_FILTER_TYPE, "index": ALL}, "value"),
        State({"type": GLOBAL_DIM_FILTER_TYPE, "index": ALL}, "id"),
        prevent_initial_call=False,
    )
    def update_global_config(
            _apply_clicks,
            _reset_clicks,
            show_significance,
            group_col,
            filter_col,
            filter_op,
            filter_text,
            dim_values,
            dim_ids
    ) -> Dict:
        ctx = callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        if trigger_id == GLOBAL_RESET_BUTTON_ID:
            report_cfg = ReportConfig(
                group_col=DEFAULT_REPORT_GROUP_COL,
                filter_by={},
                dimensions={}
            )
        else:
            group_col = group_col or DEFAULT_REPORT_GROUP_COL

            filters: Dict[str, Dict[str, object]] = {}
            if filter_col and filter_op and filter_text:
                raw = filter_text.strip()
                if raw:
                    filters[filter_col] = {"op": filter_op, "value": raw}

            dimensions = {}
            for val, id_obj in zip(dim_values or [], dim_ids or []):
                if val and val != "All":
                    dimensions[id_obj['index']] = val

            report_cfg = ReportConfig(
                group_col=group_col,
                filter_by=filters if filters else None,
                dimensions=dimensions if dimensions else None,
                is_show_significance=show_significance
            )
        
        return report_cfg.to_dict()

    @app.callback(
        Output(GLOBAL_GROUPBY_COLS_ID, "value"),
        Output(GLOBAL_FILTER_COLUMN_ID, "value"),
        Output(GLOBAL_FILTER_TEXT_ID, "value"),
        Output({"type": GLOBAL_DIM_FILTER_TYPE, "index": ALL}, "value"),
        Input(GLOBAL_RESET_BUTTON_ID, "n_clicks"),
        State({"type": GLOBAL_DIM_FILTER_TYPE, "index": ALL}, "value"),
        prevent_initial_call=True
    )
    def reset_ui_controls(_n_clicks, current_dim_values):
        # Reset dropdowns to None/Empty and all dimensions to "All"
        return None, None, "", ["All"] * len(current_dim_values)

    @app.callback(
        Output(EXPORT_CSV_DOWNLOAD_ID, "data"),
        Input(EXPORT_CSV_BUTTON_ID, "n_clicks"),
        State(GLOBAL_CONFIG_STORE_ID, "data"),
        prevent_initial_call=True,
    )
    def export_df_to_csv(n_clicks: int, global_config_dict: Dict):
        if not n_clicks:
            raise PreventUpdate

        report_cfg = ReportConfig.from_dict(global_config_dict) if global_config_dict else None
        df_filtered, group_col = apply_global_row_filters_and_grouping(df, report_cfg)

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
    def export_df_w_metadata_to_parquet(n_clicks: int, global_config_dict: Dict):
        if not n_clicks:
            raise PreventUpdate

        report_cfg = ReportConfig.from_dict(global_config_dict) if global_config_dict else None
        df_filtered, _ = apply_global_row_filters_and_grouping(df, report_cfg)

        parquet_bytes = to_parquet_bytes(df_filtered, metadata)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return dcc.send_bytes(parquet_bytes, f"{project_name}_filtered_{ts}.parquet")


    app.clientside_callback(
        ClientsideFunction(
            namespace="clientside",
            function_name="save_snapshot"
        ),
        Output(SAVE_SNAPSHOT_DOWNLOAD_ID, "data"),
        # This Output is technically dummy/unused by the JS return, but required by Dash
        Input(SAVE_SNAPSHOT_BUTTON_ID, "n_clicks"),
        prevent_initial_call=True
    )

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
        logger.debug(f"Hiding widget '{name}' — missing columns: {missing}")
        return False

    # 2) Pattern requirements (accepts str or compiled regex)
    for pat in patterns:
        pattern_str = getattr(pat, "pattern", pat)  # support compiled or plain string
        if not any(re.search(pattern_str, c) for c in cols):
            logger.debug(f"Hiding widget '{name}' — no column matches pattern: {pattern_str!r}")
            return False

    return True


def _filter_widgets(
        widgets: List[PixelPatrolWidget],
        report_config: ReportConfig | None,
) -> List[PixelPatrolWidget]:
    """Filter discovered widgets by include/exclude sets from report_config."""
    if not report_config:
        return widgets

    original_names = {w.NAME for w in widgets}

    if report_config.widgets_included:
        filtered = [w for w in widgets if w.NAME in report_config.widgets_included]
        logger.info(f"Widget filtering (include mode): {len(filtered)}/{len(widgets)} widgets included")
        missing = report_config.widgets_included - original_names
        if missing:
            logger.warning(f"Requested widget names not found: {missing}")
        return filtered

    if report_config.widgets_excluded:
        filtered = [w for w in widgets if w.NAME not in report_config.widgets_excluded]
        logger.info(f"Widget filtering (exclude mode): {len(filtered)}/{len(widgets)} widgets remaining")
        missing = report_config.widgets_excluded - original_names
        if missing:
            logger.debug(f"Some excluded widget names were not found: {missing}")
        still_present = report_config.widgets_excluded & {w.NAME for w in filtered}
        if still_present:
            logger.error(f"Widgets that should be excluded are still present: {still_present}")
        return filtered

    return widgets