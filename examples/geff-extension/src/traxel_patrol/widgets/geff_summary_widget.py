from typing import List, Dict, Set
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import polars as pl
from dash import dcc, html, Input, Output
from plotly.subplots import make_subplots

from pixel_patrol_base.core.feature_schema import get_requirements_as_patterns
from traxel_patrol.loaders.geff_loader import GeffLoader


class GeffSummaryWidget:
    # ---- Declarative spec ----
    NAME: str = "GEFF Summary"
    TAB: str = "Tracking"  # or WidgetCategories.TRACKING.value if you have it
    REQUIRES: Set[str] = {"imported_path_short", "name"}  # explicit columns we need
    REQUIRES_PATTERNS: List[str]

    # Component IDs
    CONTENT_ID = "geff-summary-content"

    def __init__(self):
        # dynamic column requirements coming from the GEFF loader
        self.REQUIRES_PATTERNS = list(get_requirements_as_patterns(GeffLoader))

    def layout(self) -> List:
        return [html.Div(id=self.CONTENT_ID)]

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output(self.CONTENT_ID, "children"),
            Input("color-map-store", "data"),
        )
        def update_geff_summary_report(color_map: Dict[str, str]):
            color_map = color_map or {}

            # 1) Filter for rows that have any GEFF-related columns populated
            df = df_global.filter(pl.any_horizontal(pl.col("^geff_.*$").is_not_null()))
            if df.is_empty():
                return dbc.Alert("No GEFF data available to generate a report.", color="warning")

            # 2) High-level summary
            num_files = df.height
            total_nodes = df.get_column("geff_num_nodes").sum() if "geff_num_nodes" in df.columns else 0
            total_lineages = df.get_column("geff_num_lineages").sum() if "geff_num_lineages" in df.columns else 0
            avg_divisions = (
                df.get_column("geff_num_divisions").mean() if "geff_num_divisions" in df.columns else float("nan")
            )

            summary_card = dbc.Card(
                dbc.CardBody(
                    [
                        html.H4("Overall GEFF Summary", className="card-title"),
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(f"GEFF Files Evaluated: {num_files}"),
                                dbc.ListGroupItem(f"Total Nodes Tracked: {total_nodes:,}"),
                                dbc.ListGroupItem(f"Total Lineages Found: {total_lineages:,}"),
                                dbc.ListGroupItem(
                                    f"Average Divisions per File: {avg_divisions:.2f}"
                                    if avg_divisions == avg_divisions  # not NaN
                                    else "Average Divisions per File: —"
                                ),
                            ],
                            flush=True,
                        ),
                    ]
                ),
                className="mb-4",
            )

            # 3) Violin plots for key metrics (only those with variance)
            metrics_to_plot = [
                {"col": "geff_num_nodes", "name": "Number of Nodes"},
                {"col": "geff_num_lineages", "name": "Number of Lineages"},
                {"col": "geff_num_divisions", "name": "Number of Divisions"},
                {"col": "geff_num_terminations", "name": "Number of Terminations"},
                {"col": "geff_num_edges", "name": "Number of Edges"},
            ]

            valid_metrics = []
            for metric in metrics_to_plot:
                col = metric["col"]
                if col in df.columns and df[col].drop_nulls().n_unique() > 1:
                    valid_metrics.append(metric)

            plots_card = None
            if valid_metrics:
                num_plots = len(valid_metrics)
                plots_per_row = 3
                num_rows = (num_plots + plots_per_row - 1) // plots_per_row

                fig = make_subplots(
                    rows=num_rows,
                    cols=plots_per_row,
                    subplot_titles=[m["name"] for m in valid_metrics],
                )

                folders = df["imported_path_short"].unique().sort().to_list()
                for i, metric in enumerate(valid_metrics):
                    row = (i // plots_per_row) + 1
                    col = (i % plots_per_row) + 1
                    metric_col = metric["col"]

                    for folder in folders:
                        folder_data = df.filter(
                            (pl.col(metric_col).is_not_null()) & (pl.col("imported_path_short") == folder)
                        )
                        if folder_data.is_empty():
                            continue

                        fig.add_trace(
                            go.Violin(
                                y=folder_data[metric_col].to_list(),
                                name=folder,
                                customdata=folder_data["name"].to_list(),
                                hovertemplate="<b>%{data.name}</b><br>Value: %{y}<br>File: %{customdata}<extra></extra>",
                                points="all",
                                spanmode="hard",
                                box_visible=True,
                                meanline_visible=True,
                                marker_color=color_map.get(folder, "blue"),
                            ),
                            row=row,
                            col=col,
                        )

                fig.update_layout(
                    showlegend=False,
                    height=num_rows * 350,
                    margin=dict(t=60, b=20, l=40, r=20),
                    title_text="Distribution of Key GEFF Metrics",
                )

                plots_card = dbc.Card(
                    dbc.CardBody([html.H4("Metric Distributions", className="card-title"), dcc.Graph(figure=fig)]),
                    className="mb-4",
                )

            layout_content = [summary_card]
            if plots_card:
                layout_content.append(plots_card)
            return layout_content
