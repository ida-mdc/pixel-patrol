import re
from typing import List, Dict

import dash_ag_grid as dag
import plotly.graph_objects as go
import polars as pl
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots

from pixel_patrol.core.loaders.traccuracy_loader import TraccuracyLoader
from pixel_patrol.core.spec_provider import get_requirements_as_patterns
from pixel_patrol.report.widget_interface import PixelPatrolWidget


class TraccuracySummaryWidget(PixelPatrolWidget):
    @property
    def tab(self) -> str:
        return "Tracking"

    @property
    def name(self) -> str:
        return "Traccuracy"

    def required_columns(self) -> List[str]:
        patterns = get_requirements_as_patterns(TraccuracyLoader())
        patterns.append("^imported_path_short$")
        return patterns

    def layout(self) -> List:
        """Defines a single container to be filled by the callback."""
        return [html.Div(id="traccuracy-final-report-content")]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output("traccuracy-final-report-content", "children"),
            Input("color-map-store", "data")
        )
        def update_traccuracy_final_report(color_map: Dict[str, str]):
            df = df_global.filter(
                pl.any_horizontal(pl.col("^traccuracy_.*$").is_not_null())
            )

            if df.is_empty():
                return dbc.Alert("No traccuracy data available to generate a report.", color="warning")

            # --- 1. High-Level Summary ---
            num_files = df.height
            traccuracy_cols = df.select(pl.col("^traccuracy_.*$")).columns
            metric_suites = sorted(list(set(
                col.split('_')[1] for col in traccuracy_cols if col.startswith("traccuracy_")
            )))

            summary_card = html.Div([
                html.H4("Overall Summary", className="card-title"),
                dbc.ListGroup([
                    dbc.ListGroupItem(f"Files Evaluated: {num_files}"),
                    dbc.ListGroupItem(f"Metric Suites Found: {', '.join(metric_suites)}"),
                ], flush=True),
            ])

            # --- 2. Generate Detailed Reports for Each Suite ---
            detailed_reports = []
            folders = df['imported_path_short'].unique().sort().to_list()

            if len(folders) <= 1:
                plots_per_row = 4
            elif len(folders) <= 2:
                plots_per_row = 3
            elif len(folders) <= 3:
                plots_per_row = 2
            else:
                plots_per_row = 1

            for suite in metric_suites:
                suite_results_cols = [col for col in df.columns if re.search(f"^traccuracy_{suite}_results_.*$", col)]
                if not suite_results_cols:
                    continue

                suite_df = df.select(pl.col("name"), pl.col("imported_path_short"), pl.col(suite_results_cols))
                cleansing_exprs = [pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in suite_results_cols]
                suite_df = suite_df.with_columns(cleansing_exprs)

                metrics_to_plot, constant_metrics_data = [], []
                for metric_col in suite_results_cols:
                    valid_data = suite_df[metric_col].drop_nulls()
                    clean_name = metric_col.replace(f"traccuracy_{suite}_results_", "").replace("_", " ").title()
                    if valid_data.n_unique() <= 1:
                        value = valid_data.unique()[0] if not valid_data.is_empty() else "N/A"
                        constant_metrics_data.append({
                            "Metric": clean_name,
                            "Constant Value": f"{value:.3f}" if isinstance(value, float) else value
                        })
                    else:
                        metrics_to_plot.append({'col_name': metric_col, 'clean_name': clean_name})

                card_body_content = [html.H5(f"{suite} Results", className="card-title")]

                if metrics_to_plot:
                    num_plots = len(metrics_to_plot)
                    num_rows = (num_plots + plots_per_row - 1) // plots_per_row

                    fig = make_subplots(rows=num_rows, cols=plots_per_row,
                                        subplot_titles=[d['clean_name'] for d in metrics_to_plot])

                    for i, plot_info in enumerate(metrics_to_plot):
                        row = (i // plots_per_row) + 1
                        col = (i % plots_per_row) + 1
                        metric_col = plot_info['col_name']

                        for folder in folders:
                            folder_data = suite_df.filter(
                                (pl.col(metric_col).is_not_null()) & (pl.col("imported_path_short") == folder)
                            )
                            if folder_data.is_empty(): continue

                            fig.add_trace(go.Violin(
                                y=folder_data[metric_col],
                                name=folder,
                                customdata=folder_data['name'],
                                hovertemplate="<b>%{data.name}</b><br>Value: %{y:.3f}<br>File: %{customdata}<extra></extra>",
                                points='all',
                                box_visible=True,
                                meanline_visible=True,
                                spanmode="hard",
                                marker_color=color_map.get(folder)
                            ), row=row, col=col)

                    fig.update_layout(
                        showlegend=False,  # Legend is removed
                        height=num_rows * 400,
                        margin=dict(t=60, b=20, l=40, r=20)
                    )
                    card_body_content.append(dcc.Graph(figure=fig))

                if constant_metrics_data:
                    card_body_content.append(html.H6(f"Constant Value Metrics for {suite}", className="mt-4"))
                    const_grid = dag.AgGrid(
                        rowData=constant_metrics_data,
                        columnDefs=[{"field": "Metric", "flex": 1}, {"field": "Constant Value", "flex": 1}],
                        dashGridOptions={"domLayout": "autoHeight"}
                    )
                    card_body_content.append(const_grid)

                table_df = suite_df.rename(
                    {col: col.replace(f"traccuracy_{suite}_results_", "") for col in suite_results_cols})
                grid = dag.AgGrid(
                    rowData=table_df.to_dicts(),
                    columnDefs=[{"field": col} for col in table_df.columns],
                    id=f"traccuracy-grid-{suite}"
                )

                card_body_content.extend([
                    html.H6(f"Detailed Data Table for {suite}", className="card-title"),
                    html.Div([grid], className="mt-4"),
                ])

                suite_card = dbc.Card(dbc.CardBody(card_body_content), className="mb-4")
                detailed_reports.append(suite_card)

            return [summary_card] + detailed_reports