from typing import List, Dict

import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import polars as pl
from dash import dcc, html, Input, Output
from plotly.subplots import make_subplots

from pixel_patrol.core.loaders.geff_loader import GeffLoader
from pixel_patrol.core.spec_provider import get_requirements_as_patterns
from pixel_patrol.report.widget_interface import PixelPatrolWidget


class GeffSummaryWidget(PixelPatrolWidget):
    @property
    def tab(self) -> str:
        return "Tracking"

    @property
    def name(self) -> str:
        return "GEFF Summary"

    def required_columns(self) -> List[str]:
        """Defines the columns required from the global DataFrame for this widget."""
        patterns = get_requirements_as_patterns(GeffLoader())
        patterns.extend(["^imported_path_short$", "^name$"])
        return patterns

    def layout(self) -> List:
        """Defines a single container to be filled by the callback."""
        return [html.Div(id="geff-summary-content")]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output("geff-summary-content", "children"),
            Input("color-map-store", "data")
        )
        def update_geff_summary_report(color_map: Dict[str, str]):
            # 1. Filter for relevant GEFF data
            df = df_global.filter(
                pl.any_horizontal(pl.col("^geff_.*$").is_not_null())
            )

            if df.is_empty():
                return dbc.Alert("No GEFF data available to generate a report.", color="warning")

            # 2. High-Level Summary Card
            num_files = df.height
            total_nodes = df.get_column("geff_num_nodes").sum()
            total_lineages = df.get_column("geff_num_lineages").sum()
            avg_divisions = df.get_column("geff_num_divisions").mean()

            summary_card = dbc.Card(dbc.CardBody([
                html.H4("Overall GEFF Summary", className="card-title"),
                dbc.ListGroup([
                    dbc.ListGroupItem(f"GEFF Files Evaluated: {num_files}"),
                    dbc.ListGroupItem(f"Total Nodes Tracked: {total_nodes:,}"),
                    dbc.ListGroupItem(f"Total Lineages Found: {total_lineages:,}"),
                    dbc.ListGroupItem(f"Average Divisions per File: {avg_divisions:.2f}"),
                ], flush=True),
            ]), className="mb-4")

            # 3. Violin Plots for Key Metrics
            metrics_to_plot = [
                {"col": "geff_num_nodes", "name": "Number of Nodes"},
                {"col": "geff_num_lineages", "name": "Number of Lineages"},
                {"col": "geff_num_divisions", "name": "Number of Divisions"},
                {"col": "geff_num_terminations", "name": "Number of Terminations"},
                {"col": "geff_num_edges", "name": "Number of Edges"},
            ]

            # Filter out metrics that are not in the dataframe or have no variance
            valid_metrics_to_plot = []
            for metric in metrics_to_plot:
                if metric["col"] in df.columns and df[metric["col"]].drop_nulls().n_unique() > 1:
                    valid_metrics_to_plot.append(metric)

            plots_card = None
            if valid_metrics_to_plot:
                num_plots = len(valid_metrics_to_plot)
                plots_per_row = 3
                num_rows = (num_plots + plots_per_row - 1) // plots_per_row

                fig = make_subplots(
                    rows=num_rows,
                    cols=plots_per_row,
                    subplot_titles=[d['name'] for d in valid_metrics_to_plot]
                )

                folders = df['imported_path_short'].unique().sort().to_list()

                for i, plot_info in enumerate(valid_metrics_to_plot):
                    row = (i // plots_per_row) + 1
                    col = (i % plots_per_row) + 1
                    metric_col = plot_info['col']

                    for folder in folders:
                        folder_data = df.filter(
                            (pl.col(metric_col).is_not_null()) & (pl.col("imported_path_short") == folder)
                        )
                        if folder_data.is_empty():
                            continue

                        fig.add_trace(go.Violin(
                            y=folder_data[metric_col],
                            name=folder,
                            customdata=folder_data['name'],
                            hovertemplate="<b>%{data.name}</b><br>Value: %{y}<br>File: %{customdata}<extra></extra>",
                            points='all',
                            spanmode="hard",
                            box_visible=True,
                            meanline_visible=True,
                            marker_color=color_map.get(folder, 'blue')
                        ), row=row, col=col)

                fig.update_layout(
                    showlegend=False,
                    height=num_rows * 350,
                    margin=dict(t=60, b=20, l=40, r=20),
                    title_text="Distribution of Key GEFF Metrics"
                )

                plots_card = dbc.Card(dbc.CardBody([
                    html.H4("Metric Distributions", className="card-title"),
                    dcc.Graph(figure=fig)
                ]), className="mb-4")

            # table_cols = (["name", "imported_path_short"] +
            #               sorted(
            #                   df.select(pl.col("^geff_(num|mean|std|min|max|version|axes|dim|axis).*$")).columns))

            # table_df = df.select(pl.col(table_cols))
            #
            # clean_column_names = {
            #     col: col.replace("geff_", "").replace("_", " ").replace("attr", "").strip().title()
            #     for col in table_df.columns
            # }
            #
            # table_grid = dag.AgGrid(
            #     rowData=table_df.to_dicts(),
            #     columnDefs=[
            #         {"field": col, "headerName": clean_column_names.get(col, col), "sortable": True,
            #          "flex": 1}
            #         for col in table_df.columns],
            #     dashGridOptions={"domLayout": "autoHeight"},
            #     defaultColDef={"resizable": True}
            # )
            # table_card = dbc.Card(dbc.CardBody([
            #     html.H4("Detailed GEFF Data", className="card-title"),
            #     table_grid
            # ]), className="mb-4")

            # Assemble final layout
            layout_content = [summary_card]
            if plots_card:
                layout_content.append(plots_card)
            # layout_content.append(table_card)

            return layout_content