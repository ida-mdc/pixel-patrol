from typing import List, Dict

import plotly.graph_objects as go
import polars as pl
from dash import html, dcc, Input, Output
from plotly.subplots import make_subplots

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.widget_interface import PixelPatrolWidget


class FileSummaryWidget(PixelPatrolWidget):
    @property
    def tab(self) -> str:
        return WidgetCategories.SUMMARY.value

    @property
    def name(self) -> str:
        return "File Data Summary"

    def required_columns(self) -> List[str]:
        # Requires columns related to file properties
        return ['size_bytes', 'file_extension']

    def layout(self) -> List:
        intro = html.Div(id="file-summary-intro", style={"marginBottom": "20px"})
        graph = dcc.Graph(id="file-summary-graph")
        table = html.Div(id="file-summary-table", style={"marginTop": "20px"})
        return [intro, graph, html.B("Aggregated File Stats", style={"marginTop": "30px"}), table]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output("file-summary-intro", "children"),
            Output("file-summary-graph", "figure"),
            Input("color-map-store", "data")
        )
        def update_file_summary(color_map: Dict[str, str]):
            # --- Aggregations for File-Specific Data ---
            summary = df_global.group_by("imported_path_short").agg(
                pl.count().alias("file_count"),
                (pl.sum("size_bytes") / (1024 * 1024)).alias("total_size_mb"),
                pl.col("file_extension").unique().sort().alias("file_types")
            ).sort("imported_path_short")

            # --- Intro Text ---
            intro_md = [html.P(f"This summary focuses on file properties across {summary.height} folders.")]
            for row in summary.iter_rows(named=True):
                ft_str = ", ".join(row["file_types"])
                intro_md.append(html.P(
                    f"Folder '{row['imported_path_short']}' contains {row['file_count']} files ({row['total_size_mb']:.1f} MB) with types: {ft_str}."))

            # --- Figure ---
            fig = make_subplots(rows=1, cols=2, subplot_titles=("File Count", "Total Size (MB)"))
            colors = [color_map.get(f, '#333333') for f in summary['imported_path_short']]

            fig.add_trace(go.Bar(x=summary['imported_path_short'], y=summary['file_count'], marker_color=colors), row=1,
                          col=1)
            fig.add_trace(go.Bar(x=summary['imported_path_short'], y=summary['total_size_mb'], marker_color=colors),
                          row=1, col=2)

            fig.update_layout(height=400, showlegend=False, margin=dict(l=40, r=40, t=80, b=40), barmode='group')

            return intro_md, fig