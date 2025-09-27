from typing import List, Dict, Set

import plotly.graph_objects as go
import polars as pl
from dash import html, dcc, Input, Output
from plotly.subplots import make_subplots

from pixel_patrol_base.report.widget_categories import WidgetCategories


class FileSummaryWidget:
    # ---- Declarative spec ----
    NAME: str = "File Data Summary"
    TAB: str = WidgetCategories.SUMMARY.value
    REQUIRES: Set[str] = {"size_bytes", "file_extension", "imported_path_short"}
    REQUIRES_PATTERNS = None

    # Component IDs
    INTRO_ID = "file-summary-intro"
    GRAPH_ID = "file-summary-graph"
    TABLE_ID = "file-summary-table"

    def layout(self) -> List:
        return [
            html.Div(id=self.INTRO_ID, style={"marginBottom": "20px"}),
            dcc.Graph(id=self.GRAPH_ID),
            html.Div(id=self.TABLE_ID, style={"marginTop": "20px"}),
        ]

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output(self.INTRO_ID, "children"),
            Output(self.GRAPH_ID, "figure"),
            Output(self.TABLE_ID, "children"),
            Input("color-map-store", "data"),
        )
        def update_file_summary(color_map: Dict[str, str]):
            color_map = color_map or {}

            # --- Aggregations for File-Specific Data ---
            summary = (
                df_global
                .group_by("imported_path_short")
                .agg(
                    pl.count().alias("file_count"),
                    (pl.sum("size_bytes") / (1024 * 1024)).alias("total_size_mb"),
                    pl.col("file_extension").unique().sort().alias("file_types"),
                )
                .sort("imported_path_short")
            )

            # --- Intro Text ---
            intro_md = [html.P(f"This summary focuses on file properties across {summary.height} folders.")]
            for row in summary.iter_rows(named=True):
                ft_str = ", ".join(row["file_types"])
                intro_md.append(
                    html.P(
                        f"Folder '{row['imported_path_short']}' contains "
                        f"{row['file_count']} files ({row['total_size_mb']:.3f} MB) with types: {ft_str}."
                    )
                )

            # --- Figure ---
            x_labels = summary["imported_path_short"].to_list()
            counts = summary["file_count"].to_list()
            sizes = summary["total_size_mb"].to_list()
            colors = [color_map.get(lbl, "#333333") for lbl in x_labels]

            fig = make_subplots(rows=1, cols=2, subplot_titles=("File Count", "Total Size (MB)"))
            fig.add_trace(go.Bar(x=x_labels, y=counts, marker_color=colors), row=1, col=1)
            fig.add_trace(go.Bar(x=x_labels, y=sizes, marker_color=colors), row=1, col=2)
            fig.update_layout(height=400, showlegend=False, margin=dict(l=40, r=40, t=80, b=40), barmode="group")

            n = len(x_labels)
            if n == 1:
                fig.update_layout(bargap=0.7)
            if n == 2:
                fig.update_layout(bargap=0.4)

            table = html.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("Folder"), html.Th("Files"), html.Th("Size (MB)"), html.Th("Types")
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td(row["imported_path_short"]),
                            html.Td(row["file_count"]),
                            html.Td(f"{row['total_size_mb']:.3f}"),
                            html.Td(", ".join(row["file_types"])),
                        ]) for row in summary.iter_rows(named=True)
                    ]),
                ],
                style={"width": "100%", "borderCollapse": "collapse"},
            )


            return intro_md, fig, table
