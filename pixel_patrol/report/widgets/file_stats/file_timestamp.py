from typing import List, Dict
from dash import html, dcc, Input, Output
import polars as pl
import os
import plotly.express as px

from pixel_patrol.report.widget_interface import PixelPatrolWidget
from pixel_patrol.report.widget_categories import WidgetCategories


class FileTimestampWidget(PixelPatrolWidget):

    @property
    def tab(self) -> str:
        return WidgetCategories.FILE_STATS.value

    @property
    def name(self) -> str:
        return "File Modification Date Distribution"

    def required_columns(self) -> List[str]:
        return ["modification_date", "imported_path", "name"]

    def layout(self) -> List:
        return [
            dcc.Graph(id="file-timestamp-bar-chart", style={"height": "500px"})
        ]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output("file-timestamp-bar-chart", "figure"),
            Input("color-map-store", "data"),
        )
        def update_file_timestamp_chart(color_map: Dict[str, str]):
            df = df_global.with_columns([
                pl.col("imported_path").map_elements(
                    lambda x: os.path.basename(x) if x else "Unknown Folder",
                    return_dtype=pl.String
                ).alias("imported_path_short"),
                pl.col("modification_date").cast(pl.Datetime).alias("modification_date")
            ]).filter(
                pl.col("modification_date").is_not_null()
            )

            grouped = df.group_by(["modification_date", "imported_path_short"]).agg([
                pl.count().alias("value"),
                pl.col("name").unique().alias("names_in_group")
            ])

            fig = px.bar(
                grouped,
                x="modification_date",
                y="value",
                color="imported_path_short",
                barmode="stack",
                color_discrete_map=color_map,
                title="File Modification Date Distribution",
                labels={
                    'modification_date': "Modification Time",
                    'value': "Number of Files",
                    'imported_path_short': 'Folder'
                },
                hover_data={
                    'names_in_group': True
                }
            )

            fig.update_layout(
                height=500,
                margin=dict(l=50, r=50, t=80, b=100),
                xaxis_title="Modification Time",
                yaxis_title="File Count",
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )

            return fig
