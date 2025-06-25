from typing import List, Dict

import dash_ag_grid as dag
import plotly.graph_objects as go
import polars as pl
from dash import html, dcc, Input, Output
from plotly.subplots import make_subplots

from pixel_patrol.report.widget_categories import WidgetCategories
from pixel_patrol.report.widget_interface import PixelPatrolWidget


class SummaryWidget(PixelPatrolWidget):
    @property
    def tab(self) -> str:
        return WidgetCategories.SUMMARY.value

    @property
    def name(self) -> str:
        return "Dataset overview"

    def required_columns(self) -> List[str]:
        return [
            "imported_path_short", "n_images", "size_bytes", "mean_intensity", "x_size",
            "file_extension", "dtype"
        ]

    def layout(self) -> List:
        intro = html.Div(id="summary-intro", style={"marginBottom": "20px"})
        graph = dcc.Graph(id="summary-graph")
        table = html.Div(id="summary-table", style={"marginTop": "20px"})
        return [intro, graph, html.B("Aggregated Folder Summary", style={"marginTop": "30px"}), table]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output("summary-intro", "children"),
            Output("summary-graph", "figure"),
            Output("summary-table", "children"),
            Input("color-map-store", "data")
        )
        def update_summary(color_map: Dict[str, str]):
            df = df_global
            # Build summaries
            folder_content = df.group_by("imported_path_short").agg(
                pl.sum("n_images").alias("image_count"),
                (pl.sum("size_bytes")/(1024*1024)).alias("total_size_mb")
            ).sort("imported_path_short")
            intensity_stats = df.group_by("imported_path_short").agg(
                pl.mean("mean_intensity"),
                pl.mean("std_intensity"),
                pl.mean("median_intensity"),
                (pl.quantile("mean_intensity",0.75)-pl.quantile("mean_intensity",0.25)).alias("intensity_iqr")
            ).sort("imported_path_short")
            dimension_stats = df.group_by("imported_path_short").agg(
                pl.mean("x_size").alias("mean_x_size"),
                pl.std("x_size").alias("std_x_size")
            ).sort("imported_path_short")
            summary = folder_content.join(intensity_stats, on="imported_path_short").join(dimension_stats, on="imported_path_short")

            # Intro markdown components
            folder_details = df.group_by("imported_path_short").agg([
                pl.col("n_images").sum().alias("image_count"),
                (pl.col("size_bytes").sum() / (1024 * 1024)).alias("total_size_mb"),
                pl.col("file_extension").unique().alias("file_types"),
                pl.col("dtype").unique().alias("data_types")
            ]).sort("imported_path_short")

            intro_md = [html.P(f"This dataset compares {folder_details.height} folders.")]

            for row in folder_details.iter_rows(named=True):
                dt_str = ", ".join(row["data_types"])
                ft_str = ", ".join(row["file_types"])
                intro_md.append(html.P(
                    f"{row['imported_path_short']}: {row['image_count']} images ({row['total_size_mb']:.1f} MB), "
                    f"types: {ft_str}, dtypes: {dt_str}."
                ))
            # Combined bar and box subplot
            folder_labels = summary['imported_path_short'].to_list()
            colors = [color_map.get(f, '#333333') for f in folder_labels]
            fig = make_subplots(rows=1, cols=3, subplot_titles=("Image Count","Total Size (MB)","Intensity Distribution"))
            fig.add_trace(go.Bar(x=folder_labels, y=summary['image_count'], marker_color=colors), row=1, col=1)
            fig.add_trace(go.Bar(x=folder_labels, y=summary['total_size_mb'], marker_color=colors), row=1, col=2)
            for f, c in zip(folder_labels, colors):
                data = df.filter(pl.col("imported_path_short")==f)["mean_intensity"].to_list()
                fig.add_trace(go.Box(y=data, name=f, marker_color=c, boxpoints='outliers', line=dict(width=1.5)), row=1, col=3)
            fig.update_layout(height=400, showlegend=False, margin=dict(l=40,r=40,t=80,b=40))

            # AgGrid table component
            grid = dag.AgGrid(
                rowData=summary.to_dicts(),
                columnDefs=[{"field": col} for col in summary.columns],
                columnSize="sizeToFit",
                id="summary_grid"
            )
            table_div = html.Div([grid])

            return intro_md, fig, table_div