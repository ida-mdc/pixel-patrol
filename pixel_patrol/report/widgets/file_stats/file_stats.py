from typing import List, Dict

import dash_ag_grid as dag
import plotly.express as px
import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol.report.widget_categories import WidgetCategories
from pixel_patrol.report.widget_interface import PixelPatrolWidget


class FileStatisticsWidget(PixelPatrolWidget):

    @property
    def tab(self) -> str:
        return WidgetCategories.FILE_STATS.value

    @property
    def name(self) -> str:
        return "File Statistics Report"

    def required_columns(self) -> List[str]:
        # Combine requirements from all three original widgets
        return ["file_extension", "size_bytes", "modification_date"]

    def layout(self) -> List:
        """A single container for the report, populated by the callback."""
        return [
            html.Div(id="file-stats-report-content")
        ]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output("file-stats-report-content", "children"),
            Input("color-map-store", "data"),
        )
        def update_file_stats_report(color_map: Dict[str, str]):
            report_elements = []
            constant_values_data = []

            df_filtered = df_global.select(
                pl.col("name"),
                pl.col("file_extension"),
                pl.col("imported_path_short"),
                pl.col("size_bytes"),
                pl.col("modification_date"),
            )

            # --- 1. File Extension Analysis ---
            ext_agg = df_filtered.group_by("file_extension").agg(pl.count()).drop_nulls()
            if ext_agg.height <= 1:
                if not ext_agg.is_empty():
                    constant_values_data.append({
                        "Property": "File Extension",
                        "Value": ext_agg["file_extension"][0]
                    })
            else:
                # Plot for Number of Files
                plot_data_ext_count = df_filtered.group_by(["file_extension", "imported_path_short"]).agg(pl.count())
                fig_ext_count = px.bar(plot_data_ext_count, x='file_extension', y='count', color='imported_path_short',
                                       barmode='stack', color_discrete_map=color_map, title="File Count by Extension")
                report_elements.append(dcc.Graph(figure=fig_ext_count))

                # Plot for Total Size
                plot_data_ext_size = df_filtered.group_by(["file_extension", "imported_path_short"]).agg(pl.sum("size_bytes"))
                fig_ext_size = px.bar(plot_data_ext_size, x='file_extension', y='size_bytes', color='imported_path_short',
                                      barmode='stack', color_discrete_map=color_map, title="Total Size by Extension")
                report_elements.append(dcc.Graph(figure=fig_ext_size))


            # --- 2. File Size Analysis ---
            bins = [1024*1024, 10*1024*1024, 100*1024*1024, 1024*1024*1024, 10*1024*1024*1024]
            labels = ["<1 MB", "1-10 MB", "10-100 MB", "100MB-1GB", "1-10 GB", ">10 GB"]
            size_df = df_filtered.with_columns(pl.col("size_bytes").cut(bins, labels=labels).alias("size_bin"))
            size_agg = size_df.group_by("size_bin").agg(pl.count()).drop_nulls()

            if size_agg.height <= 1:
                if not size_agg.is_empty():
                     constant_values_data.append({
                        "Property": "File Size Bin",
                        "Value": size_agg["size_bin"][0]
                    })
            else:
                plot_data_size = size_df.group_by(["size_bin", "imported_path_short"]).agg(pl.count())
                fig_size = px.bar(plot_data_size, x='size_bin', y='count', color='imported_path_short',
                                  barmode='stack', color_discrete_map=color_map, title="File Count by Size Bin")
                fig_size.update_layout(xaxis={'categoryorder':'array', 'categoryarray': labels})
                report_elements.append(dcc.Graph(figure=fig_size))


            # --- 3. File Timestamp Analysis ---
            ts_df = df_filtered.with_columns(pl.col("modification_date").cast(pl.Datetime))
            # Group by day for a cleaner plot
            ts_agg = ts_df.group_by(ts_df["modification_date"].dt.truncate("1d")).agg(pl.count()).drop_nulls()

            if ts_agg.height <= 1:
                 if not ts_agg.is_empty():
                    constant_values_data.append({
                        "Property": "Modification Date (Day)",
                        "Value": ts_agg["modification_date"][0].strftime('%Y-%m-%d')
                    })
            else:
                plot_data_ts = ts_df.group_by([ts_df["modification_date"].dt.truncate("1d"), "imported_path_short"]).agg(pl.count())
                fig_ts = px.bar(plot_data_ts, x='modification_date', y='count', color='imported_path_short',
                                barmode='stack', color_discrete_map=color_map, title="File Count by Modification Date")
                report_elements.append(dcc.Graph(figure=fig_ts))

            # --- 4. Add Table for Constant Values ---
            if constant_values_data:
                report_elements.append(html.H5("Properties shared between all files", className="card-title mt-4"))
                grid = dag.AgGrid(
                    rowData=constant_values_data,
                    columnDefs=[{"field": "Property"}, {"field": "Value"}],
                    columnSize="sizeToFit",
                    dashGridOptions={"domLayout": "autoHeight"}
                )
                report_elements.append(html.Div([grid]))

            return report_elements