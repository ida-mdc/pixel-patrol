from typing import List, Dict, Set

import dash_ag_grid as dag
import plotly.express as px
import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.widget_categories import WidgetCategories


class FileStatisticsWidget:
    NAME: str = "File Statistics Report"
    TAB: str = WidgetCategories.FILE_STATS.value
    REQUIRES: Set[str] = {
        "name",
        "file_extension",
        "imported_path_short",
        "size_bytes",
        "modification_date",
    }
    REQUIRES_PATTERNS = None

    CONTENT_ID = "file-stats-report-content"

    def layout(self) -> List:
        return [html.Div(id=self.CONTENT_ID)]

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output(self.CONTENT_ID, "children"),
            Input("color-map-store", "data"),
        )
        def update_file_stats_report(color_map: Dict[str, str]):
            color_map = color_map or {}
            report_elements = []
            constant_values_data = []

            df_filtered = df_global.select(
                pl.col("name"),
                pl.col("file_extension"),
                pl.col("imported_path_short"),
                pl.col("size_bytes"),
                pl.col("modification_date"),
            )

            # --- 1) File Extension Analysis ---
            ext_agg = df_filtered.group_by("file_extension").agg(pl.count()).drop_nulls()
            if ext_agg.height <= 1:
                if not ext_agg.is_empty():
                    constant_values_data.append(
                        {"Property": "File Extension", "Value": ext_agg["file_extension"][0]}
                    )
            else:
                plot_data_ext_count = df_filtered.group_by(
                    ["file_extension", "imported_path_short"]
                ).agg(pl.count())

                fig_ext_count = px.bar(
                    plot_data_ext_count,
                    x="file_extension",
                    y="count",
                    color="imported_path_short",
                    barmode="stack",
                    color_discrete_map=color_map,
                    title="File Count by Extension",
                )

                n = plot_data_ext_count["file_extension"].n_unique()
                if n == 1:
                    fig_ext_count.update_layout(bargap=0.7)
                if n == 2:
                    fig_ext_count.update_layout(bargap=0.4)

                report_elements.append(dcc.Graph(figure=fig_ext_count))

                plot_data_ext_size = df_filtered.group_by(
                    ["file_extension", "imported_path_short"]
                ).agg(pl.sum("size_bytes"))

                fig_ext_size = px.bar(
                    plot_data_ext_size,
                    x="file_extension",
                    y="size_bytes",
                    color="imported_path_short",
                    barmode="stack",
                    color_discrete_map=color_map,
                    title="Total Size by Extension",
                )
                n = plot_data_ext_size["file_extension"].n_unique()
                if n == 1:
                    fig_ext_size.update_layout(bargap=0.7)
                if n == 2:
                    fig_ext_size.update_layout(bargap=0.4)

                report_elements.append(dcc.Graph(figure=fig_ext_size))

            # --- 2) File Size Analysis ---
            bins = [1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024, 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024]
            labels = ["<1 MB", "1-10 MB", "10-100 MB", "100MB-1GB", "1-10 GB", ">10 GB"]
            size_df = df_filtered.with_columns(pl.col("size_bytes").cut(bins, labels=labels).alias("size_bin"))
            size_agg = size_df.group_by("size_bin").agg(pl.count()).drop_nulls()

            if size_agg.height <= 1:
                if not size_agg.is_empty():
                    constant_values_data.append(
                        {"Property": "File Size Bin", "Value": size_agg["size_bin"][0]}
                    )
            else:
                plot_data_size = size_df.group_by(["size_bin", "imported_path_short"]).agg(pl.count())
                fig_size = px.bar(
                    plot_data_size,
                    x="size_bin",
                    y="count",
                    color="imported_path_short",
                    barmode="stack",
                    color_discrete_map=color_map,
                    title="File Count by Size Bin",
                )
                fig_size.update_layout(xaxis={"categoryorder": "array", "categoryarray": labels})
                report_elements.append(dcc.Graph(figure=fig_size))

            # --- 3) File Timestamp Analysis ---
            ts_df = df_filtered.with_columns(pl.col("modification_date").cast(pl.Datetime))
            ts_agg = ts_df.group_by(ts_df["modification_date"].dt.truncate("1d")).agg(pl.count()).drop_nulls()

            if ts_agg.height <= 1:
                if not ts_agg.is_empty():
                    constant_values_data.append(
                        {
                            "Property": "Modification Date (Day)",
                            "Value": ts_agg["modification_date"][0].strftime("%Y-%m-%d"),
                        }
                    )
            else:
                plot_data_ts = ts_df.group_by(
                    [ts_df["modification_date"].dt.truncate("1d"), "imported_path_short"]
                ).agg(pl.count())

                fig_ts = px.bar(
                    plot_data_ts,
                    x="modification_date",
                    y="count",
                    color="imported_path_short",
                    barmode="stack",
                    color_discrete_map=color_map,
                    title="File Count by Modification Date",
                )
                report_elements.append(dcc.Graph(figure=fig_ts))

            # --- 4) Constants table ---
            if constant_values_data:
                report_elements.append(
                    html.H5("Properties shared between all files", className="card-title mt-4")
                )
                grid = dag.AgGrid(
                    rowData=constant_values_data,
                    columnDefs=[{"field": "Property"}, {"field": "Value"}],
                    columnSize="sizeToFit",
                    dashGridOptions={"domLayout": "autoHeight"},
                )
                report_elements.append(html.Div([grid]))

            return report_elements
