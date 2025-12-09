from typing import List, Dict, Set, Optional, Tuple, Any

import dash_ag_grid as dag
import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import apply_global_config, GLOBAL_CONFIG_STORE_ID
from pixel_patrol_base.report.factory import plot_bar, show_no_data_message


class FileStatisticsWidget(BaseReportWidget):
    # --- Configuration ---
    NAME: str = "File Statistics"
    TAB: str = WidgetCategories.FILE_STATS.value
    REQUIRES: Set[str] = {
        "name",
        "file_extension",
        "size_bytes",
        "modification_date",
    }

    CONTENT_ID = "file-stats-report-content"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "High-level **file statistics** for the dataset.\n\n"
            "**Charts**\n"
            "- File count by extension\n"
            "- Total size by extension\n"
            "- File count by size bin\n"
            "- File modification timeline\n\n"
            "If a property has **no variance** (e.g. all files share the same extension), "
            "it is summarized in the table instead of a chart."
        )

    def get_content_layout(self) -> List:
        return [html.Div(id=self.CONTENT_ID)]

    def register(self, app, df: pl.DataFrame):
        self._df = df

        app.callback(
            Output(self.CONTENT_ID, "children"),
            Input("color-map-store", "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._update_plot)

    def _update_plot(self, color_map: Dict[str, str], global_config: Dict):
        color_map = color_map or {}
        df = self._df

        # Apply global filters (rows) and get dynamic group column
        df_processed, group_col = apply_global_config(df, global_config)

        if df_processed.height == 0:
            return [show_no_data_message()]

        # Reduce DataFrame to needed columns for speed
        df_filtered = df_processed.select(
            pl.col("name"),
            pl.col("file_extension"),
            pl.col(group_col),
            pl.col("size_bytes"),
            pl.col("modification_date"),
        )

        report_elements: List[Any] = []
        invariant_properties = []

        # --- 1) File Extensions ---
        ext_count_df, ext_size_df, ext_constant = self._get_extension_data(df_filtered, group_col)

        if ext_constant:
            invariant_properties.append(ext_constant)
        elif ext_count_df is not None:
            # Chart 1: File Extensions Counts
            report_elements.append(dcc.Graph(figure=plot_bar(
                df=ext_count_df,
                x="file_extension",
                y="count",
                color=group_col,
                color_map=color_map,
                title="File Count by Extension",
                labels={"file_extension": "Extension", "count": "File count", group_col: "Group"}
            )))
            # Chart 2: File Extensions Sizes
            if ext_size_df is not None:
                report_elements.append(dcc.Graph(figure=plot_bar(
                    df=ext_size_df,
                    x="file_extension",
                    y="size_bytes",
                    color=group_col,
                    color_map=color_map,
                    title="Total Size by Extension",
                    labels={"file_extension": "Extension", "size_bytes": "Total size (bytes)", group_col: "Group"}
                )))

        # --- 2) File Sizes (Bins) ---
        size_df, size_constant, size_order = self._get_size_data(df_filtered, group_col)

        if size_constant:
            invariant_properties.append(size_constant)
        elif size_df is not None:
            report_elements.append(dcc.Graph(figure=plot_bar(
                df=size_df,
                x="size_bin",
                y="count",
                color=group_col,
                color_map=color_map,
                title="File Count by Size Bin",
                labels={"size_bin": "Size bin", "count": "File count", group_col: "Group"},
                order_x=size_order
            )))

        # --- 3) Modification Dates ---
        date_df, date_constant = self._get_date_data(df_filtered, group_col)

        if date_constant:
            invariant_properties.append(date_constant)
        elif date_df is not None:
            report_elements.append(dcc.Graph(figure=plot_bar(
                df=date_df,
                x="modification_date",
                y="count",
                color=group_col,
                color_map=color_map,
                title="File Count by Modification Date",
                labels={"modification_date": "Date", "count": "File count", group_col: "Group"}
            )))

        # --- 4) Invariant Properties Table ---
        if invariant_properties:
            report_elements.append(
                html.H5("Properties shared between all files", className="card-title mt-4")
            )
            grid = dag.AgGrid(
                rowData=invariant_properties,
                columnDefs=[{"field": "Property"}, {"field": "Value"}],
                columnSize="sizeToFit",
                dashGridOptions={"domLayout": "autoHeight"},
            )
            report_elements.append(html.Div([grid]))

        return report_elements

    @staticmethod
    def _get_extension_data(df: pl.DataFrame, group_col: str) -> Tuple[
        Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[Dict]]:
        """
        Returns:
            (count_df, size_df, constant_dict)
        """
        # Check variance across the whole filtered set
        ext_agg = df.group_by("file_extension").agg(pl.count()).drop_nulls()

        if ext_agg.height == 1:
            return None, None, {"Property": "File Extension", "Value": ext_agg["file_extension"][0]}

        if ext_agg.is_empty():
            return None, None, None

        # Generate aggregated plotting data
        count_df = df.group_by(["file_extension", group_col]).agg(pl.count().alias("count"))
        size_df = df.group_by(["file_extension", group_col]).agg(pl.sum("size_bytes"))

        return count_df, size_df, None

    @staticmethod
    def _get_size_data(df: pl.DataFrame, group_col: str) -> Tuple[
        Optional[pl.DataFrame], Optional[Dict], Optional[List[str]]]:
        """
        Returns:
            (plot_df, constant_dict, bin_order_list)
        """
        bins = [1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024, 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024]
        labels = ["<1 MB", "1-10 MB", "10-100 MB", "100MB-1GB", "1-10 GB", ">10 GB"]

        size_df = df.with_columns(
            pl.col("size_bytes").cut(bins, labels=labels).alias("size_bin")
        )

        size_agg = size_df.group_by("size_bin").agg(pl.count()).drop_nulls()

        if size_agg.height == 1:
            return None, {"Property": "File Size Bin", "Value": size_agg["size_bin"][0]}, labels

        if size_agg.is_empty():
            return None, None, labels

        plot_df = size_df.group_by(["size_bin", group_col]).agg(pl.count().alias("count"))
        return plot_df, None, labels

    @staticmethod
    def _get_date_data(df: pl.DataFrame, group_col: str) -> Tuple[Optional[pl.DataFrame], Optional[Dict]]:
        """
        Returns:
            (plot_df, constant_dict)
        """
        ts_df = df.with_columns(pl.col("modification_date").cast(pl.Datetime))
        ts_agg = ts_df.group_by(ts_df["modification_date"].dt.truncate("1d")).agg(pl.count()).drop_nulls()

        if ts_agg.height == 1:
            val = ts_agg["modification_date"][0]
            val_str = val.strftime("%Y-%m-%d") if val else "Unknown"
            return None, {"Property": "Modification Date (Day)", "Value": val_str}

        if ts_agg.is_empty():
            return None, None

        plot_df = ts_df.group_by(
            [ts_df["modification_date"].dt.truncate("1d"), group_col]
        ).agg(pl.count().alias("count"))

        return plot_df, None