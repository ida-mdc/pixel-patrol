from typing import List, Dict, Set, Optional, Tuple, Any

import dash_ag_grid as dag
import polars as pl
from dash import html, dcc, Input, Output
import math

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import prepare_widget_data
from pixel_patrol_base.report.factory import plot_bar, show_no_data_message
from pixel_patrol_base.report.data_utils import get_all_grouping_cols
from pixel_patrol_base.report.constants import GLOBAL_CONFIG_STORE_ID, FILTERED_INDICES_STORE_ID

MAX_DAYS = 20
SIZE_LOG_THRESHOLD = 30
SIZE_NUM_BINS = 20


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
            Input(FILTERED_INDICES_STORE_ID, "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
        )(self._update_plot)

    def _update_plot(
        self,
        color_map: Dict[str, str] | None,
        subset_indices: List[int] | None,
        global_config: Dict | None,
    ):

        df_processed, group_col, _resolved, _warning_msg, group_order = prepare_widget_data(
            self._df,
            subset_indices,
            global_config or {},
            metric_base=None,
        )

        if df_processed.is_empty():
            return [show_no_data_message()]

        cols = get_all_grouping_cols(
            ["name", "file_extension", "size_bytes", "modification_date"],
            group_col,
        )

        df_filtered = df_processed.select([pl.col(c) for c in cols])

        report_elements: List[Any] = []
        invariant_properties: List[Dict[str, Any]] = []

        # --- 1) File Extensions ---
        ext_count_df, ext_size_df, ext_constant = self._get_extension_data(df_filtered, group_col)

        if ext_constant:
            invariant_properties.append(ext_constant)
        elif ext_count_df is not None:
            labels = {"file_extension": "Extension", "count": "File count"}
            if group_col:
                labels[group_col] = "Group"

            # Chart 1: File Extensions Counts
            report_elements.append(
                dcc.Graph(
                    figure=plot_bar(
                        df=ext_count_df,
                        x="file_extension",
                        y="count",
                        order_x=group_order,
                        color=group_col,
                        color_map=color_map,
                        title="File Count by Extension",
                        labels=labels,
                        force_category_x=True
                    )
                )
            )

            # Chart 2: File Extensions Sizes
            if ext_size_df is not None:
                labels_size = {"file_extension": "Extension", "size_bytes": "Total size (bytes)"}
                if group_col:
                    labels_size[group_col] = "Group"

                report_elements.append(
                    dcc.Graph(
                        figure=plot_bar(
                            df=ext_size_df,
                            x="file_extension",
                            y="size_bytes",
                            order_x=group_order,
                            color=group_col,
                            color_map=color_map,
                            title="Total Size by Extension",
                            labels=labels_size,
                            force_category_x=True
                        )
                    )
                )

        # --- 2) File Sizes (Bins) ---
        size_df, size_constant, size_order, size_use_log = self._get_size_data(df_filtered, group_col)

        if size_constant:
            invariant_properties.append(size_constant)
        elif size_df is not None:
            x_label = "File size (log-spaced bins)" if size_use_log else "File size bin"
            labels = {"size_bin": x_label, "count": "File count"}
            if group_col:
                labels[group_col] = "Group"

            report_elements.append(
                dcc.Graph(
                    figure=plot_bar(
                        df=size_df,
                        x="size_bin",
                        y="count",
                        color=group_col,
                        color_map=color_map,
                        title="File Count by Size Bin",
                        labels=labels,
                        order_x=size_order,
                        force_category_x=True,
                        show_legend=True,
                    )
                )
            )

        # --- 3) Modification Dates ---
        date_df, date_constant = self._get_date_data(df_filtered, group_col)

        if date_constant:
            invariant_properties.append(date_constant)
        elif date_df is not None:
            labels = {"modification_date": "Date", "count": "File count"}
            if group_col:
                labels[group_col] = "Group"

            report_elements.append(
                dcc.Graph(
                    figure=plot_bar(
                        df=date_df,
                        x="modification_date",
                        y="count",
                        color=group_col,
                        color_map=color_map,
                        title="File Count by Modification Date",
                        labels=labels,
                        force_category_x=True,
                        show_legend=True,
                    )
                )
            )

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
    def _get_extension_data(
        df: pl.DataFrame,
        group_col: Optional[str],
    ) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[Dict]]:
        """
        Returns:
            (count_df, size_df, constant_dict)
        """
        # Check variance across the whole filtered set
        ext_agg = df.group_by("file_extension").agg(pl.count()).drop_nulls()

        if ext_agg.is_empty():
            return None, None, None

        if ext_agg.height == 1:
            return None, None, {
                "Property": "File Extension",
                "Value": ext_agg["file_extension"][0],
            }

        count_df = _group_and_agg(
            df = df,
            base_key = "file_extension",
            group_col = group_col,
            metric = pl.count(),
            out_col = "count",
        )

        size_df = _group_and_agg(
            df = df,
            base_key = "file_extension",
            group_col = group_col,
            metric = pl.sum("size_bytes"),
            out_col = "size_bytes",
        )

        return count_df, size_df, None

    @staticmethod
    def _get_size_data(
        df: pl.DataFrame,
        group_col: Optional[str],
    ) -> Tuple[Optional[pl.DataFrame], Optional[Dict], Optional[List[str]], bool]:
        """Build binned size data for the size histogram.

        Returns:
            (plot_df, constant_dict, bin_order_list, use_log)

        - If all sizes are identical -> constant_dict only.
        - Else:
            * If max_size / min_positive < SIZE_LOG_THRESHOLD -> linear bins.
            * Otherwise -> equal-width bins in log10(size) space.
        """
        if "size_bytes" not in df.columns:
            return None, None, [], False

        size_series = df.get_column("size_bytes").drop_nulls()
        if size_series.is_empty():
            return None, None, [], False

        # All files same size -> treat as invariant
        if size_series.n_unique() == 1:
            value = int(size_series[0])
            return None, {
                "Property": "File Size",
                "Value": f"{value} bytes",
            }, [], False

        max_size = float(size_series.max())
        min_size = float(size_series.min())
        n_unique = int(size_series.n_unique())

        positive = size_series.filter(size_series > 0)
        if positive.is_empty():
            min_positive: Optional[float] = None
        else:
            min_positive = float(positive.min())

        # Number of bins: at most SIZE_NUM_BINS, but not more than distinct sizes
        effective_bins = SIZE_NUM_BINS
        if n_unique > 0:
            effective_bins = min(SIZE_NUM_BINS, n_unique)

        # Decide whether to use log binning
        use_log = False
        if min_positive is not None and min_positive > 0.0:
            ratio = max_size / min_positive
            if ratio >= SIZE_LOG_THRESHOLD:
                use_log = True

        breaks: List[float] = []

        if use_log and min_positive is not None and min_positive > 0.0 and effective_bins > 1:
            log_min = math.log10(min_positive)
            log_max = math.log10(max_size)
            if log_max > log_min:
                step_log = (log_max - log_min) / effective_bins
                breaks = [
                    10 ** (log_min + step_log * i)
                    for i in range(1, effective_bins)
                ]

        # Fallback / linear strategy
        if not breaks:
            if max_size <= min_size or effective_bins <= 1:
                return None, None, [], False
            step = (max_size - min_size) / effective_bins
            if step <= 0:
                return None, None, [], False
            breaks = [
                min_size + step * i
                for i in range(1, effective_bins)
            ]

        if not breaks:
            return None, None, [], False

        # Build human-readable labels as full ranges:
        # edges = [min, b1, b2, ..., b_{k-1}, max]
        # labels = ["min–b1", "b1–b2", ..., "b_{k-1}–max"]
        labels: List[str] = []
        lower_bound = min_size
        upper_bound = max_size
        edges: List[float] = [lower_bound] + breaks + [upper_bound]

        for left, right in zip(edges[:-1], edges[1:]):
            labels.append(f"{_format_bytes(left)}–{_format_bytes(right)}")

        size_df = df.with_columns(
            pl.col("size_bytes").cut(breaks, labels=labels).alias("size_bin")
        )

        size_agg = size_df.group_by("size_bin").agg(pl.count()).drop_nulls()
        if size_agg.is_empty():
            return None, None, labels, use_log

        # If everything fell into one bin, treat that as invariant
        if size_agg.height == 1:
            return None, {
                "Property": "File Size Bin",
                "Value": size_agg["size_bin"][0],
            }, labels, use_log

        if group_col:
            group_cols = ["size_bin", group_col]
        else:
            group_cols = ["size_bin"]

        plot_df = size_df.group_by(group_cols).agg(
            pl.count().alias("count")
        )

        return plot_df, None, labels, use_log


    @staticmethod
    def _get_date_data(
        df: pl.DataFrame,
        group_col: Optional[str],
    ) -> Tuple[Optional[pl.DataFrame], Optional[Dict]]:

        # Cast once to datetime
        ts_df = df.with_columns(
            pl.col("modification_date")
            .cast(pl.Datetime)
            .alias("modification_dt")
        )

        if ts_df.is_empty():
            return None, None

        # ---- 1) Day-level buckets (strings -> categorical axis) ----
        ts_days = ts_df.with_columns(
            pl.col("modification_dt")
            .dt.truncate("1d")
            .dt.strftime("%Y-%m-%d")
            .alias("modification_bucket")
        )

        day_agg = _group_and_agg(
            df=ts_days,
            base_key="modification_bucket",
            group_col=None,
            metric=pl.count(),
            out_col="count",
        ).drop_nulls()

        if day_agg.is_empty():
            return None, None

        # All files on same day -> invariant
        if day_agg.height == 1:
            val_str = day_agg["modification_bucket"][0] or "Unknown"
            return None, {"Property": "Modification Date (Day)", "Value": val_str}

        # Few days -> show per-day bars
        if day_agg.height <= MAX_DAYS:
            plot_df = _group_and_agg(
                df=ts_days,
                base_key="modification_bucket",
                group_col=group_col,
                metric=pl.count(),
                out_col="count",
            )
            # Keep external name stable for plotting code
            plot_df = plot_df.rename({"modification_bucket": "modification_date"})
            return plot_df, None

        # ---- 2) Too many days -> month-level buckets ----
        ts_months = ts_df.with_columns(
            pl.col("modification_dt")
            .dt.truncate("1mo")
            .dt.strftime("%Y-%m")
            .alias("modification_bucket")
        )

        month_agg = _group_and_agg(
            df=ts_months,
            base_key="modification_bucket",
            group_col=None,
            metric=pl.count(),
            out_col="count",
        ).drop_nulls()

        if month_agg.is_empty():
            return None, None

        plot_df = _group_and_agg(
            df=ts_months,
            base_key="modification_bucket",
            group_col=group_col,
            metric=pl.count(),
            out_col="count",
        )
        plot_df = plot_df.rename({"modification_bucket": "modification_date"})

        return plot_df, None


def _group_and_agg(
    df: pl.DataFrame,
    base_key: str,
    group_col: Optional[str],
    metric: pl.Expr,
    out_col: str,
) -> pl.DataFrame:

    keys = get_all_grouping_cols([base_key], group_col)

    return df.group_by(keys).agg(metric.alias(out_col)).sort(keys)


def _format_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(value)
    u_idx = 0
    while v >= 1024.0 and u_idx < len(units) - 1:
        v /= 1024.0
        u_idx += 1
    if v >= 10:
        return f"{v:.0f} {units[u_idx]}"
    return f"{v:.1f} {units[u_idx]}"
