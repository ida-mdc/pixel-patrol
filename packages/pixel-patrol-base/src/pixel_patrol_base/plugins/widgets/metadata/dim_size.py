from typing import List, Dict, Set, Optional

import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import (
    prepare_widget_data,
    GLOBAL_CONFIG_STORE_ID,
    FILTERED_INDICES_STORE_ID,
)
from pixel_patrol_base.report.factory import plot_scatter, plot_strip, show_no_data_message

class DimSizeWidget(BaseReportWidget):
    NAME: str = "Dimension Size Distribution"
    TAB: str = WidgetCategories.METADATA.value
    REQUIRES: Set[str] = {"name"}
    REQUIRES_PATTERNS: List[str] = [r"^[a-zA-Z]_size$"]  # dynamic size columns
    CONTENT_ID = "dim-size-content"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "Shows how **image dimensions** (X, Y, Z, T, …) vary across the dataset.\n\n"
            "Includes one X/Y size plot and per-dimension size plot.\n\n"
            "**Use this to identify**\n"
            "- unexpected dimension sizes\n"
            "- mismatched shapes between groupings\n"
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
        color_map: Dict[str, str],
        subset_indices: List[int] | None,
        global_config: Dict | None,
    ):

        df_filtered, group_col, _resolved, _warning, group_order = prepare_widget_data(
                                self._df,
                               subset_indices,
                               global_config or {},
                               metric_base = None,
        )

        # Identify size columns (fast string check)
        dimension_size_cols = [col for col in df_filtered.columns if col.endswith("_size")]

        if df_filtered.height == 0 or not dimension_size_cols:
            return [show_no_data_message()]

        # Select only size columns + metadata
        cols_needed = dimension_size_cols + ["name"]
        if group_col:
            cols_needed.append(group_col)

        df_slim = df_filtered.select([c for c in cols_needed if c in df_filtered.columns])

        # Pre-filter rows where at least one size dimension is a valid number (>1)
        filtered_df = df_slim.filter(
            pl.any_horizontal([pl.col(c).is_not_null() & (pl.col(c) > 1) for c in dimension_size_cols])
        )

        if filtered_df.height == 0:
            return [show_no_data_message("No data available with current filters.")]

        ratio_div = html.Div(
            self._get_availability_ratios(filtered_df, df_filtered.height, dimension_size_cols),
            style={"marginBottom": "15px"}
        )
        xy_plots = self._create_xy_scatter_plot(filtered_df, group_col, color_map)
        strip_plots = self._create_dim_strip_plots(filtered_df, group_col, dimension_size_cols, color_map)

        # Return combined list
        return [ratio_div] + xy_plots + strip_plots

    @staticmethod
    def _get_availability_ratios(df_filtered: pl.DataFrame, total_files: int, cols: List[str]) -> List:
        """Generates the availability summary text for each dimension."""
        ratio_spans: List = []
        for column in cols:
            present = df_filtered.filter(pl.col(column) > 1).height

            # Format text: e.g. "X Size: 100 of 100 files (100.00%)."
            col_label = column.replace('_', ' ').title()
            text = (
                f"{col_label}: {present} of {total_files} files "
                f"({(present / total_files) * 100:.2f}%)."
                if total_files > 0 else f"{col_label}: No files."
            )
            ratio_spans.extend([html.Span(text), html.Br()])

        return [html.P(html.B("Data Availability by Dimension:")), html.P(ratio_spans)]

    @staticmethod
    def _create_xy_scatter_plot(df: pl.DataFrame, group_col: Optional[str], color_map: Dict[str, str]) -> List:

        x_col, y_col = "X_size", "Y_size"
        if x_col not in df.columns or y_col not in df.columns:
            return [html.P("No valid data for X/Y plot. Requires 'X_size' and 'Y_size' columns with data > 1.")]

        xy_plot_data = df.filter((pl.col(x_col) > 1) & (pl.col(y_col) > 1))

        if xy_plot_data.height == 0:
            return [show_no_data_message("No valid data for X/Y plot.")]

        group_keys = [x_col, y_col] + ([group_col] if group_col else [])
        bubble_data_agg = (xy_plot_data.
                           group_by(group_keys).
                           agg(pl.count().alias("bubble_size"))
                           .sort([x_col, y_col] + ([group_col] if group_col else []))
                           )

        fig = plot_scatter(
            df=bubble_data_agg,
            x=x_col,
            y=y_col,
            size="bubble_size",
            color=group_col if group_col else None,
            color_map=color_map,
            title="Distribution of X and Y Dimension Sizes",
            labels={"bubble_size": "Count", group_col: "Group"} if group_col else {"bubble_size": "Count"}
        )
        return [dcc.Graph(figure=fig)]

    @staticmethod
    def _create_dim_strip_plots(df: pl.DataFrame, group_col: str, cols: List[str], color_map: Dict[str, str]) -> List:
        """Creates the faceted strip plot for all dimensions."""
        # Clean data for plotting: Replace "X_size" with "X Size" directly in the dataframe
        melted_df = (
            df.unpivot(
                index=[group_col, "name"],
                on=cols,
                variable_name="dimension_name",
                value_name="dimension_value",
            )
            .filter(pl.col("dimension_value") > 1)
            .with_columns(
                pl.col("dimension_name")
                .str.replace("_size", " Size")
                .str.replace("_", " ")
            )
        )

        if melted_df.height == 0:
            return [show_no_data_message("No data to plot for individual dimension sizes.")]

        # Calculate dynamic height based on number of facets (cols // 3 rows)
        plot_height = 200 * ((len(cols) + 2) // 3)

        fig = plot_strip(
            df=melted_df,
            x=group_col,
            y="dimension_value",
            color=group_col,
            facet_col="dimension_name",
            facet_col_wrap=3,
            color_map=color_map,
            title="Individual Dimension Sizes per Dataset",
            labels={"dimension_value": "Size", group_col: "Group", "dimension_name": "Dimension"},
            hover_data=["name"],
            height=plot_height
        )

        return [dcc.Graph(figure=fig)]