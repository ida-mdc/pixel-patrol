from collections import defaultdict
from typing import List, Dict, Set

import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.data_utils import parse_metric_dimension_column
from pixel_patrol_base.report.factory import show_no_data_message, plot_aggregated_scatter
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import prepare_widget_data
from pixel_patrol_base.report.constants import GLOBAL_CONFIG_STORE_ID, FILTERED_INDICES_STORE_ID


class MetricsAcrossDimensionsWidget(BaseReportWidget):
    """
    Reusable base for widgets that display stats across dimensions in a table.
    Inherits from BaseReportWidget to provide standard Card layout.

    IMPROVED: Now shows distribution info (std bands, sample counts) not just means.
    """
    # Subclasses set these:
    NAME: str = "Metrics Across Dimensions"
    TAB: str = ""
    REQUIRES = set()
    REQUIRES_PATTERNS = None

    def __init__(self, widget_id: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.widget_id = widget_id
        self._df: pl.DataFrame | None = None

    def get_content_layout(self) -> List:
        """Defines the generic layout with unique IDs derived from widget_id."""
        return [
            html.Div(id=f"{self.widget_id}-table-container"),
        ]

    def register(self, app, df: pl.DataFrame) -> None:
        self._df = df

        app.callback(
            Output(f"{self.widget_id}-table-container", "children"),
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
        """Render the table of metrics vs. dimensions with distribution-aware sparklines."""

        df_filtered, group_col, _resolved, _warning, _order = prepare_widget_data(
            self._df,
            subset_indices,
            global_config,
            metric_base=None,
        )

        if df_filtered.is_empty():
            return show_no_data_message()

        # Parse metric/dimension columns from the filtered dataframe
        supported_metrics = self.get_supported_metrics()
        parsed_cols = []
        for col in df_filtered.columns:
            parsed = parse_metric_dimension_column(col, supported_metrics=supported_metrics)
            if parsed is None:
                continue
            metric, dims = parsed
            parsed_cols.append({"col": col, "metric": metric, "dims": dims})

        if not parsed_cols:
            return show_no_data_message()

        # --- Dimension filter (from global controls) ---
        dims_selection_raw = global_config.get("dimensions") or {}

        # Build dim_filter: {dim_name: index_value} for active filters
        dim_filter: Dict[str, str] = {}
        for dim, val in dims_selection_raw.items():
            if val and val != "All":
                # Normalize: if val starts with dim letter, strip it
                idx_val = str(val)[len(dim):] if str(val).startswith(dim) else str(val)
                dim_filter[dim.lower()] = idx_val

        filtered_dims_set: Set[str] = set(dim_filter.keys())

        # Filter parsed columns:
        # - Must contain ALL filtered dimensions with matching indices
        # - Must have exactly (len(filtered_dims) + 1) dimensions (one extra to plot across)
        expected_dim_count = len(filtered_dims_set) + 1

        filtered_parsed = []
        for pc in parsed_cols:
            col_dims = pc["dims"]

            # Must have exactly the right number of dimensions
            if len(col_dims) != expected_dim_count:
                continue

            # Must contain all filtered dimensions with correct values
            matches_filter = all(
                d in col_dims and str(col_dims[d]) == idx
                for d, idx in dim_filter.items()
            )
            if matches_filter:
                filtered_parsed.append(pc)

        if not filtered_parsed:
            return show_no_data_message()

        # --- Row-filter awareness: drop slice-columns that have no data after filtering rows ---
        cols_to_check = [pc["col"] for pc in filtered_parsed]
        present_flags = df_filtered.select(
            [pl.col(c).is_not_null().any().alias(c) for c in cols_to_check]).to_dicts()[0]
        filtered_parsed = [pc for pc in filtered_parsed if present_flags.get(pc["col"], False)]
        if not filtered_parsed:
            return show_no_data_message()

        # --- Decide which metrics and dimensions to show ---
        metrics_to_show = sorted({p["metric"] for p in filtered_parsed})

        dim_slices = defaultdict(set)
        for p in filtered_parsed:
            for dname, didx in p["dims"].items():
                # Only collect dimensions NOT in the filter
                if dname not in filtered_dims_set:
                    dim_slices[dname].add(didx)

        dims_to_plot = sorted(
            dname for dname, idxs in dim_slices.items() if len(idxs) >= 2
        )

        if not metrics_to_show or not dims_to_plot:
            return show_no_data_message()

        # --- Build table (metric as row title) ---
        header = [html.Th(f"Across '{d.upper()}' Slices") for d in dims_to_plot]

        table_rows = []
        n_cols = len(dims_to_plot)

        metric_title_style = {
            "fontWeight": "500",
            "fontSize": "1.4rem",
            "color": "#343a40",
            "padding": "6px 10px",
            "borderTop": "1px solid #e9ecef",
            "backgroundColor": "#f8f9fa",
        }

        for metric in metrics_to_show:
            metric_title = metric.replace("_", " ").title()

            # 1) Full-width "row title"
            table_rows.append(
                html.Tr(
                    [html.Td(metric_title, colSpan=n_cols, style=metric_title_style)]
                )
            )

            # 2) Plots row (only dims)
            row_cells = []
            for plot_dim in dims_to_plot:
                cols_for_cell = [
                    pc["col"]
                    for pc in filtered_parsed
                    if pc["metric"] == metric and plot_dim in pc["dims"]
                ]
                slice_idxs = {
                    pc["dims"][plot_dim]
                    for pc in filtered_parsed
                    if pc["metric"] == metric and plot_dim in pc["dims"]
                }

                # Require at least 2 slices for that dim
                if cols_for_cell and len(slice_idxs) > 1:
                    # Use new aggregation that includes distribution stats
                    agg = _aggregate_cell_series_with_distribution(
                        df_filtered,
                        group_col=group_col,
                        cols=cols_for_cell,
                        dim_name=plot_dim,
                        parsed_cols_info=filtered_parsed,
                    )

                    fig = plot_aggregated_scatter(
                        agg,
                        x_col="x",
                        y_col="y_mean",
                        y_std_col="y_std",
                        n_col="n",
                        group_col=group_col,
                        color_map=color_map or {},
                        show_legend=False,
                        height=140,  # Slightly taller to accommodate bands
                    )

                    cell_content = dcc.Graph(
                        figure=fig,
                        config={"displayModeBar": False},
                        style={"height": "140px", "width": "280px"},
                    )

                else:
                    cell_content = html.Div(
                        "N/A",
                        style={
                            "textAlign": "center",
                            "padding": "15px",
                            "color": "#6c757d",
                        },
                    )

                row_cells.append(html.Td(cell_content, style={"width": "280px"}))

            table_rows.append(html.Tr(row_cells))

        return html.Div(
            html.Table(
                [html.Thead(html.Tr(header)), html.Tbody(table_rows)],
                style={
                    "width": "max-content",
                    "borderCollapse": "collapse",
                    "tableLayout": "fixed",
                },
            ),
            style={"overflowX": "auto"},
        )

    def get_supported_metrics(self) -> List[str]:
        raise NotImplementedError("Subclasses must return the list of supported metric base names.")



def _aggregate_cell_series_with_distribution(
        df: pl.DataFrame,
        *,
        group_col: str,
        cols: List[str],
        dim_name: str,
        parsed_cols_info: List[Dict],
) -> pl.DataFrame:
    """
    Aggregate values across groups for a set of columns that vary along dim_name.

    For each column in `cols`, extracts the dim_name index value as the x-coordinate,
    then computes mean, std, and count per group.
    """
    # Build mapping: col_name -> x_val (the index along dim_name)
    col_to_x = {}
    cols_set = set(cols)
    for pc in parsed_cols_info:
        if pc["col"] not in cols_set:
            continue
        if dim_name not in pc["dims"]:
            continue
        col_to_x[pc["col"]] = str(pc["dims"][dim_name])

    if not col_to_x:
        return pl.DataFrame()

    valid_cols = list(col_to_x.keys())

    # Single unpivot
    long = df.select([group_col, *valid_cols]).unpivot(
        index=[group_col],
        on=valid_cols,
        variable_name="var",
        value_name="val",
    )

    # Map column name to x value
    long = long.with_columns(
        pl.col("var").replace(col_to_x).alias("x")
    ).drop_nulls(["val"])

    # Single group_by
    return (
        long
        .group_by([group_col, "x"])
        .agg([
            pl.mean("val").alias("y_mean"),
            pl.std("val").alias("y_std"),
            pl.count("val").alias("n"),
        ])
        .with_columns(pl.col("y_std").fill_null(0))
        .sort("x")
    )