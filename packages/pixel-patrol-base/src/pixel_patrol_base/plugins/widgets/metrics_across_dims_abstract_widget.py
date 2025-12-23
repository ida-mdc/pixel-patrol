from collections import defaultdict
from typing import List, Dict

import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol_base.report.data_utils import parse_metric_dimension_column
from pixel_patrol_base.report.factory import plot_grouped_scatter, show_no_data_message
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.global_controls import (
    prepare_widget_data,
    GLOBAL_CONFIG_STORE_ID,
    FILTERED_INDICES_STORE_ID,
)


class MetricsAcrossDimensionsWidget(BaseReportWidget):
    """
    Reusable base for widgets that display stats across dimensions in a table.
    Inherits from BaseReportWidget to provide standard Card layout.
    """
    # Subclasses set these:
    NAME: str = "Metrics Across Dimensions"
    TAB: str = ""
    REQUIRES = set()
    REQUIRES_PATTERNS = None

    def __init__(self, widget_id: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Unique ID prefix to avoid Dash callback collisions
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
        """Render the table of metrics vs. dimensions with sparklines."""

        df_filtered, group_col, _resolved, _warning = prepare_widget_data(
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

        # Normalize values so they can be compared to parsed dim indices (ints)
        # Values may be "0" or "t0"; we always keep just the numeric index as string.
        dim_filter = {
            dim.lower(): (
                str(val)[len(dim):] if str(val).startswith(dim) else str(val)
            )
            for dim, val in dims_selection_raw.items()
            if val and val != "All"
        }

        # Filter parsed columns by the selected dimensions
        if dim_filter:
            filtered_parsed = [
                pc
                for pc in parsed_cols
                if all(
                    d in pc["dims"] and str(pc["dims"][d]) == idx
                    for d, idx in dim_filter.items()
                )
            ]
        else:
            filtered_parsed = parsed_cols

        if not filtered_parsed:
            return show_no_data_message()

        # --- Decide which metrics and dimensions to show ---
        metrics_to_show = sorted({p["metric"] for p in filtered_parsed})

        dim_slices = defaultdict(set)
        for p in filtered_parsed:
            for dname, didx in p["dims"].items():
                dim_slices[dname].add(didx)

        # Only keep dims that still have at least 2 slices after filtering
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
            "fontWeight": "500",  # less “header-ish”
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

                # Require at least 2 slices for that dim; otherwise no plot
                if cols_for_cell and len(slice_idxs) > 1:
                    agg = _aggregate_cell_series(
                        df_filtered,
                        group_col=group_col,
                        cols=cols_for_cell,
                        dim_name=plot_dim,
                    )

                    fig = plot_grouped_scatter(
                        agg,
                        x_col="x",
                        y_col="y",
                        group_col=group_col,
                        mode="lines+markers",
                        color_map=color_map or {},
                        show_legend=False,
                        height=120,
                    )

                    cell_content = dcc.Graph(
                        figure=fig,
                        config={"displayModeBar": False},
                        style={"height": "120px", "width": "260px"},
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

                row_cells.append(html.Td(cell_content, style={"width": "260px"}))

            table_rows.append(html.Tr(row_cells))

        return html.Div(
            html.Table(
                [html.Thead(html.Tr(header)), html.Tbody(table_rows)],
                style={
                    "width": "max-content",  # <- don't stretch when few dims
                    "borderCollapse": "collapse",
                    "tableLayout": "fixed",
                },
            ),
            style={"overflowX": "auto"},
        )


    def get_supported_metrics(self) -> List[str]:
        raise NotImplementedError("Subclasses must return the list of supported metric base names.")


def _aggregate_cell_series(
        df: pl.DataFrame,
        *,
        group_col: str,
        cols: List[str],
        dim_name: str,
) -> pl.DataFrame:
    """
    Returns long aggregated df with columns: group_col, x, y
    x is the discrete dim index (as string), y is mean value per group per x.
    """
    if not cols:
        return pl.DataFrame()

    # Keep only what we need
    base = df.select([group_col, *cols])

    # Long form
    long = base.unpivot(
        index=[group_col],
        on=cols,
        variable_name="var",
        value_name="val",
    )

    # Extract dim index from column name, aggregate
    # expects something like "..._{dim_name}12..." somewhere in var
    out = (
        long.with_columns(
            pl.col("var").str.extract(f"_{dim_name}(\\d+)", 1).alias("x")
        )
        .drop_nulls(["x", "val"])
        .group_by([group_col, "x"])
        .agg(pl.mean("val").alias("y"))
        .sort(["x"])
        .select([group_col, "x", "y"])
    )
    return out
