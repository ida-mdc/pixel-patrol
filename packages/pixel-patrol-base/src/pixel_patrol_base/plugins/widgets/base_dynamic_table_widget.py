from collections import defaultdict
from typing import List

import polars as pl
from dash import html, dcc, Input, Output, ALL, ctx

from pixel_patrol_base.report.utils import _parse_dynamic_col, _create_sparkline


class BaseDynamicTableWidget:
    """
    Reusable base for widgets that display dynamic stats in a table with dimension filters.
    Not a Protocol, just a convenience base for shared layout + callbacks.
    Subclasses should provide:
      - NAME, TAB, REQUIRES, REQUIRES_PATTERNS (class attributes, for gating)
      - get_supported_metrics(self) -> List[str]
    """

    # Subclasses set these:
    NAME: str = "Dynamic Table"
    TAB: str = ""
    REQUIRES = set()
    REQUIRES_PATTERNS = None

    def __init__(self, widget_id: str):
        # Unique ID prefix to avoid Dash callback collisions
        self.widget_id = widget_id

    def get_supported_metrics(self) -> List[str]:
        raise NotImplementedError("Subclasses must return the list of supported metric base names.")

    def layout(self) -> List:
        """Defines the generic layout with unique IDs derived from widget_id."""
        return [
            html.Div([
                html.P("Filter the dataset by specific dimension slices."),
                html.Div(id=f"{self.widget_id}-filters-container", className="row"),
            ]),
            html.Div(id=f"{self.widget_id}-table-container"),
        ]

    def register(self, app, df_global: pl.DataFrame):
        """Registers generic callbacks using the unique widget_id."""

        # Populate filters based on the dynamic columns present
        @app.callback(
            Output(f"{self.widget_id}-filters-container", "children"),
            Input("color-map-store", "data"),  # just a trigger; not used directly
        )
        def populate_filters(_color_map_data):
            all_dims = defaultdict(set)
            supported_metrics = self.get_supported_metrics()

            for col in df_global.columns:
                parsed = _parse_dynamic_col(col, supported_metrics=supported_metrics)
                if parsed:
                    _, dims = parsed
                    for dim_name, dim_idx in dims.items():
                        all_dims[dim_name].add(dim_idx)

            dropdowns = []
            for dim_name, indices in sorted(all_dims.items()):
                if len(indices) <= 1:
                    continue
                dropdown_id = {"type": f"dynamic-filter-{self.widget_id}", "dim": dim_name}
                dropdowns.append(
                    html.Div(
                        [
                            html.Label(f"Dimension '{dim_name.upper()}'"),
                            dcc.Dropdown(
                                id=dropdown_id,
                                options=[{"label": "All", "value": "all"}]
                                        + [{"label": i, "value": i} for i in sorted(indices)],
                                value="all",
                                clearable=False,
                            ),
                        ],
                        className="three columns",
                    )
                )
            return dropdowns

        @app.callback(
            Output(f"{self.widget_id}-table-container", "children"),
            Input({"type": f"dynamic-filter-{self.widget_id}", "dim": ALL}, "value"),
        )
        def update_stats_table(filter_values):
            # Pattern-matching Input gives us ctx.inputs_list
            inputs_list = ctx.inputs_list[0] if ctx.inputs_list else []
            filters = {prop["id"]["dim"]: value for prop, value in zip(inputs_list, filter_values)}
            supported_metrics = self.get_supported_metrics()

            parsed_cols = []
            for col in df_global.columns:
                parsed = _parse_dynamic_col(col, supported_metrics=supported_metrics)
                if parsed:
                    parsed_cols.append({"col": col, "metric": parsed[0], "dims": parsed[1]})

            metrics_to_show = sorted({p["metric"] for p in parsed_cols})
            from collections import defaultdict
            all_dims = defaultdict(set)

            for p in parsed_cols:
                for dname, didx in p["dims"].items():
                    all_dims[dname].add(didx)

            dims_to_plot = sorted([d for d, idxs in all_dims.items() if len(idxs) > 1])

            if not metrics_to_show or not dims_to_plot:
                return html.P("No matching dynamic statistics found.")

            header = [html.Th("Metric")] + [html.Th(f"Trend across '{d.upper()}'") for d in dims_to_plot]
            table_rows = []
            for metric in metrics_to_show:
                row_cells = [html.Td(metric.replace("_", " ").title())]
                for plot_dim in dims_to_plot:
                    cols_for_cell, slice_idxs = [], set()  # track which slice indices exist for this dim
                    for pc in parsed_cols:
                        if pc["metric"] == metric and plot_dim in pc["dims"]:
                            # honor dropdown filters; 'all' passes through
                            if all(f_val == "all" or pc["dims"].get(f_dim) == f_val for f_dim, f_val in
                                   filters.items()):
                                cols_for_cell.append(pc["col"])
                                slice_idxs.add(pc["dims"][plot_dim])

                    # require at least 2 slices for that dim; otherwise no plot
                    if cols_for_cell and len(slice_idxs) > 1:
                        fig = _create_sparkline(df_global, plot_dim, cols_for_cell)
                        fig.update_layout(height=120, margin=dict(l=30, r=10, t=10, b=30))
                        fig.update_xaxes(visible=True, showticklabels=True, showgrid=True, ticks="outside", title_text=None,
                                         tickmode="linear", dtick=1, tickformat="d")
                        fig.update_yaxes(visible=True, showticklabels=True, ticks="outside", tickformat=".2f")
                        cell_content = dcc.Graph(figure=fig, config={"displayModeBar": False})
                    else:
                        cell_content = html.Div("N/A", style={"textAlign": "center", "padding": "15px"})
                    row_cells.append(html.Td(cell_content))
                table_rows.append(html.Tr(row_cells))

            return html.Table(
                [html.Thead(html.Tr(header)), html.Tbody(table_rows)],
                className="striped-table",
                style={"width": "100%"},
            )

