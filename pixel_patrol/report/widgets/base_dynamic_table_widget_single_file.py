from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Union

import polars as pl
from dash import html, dcc, Input, Output, ALL, ctx, no_update

from pixel_patrol.report.utils import _parse_dynamic_col, _create_sparkline
from pixel_patrol.report.widget_interface import PixelPatrolWidget


class BaseDynamicTableWidget(PixelPatrolWidget, ABC):
    """
    A reusable base class for widgets that display dynamic stats in a table
    with dimension filters. It can operate on a full dataframe (for group stats)
    or subscribe to a dcc.Store (for individual item stats).
    """

    def __init__(self, widget_id: str):
        super().__init__()
        # A unique ID to prevent Dash callback collisions
        self.widget_id = widget_id

    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """Each subclass must implement this to provide its list of metrics."""
        pass

    def layout(self) -> List:
        """Defines the generic layout with unique IDs."""
        return [
            html.Div([
                html.P("Filter the dataset by specific dimension slices."),
                html.Div(id=f"{self.widget_id}-filters-container", className="row"),
            ]),
            html.Div(id=f"{self.widget_id}-table-container"),
        ]

    def register_callbacks(self, app, data_source: Union[pl.DataFrame, str]):
        """
        Registers optimized callbacks that can handle both a static DataFrame
        and a dynamic data store.
        """
        is_dynamic_source = isinstance(data_source, str)
        store_id = data_source if is_dynamic_source else None

        # --- Pre-processing Function ---
        def _get_parsed_dataframe(df: pl.DataFrame, supported_metrics: list) -> pl.DataFrame:
            """Parses all dynamic columns once and returns a structured DataFrame."""
            records = []
            for col in df.columns:
                parsed = _parse_dynamic_col(col, supported_metrics=supported_metrics)
                if parsed:
                    # Assuming parsed is a tuple: (combined_metric, dims_dict)
                    records.append({"col_name": col, "metric": parsed[0], "dims": parsed[1]})
            if not records:
                return pl.DataFrame()
            return pl.from_records(records)

        # --- Callback to create the dimension filter dropdowns ---
        @app.callback(
            Output(f"{self.widget_id}-filters-container", "children"),
            [Input("color-map-store", "data")] + ([Input(store_id, "data")] if is_dynamic_source else [])
        )
        def populate_filters(*args):
            df = pl.from_dicts(args[1]) if is_dynamic_source else data_source
            if df is None or (is_dynamic_source and not args[1]):
                return html.P("Select an item to see available filters.", className="text-muted")

            parsed_df = _get_parsed_dataframe(df, self.get_supported_metrics())
            if parsed_df.is_empty():
                return html.P("No filterable dimensions found for this item.", className="text-muted")

            all_dims = defaultdict(set)
            for dims_dict in parsed_df.get_column("dims"):
                for dim_name, dim_idx in dims_dict.items():
                    all_dims[dim_name].add(dim_idx)

            dropdowns = []
            for dim_name, indices in sorted(all_dims.items()):
                dropdowns.append(html.Div([
                    html.Label(f"Dimension '{dim_name.upper()}'"),
                    dcc.Dropdown(
                        id={'type': f'dynamic-filter-{self.widget_id}', 'dim': dim_name},
                        options=[{'label': 'All', 'value': 'all'}] + [{'label': i, 'value': i} for i in
                                                                      sorted(indices)],
                        value='all',
                        clearable=False
                    )
                ], className="three columns"))
            return dropdowns

        # --- Callback to update the table based on filter selections ---
        @app.callback(
            Output(f"{self.widget_id}-table-container", "children"),
            [Input({'type': f'dynamic-filter-{self.widget_id}', 'dim': ALL}, 'value')] + (
            [Input(store_id, "data")] if is_dynamic_source else [])
        )
        def update_stats_table(*args):
            # BUG FIX: Get dropdown values from the callback arguments, not a separate variable
            dropdown_values = args[0]

            if not ctx.triggered_id and is_dynamic_source:
                return no_update

            df = pl.from_dicts(args[1]) if is_dynamic_source else data_source
            if df is None or (is_dynamic_source and not args[1]):
                return html.P("Select an item to view its stats.", className="text-muted")

            # Create a dictionary of the active filters from the dropdowns
            filters = {prop['id']['dim']: value for prop, value in zip(ctx.inputs_list[0], dropdown_values)}

            # PERFORMANCE: Parse columns once into a structured DataFrame
            parsed_df = _get_parsed_dataframe(df, self.get_supported_metrics())

            if parsed_df.is_empty():
                return html.P("No matching dynamic statistics found.", className="text-warning")

            # Get unique metrics and dimensions directly from the parsed data
            metrics_to_show = sorted(parsed_df["metric"].unique().to_list())
            dims_to_plot = sorted({d for dims_list in parsed_df["dims"] for d in dims_list})

            header = [html.Th("Metric")] + [html.Th(f"Trend across '{d.upper()}'") for d in dims_to_plot]
            table_rows = []

            for metric in metrics_to_show:
                row_cells = [html.Td(metric.replace('_', ' ').title())]
                for plot_dim in dims_to_plot:

                    # Pre-filter for the metric and the dimension-to-be-plotted
                    cell_data_base = parsed_df.filter(
                        (pl.col("metric") == metric)
                    )

                    cols_for_cell = []
                    # Iterate over the much smaller, pre-filtered DataFrame
                    for row in cell_data_base.iter_rows(named=True):
                        # Check if the row's dimensions match the user's filter selections
                        if all(f_val == 'all' or row['dims'].get(f_dim) == f_val for f_dim, f_val in filters.items()):
                            cols_for_cell.append(row['col_name'])

                    if cols_for_cell:
                        cell_content = dcc.Graph(
                            figure=_create_sparkline(df, plot_dim, cols_for_cell),
                            config={'displayModeBar': False}
                        )
                    else:
                        cell_content = html.Div("N/A", style={'textAlign': 'center', 'padding': '15px'})
                    row_cells.append(html.Td(cell_content))
                table_rows.append(html.Tr(row_cells))

            return html.Table([html.Thead(html.Tr(header)), html.Tbody(table_rows)], className="striped-table",
                              style={'width': '100%'})