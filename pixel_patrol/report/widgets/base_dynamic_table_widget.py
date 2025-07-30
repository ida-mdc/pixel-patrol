from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List

import polars as pl
from dash import html, dcc, Input, Output, ALL, ctx

from pixel_patrol.report.utils import _parse_dynamic_col, _create_sparkline
from pixel_patrol.report.widget_interface import PixelPatrolWidget


class BaseDynamicTableWidget(PixelPatrolWidget, ABC):
    """
    A reusable base class for widgets that display dynamic stats in a table
    with dimension filters.
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

    def register_callbacks(self, app, df_global: pl.DataFrame):
        """Registers the generic callbacks using the unique widget ID."""

        # This callback populates the filters
        @app.callback(
            Output(f"{self.widget_id}-filters-container", "children"),
            Input("color-map-store", "data")
        )
        def populate_filters(color_map_data):
            all_dims = defaultdict(set)
            # Use the abstract method to get metrics
            supported_metrics = self.get_supported_metrics()
            for col in df_global.columns:
                parsed = _parse_dynamic_col(col, supported_metrics=supported_metrics)
                if parsed:
                    _, dims = parsed
                    for dim_name, dim_idx in dims.items():
                        all_dims[dim_name].add(dim_idx)

            # Create dropdowns using pattern-matching IDs with the widget_id
            dropdowns = []
            for dim_name, indices in sorted(all_dims.items()):
                dropdown_id = {'type': f'dynamic-filter-{self.widget_id}', 'dim': dim_name}
                dropdowns.append(
                    html.Div([
                        html.Label(f"Dimension '{dim_name.upper()}'"),
                        dcc.Dropdown(
                            id=dropdown_id,
                            options=[{'label': 'All', 'value': 'all'}] + [{'label': i, 'value': i} for i in
                                                                          sorted(indices)],
                            value='all',
                            clearable=False
                        )
                    ], className="three columns")
                )
            return dropdowns

        # This callback updates the table
        @app.callback(
            Output(f"{self.widget_id}-table-container", "children"),
            Input({'type': f'dynamic-filter-{self.widget_id}', 'dim': ALL}, 'value')
        )
        def update_stats_table(filter_values):
            filters = {prop['id']['dim']: value for prop, value in zip(ctx.inputs_list[0], filter_values)}
            supported_metrics = self.get_supported_metrics()

            parsed_cols = []
            for col in df_global.columns:
                parsed = _parse_dynamic_col(col, supported_metrics=supported_metrics)
                if parsed:
                    parsed_cols.append({'col': col, 'metric': parsed[0], 'dims': parsed[1]})

            metrics_to_show = sorted({p['metric'] for p in parsed_cols})
            dims_to_plot = sorted({d for p in parsed_cols for d in p['dims']})

            if not metrics_to_show or not dims_to_plot:
                return html.P("No matching dynamic statistics found.")

            header = [html.Th("Metric")] + [html.Th(f"Trend across '{d.upper()}'") for d in dims_to_plot]
            table_rows = []
            for metric in metrics_to_show:
                row_cells = [html.Td(metric.replace('_', ' ').title())]
                for plot_dim in dims_to_plot:
                    cols_for_cell = []
                    for pc in parsed_cols:
                        if pc['metric'] == metric and plot_dim in pc['dims']:
                            if all(f_val == 'all' or pc['dims'].get(f_dim) == f_val for f_dim, f_val in
                                   filters.items()):
                                cols_for_cell.append(pc['col'])

                    if cols_for_cell:
                        cell_content = dcc.Graph(
                            figure=_create_sparkline(df_global, plot_dim, cols_for_cell),
                            config={'displayModeBar': False}
                        )
                    else:
                        cell_content = html.Div("N/A", style={'textAlign': 'center', 'padding': '15px'})
                    row_cells.append(html.Td(cell_content))
                table_rows.append(html.Tr(row_cells))

            return html.Table(
                [html.Thead(html.Tr(header)), html.Tbody(table_rows)],
                className="striped-table",
                style={'width': '100%'}
            )