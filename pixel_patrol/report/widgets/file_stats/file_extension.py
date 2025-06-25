from typing import List, Dict
from dash import html, dcc, Input, Output
import polars as pl
import os
import plotly.express as px


from pixel_patrol.report.widget_interface import PixelPatrolWidget
from pixel_patrol.report.widget_categories import WidgetCategories



class FileExtensionWidget(PixelPatrolWidget):

    @property
    def tab(self) -> str:
        return WidgetCategories.FILE_STATS.value

    @property
    def name(self) -> str:
        return "File Extension Distribution"

    def required_columns(self) -> List[str]:
        # 'file_extension' for x-axis, 'size' for size option, 'imported_path' for color, 'name' for hover
        return ["file_extension", "size_bytes", "imported_path", "name"]

    def layout(self) -> List:
        """Defines the layout of the File Extension Distribution widget."""
        return [
            html.Div(id="file-extension-controls", children=[
                dcc.RadioItems(
                    id="file-extension-yaxis-param",
                    options=[
                        {'label': 'Number of Files', 'value': 'number_of_files'},
                        {'label': 'Total Size of Files', 'value': 'total_size'}
                    ],
                    value='number_of_files', # Default selection
                    inline=True,
                    style={"marginBottom": "20px"}
                )
            ]),
            dcc.Graph(id="file-extension-bar-chart", style={"height": "500px"})
        ]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        """Registers callbacks for the File Extension Distribution widget."""
        @app.callback(
            Output("file-extension-bar-chart", "figure"),
            Input("color-map-store", "data"),
            Input("file-extension-yaxis-param", "value"), # Input for the Y-axis parameter
        )
        def update_file_extension_chart(color_map: Dict[str, str], y_axis_param: str):
            # --- Data Preprocessing (all in Polars) ---

            # Prepare common columns: short folder name
            processed_df = df_global.with_columns([
                pl.col("imported_path").map_elements(
                    lambda x: os.path.basename(x) if x is not None else "Unknown Folder",
                    return_dtype=pl.String
                ).alias("imported_path_short"),
            ])

            # Determine the value column and its aggregation based on y_axis_param
            y_col_name = "count"
            y_axis_title = "Number of Files"
            aggregation_expression = pl.count().alias("count")

            if y_axis_param == "total_size":
                y_col_name = "total_size_bytes"
                y_axis_title = "Total Size (Bytes)"
                # Ensure 'size' column is present and is a numerical type before summing
                if "size_bytes" in processed_df.columns:
                    aggregation_expression = pl.sum("size_bytes").alias("total_size_bytes")
                else:
                    # Fallback if 'size' column is missing, though required_columns should prevent this
                    aggregation_expression = pl.count().alias("count")
                    y_col_name = "count"
                    y_axis_title = "Number of Files (Size Data Missing)"


            # Aggregate data by file_extension and imported_path_short
            plot_data_agg = processed_df.group_by(
                ["file_extension", "imported_path_short"]
            ).agg(
                aggregation_expression,
                pl.col("name").unique().alias("names_in_group"), # Collect names for hover
                pl.col("size_bytes").sum().alias("sum_size_for_hover") # Keep total size for hover
            ).sort(
                ["file_extension", "imported_path_short"]
            )

            # Add the 'color' column for Plotly Express
            plot_data_agg = plot_data_agg.with_columns(
                pl.col("imported_path_short").map_elements(
                    lambda f: color_map.get(f, '#333333'), # Fallback color
                    return_dtype=pl.String
                ).alias("color")
            )

            # --- Plotting using plotly.express ---
            fig = px.bar(
                plot_data_agg,
                x='file_extension',
                y=y_col_name,
                color='imported_path_short',
                barmode='stack', # Stacked bars for this distribution
                color_discrete_map=color_map, # Apply the provided color map
                title="File Extension Distribution",
                labels={
                    'file_extension': "File Extension",
                    y_col_name: y_axis_title,
                    'imported_path_short': 'Folder'
                },
                # Hover data: display folder, count/size, names in group
                hover_data={
                    'imported_path_short': True,
                    y_col_name: True,
                    'names_in_group': True, # This will show the list of names
                    'sum_size_for_hover': ":.2f" # Format sum_size for hover
                }
            )

            # Customize plot appearance
            fig.update_traces(
                marker_line_color="white",
                marker_line_width=0.5,
                opacity=1,
            )
            fig.update_layout(
                height=500,
                margin=dict(l=50, r=50, t=80, b=100),
                hovermode='closest',
                bargap=0.1,
                bargroupgap=0.05,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )

            return fig