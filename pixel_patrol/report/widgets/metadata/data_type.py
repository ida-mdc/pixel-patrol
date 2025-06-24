from typing import List, Dict
from dash import html, dcc, Input, Output
import polars as pl
import os
import plotly.express as px

from pixel_patrol.report.widget_interface import PixelPatrolWidget
from pixel_patrol.report.widget_categories import WidgetCategories


class DataTypeWidget(PixelPatrolWidget):

    @property
    def tab(self) -> str:
        return WidgetCategories.METADATA.value

    @property
    def name(self) -> str:
        return "Data Type Distribution"

    def required_columns(self) -> List[str]:
        # Only these columns are strictly needed for the simplified plot
        return ["dtype", "imported_path", "name"]

    def layout(self) -> List:
        """Defines the layout of the Data Type Distribution widget."""
        return [
            html.Div(id="dtype-present-ratio", style={"marginBottom": "15px"}),
            dcc.Graph(id="data-type-bar-chart", style={"height": "500px"})
        ]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        """Registers callbacks for the Data Type Distribution widget."""
        @app.callback(
            Output("data-type-bar-chart", "figure"),
            Output("dtype-present-ratio", "children"),
            Input("color-map-store", "data"), # Input for colors
        )
        def update_data_type_chart(color_map: Dict[str, str]):
            # --- Data Preprocessing (all in Polars) ---

            # Prepare data: extract short folder name, filter null dtypes, add 'value' for counting
            processed_df = df_global.with_columns([
                pl.col("imported_path").map_elements(
                    lambda x: os.path.basename(x) if x is not None else "Unknown Folder",
                    return_dtype=pl.String
                ).alias("imported_path_short"),
                pl.lit(1).alias("value_count") # Add a column for counting
            ]).filter(pl.col("dtype").is_not_null()) # Filter out rows with null dtype

            # Calculate ratio of files with 'Data Type' information
            dtype_present_count = processed_df.height
            total_files = df_global.height
            dtype_ratio_text = (
                f"{dtype_present_count} of {total_files} files ({((dtype_present_count / total_files) * 100):.2f}%) have 'Data Type' information."
                if total_files > 0 else "No files to display data type information."
            )

            # Aggregate data for the bar chart: count occurrences of (dtype, folder)
            plot_data_agg = processed_df.group_by(
                ["dtype", "imported_path_short"]
            ).agg(
                pl.sum("value_count").alias("count"), # Sum the 'value_count'
                pl.col("name").unique().alias("names_in_group") # Collect names for hover
            ).sort(
                ["dtype", "imported_path_short"]
            )

            # Add the 'color' column to the aggregated Polars DataFrame by mapping
            plot_data_agg = plot_data_agg.with_columns(
                pl.col("imported_path_short").map_elements(
                    lambda f: color_map.get(f, '#333333'), # Fallback color
                    return_dtype=pl.String
                ).alias("color")
            )

            # --- Plotting using plotly.express ---
            fig = px.bar(
                plot_data_agg, # Pass Polars DataFrame directly
                x='dtype',
                y='count', # Use the aggregated 'count' column
                color='imported_path_short', # Color by folder
                barmode='stack', # Default to stacked bars for simplicity
                color_discrete_map=color_map, # Apply the provided color map
                title="Data Type Distribution",
                labels={
                    'dtype': "Data Type",
                    'count': "Number of Files",
                    'imported_path_short': 'Folder'
                },
                # Hover data: display folder, count, and names in group
                hover_data=['imported_path_short', 'count', 'names_in_group']
            )

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

            return fig, dtype_ratio_text