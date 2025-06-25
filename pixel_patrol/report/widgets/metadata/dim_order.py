from typing import List, Dict
from dash import html, dcc, Input, Output
import polars as pl
import os
import plotly.express as px 


from pixel_patrol.report.widget_interface import PixelPatrolWidget
from pixel_patrol.report.widget_categories import WidgetCategories

class DimOrderWidget(PixelPatrolWidget):

    @property
    def tab(self) -> str:
        return WidgetCategories.METADATA.value

    @property
    def name(self) -> str:
        return "Dim Order Distribution"

    def required_columns(self) -> List[str]:
        # These are the columns directly used for this simplified plot
        return ["dim_order", "imported_path", "name"]

    def layout(self) -> List:
        """Defines the layout of the Dim Order Distribution widget."""
        return [
            html.Div(id="dim-order-present-ratio", style={"marginBottom": "15px"}),
            dcc.Graph(id="dim-order-bar-chart", style={"height": "500px"})
        ]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        """Registers callbacks for the Dim Order Distribution widget."""
        @app.callback(
            Output("dim-order-bar-chart", "figure"),
            Output("dim-order-present-ratio", "children"),
            Input("color-map-store", "data"), # Input for color mapping
        )
        def update_dim_order_chart(color_map: Dict[str, str]):
            # --- Data Preprocessing (all in Polars) ---

            # Extract short folder name, filter out null dim_orders, and add a 'value_count' column
            processed_df = df_global.with_columns([
                pl.col("imported_path").map_elements(
                    lambda x: os.path.basename(x) if x is not None else "Unknown Folder",
                    return_dtype=pl.String
                ).alias("imported_path_short"),
                pl.lit(1).alias("value_count") # Column for counting files
            ]).filter(pl.col("dim_order").is_not_null()) # Ensure 'dim_order' is not null

            # Calculate and display the ratio of files with 'dim_order' information
            dim_order_present_count = processed_df.height
            total_files = df_global.height
            dim_order_ratio_text = (
                f"{dim_order_present_count} of {total_files} files ({((dim_order_present_count / total_files) * 100):.2f}%) have 'Dim Order' information."
                if total_files > 0 else "No files to display Dim Order information."
            )

            # Aggregate data for the bar chart: count occurrences of (dim_order, folder)
            plot_data_agg = processed_df.group_by(
                ["dim_order", "imported_path_short"]
            ).agg(
                pl.sum("value_count").alias("count"), # Sum the 'value_count'
                pl.col("name").unique().alias("names_in_group") # Collect unique names for hover
            ).sort(
                ["dim_order", "imported_path_short"]
            )

            # Add the 'color' column to the aggregated Polars DataFrame by mapping from color_map
            plot_data_agg = plot_data_agg.with_columns(
                pl.col("imported_path_short").map_elements(
                    lambda f: color_map.get(f, '#333333'), # Fallback color if folder not in map
                    return_dtype=pl.String
                ).alias("color")
            )

            # --- Plotting using plotly.express ---
            fig = px.bar(
                plot_data_agg, # Pass the Polars DataFrame directly
                x='dim_order',
                y='count', # Use the aggregated 'count'
                color='imported_path_short', # Color bars by folder
                barmode='stack', # Default to stacked bars
                color_discrete_map=color_map, # Apply the provided color map
                title="Dim Order Distribution",
                labels={
                    'dim_order': "Dimension Order",
                    'count': "Number of Files",
                    'imported_path_short': 'Folder'
                },
                # Hover data: display folder, count, and names in group
                hover_data=['imported_path_short', 'count', 'names_in_group']
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

            return fig, dim_order_ratio_text