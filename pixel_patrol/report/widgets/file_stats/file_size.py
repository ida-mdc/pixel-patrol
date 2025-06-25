from typing import List, Dict
from dash import html, dcc, Input, Output
import polars as pl
import os
import plotly.express as px


from pixel_patrol.report.widget_interface import PixelPatrolWidget
from pixel_patrol.report.widget_categories import WidgetCategories

class FileSizeWidget(PixelPatrolWidget):

    @property
    def tab(self) -> str:
        return WidgetCategories.FILE_STATS.value

    @property
    def name(self) -> str:
        return "File Size Distribution"

    def required_columns(self) -> List[str]:
        # 'size' for binning, 'imported_path' for color, 'name' for hover
        return ["size_bytes", "imported_path", "name"]

    def layout(self) -> List:
        """Defines the layout of the File Size Distribution widget."""
        return [
            dcc.Graph(id="file-size-bar-chart", style={"height": "500px"})
        ]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        """Registers callbacks for the File Size Distribution widget."""
        @app.callback(
            Output("file-size-bar-chart", "figure"),
            Input("color-map-store", "data"), # Input for color mapping
        )
        def update_file_size_chart(color_map: Dict[str, str]):
            # Define bins and labels for file size categories
            bins = [
                1 * 1024 * 1024,  # 1 MB
                10 * 1024 * 1024,  # 10 MB
                100 * 1024 * 1024,  # 100 MB
                1 * 1024 * 1024 * 1024,  # 1 GB
                10 * 1024 * 1024 * 1024,  # 10 GB
                100 * 1024 * 1024 * 1024, # 100 GB
            ]
            labels = [
                "<1 MB",
                "1 MB - 10 MB",
                "10 MB - 100 MB",
                "100 MB - 1 GB",
                "1 GB - 10 GB",
                "10 GB - 100 GB",
                ">100 GB"
            ]

            # --- Data Preprocessing (all in Polars) ---
            # Extract short folder name and create size_bin
            processed_df = df_global.with_columns([
                pl.col("imported_path").map_elements(
                    lambda x: os.path.basename(x) if x is not None else "Unknown Folder",
                    return_dtype=pl.String
                ).alias("imported_path_short"),
                pl.col("size_bytes").cut(bins, labels=labels).alias("size_bin"),
            ]).filter(
                pl.col("size_bin").is_not_null() & # Filter out entries that didn't fall into a bin
                pl.col("size_bytes").is_not_null() # Ensure 'size' itself isn't null
            )

            # Polars' cut creates a categorical type, but its order might not match `labels`.
            # To ensure the correct order on the x-axis, we'll convert to a categorical with explicit ordering.
            # This is crucial for plotting.
            processed_df = processed_df.with_columns(
                pl.col("size_bin").cast(pl.Categorical)
            )

            # Aggregate data by size_bin and imported_path_short
            plot_data_agg = processed_df.group_by(
                ["size_bin", "imported_path_short"]
            ).agg(
                pl.count().alias("value"), # Count files in each bin
                pl.col("name").unique().alias("names_in_group"), # Collect names for hover
                pl.col("size_bytes").sum().alias("sum_size_for_hover") # Total size for hover
            ).sort(
                "size_bin" # Sort by the ordered categorical column
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
                x='size_bin',
                y='value',
                color='imported_path_short',
                barmode='stack', # Always stacked for simplicity
                color_discrete_map=color_map, # Apply the provided color map
                title="File Size Distribution",
                labels={
                    'size_bin': "File Size Range",
                    'value': "Number of Files",
                    'imported_path_short': 'Selected Folder'
                },
                hover_data={
                    'imported_path_short': True,
                    'value': True,
                    'names_in_group': True, # Show list of names
                    'sum_size_for_hover': ":.2f" # Format total size for hover
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
                ),
                xaxis={'categoryorder': 'array', 'categoryarray': labels} # Explicitly ensure x-axis order
            )

            return fig