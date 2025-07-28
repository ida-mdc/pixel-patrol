import os
from typing import List, Dict

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import html, dcc, Input, Output

from pixel_patrol.core.loaders.bioio_loader import BioIoLoader
from pixel_patrol.core.spec_provider import get_requirements_as_patterns
from pixel_patrol.report.widget_interface import PixelPatrolWidget
from pixel_patrol.report.widget_categories import WidgetCategories


class DimSizeWidget(PixelPatrolWidget):

    @property
    def tab(self) -> str:
        return WidgetCategories.METADATA.value

    @property
    def name(self) -> str:
        return "Dimension Size Distribution"

    def required_columns(self) -> List[str]:
        return get_requirements_as_patterns(BioIoLoader())

    def layout(self) -> List:
        """Defines the layout of the Dimension Size Distribution widget."""
        return [
            html.Div(id="dim-size-info", style={"marginBottom": "15px"}), # Combined info/ratio
            html.Div(id="xy-size-plot-area", children=[
                html.P("No valid data to plot for X and Y dimension sizes.")
            ]), # Placeholder for XY plot or info message
            html.Div(id="individual-dim-plots-area") # Container for individual histograms
        ]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        """Registers callbacks for the Dimension Size Distribution widget."""
        @app.callback(
            Output("dim-size-info", "children"),
            Output("xy-size-plot-area", "children"),
            Output("individual-dim-plots-area", "children"), # Corrected ID
            Input("color-map-store", "data"), # Input for colors
        )
        def update_dim_size_charts(color_map: Dict[str, str]):
            # Ensure the output ID matches the layout ID
            # layout ID: "individual-dim-plots-area"
            # output ID: "individual-dim_plots_area" - Mismatch here, fixed in the return section below too.

            # --- Data Preprocessing (all in Polars) ---
            # Assume imported_path_short and _size columns are already correctly typed and named in df_global.
            # If imported_path_short needs to be re-calculated or other size columns need casting,
            # ensure that logic is here. For simplicity, assuming df_global is ready.
            processed_df = df_global.clone() # Start with a clone to avoid modifying df_global directly

            # --- 1. X and Y Size Distribution (Bubble Chart) ---
            x_col, y_col = "X_size", "Y_size"

            # Explicitly select necessary columns from processed_df *before* filtering
            # This makes the column names explicit in the projection plan
            initial_xy_data = processed_df.select(
                pl.col(x_col),
                pl.col(y_col),
                pl.col("imported_path_short"),
                pl.col("name")  # Needed for bubble_data_agg and hover_data
            )

            # Filter for rows where both x_size and y_size are valid numbers (>1, assuming 1 is null-like)
            xy_plot_data = initial_xy_data.filter(  # Filter on initial_xy_data
                (pl.col(x_col).is_not_null()) & (pl.col(y_col).is_not_null()) &
                (pl.col(x_col) > 1) & (pl.col(y_col) > 1)
            ).with_columns(
                pl.lit(1).alias("value_count")
            )

            if xy_plot_data.height == 0:
                xy_size_plot_children = [html.P("No valid data to plot for X and Y dimension sizes.")]
            else:
                # Aggregate for bubble size: count occurrences of (x_size, y_size, folder)
                bubble_data_agg = xy_plot_data.group_by(
                    [x_col, y_col, "imported_path_short"]
                ).agg(
                    pl.sum("value_count").alias("bubble_size"),
                    pl.col("name").unique().alias("names_in_group")  # Collect names for hover
                ).sort(
                    [x_col, y_col, "imported_path_short"]
                )

                fig_bubble = px.scatter(
                    bubble_data_agg,
                    x=x_col,
                    y=y_col,
                    size='bubble_size',
                    color='imported_path_short',
                    color_discrete_map=color_map, # Apply color map directly from input
                    title="Distribution of X and Y Dimension Sizes",
                    labels={
                        x_col: x_col.replace('_', ' ').title(),
                        y_col: y_col.replace('_', ' ').title(),
                        'bubble_size': 'Count',
                        'imported_path_short': 'Folder'
                    },
                    hover_data=['imported_path_short', 'bubble_size', 'names_in_group'],
                )

                fig_bubble.update_layout(
                    height=500,
                    margin=dict(l=50, r=50, t=80, b=100),
                    hovermode='closest',
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    ),
                    template='plotly' # Add template for robustness
                )
                xy_size_plot_children = [dcc.Graph(figure=fig_bubble)]


            # --- 2. Individual Dimension Histograms (SIMPLIFIED) ---
            individual_dim_plots = []
            all_ratios_text_components = [] # List to hold html.Span and html.Br components

            # Identify all columns ending with '_size' dynamically
            dimension_size_cols = [col for col in processed_df.columns if col.endswith('_size')]

            # Prepare data for all individual histograms at once using Polars melt
            # We'll melt all _size columns into 'dimension_name' and 'dimension_value'
            # Filter for valid values (>1 and not null) before melting
            melted_df = processed_df.filter(
                pl.any_horizontal([pl.col(c).is_not_null() & (pl.col(c) > 1) for c in dimension_size_cols])
            ).melt(
                id_vars=["imported_path_short", "name"], # Keep these columns as identifiers
                value_vars=dimension_size_cols,           # These are the columns to melt
                variable_name="dimension_name",          # New column for original column names (e.g., 'X_size')
                value_name="dimension_value"             # New column for the values (e.g., 1024)
            ).filter(
                pl.col("dimension_value").is_not_null() & (pl.col("dimension_value") > 1)
            )

            # Calculate and display availability ratios (can be done from processed_df or melted_df)
            for column in dimension_size_cols:
                col_present_count = processed_df.filter(
                    (pl.col(column).is_not_null()) & (pl.col(column) > 1)
                ).height
                col_total_files = df_global.height
                col_ratio_text = (
                    f"{column.replace('_', ' ').title()}: {col_present_count} of {col_total_files} files ({((col_present_count / col_total_files) * 100):.2f}%)."
                    if col_total_files > 0 else f"{column.replace('_', ' ').title()}: No files."
                )
                all_ratios_text_components.append(html.Span(col_ratio_text))
                all_ratios_text_components.append(html.Br())


            if melted_df.height == 0:
                individual_dim_plots = [html.P("No valid data to plot for individual dimension sizes.")]
            else:
                # Use px.histogram with faceting for multiple plots
                # Each 'facet_col' will be a different dimension (X_size, Y_size, etc.)
                fig_hist_all = px.histogram(
                    melted_df,
                    x="dimension_value",
                    color="imported_path_short",
                    facet_col="dimension_name",
                    facet_col_wrap=3, # Wrap plots after 3 columns
                    color_discrete_map=color_map,
                    title="Distribution of Individual Dimension Sizes by Folder",
                    labels={
                        "dimension_value": "Dimension Size",
                        "dimension_name": "Dimension",
                        "imported_path_short": "Folder",
                        "count": "Number of Files"
                    },
                    barmode='stack', # Stack bars for different folders
                    hover_data=['name'] # Show 'name' (original file names) on hover
                )

                # Update facet titles for readability
                fig_hist_all.for_each_annotation(lambda a: a.update(text=a.text.replace("dimension_name=", "").replace("_size", " Size")))

                fig_hist_all.update_layout(
                    height=600, # Adjust height as needed for multiple facets
                    margin=dict(l=50, r=50, t=80, b=100),
                    hovermode='x unified', # Unified hover for stacked bars
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5
                    ),
                    template='plotly' # Add template for robustness
                )
                # You can adjust binning globally if needed, e.g., fig_hist_all.update_traces(xbins=dict(size=10))
                # However, with different ranges for different dimensions, dynamic binning per facet is harder with px.
                # Plotly's auto-binning is usually a good starting point.

                individual_dim_plots = [dcc.Graph(figure=fig_hist_all)]

            # Combine all ratio texts for the info div
            dim_size_info_children = [
                html.P(html.B("Overall data availability:")),
                html.P(all_ratios_text_components) # Pass the list directly to html.P
            ]

            return dim_size_info_children, xy_size_plot_children, individual_dim_plots