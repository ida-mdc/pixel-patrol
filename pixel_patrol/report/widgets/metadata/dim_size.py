import os
from typing import List, Dict

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import html, dcc, Input, Output

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
        # List all possible numerical columns that might be used
        return [
            "x_size", "y_size", "z_size", "t_size", "c_size", "s_size", "n_images",
            "imported_path", "name" # Required for plotting and hover info
        ]

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
            Output("individual-dim-plots-area", "children"),
            Input("color-map-store", "data"), # Input for colors
        )
        def update_dim_size_charts(color_map: Dict[str, str]):
            # --- Data Preprocessing (all in Polars) ---
            processed_df = df_global.with_columns([
                pl.col("imported_path").map_elements(
                    lambda x: os.path.basename(x) if x is not None else "Unknown Folder",
                    return_dtype=pl.String
                ).alias("imported_path_short"),
            ])

            # --- 1. X and Y Size Distribution (Bubble Chart) ---
            xy_size_plot_children = []
            x_col, y_col = "x_size", "y_size"

            # Filter for rows where both x_size and y_size are valid numbers (>1, assuming 1 is null-like)
            xy_plot_data = processed_df.filter(
                (pl.col(x_col).is_not_null()) & (pl.col(y_col).is_not_null()) &
                (pl.col(x_col) > 1) & (pl.col(y_col) > 1)
            ).with_columns(
                pl.lit(1).alias("value_count") # Add a column for counting points in bubble size
            )

            if xy_plot_data.height == 0:
                xy_size_plot_children = [html.P("No valid data to plot for X and Y dimension sizes.")]
            else:
                # Aggregate for bubble size: count occurrences of (x_size, y_size, folder)
                bubble_data_agg = xy_plot_data.group_by(
                    [x_col, y_col, "imported_path_short"]
                ).agg(
                    pl.sum("value_count").alias("bubble_size"),
                    pl.col("name").unique().alias("names_in_group") # Collect names for hover
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
                    )
                )
                xy_size_plot_children = [dcc.Graph(figure=fig_bubble)]


            # --- 2. Individual Dimension Histograms ---
            individual_dim_plots = []
            numerical_columns = ['y_size', 'x_size', 'z_size', 't_size', 'c_size', 's_size', 'n_images']
            all_ratios_text_components = [] # List to hold html.Span and html.Br components

            for column in numerical_columns:
                # Filter for valid data in the current column (value > 1 and not null)
                # Ensure column exists before filtering
                if column not in processed_df.columns:
                    col_ratio_text = f"{column.replace('_', ' ').title()}: Column not found in data."
                    all_ratios_text_components.append(html.Span(col_ratio_text))
                    all_ratios_text_components.append(html.Br())
                    continue

                col_plot_data = processed_df.filter(
                    (pl.col(column).is_not_null()) & (pl.col(column) > 1)
                ).with_columns(pl.lit(1).alias("value_count"))

                col_present_count = col_plot_data.height
                col_total_files = df_global.height # Ratio against total files in df_global
                col_ratio_text = (
                    f"{column.replace('_', ' ').title()}: {col_present_count} of {col_total_files} files ({((col_present_count / col_total_files) * 100):.2f}%)."
                    if col_total_files > 0 else f"{column.replace('_', ' ').title()}: No files."
                )
                all_ratios_text_components.append(html.Span(col_ratio_text))
                all_ratios_text_components.append(html.Br()) # Add a break after each ratio text

                if col_plot_data.height == 0:
                    # No data for this column, skip plotting
                    continue

                # Determine bin size
                x_min_s = col_plot_data.select(pl.col(column).min())
                x_max_s = col_plot_data.select(pl.col(column).max())

                if x_min_s.is_empty() or x_max_s.is_empty(): # Handle case where min/max might still be empty after filter
                    continue # Skip plotting if no valid min/max

                x_min, x_max = x_min_s.item(), x_max_s.item()

                if x_min is None or x_max is None: # Further check for None values
                    continue

                range_value = x_max - x_min
                if range_value <= 50: bin_size = 1
                elif range_value <= 500: bin_size = 10
                else: bin_size = 100

                fig_hist = go.Figure()

                # Iterate through unique folders for plotting traces
                unique_folders_in_data = col_plot_data.select(pl.col("imported_path_short")).unique().to_series().to_list()

                for folder in unique_folders_in_data:
                    # Get the color for this folder directly from the color_map
                    folder_color = color_map.get(folder, '#333333') # Fallback color

                    # Filter data for the current folder AND THEN SORT for group_by_dynamic
                    binned_df = col_plot_data.filter(pl.col("imported_path_short") == folder).sort(column).group_by_dynamic(
                        index_column=column,
                        every=f"{bin_size}i",
                        closed="left", # Use 'left' for [start, end) bins
                        by="imported_path_short" # Still group by folder, but it will be single group now
                    ).agg(
                        pl.sum("value_count").alias("count"), # Sum up the value_count for each bin
                        pl.col("name").unique().alias("names_in_group")
                    ).sort(column) # Sort aggregated bins for consistent x-axis order

                    if binned_df.height == 0: # Skip if no data for this folder in this column
                        continue

                    # Prepare x-axis labels for bins
                    binned_x_labels = []
                    for row_idx in range(binned_df.height):
                        bin_start = binned_df.row(row_idx, named=True)[column]
                        if bin_size == 1:
                            binned_x_labels.append(str(int(bin_start)))
                        else:
                            binned_x_labels.append(f"{int(bin_start)}-{int(bin_start + bin_size - 1)}")

                    binned_y = binned_df.select(pl.col("count")).to_series().to_list()
                    binned_names = binned_df.select(pl.col("names_in_group")).to_series().to_list()

                    hover_texts_hist = []
                    for row_idx in range(binned_df.height):
                        row = binned_df.row(row_idx, named=True)
                        names = row['names_in_group']
                        hover_items = [
                            f"{column.replace('_', ' ').title()}: {binned_x_labels[row_idx]}",
                            f"Folder: {row['imported_path_short']}",
                            f"Count: {row['count']}",
                        ]
                        if names:
                            hover_items.append(f"Files: {', '.join(names[:5])}{'...' if len(names) > 5 else ''}")
                        hover_texts_hist.append("<br>".join(hover_items))

                    fig_hist.add_trace(go.Bar(
                        x=binned_x_labels, # Use the generated labels for x-axis
                        y=binned_y,
                        name=folder,
                        marker_color=folder_color,
                        hovertext=hover_texts_hist,
                        hoverinfo="text",
                        showlegend=True,
                    ))

                fig_hist.update_layout(
                    barmode='stack',
                    title=f"Distribution of {column.replace('_', ' ').title()}",
                    xaxis_title=column.replace('_', ' ').title(),
                    yaxis_title="Number of Files",
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
                    xaxis=dict(tickangle=0)
                )
                individual_dim_plots.append(dcc.Graph(figure=fig_hist))

            # Combine all ratio texts for the info div
            # Corrected line: create a list of components explicitly
            dim_size_info_children = [
                html.P(html.B("Overall data availability:")),
                html.P(all_ratios_text_components) # Pass the list directly to html.P
            ]

            return dim_size_info_children, xy_size_plot_children, individual_dim_plots