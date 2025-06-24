import itertools
from pathlib import Path
from typing import List, Dict

import plotly.graph_objects as go
import polars as pl
import statsmodels.stats.multitest as smm  # For Bonferroni correction
from dash import html, dcc, Input, Output  # Import necessary Dash components
from scipy.stats import mannwhitneyu  # For statistical tests

from pixel_patrol.report.widget_interface import PixelPatrolWidget  # Corrected import path
from pixel_patrol.report.widget_categories import WidgetCategories


class DatasetStatsWidget(PixelPatrolWidget):
    @property
    def tab(self) -> str:
        return WidgetCategories.DATASET_STATS.value

    @property
    def name(self) -> str:
        return "Pixel Value Statistics"

    def required_columns(self) -> List[str]:
        # These are the columns needed for statistical analysis and plotting
        # Assuming 'mean', 'median', 'std', 'min', 'max' are direct columns now
        # Also need 'name' for hover info and 'imported_path' for grouping/colors
        return ["mean", "median", "std", "min", "max", "name", "imported_path"]

    def layout(self) -> List:
        """Defines the layout of the Pixel Value Statistics widget."""
        return [
            html.P(id="dataset-stats-warning", className="text-warning", style={"marginBottom": "15px"}),
            # Warning for no data
            html.Div([
                html.Label("Select value to plot:"),
                dcc.Dropdown(
                    id="stats-value-to-plot-dropdown",
                    options=[],  # Options will be populated by callback
                    value=None,  # Default value will be set by callback
                    clearable=False,  # Don't allow clearing the selection
                    style={"width": "300px", "marginTop": "10px", "marginBottom": "20px"}
                )
            ]),
            dcc.Graph(id="stats-violin-chart", style={"height": "600px"}),  # Set a fixed height for clarity
            html.Div(className="markdown-content", children=[  # Static markdown content
                html.H4("Description of the test"),
                html.P([
                    html.Strong("Selectable values to plot: "),
                    "The selected representation of intensities within an image is plotted on the y-axis, while the x-axis shows the different groups (folders) selected. This is calculated on each individual image in the selected folders."
                ]),
                html.P([
                    "Each image is represented by a dot, and the boxplot shows the distribution of the selected value for each group."
                ]),
                html.P([
                    html.Strong("Images with more than 2 dimensions: "),
                    "As images can contain multiple time points (t), channels (c), and z-slices (z), the statistics are calculated across all dimensions. To e.g. visualize the distribution of mean intensities across all z-slices and channels at time point t0, please select e.g. ",
                    html.Code("mean_intensity_t0"), "."
                ]),
                html.P([
                    "If you want to display the mean intensity across the whole image, select ", html.Code("mean_intensity"),
                    " (without any suffix)."
                ]),
                html.P([
                    html.Strong("Higher dimensional images that include RGB data: "),
                    "When an image with Z-slices or even time points contains RGB data, the S-dimension is added. Therefore, the RGB color is indicated by the suffix ",
                    html.Code("s0"), ", ", html.Code("s1"), ", and ", html.Code("s2"),
                    " for red, green, and blue channels, respectively. This allows for images with multiple channels, where each channels consists of an RGB image itself, while still being able to select the color channel."
                ]),
                html.P([
                    "The suffixes are as follows:", html.Br(),
                    html.Ul([
                        html.Li(html.Code("t: time point")),
                        html.Li(html.Code("c: channel")),
                        html.Li(html.Code("z: z-slice")),
                        html.Li(html.Code("s: color in RGB images (red, green, blue)"))
                    ])
                ]),
                html.H4("Statistical hints:"),
                html.P([
                    "The symbols (", html.Code("*"), " or ", html.Code("ns"),
                    ") shown above indicate the significance of the differences between two groups, with more astersisk indicating a more significant difference. The Mann-Whitney U test is applied to compare the distributions of the selected value between pairs of groups. This non-parametric test is used as a first step to assess whether the distributions of two independent samples. The results are adjusted with a Bonferroni correction to account for multiple comparisons, reducing the risk of false positives."
                ]),
                html.P([
                    "Significance levels:", html.Br(),
                    html.Ul([
                        html.Li(html.Code("ns: not significant")),
                        html.Li(html.Code("*: p < 0.05")),
                        html.Li(html.Code("**: p < 0.01")),
                        html.Li(html.Code("***: p < 0.001"))
                    ])
                ]),
                html.H5("Disclaimer:"),
                html.P(
                    "Please do not interpret the results as a final conclusion, but rather as a first step to assess the differences between groups. This may not be the appropriate test for your data, and you should always consult a statistician for a more detailed analysis.")
            ])
        ]

    def register_callbacks(self, app, df_global: pl.DataFrame):
        """Registers callbacks for the Pixel Value Statistics widget."""

        # NEW CALLBACK: To populate the dropdown options dynamically
        @app.callback(
            Output("stats-value-to-plot-dropdown", "options"),
            Output("stats-value-to-plot-dropdown", "value"),
            Input("color-map-store", "data"), # This input can serve as a trigger for initial load
            prevent_initial_call=False # Allow this callback to run on initial load
        )
        def set_stats_dropdown_options(color_map: Dict[str, str]):
            # Filter for numeric columns that are likely candidates for plotting stats
            # We use df_global here, which is available in the register_callbacks scope
            numeric_cols_for_plot = [
                col for col in df_global.columns
                if df_global[col].dtype.is_numeric() and any(metric in col for metric in self.required_columns()[:-2])
            ]

            dropdown_options = [{'label': col, 'value': col} for col in numeric_cols_for_plot]

            # Set a default value if available
            default_value_to_plot = 'mean' if 'mean' in numeric_cols_for_plot else (
                numeric_cols_for_plot[0] if numeric_cols_for_plot else None)

            return dropdown_options, default_value_to_plot

        @app.callback(
            Output("stats-violin-chart", "figure"),
            Output("dataset-stats-warning", "children"),
            Input("color-map-store", "data"),
            Input("stats-value-to-plot-dropdown", "value")
        )
        def update_stats_chart(color_map: Dict[str, str], value_to_plot: str):
            if not value_to_plot:
                return go.Figure(), "Please select a value to plot."

            # --- Data Preprocessing (all in Polars) ---
            # Ensure necessary columns are available and handle potential None values
            processed_df = df_global.filter(
                pl.col("imported_path").is_not_null()  # Ensure folder is present for grouping
            ).with_columns([
                pl.col("imported_path").map_elements(
                    lambda x: Path(x).name if x is not None else "Unknown Folder",
                    # Use Path.name for short folder names
                    return_dtype=pl.String
                ).alias("imported_path_short"),
            ])

            # Filter out rows where the selected value_to_plot is null
            if value_to_plot not in processed_df.columns:
                return go.Figure(), html.P(f"Error: Column '{value_to_plot}' not found in data.",
                                           className="text-danger")

            plot_data = processed_df.filter(
                pl.col(value_to_plot).is_not_null()
            )

            if plot_data.is_empty():
                return go.Figure(), html.P(f"No valid data found for '{value_to_plot}' in selected folders.",
                                           className="text-warning")

            warning_message = ""  # Clear warning if data is found

            # --- Violin plots ---
            chart = go.Figure()

            # Get unique folders in the order they appear in the data for consistent legend/axis
            groups = plot_data.get_column("imported_path_short").unique().to_list()

            # Sort groups alphabetically for consistent display unless a specific order is desired
            groups.sort()

            for imported_path_short in groups:
                df_group = plot_data.filter(
                    pl.col("imported_path_short") == imported_path_short
                )

                # Convert Polars Series to Python list for Plotly trace
                data_values = df_group.get_column(value_to_plot).to_list()
                file_names = df_group.get_column("name").to_list()
                file_names_short = [str(Path(x).name) if x is not None else "Unknown File" for x in
                                    file_names]  # Shorten file names

                # Get color for the group from the color_map
                group_color = color_map.get(imported_path_short, '#333333')  # Fallback color

                chart.add_trace(
                    go.Violin(
                        y=data_values,
                        name=imported_path_short,
                        customdata=file_names_short,  # Pass short file names for hover
                        marker_color=group_color,
                        opacity=0.9,
                        showlegend=True,
                        points="all",  # Display individual points
                        pointpos=0,
                        box_visible=True,  # Show box plot inside the violin plot
                        meanline=dict(visible=True),
                        # Use '%{y}<br>Filename: %{customdata}' for hover template
                        hovertemplate=f"<b>Group: {imported_path_short}</b><br>" +
                                      f"Value: %{{y:.2f}}<br>Filename: %{{customdata}}<extra></extra>"
                    )
                )

            # Set black outlines to our marker and box plots
            chart.update_traces(
                marker=dict(line=dict(width=1, color="black")),
                box=dict(line_color="black")
            )

            # ---------------------------------------------------------------------
            # Add statistical annotations using pairwise Mann-Whitney U tests with Bonferroni correction

            if len(groups) > 1:  # Only perform tests if there's more than one group
                # Perform all pairwise comparisons for p-values
                comparisons = list(itertools.combinations(groups, 2))
                p_values = []
                for group1, group2 in comparisons:
                    data1 = plot_data.filter(pl.col("imported_path_short") == group1).get_column(
                        value_to_plot).to_list()
                    data2 = plot_data.filter(pl.col("imported_path_short") == group2).get_column(
                        value_to_plot).to_list()

                    if len(data1) > 0 and len(data2) > 0:  # Ensure groups have data for test
                        stat_val, p_val = mannwhitneyu(data1, data2, alternative="two-sided")
                        p_values.append(p_val)
                    else:
                        p_values.append(1.0)  # Assign 1.0 (not significant) if a group is empty

                # Apply Bonferroni correction
                if p_values:  # Only apply correction if there were p_values
                    reject, pvals_corrected, _, _ = smm.multipletests(p_values, alpha=0.05, method="bonferroni")
                else:
                    reject, pvals_corrected = [], []  # No rejections or corrected p-values if no data

                # Ensure consistent ordering on the x-axis
                chart.update_layout(xaxis=dict(categoryorder="array", categoryarray=groups))
                positions = {group: i for i, group in enumerate(groups)}

                # Calculate a y-offset based on the range of the values
                overall_y_min = plot_data.get_column(value_to_plot).min()
                overall_y_max = plot_data.get_column(value_to_plot).max()
                y_range = overall_y_max - overall_y_min
                y_offset = y_range * 0.05  # Base offset

                # Dynamically adjust vertical spacing for annotations based on number of comparisons
                # This logic is simplified; for many groups, lines might overlap.
                # A more sophisticated approach might involve a list of y-levels for annotations.
                annotation_y_levels = {}
                for i in range(len(groups)):
                    annotation_y_levels[groups[i]] = overall_y_max  # Starting y for each group's highest annotation

                # Filter comparisons to adjacent groups for cleaner visualization
                comparisons_to_annotate = []
                for i in range(len(groups) - 1):
                    comparisons_to_annotate.append((groups[i], groups[i + 1]))

                # Add bracket annotations for each adjacent comparison
                for i, (group1, group2) in enumerate(comparisons_to_annotate):
                    # Find the corresponding p-value from the full `comparisons` list
                    try:
                        original_comparison_index = comparisons.index((group1, group2))
                    except ValueError:
                        original_comparison_index = comparisons.index((group2, group1)) # Check reversed order

                    p_corr = pvals_corrected[original_comparison_index] if original_comparison_index < len(pvals_corrected) else 1.0

                    if p_corr < 0.001:
                        sig = "***"
                    elif p_corr < 0.01:
                        sig = "**"
                    elif p_corr < 0.05:
                        sig = "*"
                    else:
                        sig = "ns"

                    # Get max y value for the two current groups
                    y_max1 = plot_data.filter(pl.col("imported_path_short") == group1).get_column(value_to_plot).max()
                    y_max2 = plot_data.filter(pl.col("imported_path_short") == group2).get_column(value_to_plot).max()

                    # Determine the y position for the bracket and annotation
                    # Find a y-level that is just above the highest existing annotation for these groups
                    current_y_level = max(annotation_y_levels.get(group1, overall_y_max),
                                          annotation_y_levels.get(group2, overall_y_max))
                    y_bracket = max(y_max1, y_max2,
                                    current_y_level) + y_offset  # Place bracket above highest data point + offset

                    # Update annotation levels for future comparisons
                    annotation_y_levels[group1] = y_bracket + y_offset
                    annotation_y_levels[group2] = y_bracket + y_offset

                    pos1 = positions[group1]
                    pos2 = positions[group2]
                    x_offset_line = 0.05  # Small offset for line endpoints

                    # Add horizontal line connecting the two groups
                    chart.add_shape(
                        type="line",
                        x0=pos1 + x_offset_line, x1=pos2 - x_offset_line,
                        y0=y_bracket, y1=y_bracket,
                        line=dict(color="black", width=1.5), xref="x", yref="y",
                    )
                    # Add vertical lines (brackets) at the ends
                    chart.add_shape(
                        type="line",
                        x0=pos1 + x_offset_line, x1=pos1 + x_offset_line,
                        y0=y_bracket, y1=y_bracket - y_offset / 2,  # Shorter vertical line
                        line=dict(color="black", width=1.5), xref="x", yref="y",
                    )
                    chart.add_shape(
                        type="line",
                        x0=pos2 - x_offset_line, x1=pos2 - x_offset_line,
                        y0=y_bracket, y1=y_bracket - y_offset / 2,  # Shorter vertical line
                        line=dict(color="black", width=1.5), xref="x", yref="y",
                    )

                    # Compute the midpoint for the annotation
                    x_mid = (pos1 + pos2) / 2
                    chart.add_annotation(
                        x=x_mid,
                        y=y_bracket + y_offset / 4,  # Position annotation slightly above the line
                        text=sig,
                        showarrow=False,
                        font=dict(color="black"),
                        xref="x",
                        yref="y",
                    )

            # Update layout titles and legend
            chart.update_layout(
                title_text=f"Distribution of {value_to_plot.replace('_', ' ').title()}",
                xaxis_title="Folder",
                yaxis_title=value_to_plot.replace('_', ' ').title(),
                height=600,
                margin=dict(l=50, r=50, t=80, b=100),
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15,  # Adjust legend position to avoid overlapping with annotations
                    xanchor="center",
                    x=0.5
                )
            )

            return chart, warning_message