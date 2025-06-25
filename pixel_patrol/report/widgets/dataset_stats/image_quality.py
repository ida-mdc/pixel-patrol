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


class ImageQualityWidget(PixelPatrolWidget):
    @property
    def tab(self) -> str:
        return WidgetCategories.DATASET_STATS.value

    @property
    def name(self) -> str:
        return "Image Quality"

    def required_columns(self) -> List[str]:
        return [
            "laplacian_variance", "tenengrad", "brenner",
            "noise_estimation", "wavelet_energy",
            "blocking_artifacts", "ringing_artifacts",
            "name", "imported_path"
        ]

    def layout(self) -> List:  # Removed df from layout function
        return [
            html.P(id="dataset-quality-warning", className="text-warning", style={"marginBottom": "15px"}),
            html.Div([
                html.Label("Select value to plot:"),
                # Dropdown options will be populated by a callback
                dcc.Dropdown(id="quality-value-to-plot-dropdown", style={"width": "300px"})
            ]),
            html.P(id="dataset-quality-description", style={"marginBottom": "15px"}),
            dcc.Graph(id="quality-violin-chart", style={"height": "600px"})
        ]

    def get_descriptions(self):
        descriptions = {
            "laplacian_variance": (
                "Measures the sharpness of an image by calculating the variance of the Laplacian. "
                "The Laplacian operator highlights regions of rapid intensity change, such as edges. "
                "A higher value indicates a sharper image with more pronounced edges, while a lower value suggests a blurrier image."
            ),
            "tenengrad": (
                "Reflects the strength of edges in an image by computing the gradient magnitude using the Sobel operator. "
                "Stronger edges typically indicate a clearer and more detailed image. "
                "This metric is often used to assess image focus and sharpness."
            ),
            "brenner": (
                "Captures the level of detail in an image by measuring intensity differences between neighboring pixels. "
                "A higher Brenner score indicates more fine details and textures, while a lower score suggests a smoother or blurrier image. "
                "This metric is particularly useful for evaluating image focus."
            ),
            "noise_estimation": (
                "Estimates the level of random noise present in an image. "
                "Noise can appear as graininess or speckles and is often caused by low light conditions or sensor limitations. "
                "A higher noise level can reduce image clarity and make it harder to distinguish fine details."
            ),
            "wavelet_energy": (
                "Summarizes the amount of high-frequency detail in an image using wavelet transforms. "
                "Wavelets decompose an image into different frequency components, and the energy in the high-frequency bands reflects fine details and textures. "
                "A higher wavelet energy indicates more intricate details, while a lower value suggests a smoother image."
            ),
            "blocking_artifacts": (
                "Detects compression artifacts known as 'blocking,' which occur when an image is heavily compressed (e.g., in JPEG format). "
                "Blocking artifacts appear as visible 8x8 pixel blocks, especially in smooth or gradient regions. "
                "A higher score indicates more severe blocking artifacts, which can degrade image quality."
            ),
            "ringing_artifacts": (
                "Identifies compression artifacts known as 'ringing,' which appear as ghosting or oscillations near sharp edges. "
                "Ringing artifacts are common in compressed images and can make edges look blurry or distorted. "
                "A higher score indicates more pronounced ringing artifacts, which can reduce image clarity."
            ),
        }
        return descriptions

    def register_callbacks(self, app, df_global: pl.DataFrame):
        """Registers callbacks for the Pixel Value Statistics widget."""

        # New callback to populate the dropdown options
        @app.callback(
            Output("quality-value-to-plot-dropdown", "options"),
            Output("quality-value-to-plot-dropdown", "value"),
            Input("color-map-store", "data"), # This input can trigger the initial load
        )
        def set_dropdown_options(color_map: Dict[str, str]):
            # Get the keys from get_descriptions
            description_keys = list(self.get_descriptions().keys())

            # Filter numerical columns to only include those in description_keys
            # Corrected: Use df_global[col].dtype.is_numeric() instead of .is_numeric()
            available_plot_columns = [
                col for col in description_keys
                if col in df_global.columns and df_global[col].dtype.is_numeric()
            ]

            options = [{'label': col.replace('_', ' ').title(), 'value': col} for col in available_plot_columns]

            # Set a default value if options exist
            default_value = available_plot_columns[0] if available_plot_columns else None
            return options, default_value

        @app.callback(
            Output("quality-violin-chart", "figure"),
            Output("dataset-quality-warning", "children"),
            Output("dataset-quality-description", "children"),
            Input("color-map-store", "data"),
            Input("quality-value-to-plot-dropdown", "value")
        )
        def update_quality_chart(color_map: Dict[str, str], value_to_plot: str):
            if not value_to_plot:
                return go.Figure(), "Please select a value to plot.", ""

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
                                           className="text-danger"), ""

            plot_data = processed_df.filter(
                pl.col(value_to_plot).is_not_null()
            )

            if plot_data.is_empty():
                return go.Figure(), html.P(f"No valid data found for '{value_to_plot}' in selected folders.",
                                           className="text-warning"), ""

            warning_message = ""  # Clear warning if data is found
            description_message = self.get_descriptions().get(value_to_plot)

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
                    # We need to map the flat p_values list to the comparison
                    # The `comparisons` list (all combinations) is in the same order as `p_values`
                    try:
                        original_comparison_index = comparisons.index((group1, group2))
                    except ValueError:
                        # Handle cases where the order might be reversed in `comparisons` if it's not strictly sorted
                        original_comparison_index = -1
                        for idx, (c1, c2) in enumerate(comparisons):
                            if (c1 == group1 and c2 == group2) or (c1 == group2 and c2 == group1):
                                original_comparison_index = idx
                                break

                    p_corr = pvals_corrected[original_comparison_index] if original_comparison_index != -1 and original_comparison_index < len(pvals_corrected) else 1.0


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

            return chart, warning_message, description_message