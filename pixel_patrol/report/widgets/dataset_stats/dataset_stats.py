import itertools
from pathlib import Path
from typing import List, Dict, Tuple

import plotly.graph_objects as go
import polars as pl
import statsmodels.stats.multitest as smm
from dash import html, dcc, Input, Output, dash_table
from scipy.stats import mannwhitneyu

from pixel_patrol.core.processors.basic_stats_processor import BasicStatsProcessor
from pixel_patrol.core.spec_provider import get_requirements_as_patterns
from pixel_patrol.report.utils import get_sortable_columns
from pixel_patrol.report.widget_categories import WidgetCategories
from pixel_patrol.report.widget_interface import PixelPatrolWidget


class DatasetStatsWidget(PixelPatrolWidget):
    @property
    def tab(self) -> str:
        return WidgetCategories.DATASET_STATS.value

    @property
    def name(self) -> str:
        return "Pixel Value Statistics"

    def required_columns(self) -> List[str]:
        return get_requirements_as_patterns(BasicStatsProcessor())

    def layout(self) -> List:
        """Defines the new layout with a container for all plots and tables."""
        return [
            # This single container will be populated by the callback
            html.Div(id="dataset-stats-container"),

            # The static markdown content remains unchanged
            html.Div(className="markdown-content", children=[
                html.H4("Description of the plots"),
                # ... (rest of your markdown content is unchanged) ...
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
                    "If you want to display the mean intensity across the whole image, select ",
                    html.Code("mean_intensity"),
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

    def _create_single_violin_plot(
            self,
            plot_data: pl.DataFrame,
            value_to_plot: str,
            groups: List[str],
            color_map: Dict[str, str],
    ) -> go.Figure:
        """Helper method to generate one violin plot figure with stats."""
        chart = go.Figure()

        # Add a violin trace for each group
        for group_name in groups:
            df_group = plot_data.filter(pl.col("imported_path_short") == group_name)
            group_color = color_map.get(group_name, '#333333')

            chart.add_trace(go.Violin(
                y=df_group.get_column(value_to_plot),
                name=group_name,
                customdata=df_group.get_column("name").map_elements(lambda p: Path(p).name),
                marker_color=group_color,
                opacity=0.9,
                showlegend=True,
                points="all",
                pointpos=0,
                box_visible=True,
                meanline=dict(visible=True),
                hovertemplate=f"<b>Group: {group_name}</b><br>Value: %{{y:.2f}}<br>Filename: %{{customdata}}<extra></extra>"
            ))

        chart.update_traces(
            marker=dict(line=dict(width=1, color="black")),
            box=dict(line_color="black")
        )

        # Add statistical annotations if more than one group exists
        if len(groups) > 1:
            # (The complex statistical annotation logic is moved here)
            # This logic remains the same as your original code, operating on the `chart` object.
            comparisons = list(itertools.combinations(groups, 2))
            p_values = [
                mannwhitneyu(
                    plot_data.filter(pl.col("imported_path_short") == g1).get_column(value_to_plot),
                    plot_data.filter(pl.col("imported_path_short") == g2).get_column(value_to_plot)
                ).pvalue for g1, g2 in comparisons
            ]

            if p_values:
                reject, pvals_corrected, _, _ = smm.multipletests(p_values, alpha=0.05, method="bonferroni")

                chart.update_layout(xaxis=dict(categoryorder="array", categoryarray=groups))
                positions = {group: i for i, group in enumerate(groups)}
                y_range = plot_data.get_column(value_to_plot).max() - plot_data.get_column(value_to_plot).min()
                y_offset = y_range * 0.05 if y_range > 0 else 1

                # Simplified annotation for adjacent pairs to keep it clean
                for i, (g1, g2) in enumerate(comparisons):
                    if abs(positions[g1] - positions[g2]) > 1: continue  # Only adjacent for now

                    p_corr = pvals_corrected[i]
                    sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"

                    y_max = max(
                        plot_data.filter(pl.col("imported_path_short") == g1).get_column(value_to_plot).max(),
                        plot_data.filter(pl.col("imported_path_short") == g2).get_column(value_to_plot).max()
                    )
                    y_bracket = y_max + y_offset * (abs(positions[g1] - positions[g2]))

                    chart.add_shape(type="line", x0=positions[g1], y0=y_bracket, x1=positions[g2], y1=y_bracket,
                                    line=dict(color="black", width=1.5))
                    chart.add_annotation(x=(positions[g1] + positions[g2]) / 2, y=y_bracket + y_offset / 4, text=sig,
                                         showarrow=False)

        # Final layout updates for the single plot
        chart.update_layout(
            title_text=f"Distribution of {value_to_plot.replace('_', ' ').title()}",
            xaxis_title="Folder",
            yaxis_title=value_to_plot.replace('_', ' ').title(),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode='closest',
            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5)
        )
        return chart

    def register_callbacks(self, app, df_global: pl.DataFrame):
        """Registers a single callback to generate all plots and tables."""

        @app.callback(
            Output("dataset-stats-container", "children"),
            Input("color-map-store", "data")
        )
        def update_dataset_stats_layout(color_map: Dict[str, str]):

            groups = df_global["imported_path_short"].unique().sort().to_list()
            if not groups:
                return html.P("No data available to generate statistics.", className="text-warning")

            # --- 2. Partition Columns (Plot vs. Table) ---
            numeric_cols = BasicStatsProcessor().get_specification().keys()
            cols_to_plot = []
            no_variance_data = []

            for col in numeric_cols:
                series = df_global.get_column(col).drop_nulls()
                if series.n_unique() == 1:
                    no_variance_data.append({"Metric": col.replace('_', ' ').title(), "Value": f"{series[0]:.4f}"})
                elif series.n_unique() > 1:
                    cols_to_plot.append(col)

            # --- 3. Generate Plot Components ---

            # Determine column width based on number of groups to keep layout clean
            num_groups = len(groups)
            if num_groups <= 2:
                col_class = "four columns"  # 3 plots per row
            elif num_groups == 3:
                col_class = "six columns"  # 2 plots per row
            else:  # 4 or more groups
                col_class = "twelve columns"  # 1 plot per row

            plot_divs = [
                html.Div(
                    dcc.Graph(figure=self._create_single_violin_plot(df_global, col_name, groups, color_map)),
                    className=col_class,
                    style={"marginBottom": "20px"}
                ) for col_name in cols_to_plot
            ]
            plots_container = html.Div(plot_divs, className="row")

            table_component = []
            if no_variance_data:
                table_component = [
                    html.Hr(),
                    html.H4("Metrics with No Variance", style={"marginTop": "30px", "marginBottom": "15px"}),
                    dash_table.DataTable(
                        data=no_variance_data,
                        columns=[{"name": i, "id": i} for i in ["Metric", "Value"]],
                        style_cell={'textAlign': 'left'},
                        style_header={'fontWeight': 'bold'},
                        style_as_list_view=True,
                    )
                ]

            return [plots_container] + table_component