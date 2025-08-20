from pathlib import Path
from typing import List

import plotly.graph_objects as go
import polars as pl
from dash import html, dcc, dash_table


def get_sortable_columns(df: pl.DataFrame) -> List[str]:
    """
    Returns a list of sortable column names, excluding columns that end
    with coordinate-like suffixes (e.g., '_x1', '_c0_z5').
    """
    # Regex to find column names ending with one or more _LetterNumber segments.
    slice_pattern = re.compile(r'(_[a-zA-Z]\d+)+$')

    sortable_columns = [
        col for col in df.columns
        if (
            df[col].dtype.is_numeric() and
            not slice_pattern.search(col) and
            df[col].dtype not in [pl.List, pl.Struct, pl.Array]
        )
    ]

    return sortable_columns


import re
from typing import List, Tuple, Dict, Optional


def _parse_dynamic_col(
    col_name: str, supported_metrics: List[str]
) -> Optional[Tuple[str, Dict[str, int]]]:
    """
    Parses a column name that may contain a metric, an optional attribute, and
    one or more dimension suffixes.

    This function correctly handles names in two formats:
    1. metric + dimensions (e.g., 'mean_T1_C0')
    2. metric + attribute + dimensions (e.g., 'mean_intensity_T1_C0')

    Args:
        col_name: The column name to parse.
        supported_metrics: A list of valid base metrics (e.g., ['mean', 'std']).

    Returns:
        A tuple of (base_metric, attribute, dims_dict) if parsing is successful.
        The 'attribute' is a string if present, otherwise None.
        Returns None if the column name doesn't match the expected patterns.
    """
    if not supported_metrics:
        return None

    # 1. Isolate the dimension suffix (e.g., _T1_C0) from the end of the string.
    #    This part of the name has a predictable structure.
    suffix_pattern = re.compile(r"((?:_[a-zA-Z]\d+)+)$")
    suffix_match = suffix_pattern.search(col_name)

    if not suffix_match:
        return None # No valid dimension suffix found.

    suffix_part = suffix_match.group(1)
    prefix_part = col_name[:suffix_match.start(1)]

    # 2. Parse the remaining prefix to find the base metric and optional attribute.
    #    Sort metrics by length (descending) to correctly handle cases where one metric
    #    is a prefix of another (e.g., 'count_total' vs 'count').
    sorted_metrics = sorted(supported_metrics, key=len, reverse=True)

    base_metric = None
    attribute = None

    for metric in sorted_metrics:
        # Case 1: The prefix exactly matches a metric (e.g., prefix is 'mean').
        if prefix_part == metric:
            base_metric = metric
            attribute = None
            break
        # Case 2: The prefix starts with a metric (e.g., prefix is 'mean_intensity').
        elif prefix_part.startswith(metric + '_'):
            base_metric = metric
            attribute = prefix_part[len(metric) + 1:] # Extract the attribute part.
            break

    if base_metric is None:
        return None # The prefix didn't match any supported metric.

    # 3. Parse the individual dimensions from the captured suffix string.
    dim_pattern = re.compile(r'_([a-zA-Z])(\d+)')
    dims = {m.group(1): int(m.group(2)) for m in dim_pattern.finditer(suffix_part)}

    return base_metric + "_" + attribute if attribute else base_metric, dims

def _create_sparkline(df: pl.DataFrame, dim_name: str, cols: List[str]) -> go.Figure:
    """Creates a minimalist, aggregated line plot for a table cell."""
    dim_pattern = re.compile(rf'_{dim_name}(\d+)')

    # Melt the dataframe to a long format and extract the dimension index
    long_df = df.select(cols).unpivot(variable_name="variable", value_name="value").with_columns(
        pl.col("variable").map_elements(lambda x: int(dim_pattern.search(x).group(1)), return_dtype=pl.Int32).alias(
            dim_name)
    )

    # Aggregate the data: calculate mean at each slice point
    agg_df = long_df.group_by(dim_name).agg(pl.mean("value").alias("mean")).sort(dim_name)

    fig = go.Figure(data=go.Scatter(
        x=agg_df[dim_name],
        y=agg_df["mean"],
        mode='lines',
        line=dict(color='rgb(0,100,80)', width=2)
    ))

    # Style the figure to be minimalist
    fig.update_layout(
        showlegend=False,
        margin=dict(l=2, r=2, t=2, b=2),
        height=50,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def generate_column_violin_plots(df_global, color_map, numeric_cols):
    df_filtered = df_global.select(
        pl.col("name"),
        pl.col("imported_path_short"),
        pl.col(numeric_cols),
    )
    # Add this line to remove rows with nulls in the numeric columns
    df_filtered = df_filtered.filter(
        pl.all_horizontal([pl.col(c).is_not_null() for c in numeric_cols])
    )
    groups = df_filtered["imported_path_short"].unique().sort().to_list()
    if not groups:
        return html.P("No data available to generate statistics.", className="text-warning")
    # --- 2. Partition Columns (Plot vs. Table) ---
    cols_to_plot = []
    no_variance_data = []
    for col in numeric_cols:
        series = df_filtered.get_column(col).drop_nulls()
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
    plot_divs = []
    for col_name in cols_to_plot:
        plot = _create_single_violin_plot(df_filtered, col_name, groups, color_map)
        plot_divs.append(html.Div(
            dcc.Graph(figure=plot),
            className=col_class,
            style={"marginBottom": "20px"}
        ))
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

def _create_single_violin_plot(
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
            customdata=df_group.get_column("name").map_elements(lambda p: Path(p).name, return_dtype=pl.String),
            marker_color=group_color,
            opacity=0.9,
            showlegend=True,
            points="all",
            spanmode="hard",
            pointpos=0,
            box_visible=True,
            meanline=dict(visible=True),
            hovertemplate=f"<b>Group: {group_name}</b><br>Value: %{{y:.2f}}<br>Filename: %{{customdata}}<extra></extra>"
        ))

    chart.update_traces(
        marker=dict(line=dict(width=1, color="black")),
        box=dict(line_color="black")
    )

    # # Add statistical annotations if more than one group exists
    # if len(groups) > 1:
    #     # (The complex statistical annotation logic is moved here)
    #     # This logic remains the same as your original code, operating on the `chart` object.
    #     comparisons = list(itertools.combinations(groups, 2))
    #     p_values = [
    #         mannwhitneyu(
    #             plot_data.filter(pl.col("imported_path_short") == g1).get_column(value_to_plot),
    #             plot_data.filter(pl.col("imported_path_short") == g2).get_column(value_to_plot)
    #         ).pvalue for g1, g2 in comparisons
    #     ]
    #
    #     if p_values:
    #         reject, pvals_corrected, _, _ = smm.multipletests(p_values, alpha=0.05, method="bonferroni")
    #
    #         chart.update_layout(xaxis=dict(categoryorder="array", categoryarray=groups))
    #         positions = {group: i for i, group in enumerate(groups)}
    #         y_range = plot_data.get_column(value_to_plot).max() - plot_data.get_column(value_to_plot).min()
    #         y_offset = y_range * 0.05 if y_range > 0 else 1
    #
    #         # Simplified annotation for adjacent pairs to keep it clean
    #         for i, (g1, g2) in enumerate(comparisons):
    #             if abs(positions[g1] - positions[g2]) > 1: continue  # Only adjacent for now
    #
    #             p_corr = pvals_corrected[i]
    #             sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
    #
    #             y_max = max(
    #                 plot_data.filter(pl.col("imported_path_short") == g1).get_column(value_to_plot).max(),
    #                 plot_data.filter(pl.col("imported_path_short") == g2).get_column(value_to_plot).max()
    #             )
    #             y_bracket = y_max + y_offset * (abs(positions[g1] - positions[g2]))
    #
    #             chart.add_shape(type="line", x0=positions[g1], y0=y_bracket, x1=positions[g2], y1=y_bracket,
    #                             line=dict(color="black", width=1.5))
    #             chart.add_annotation(x=(positions[g1] + positions[g2]) / 2, y=y_bracket + y_offset / 4, text=sig,
    #                                  showarrow=False)
    #
    # Final layout updates for the single plot
    chart.update_layout(
        title_text=f"Distribution of {value_to_plot.replace('_', ' ').title()}",
        xaxis_title="Folder",
        yaxis_title=value_to_plot.replace('_', ' ').title(),
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode='closest',
        showlegend=False
    )
    return chart

import plotly.graph_objects as go
import polars as pl
from typing import List

def _create_mean_sparkline(df: pl.DataFrame, cols_for_cell: List[str]) -> go.Figure:
    """
    Creates a sparkline figure with individual faint lines and a bold mean line.

    Args:
        df: The main Polars DataFrame containing the series data.
        cols_for_cell: A list of column names to plot.

    Returns:
        A Plotly graph objects Figure.
    """
    fig = go.Figure()

    if not cols_for_cell:
        # Return an empty figure if there's no data to plot
        return fig

    # --- Use Polars to efficiently calculate the mean across all relevant columns ---
    mean_series = df.select(cols_for_cell).mean(axis=1)

    # --- Add a faint trace for each individual column ---
    for col in cols_for_cell:
        fig.add_trace(
            go.Scatter(
                y=df.get_column(col),
                mode='lines',
                line=dict(color='rgba(200, 200, 200, 0.5)', width=1),
                hoverinfo='none' # Hide tooltips for faint lines
            )
        )

    # --- Add the bold mean line on top ---
    fig.add_trace(
        go.Scatter(
            y=mean_series,
            mode='lines',
            name='Mean', # Name for the hover label
            line=dict(color='#1f77b4', width=3), # A prominent blue color
            hovertemplate='Mean: %{y:.2f}<extra></extra>'
        )
    )

    # --- Apply sparkline styling ---
    fig.update_layout(
        showlegend=False,
        margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    return fig