import re
from typing import List, Optional, Tuple, Dict

import plotly.graph_objects as go
import polars as pl


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


def _parse_dynamic_col(
        col_name: str, supported_metrics: List[str]
) -> Optional[Tuple[str, Dict[str, int]]]:
    """
    Parses a column name that starts with a supported metric and ends with dynamic
    dimension suffixes. This is more performant by building the metric options
    into the initial regex match.
    """
    if not supported_metrics:
        return None

    # 1. Dynamically create a regex pattern from the list of supported metrics.
    #    e.g., for ['mean', 'std'], this becomes '(mean|std)'
    metric_options = '|'.join(re.escape(m) for m in supported_metrics)

    # 2. Construct the full pattern to capture the metric and suffix in one go.
    #    ^({metric_options})   - At the start, capture one of the supported metrics.
    #    ((?:_[a-zA-Z]\d+)+)$  - Capture the entire group of one or more dimension suffixes at the end.
    main_pattern = re.compile(rf"^({metric_options})((?:_[a-zA-Z]\d+)+)$")

    match = main_pattern.match(col_name)
    if not match:
        return None

    # 3. Extract the captured parts.
    base_metric = match.group(1)
    suffix_part = match.group(2)

    # 4. Parse the individual dimensions from the captured suffix string.
    dim_pattern = re.compile(r'_([a-zA-Z])(\d+)')
    dims = {m.group(1): int(m.group(2)) for m in dim_pattern.finditer(suffix_part)}

    return base_metric, dims

def _create_sparkline(df: pl.DataFrame, dim_name: str, cols: List[str]) -> go.Figure:
    """Creates a minimalist, aggregated line plot for a table cell."""
    dim_pattern = re.compile(rf'_{dim_name}(\d+)')

    # Melt the dataframe to a long format and extract the dimension index
    long_df = df.select(cols).melt(variable_name="variable", value_name="value").with_columns(
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
