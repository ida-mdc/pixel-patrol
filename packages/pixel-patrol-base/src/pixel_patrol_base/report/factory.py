import re
from typing import Dict, Optional, List, Any
from collections import defaultdict
import statistics

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import html
import dash_bootstrap_components as dbc

# =============================================================================
#  SECTION 1: GLOBAL STYLES
# =============================================================================

STANDARD_LAYOUT_KWARGS = dict(
    margin=dict(l=50, r=50, t=50, b=50),
    hovermode="closest",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
)


def _apply_standard_styling(fig: go.Figure, n_categories: int = 0):
    """Applies global style rules and fixes bar chart gaps."""
    fig.update_layout(**STANDARD_LAYOUT_KWARGS)

    # Aesthetic fix: Plotly bars are too thin if there are few categories
    if n_categories == 1:
        fig.update_layout(bargap=0.7)
    elif n_categories == 2:
        fig.update_layout(bargap=0.4)
    else:
        fig.update_layout(bargap=0.1)


# =============================================================================
#  SECTION 2: HTML CONTAINERS (Formerly in layouts.py)
# =============================================================================

def create_info_icon(widget_id: str, help_text: str):
    """Creates a standardized 'i' icon that reveals a popover on hover/click."""
    if not help_text:
        return None

    target_id = f"info-target-{widget_id}"

    return html.Div(
        [
            html.I(
                className="bi bi-info-circle-fill",
                id=target_id,
                style={
                    "cursor": "pointer",
                    "color": "#6c757d",
                    "fontSize": "1.2rem",
                    "marginLeft": "10px"
                }
            ),
            dbc.Popover(
                [
                    dbc.PopoverHeader("Widget Info"),
                    dbc.PopoverBody(help_text),
                ],
                target=target_id,
                trigger="legacy",
                placement="left",
            ),
        ],
        className="d-flex align-items-center"
    )


def create_widget_card(title: str, content: list, widget_id: str, help_text: str = None):
    """Wraps widget content in a standardized card with a header."""
    header_children = [
        html.H4(title, className="m-0 text-primary"),
    ]

    if help_text:
        header_children.append(
            html.Div(create_info_icon(widget_id, help_text), className="ms-auto")
        )

    return dbc.Card(
        [
            dbc.CardHeader(
                html.Div(header_children, className="d-flex align-items-center w-100"),
                className="bg-light"
            ),
            dbc.CardBody(
                content,
                className="p-3"
            ),
        ],
        className="mb-4 shadow-sm",
        style={"borderRadius": "8px"}
    )


# =============================================================================
#  SECTION 3: PLOTLY CHARTS (Standardized Generators)
# =============================================================================

def plot_bar(
        df: pl.DataFrame,
        x: str,
        y: str,
        color: str,
        color_map: Dict[str, str],
        title: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        barmode: str = "stack",
        order_x: Optional[List[str]] = None
) -> go.Figure:
    """Standardized Bar Chart."""
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        barmode=barmode,
        color_discrete_map=color_map,
        title=title,
        labels=labels
    )

    if order_x:
        fig.update_layout(xaxis={"categoryorder": "array", "categoryarray": order_x})

    try:
        n = df[x].n_unique()
    except Exception:
        n = 0

    _apply_standard_styling(fig, n)
    return fig


def plot_scatter(
        df: pl.DataFrame,
        x: str,
        y: str,
        size: str,
        color: str,
        color_map: Dict[str, str],
        title: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        hover_data: Optional[List[str]] = None
) -> go.Figure:
    """Standardized Scatter/Bubble Chart."""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size,
        color=color,
        color_discrete_map=color_map,
        title=title,
        labels=labels,
        hover_data=hover_data
    )
    _apply_standard_styling(fig)
    return fig


def plot_violin_distribution(
        df: pl.DataFrame,
        y: str,
        group_col: str,
        color_map: Dict[str, str],
        title: str,
        custom_data_cols: List[str] = None
) -> go.Figure:
    """Standardized Violin/Distribution Plot."""
    chart = go.Figure()
    groups = df.get_column(group_col).unique().sort().to_list()

    for group_name in groups:
        df_group = df.filter(pl.col(group_col) == group_name)

        group_color = color_map.get(group_name, "#333333") if group_col in ["imported_path",
                                                                            "imported_path_short"] else None

        custom_data = None
        if custom_data_cols:
            custom_data = df_group.get_column(custom_data_cols[0]).to_list()

        violin_kwargs = dict(
            y=df_group.get_column(y).to_list(),
            name=group_name,
            customdata=custom_data,
            opacity=0.9,
            showlegend=True,
            points="all",
            pointpos=0,
            box_visible=True,
            meanline=dict(visible=True),
        )
        if group_color:
            violin_kwargs["marker_color"] = group_color

        chart.add_trace(go.Violin(**violin_kwargs))

    chart.update_traces(
        marker=dict(line=dict(width=1, color="black")),
        box=dict(line_color="black")
    )

    layout = STANDARD_LAYOUT_KWARGS.copy()
    layout.update(dict(
        title_text=title,
        yaxis_title=y.replace('_', ' ').title(),
        xaxis_title="Group"
    ))
    chart.update_layout(**layout)
    return chart


def create_sparkline(df: pl.DataFrame, dim_name: str, cols: List[str]) -> go.Figure:
    """
    Creates a minimalist sparkline plot.
    Aggregates multiple columns mapping to the same index (e.g. t0_c0, t0_c1 -> t0).
    """
    pattern = re.compile(f"_{dim_name}(\\d+)")

    # 1. Get mean of each column across the whole filtered dataframe
    stats_row = df.select([pl.col(c).mean() for c in cols]).row(0)

    # 2. Group values by their dimension index (e.g. {0: [val_c0, val_c1], 1: [...]})
    grouped_points = defaultdict(list)

    for col, val in zip(cols, stats_row):
        if val is None:
            continue
        m = pattern.search(col)
        if m:
            idx = int(m.group(1))
            grouped_points[idx].append(val)

    if not grouped_points:
        return go.Figure()

    # 3. Average the values for each index to ensure 1 point per X
    final_points = []
    for idx in sorted(grouped_points.keys()):
        avg_val = statistics.mean(grouped_points[idx])
        final_points.append((idx, avg_val))

    x_vals, y_vals = zip(*final_points)

    fig = go.Figure(data=go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines+markers',
        line=dict(width=2, color='#007eff'),
        marker=dict(size=4)
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig