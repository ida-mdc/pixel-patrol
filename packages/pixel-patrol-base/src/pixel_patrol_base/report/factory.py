import re
from typing import Dict, Optional, List, Any, Tuple
from collections import defaultdict
import statistics
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from dash import html, dcc, dash_table
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
#  SECTION 2: HTML CONTAINERS
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
                    "fontSize": "1.6rem",
                    "marginLeft": "8px"
                }
            ),
            dbc.Popover(
                [
                    dbc.PopoverHeader("Widget Info"),
                    dbc.PopoverBody(
                        dcc.Markdown(help_text, style={"marginBottom": 0, "fontSize": "1.4rem"})
                        ),
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
            html.Div(create_info_icon(widget_id, help_text), className="ms-2")
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
#  SECTION 3: PLOTLY CHARTS
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


def generate_column_violin_plots(
    df_global: pl.DataFrame,
    color_map: Dict[str, str],
    numeric_cols: List[str],
):
    """
    Build a grid of violin plots (one per metric column) grouped by folder,
    plus an optional table listing metrics that have no variance.
    """
    df_filtered = df_global.select(
        pl.col("name"),
        pl.col("imported_path_short"),
        pl.col(numeric_cols),
    )

    # Drop rows where any of the numeric columns is null
    df_filtered = df_filtered.filter(
        pl.all_horizontal([pl.col(c).is_not_null() for c in numeric_cols])
    )

    groups = df_filtered["imported_path_short"].unique().sort().to_list()
    if not groups:
        return html.P(
            "No data available to generate statistics.",
            className="text-warning",
        )

    # Decide which metrics get a plot vs. go into the “no variance” table
    cols_to_plot: List[str] = []
    no_variance_data: List[Dict[str, str]] = []

    for col in numeric_cols:
        series = df_filtered.get_column(col).drop_nulls()
        if series.n_unique() == 1:
            no_variance_data.append(
                {
                    "Metric": col.replace("_", " ").title(),
                    "Value": f"{series[0]:.4f}",
                }
            )
        elif series.n_unique() > 1:
            cols_to_plot.append(col)

    # Layout width based on number of groups
    num_groups = len(groups)
    if num_groups <= 2:
        col_class = "four columns"      # 3 plots per row
    elif num_groups == 3:
        col_class = "six columns"       # 2 plots per row
    else:
        col_class = "twelve columns"    # 1 plot per row

    plot_divs: List[html.Div] = []
    for col_name in cols_to_plot:
        fig = _create_single_violin_plot(
            plot_data=df_filtered,
            value_to_plot=col_name,
            groups=groups,
            color_map=color_map,
        )
        plot_divs.append(
            html.Div(
                dcc.Graph(figure=fig),
                className=col_class,
                style={"marginBottom": "20px"},
            )
        )

    plots_container = html.Div(plot_divs, className="row")

    table_component: List = []
    if no_variance_data:
        table_component = [
            html.Hr(),
            html.H4(
                "Metrics with No Variance",
                style={"marginTop": "30px", "marginBottom": "15px"},
            ),
            dash_table.DataTable(
                data=no_variance_data,
                columns=[{"name": i, "id": i} for i in ["Metric", "Value"]],
                style_cell={"textAlign": "left"},
                style_header={"fontWeight": "bold"},
                style_as_list_view=True,
            ),
        ]

    return [plots_container] + table_component


def _create_single_violin_plot(
    plot_data: pl.DataFrame,
    value_to_plot: str,
    groups: List[str],
    color_map: Dict[str, str],
) -> go.Figure:
    """Generate one violin plot figure for a single metric grouped by folder."""
    chart = go.Figure()

    for group_name in groups:
        df_group = plot_data.filter(pl.col("imported_path_short") == group_name)
        group_color = color_map.get(group_name, "#333333")

        chart.add_trace(
            go.Violin(
                y=df_group.get_column(value_to_plot),
                name=group_name,
                customdata=df_group.get_column("name").map_elements(
                    lambda p: Path(p).name, return_dtype=pl.String
                ),
                marker_color=group_color,
                opacity=0.9,
                showlegend=True,
                points="all",
                spanmode="hard",
                pointpos=0,
                box_visible=True,
                meanline=dict(visible=True),
                hovertemplate=(
                    f"<b>Group: {group_name}</b>"
                    "<br>Value: %{y:.2f}"
                    "<br>Filename: %{customdata}"
                    "<extra></extra>"
                ),
            )
        )

    chart.update_traces(
        marker=dict(line=dict(width=1, color="black")),
        box=dict(line_color="black"),
    )

    nice_name = value_to_plot.replace("_", " ").title()
    chart.update_layout(
        title_text=f"Distribution of {nice_name}",
        xaxis_title="Folder",
        yaxis_title=nice_name,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="closest",
        showlegend=False,
    )
    return chart


# =============================================================================
#  SECTION 4: CONTROLS
# =============================================================================

def create_labeled_dropdown(
        label: str,
        id: str | Dict,
        options: List[Dict],
        value: Any = None,
        clearable: bool = False,
        multi: bool = False,
        width: str = "300px",
        style: Dict = None,
) -> html.Div:
    """Creates a standardized labeled dropdown."""
    default_style = {"width": width, "marginBottom": "20px"}
    if style:
        default_style.update(style)

    return html.Div(
        [
            html.Label(label, style={"marginBottom": "5px", "fontWeight": "500"}),
            dcc.Dropdown(
                id=id,
                options=options,
                value=value,
                clearable=clearable,
                multi=multi,
            ),
        ],
        style=default_style,
    )


def create_dimension_selectors(
        tokens: Dict[str, List[str]],
        id_type: str
) -> Tuple[List[html.Div], List[str]]:
    """
    Creates standardized T/C/Z/S dimension selectors based on extracted tokens.
    Returns: (List of Components, List of dimensions found)
    """
    children = []
    dims_order = []

    # Preferred stable order: t, c, z, s (TCZS)
    for dim_name in ['t', 'c', 'z', 's']:
        if dim_name in tokens and tokens[dim_name]:
            dims_order.append(dim_name)
            dropdown_id = {"type": id_type, "dim": dim_name}

            options = [{"label": "All", "value": "All"}] + [
                {"label": tok, "value": tok} for tok in tokens[dim_name]
            ]

            children.append(
                html.Div(
                    [
                        html.Label(f"{dim_name.upper()} slice"),
                        dcc.Dropdown(id=dropdown_id, options=options, value="All", clearable=False),
                    ],
                    style={"display": "inline-block", "marginRight": "15px", "width": "100px"}
                )
            )

    return children, dims_order