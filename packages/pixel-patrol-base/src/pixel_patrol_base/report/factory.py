from typing import Dict, Optional, List, Any, Tuple

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from numpy import ndarray
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import math

from pixel_patrol_base.report.data_utils import prettify_col_name
from pixel_patrol_base.report.constants import NO_GROUPING_COL
from pixel_patrol_base.report.stats_annotations import annotate_plot_with_significance

# =============================================================================
#  SECTION 1: GLOBAL STYLES
# =============================================================================

_STANDARD_LAYOUT_KWARGS = dict(
    margin=dict(l=50, r=50, t=50, b=50),
    hovermode="closest",
    template="plotly_white",
    showlegend=False,
)

_STANDARD_LEGEND_KWARGS = dict(
    orientation="v",
    yanchor="top",
    y=1,
    xanchor="left",
    x=1.02,
    title=None,
)

def _apply_standard_styling(fig: go.Figure, n_categories: int = 0):
    """Applies global style rules and fixes bar chart gaps."""
    fig.update_layout(**_STANDARD_LAYOUT_KWARGS)

    # Aesthetic fix: Plotly bars are too thin if there are few categories
    if n_categories == 1:
        fig.update_layout(bargap=0.7)
    elif n_categories == 2:
        fig.update_layout(bargap=0.4)
    else:
        fig.update_layout(bargap=0.1)


def _get_category_orders(
    df: pl.DataFrame,
    x: Optional[str],
    color: Optional[str],
    order_x: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Helper to calculate sorted category orders for Plotly Express functions.
    Ensures x-axis and Legend (color) are sorted alphabetically unless order_x is provided.
    """
    orders = {}

    # 1. Handle X-Axis Order
    if x and x in df.columns:
        if order_x:
            # Respect explicit order (e.g. for Size Bins)
            orders[x] = order_x
        elif df[x].dtype in (pl.Utf8, pl.Categorical, pl.Boolean):
            # Default: Sort strings/booleans alphabetically
            orders[x] = df[x].unique().drop_nulls().sort().to_list()

    # 2. Handle Legend (Color) Order
    if color and color in df.columns:
        if df[color].dtype in (pl.Utf8, pl.Categorical, pl.Boolean):
            orders[color] = df[color].unique().drop_nulls().sort().to_list()

    return orders


def _sanitize_labels(labels: Optional[Dict[str, str]], *cols: Optional[str]) -> Dict[str, str]:
    out = dict(labels or {})
    for c in cols:
        if c:
            pretty = prettify_col_name(c)
            if pretty and c not in out:
                out[c] = pretty
    return out


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string with given alpha."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


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
                    dbc.PopoverHeader("Info"),
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


def show_no_data_message(text: str = "No data available after filtering.") -> html.Div:
    return html.Div(
        text,
        className="text-warning p-3",
        style={"textAlign": "center"}
    )

# =============================================================================
#  SECTION 3: PLOTLY CHARTS
# =============================================================================

def plot_bar(
        df: pl.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        color_map: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        barmode: str = "stack",
        order_x: Optional[List[str]] = None,
        force_category_x: bool = True,
        show_legend: bool = False,
) -> go.Figure:

    cat_orders = _get_category_orders(df, x, color, order_x)

    labels = _sanitize_labels(labels, x, color)

    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        barmode=barmode,
        color_discrete_map=color_map or {},
        category_orders=cat_orders,
        title=title,
        labels=labels,
    )

    if force_category_x:
        fig.update_xaxes(type="category")

    if order_x:
        fig.update_layout(
            xaxis={"categoryorder": "array", "categoryarray": order_x}
        )

    try:
        n_categories = df[x].n_unique()
    except (KeyError, AttributeError):
        n_categories = 0

    fig.update_xaxes(title_text=prettify_col_name(x))

    _apply_standard_styling(fig, n_categories)

    if show_legend and color!=NO_GROUPING_COL:
        fig.update_layout(showlegend=True, legend=_STANDARD_LEGEND_KWARGS)
    else:
        fig.update_layout(showlegend=False, margin=dict(b=80))

    return fig


def plot_scatter(
        df: pl.DataFrame,
        x: str,
        y: str,
        size: Optional[str] = None,
        color: Optional[str] = None,
        color_map: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        hover_data: Optional[List[str]] = None,
        show_legend: bool = True,
) -> go.Figure:

    cat_orders = _get_category_orders(df, x, color)

    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size,
        color=color,
        color_discrete_map=color_map or {},
        category_orders=cat_orders,
        title=title,
        labels=labels,
        hover_data=hover_data,
    )
    _apply_standard_styling(fig)
    if show_legend and color!=NO_GROUPING_COL:
        fig.update_layout(showlegend=True, legend=_STANDARD_LEGEND_KWARGS)
    return fig


def plot_strip(
        df: pl.DataFrame,
        x: str,
        y: str,
        color: Optional[str] = None,
        color_map: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        hover_data: Optional[List[str]] = None,
        height: Optional[int] = None,
) -> go.Figure:
    """Creates a single strip plot (no faceting)."""
    cat_orders = _get_category_orders(df, x, color)

    fig = px.strip(
        df,
        x=x,
        y=y,
        color=color,
        color_discrete_map=color_map or {},
        category_orders=cat_orders,
        title=title,
        labels=labels,
        hover_data=hover_data,
    )

    fig.update_xaxes(showline=True, showticklabels=True, title=None)
    fig.update_yaxes(showline=True, showticklabels=True)

    _apply_standard_styling(fig)
    fig.update_layout(showlegend=False)

    if height:
        fig.update_layout(height=height)

    return fig


def create_strip_plot_grid(
        df: pl.DataFrame,
        x: str,
        y: str,
        facet_col: str,
        color: Optional[str] = None,
        color_map: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None,
        hover_data: Optional[List[str]] = None,
        n_cols: int = 3,
        plot_height: int = 250,
        title: Optional[str] = None,
) -> html.Div:
    """
    Creates a grid of individual strip plots, one per facet value.
    Returns a Dash Div containing the grid layout.
    """
    facet_values = df[facet_col].unique().sort().to_list()

    plots = []
    for facet_val in facet_values:
        facet_df = df.filter(pl.col(facet_col) == facet_val)

        fig = plot_strip(
            df=facet_df,
            x=x,
            y=y,
            color=color,
            color_map=color_map,
            title=str(facet_val),
            labels=labels,
            hover_data=hover_data,
            height=plot_height,
        )

        plots.append(
            html.Div(
                dcc.Graph(figure=fig),
                style={"flex": f"1 1 {100 // n_cols}%", "minWidth": "300px"}
            )
        )

    title_div = html.H5(title, style={"marginBottom": "10px"}) if title else None

    grid = html.Div(
        plots,
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "10px",
        }
    )

    if title_div:
        return html.Div([title_div, grid])
    return grid


def plot_violin(
        df: pl.DataFrame,
        y: str,
        group_col: str,
        color_map: Optional[Dict[str, str]] = None,
        title: Optional[str] = None,
        custom_data_col: Optional[str] = None,
        height: Optional[int] = None,
        show_legend: bool = False,
        group_order: Optional[List[str]] = None,
        annotate_significance=False,
) -> go.Figure:

    color_map = color_map or {}
    fig = go.Figure()

    if group_order:
        present = set(df.get_column(group_col).drop_nulls().unique().to_list())
        groups = [g for g in group_order if g in present]
        groups += [g for g in df.get_column(group_col).unique().drop_nulls().sort().to_list()
                        if g not in set(groups)]
    else:
        groups = df.get_column(group_col).unique().drop_nulls().sort().to_list()

    for group_name in groups:
        df_group = df.filter(pl.col(group_col) == group_name)

        group_color = color_map.get(str(group_name), "#333333")

        custom_data = None
        if custom_data_col and custom_data_col in df_group.columns:
            custom_data = df_group.get_column(custom_data_col).to_list()

        violin_kwargs = dict(
            y=df_group.get_column(y).to_list(),
            name=group_name,
            customdata=custom_data,
            opacity=0.9,
            showlegend=show_legend,
            points="all",
            pointpos=0,
            box=dict(visible=True),
            meanline=dict(visible=True),
        )

        violin_kwargs["marker"] = dict(color=group_color)

        fig.add_trace(
            go.Violin(
                **violin_kwargs,
                hovertemplate=(
                        "<b>Group:</b> %{x}<br>"
                        "<b>Value:</b> %{y:.2f}"
                        + (f"<br><b>{'Name' if custom_data_col == 'name' else (custom_data_col or 'Info')}:</b> %{{customdata}}"
                        if custom_data is not None else "")
                        + "<extra></extra>"
                ),
            )
        )

    fig.update_traces(
        marker=dict(line=dict(width=1, color="black")),
        box=dict(line_color="black"),
    )

    layout = _STANDARD_LAYOUT_KWARGS.copy()
    layout.update(
        dict(
            title_text=title,
            yaxis_title=y.replace("_", " ").title(),
            xaxis_title=prettify_col_name(group_col),
            showlegend=show_legend,
        )
    )

    if height is not None:
        fig.update_layout(**layout, height=height)
    else:
        fig.update_layout(**layout)

    if annotate_significance:
        fig = annotate_plot_with_significance(
            fig=fig, df=df, value_col=y, group_col=group_col, group_order=groups
        )

    return fig



def plot_aggregated_scatter(
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        y_std_col: str,
        n_col: str,
        group_col: str,
        *,
        color_map: Dict[str, str] = None,
        show_legend: bool = False,
        height: int = 140,
) -> go.Figure:
    """
    grouped XY plot showing:
    - Line for mean values
    - Shaded band for ±1 std deviation
    - Marker size proportional to log(n) to show sample count
    - Rich hover info with n, mean, std
    """
    color_map = color_map or {}
    fig = go.Figure()

    groups = (
        df.get_column(group_col)
        .unique()
        .drop_nulls()
        .sort()
        .to_list()
    )

    for g in groups:
        dfg = df.filter(pl.col(group_col) == g).sort(x_col)

        x_vals = dfg[x_col].to_list()
        y_mean = dfg[y_col].to_list()
        y_std = dfg[y_std_col].to_list()
        n_vals = dfg[n_col].to_list()

        color = color_map.get(str(g), "#333333")

        # Calculate upper and lower bounds for the band
        y_upper = [m + s for m, s in zip(y_mean, y_std)]
        y_lower = [m - s for m, s in zip(y_mean, y_std)]

        # Marker sizes: scale by log(n) for visibility, with min/max bounds
        marker_sizes = [max(4, min(12, 3 + 3 * math.log10(max(n, 1)))) for n in n_vals]

        # Add shaded std band (fill between upper and lower)
        # Upper bound line (invisible)
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Lower bound with fill to previous trace
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=_hex_to_rgba(color, 0.2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Main line with markers
        hover_text = [
            f"<b>{g}</b><br>"
            f"Slice: {x}<br>"
            f"Mean: {m:.3f}<br>"
            f"Std: {s:.3f}<br>"
            f"<b>n={n}</b>"
            for x, m, s, n in zip(x_vals, y_mean, y_std, n_vals)
        ]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_mean,
                mode="lines+markers",
                name=str(g),
                line=dict(width=2, color=color),
                marker=dict(
                    size=marker_sizes,
                    color=color,
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="%{text}<extra></extra>",
                text=hover_text,
            )
        )

    layout = _STANDARD_LAYOUT_KWARGS.copy()
    layout.update(
        margin=dict(l=30, r=10, t=10, b=25),
        showlegend=show_legend,
        xaxis=dict(
            type="category",
            showgrid=False,
            zeroline=False,
            showline=True,
            mirror=True,
            ticks="outside",
            title=None,
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            mirror=True,
            ticks="outside",
            title=None,
        ),
    )
    fig.update_layout(**layout, height=height)
    return fig


def build_violin_grid(
        df: pl.DataFrame,
        color_map: Dict[str, str],
        numeric_cols: List[str],
        group_col: str,
        order_x: Optional[List[str]] = None,
        annotate_significance=False,
):
    """
    Build a grid of violin plots (one per metric column) grouped by folder,
    plus an optional table listing metrics that have no variance.
    """
    df_filtered = df.select(
        pl.col("name"),
        pl.col(group_col),
        pl.col(numeric_cols),
    )

    # Drop rows where any of the numeric columns is null
    df_filtered = df_filtered.filter(
        pl.all_horizontal([pl.col(c).is_not_null() for c in numeric_cols])
    )

    groups = df_filtered.get_column(group_col).unique().sort().to_list()
    if not groups:
        return html.P(
            "No data available to generate statistics.",
            className="text-warning",
        )

    # Decide which metrics get a plot vs. go into the "no variance" table
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

    # Determine number of plots per row based on number of groups
    # (wider violins need more horizontal space)
    num_groups = len(groups)
    if num_groups <= 2:
        plots_per_row = 3
    elif num_groups == 3:
        plots_per_row = 2
    else:
        plots_per_row = 1

    # Calculate flex basis percentage with gap allowance
    flex_basis_pct = max(30, 100 // plots_per_row - 2)

    plot_divs: List[html.Div] = []
    for col_name in cols_to_plot:
        nice_name = col_name.replace("_", " ").title()
        fig = plot_violin(
            df=df_filtered,
            y=col_name,
            group_col=group_col,
            color_map=color_map,
            title=f"Distribution of {nice_name}",
            custom_data_col="name",
            show_legend=False,
            group_order=order_x,
            annotate_significance=annotate_significance,
        )
        plot_divs.append(
            html.Div(
                dcc.Graph(figure=fig, style={"width": "100%", "height": "100%"}),
                style={
                    "flex": f"0 0 {flex_basis_pct}%",
                    "minWidth": "300px",
                    "marginBottom": "20px",
                    "boxSizing": "border-box",
                },
            )
        )

    # Use flexbox container with wrap for responsive grid
    plots_container = html.Div(
        plot_divs,
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "15px",
            "width": "100%",
        }
    )

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


def plot_grouped_histogram(
        group_data: Dict[str, Tuple[Any, Any]],  # {group_name: (x_centers, y_counts)}
        color_map: Dict[str, str],
        title: str = None,
        xaxis_title: str = "Intensity",
        yaxis_title: str = "Normalized Count",
        overlay_data: Optional[Dict] = None,  # {x, y, width, name} for a specific file bar chart
        group_order: Optional[List[str]] = None,
) -> go.Figure:
    """
    Plots aggregated line histograms for groups, optionally overlaying a bar chart for a single item.
    """
    fig = go.Figure()

    # 1. Plot the overlay bar chart first (so lines appear on top, or vice versa depending on pref)
    # Usually bars for single file, lines for group means.
    if overlay_data:
        fig.add_trace(
            go.Bar(
                x=overlay_data["x"],
                y=overlay_data["y"],
                width=overlay_data.get("width"),
                name=overlay_data.get("name", "Selected"),
                marker=dict(color="black"),
                opacity=0.3,
            )
        )

    # 2. Plot group lines
    if group_order:
        keys = [k for k in group_order if k in group_data]
        keys += [k for k in group_data.keys() if k not in set(keys)]
    else:
        keys = list(group_data.keys())

    for group_name in keys:
        x_vals, y_vals = group_data[group_name]
        color = color_map.get(group_name, "#333333")
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=str(group_name),
                line=dict(color=color, width=2),
                fill="tozeroy",
                opacity=0.6,
            )
        )

    # 3. Styling
    layout = _STANDARD_LAYOUT_KWARGS.copy()
    layout.update(
        title_text=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        bargap=0.0,
    )
    fig.update_layout(**layout)
    if len(group_data) > 1:
        fig.update_layout(showlegend=True, legend=_STANDARD_LEGEND_KWARGS)

    return fig


def plot_sunburst(
        ids: List[str],
        labels: List[str],
        parents: List[str],
        values: List[int],
        colors: Optional[List[str]] = None,
        hovertemplate: Optional[str] = None,
) -> go.Figure:
    marker = dict(colors=colors) if colors is not None else None

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            marker=marker,
            branchvalues="total",
            hovertemplate=hovertemplate or "<b>%{label}</b><br>%{value}<extra></extra>",
        )
    )

    _apply_standard_styling(fig)
    return fig



def plot_image_mosaic(
    sprite_np: ndarray,
    *,
    unique_groups: List[str],
    color_map: Dict[str, str],
    height: int,
    hover_info: Optional[Dict] = None,
) -> go.Figure:
    fig = px.imshow(sprite_np)
    fig.update_traces(hoverinfo='skip', hovertemplate=None, selector=dict(type="image"))

    # Add invisible scatter points for hover (if hover_info provided)
    if hover_info and hover_info.get("names"):
        fig.add_trace(
            go.Scatter(
                x=hover_info["x"],
                y=hover_info["y"],
                mode="markers",
                marker=dict(
                    size=hover_info.get("marker_size", 32),
                    opacity=0,  # Invisible markers
                ),
                text=hover_info["names"],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )

    # Add legend entries for groups
    for label in unique_groups:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=color_map.get(label, "#333333")),
                name=label,
                showlegend=True,
            )
        )

    fig.update_layout(
        autosize=True,
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode="closest",  # CHANGED: Enable hover
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.04,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


# =============================================================================
#  SECTION 4: CONTROLS
# =============================================================================

def create_labeled_dropdown(
        label: str,
        component_id: str | Dict,
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
                id=component_id,
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
                    style={"display": "inline-block", "marginRight": "15px", "width": "100px"}  # type: ignore[arg-type]
                )
            )

    return children, dims_order
