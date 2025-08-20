import re
from pathlib import Path
from typing import List, Dict

import dash_bootstrap_components as dbc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import polars as pl
from dash import Dash, html, dcc
from dash.dependencies import Input, Output

from pixel_patrol_base.core.project import Project
from pixel_patrol_base.report.widget import organize_widgets_by_tab
from pixel_patrol_base.report.widget_interface import PixelPatrolWidget
from pixel_patrol_base.plugins import discover_widget_plugins


def load_and_concat_parquets(
        paths: List[str]
) -> pl.DataFrame:
    """
    Read parquet files or directories and concatenate into a single DataFrame
    without altering content (assumes each file already has a imported_path_short column).
    """
    dfs = []
    for base_str in paths:
        base = Path(base_str)
        files = sorted(base.rglob("*.parquet")) if base.is_dir() else []
        if base.is_file() and base.suffix == ".parquet":
            files = [base]
        for file in files:
            dfs.append(pl.read_parquet(file))
    return pl.concat(dfs, how="diagonal", rechunk=True) if dfs else pl.DataFrame()


def create_app(
        project: Project
) -> Dash:
    return _create_app(project.images_df, project.get_settings().cmap, pixel_patrol_flavor=project.get_settings().pixel_patrol_flavor)


def _create_app(
        df: pl.DataFrame,
        default_palette_name: str = 'tab10',
        pixel_patrol_flavor = ""
) -> Dash:
    """
    Instantiate Dash app, register callbacks, and assign layout.
    Accepts DataFrame and palette name as arguments.
    """
    df = df.with_row_index(name="unique_id")
    external_stylesheets = [dbc.themes.BOOTSTRAP, 'https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    group_widgets = discover_widget_plugins()

    for w in group_widgets:
        if hasattr(w, 'register_callbacks'):
            w.register_callbacks(app, df)

    def serve_layout_closure() -> html.Div:
        DEFAULT_WIDGET_WIDTH = 12

        # --- Header (Unchanged) ---
        header_row = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Img(src=app.get_asset_url('prevalidation.png'),
                                     style={'height': '110px', 'marginRight': '15px'}),
                            html.H1('Pixel Patrol', className='m-0'),
                            html.Span(
                                pixel_patrol_flavor,
                                style={
                                    'color': '#d9534f', 'fontSize': '2rem', 'fontWeight': 'bold',
                                    'fontFamily': 'cursive', 'transform': 'rotate(-6deg)',
                                    'marginLeft': '15px', 'marginTop': '10px'
                                }
                            ),
                            dbc.Col(
                                html.Div(),
                                width="auto",
                            ),
                        ],
                        className="d-flex align-items-center"
                    ),
                    width=True,
                ),
                dbc.Col(html.Div(["This is a prototype. Data may be incomplete or inaccurate.",
                                    html.Br(),"Use for experimental purposes only."], style={"color": '#d9534f', "textAlign": "right"}), width="auto"),
            ],
            align="center",
            className="my-3"
        )

        # --- NEW: Individual Item Selector (Moved and Restyled) ---
        label_col = 'name' if 'name' in df.columns else 'imported_path_short'
        item_selector_controls = dbc.Row(
            [
                dbc.Col(
                    html.Div(
                    ),
                    width=True,
                ),
                dbc.Col(
                html.Div(
                    [
                        html.Label('Color Palette:', className='me-2'),
                        dcc.Dropdown(
                            id='palette-selector',
                            options=[{'label': name, 'value': name} for name in sorted(plt.colormaps())],
                            value=default_palette_name,
                            clearable=False,
                            style={'width': '200px'}
                        )
                    ],
                    className="d-flex align-items-center justify-content-end"
                ),
                width="auto",
            ),
            # dbc.Col(html.Div([
            #     html.Label("Highlight a specific item in the report:", className="me-2"),
            #     dcc.Dropdown(
            #         id='item-selector',
            #         options=[
            #             {'label': row[label_col], 'value': row['unique_id']}
            #             for row in df.select(['unique_id', label_col]).to_dicts()
            #         ],
            #         placeholder="Select an item to highlight...",
            #         clearable=True,
            #         style={'width': '300px'}
            #     )], className="d-flex align-items-center justify-content-end"), width="auto")
            ]
        )

        # --- Group Widget Layout Generation (Logic Unchanged) ---
        group_widget_content = []
        tabbed_group_widgets = organize_widgets_by_tab(group_widgets)
        for group_name, ws in tabbed_group_widgets.items():
            group_widget_content.append(dbc.Row(dbc.Col(html.H3(group_name, className='my-3 text-primary'))))
            # This logic remains the same, building rows of widgets
            current_group_cols = []
            current_row_width = 0
            for w in ws:
                if should_display_widget(w, df.columns):
                    widget_width = getattr(w, 'width', DEFAULT_WIDGET_WIDTH)
                    if current_row_width + widget_width > 12:
                        group_widget_content.append(dbc.Row(current_group_cols, className='g-4 p-3'))
                        current_group_cols, current_row_width = [], 0

                    current_group_cols.append(dbc.Row(dbc.Col(html.H4(w.name, className='my-3 text-primary'))))
                    current_group_cols.append(dbc.Col(html.Div(w.layout()), width=widget_width, className='mb-3'))
                    current_row_width += widget_width

            if current_group_cols:
                group_widget_content.append(dbc.Row(current_group_cols, className='g-4 p-3'))

        # --- Data Stores ---
        stores = html.Div([
            dcc.Store(id='color-map-store'),
            dcc.Store(id='tb-process-store-tensorboard-embedding-projector', data={}),
            dcc.Store(id='individual-data-store'),
        ])

        # --- Final Layout Assembly (Single Column) ---
        return html.Div(
            dbc.Container(
                [
                    header_row,
                    stores,
                    item_selector_controls, # Add the new global item selector here
                    html.Hr(),
                    # The container for the highlighted individual analysis.
                    # It's empty and invisible by default.
                    html.Div(id='highlighted-individual-wrapper'),
                    # All the standard group widgets are placed after the highlight section.
                    *group_widget_content,
                ],
                fluid=True,
                style={'maxWidth': '1200px', 'margin': '0 auto'},
            )
        )

    app.layout = serve_layout_closure

    # --- Callbacks (update_individual_widgets_layout is modified) ---
    @app.callback(
        Output('color-map-store', 'data'),
        Input('palette-selector', 'value')
    )
    def update_color_map(palette: str) -> Dict[str, str]:
        folders = df.select(pl.col('imported_path_short')).unique().to_series().to_list()
        cmap = cm.get_cmap(palette, len(folders))
        return {
            f: f"#{int(cmap(i)[0] * 255):02x}{int(cmap(i)[1] * 255):02x}{int(cmap(i)[2] * 255):02x}"
            for i, f in enumerate(folders)
        }

    @app.callback(
        Output('individual-data-store', 'data'),
        Input('item-selector', 'value')
    )
    def update_individual_store(selected_item_id: int):
        if selected_item_id is None:
            return None
        df_single_row = df.filter(pl.col("unique_id") == selected_item_id)
        if df_single_row.is_empty():
            return None
        df_cleaned = df_single_row.select([s.name for s in df_single_row if not s.is_null().all()])
        return df_cleaned.to_dicts()

    @app.callback(
        Output('highlighted-individual-wrapper', 'children'),
        Output('highlighted-individual-wrapper', 'style'),
        Input('individual-data-store', 'data')
    )
    def update_highlighted_individual_section(item_data: List[Dict]):
        # Define the style for the highlighted container
        highlight_style = {
            'border': '3px dashed #d9534f', # A prominent red-orange border
            'borderRadius': '10px',
            'backgroundColor': 'rgba(252, 248, 227, 0.6)', # A light warning/info background
            'padding': '1.5rem',
            'marginBottom': '2rem', # Space between this and the group widgets
            'transition': 'all 0.3s ease-in-out'
        }

        # If no item is selected, hide the container completely
        if not item_data:
            return None, {'display': 'none'}

        df_cleaned = pl.from_dicts(item_data)
        item_name = df_cleaned.item(0, 'name') if 'name' in df_cleaned.columns else "Selected Item"

        individual_widget_layouts = []
        for w in individual_widgets:
            if should_display_widget(w, df_cleaned.columns):
                layout_content = [
                    dbc.Row(dbc.Col(html.H5(w.name, className='my-3 text-secondary'))),
                    dbc.Row(dbc.Col(html.Div(w.layout()), className='mb-3')),
                    html.Hr()
                ]
                individual_widget_layouts.extend(layout_content)

        if not individual_widget_layouts:
            children = [html.P("No applicable widgets for this item.", className="text-warning mt-3")]
        else:
            # Build the full content for the highlighted box
            children = [
                html.H3(f"🔎 Analysis for: {Path(item_name).name}", className="text-danger"),
                html.P("This section shows a detailed breakdown of the item selected above."),
                html.Hr(className="my-3"),
                *individual_widget_layouts
            ]

        # Return the content and the style to make it visible
        return children, highlight_style

    return app

def should_display_widget(widget: PixelPatrolWidget, available_columns: List[str]) -> bool:
    """
    Checks if a widget should be displayed by ensuring ALL its required
    column patterns match at least one column in the available data.
    Includes a debug print on failure.
    """
    required_patterns = widget.required_columns()
    if not required_patterns:
        return True  # Widget has no column requirements

    for pattern in required_patterns:
        has_match = any(re.search(pattern, col) for col in available_columns)
        if not has_match:
            widget_name = getattr(widget, 'name', widget.__class__.__name__)
            print(
                f"DEBUG: Hiding widget '{widget_name}' because required pattern "
                f"was not found: '{pattern}' in available columns."
            )
            return False

    return True