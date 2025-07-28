import re
from pathlib import Path
from typing import List, Dict, Tuple

import dash_bootstrap_components as dbc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import polars as pl
from dash import Dash, html, dcc
from dash.dependencies import Input, Output

from pixel_patrol.core.project import Project
from pixel_patrol.plugins import discover_report_plugins
from pixel_patrol.report.widget import organize_widgets_by_tab
from pixel_patrol.report.widget_interface import PixelPatrolWidget


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


def load_widgets() -> Tuple[List[PixelPatrolWidget], List[PixelPatrolWidget]]:
    """
    Recursively discover and load widget classes from Python files under `root`.
    Returns instances of classes inheriting from PixelPatrolWidget.
    """
    plugins = discover_report_plugins()
    return plugins.get("group_widgets", []), plugins.get("individual_widgets", [])


def create_app(
        project: Project
) -> Dash:
    return _create_app(project.images_df, project.get_settings().cmap)


def _create_app(
        df: pl.DataFrame,
        default_palette_name: str = 'tab10',
        widget_root: str = "widgets"
) -> Dash:
    """
    Instantiate Dash app, register callbacks, and assign layout.
    Accepts DataFrame and palette name as arguments.
    """
    # Add a stable, unique ID for row selection in the dropdown
    df = df.with_row_count(name="unique_id")

    external_stylesheets = [dbc.themes.BOOTSTRAP, 'https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

    group_widgets, individual_widgets = load_widgets()

    all_widgets = group_widgets + individual_widgets
    for w in all_widgets:
        if hasattr(w, 'register_callbacks'):
            w.register_callbacks(app, df)

    def serve_layout_closure() -> html.Div:
        DEFAULT_WIDGET_WIDTH = 12

        # --- Header and Global Controls ---
        header = dbc.Row(dbc.Col(html.H1('Pixel Patrol', className='mt-3 mb-2')))

        disclaimer = dbc.Row(
            dbc.Col(
                dbc.Alert(
                    [
                        html.P(
                            "This application is a prototype. "
                            "The data may be inaccurate, incomplete, or subject to change. "
                            "Please use this tool for experimental purposes only and do not rely on its "
                            "output for critical decisions."
                        ),
                        html.Hr(),
                        html.P("Your feedback is welcome!", className="mb-0"),
                    ],
                    color="warning",
                    className="my-4"
                )
            )
        )

        palette_dropdown = dcc.Dropdown(
            id='palette-selector',
            options=[{'label': name, 'value': name} for name in sorted(plt.colormaps())],
            value=default_palette_name,
            clearable=False,
            style={'width': '250px'}
        )
        palette_row = dbc.Row(
            dbc.Col(html.Div([html.Label('Color Palette:'), palette_dropdown]), className='mb-3')
        )

        # --- Group Widget Layout (Left Column) ---
        group_widget_content = []
        tabbed_group_widgets = organize_widgets_by_tab(group_widgets)
        for group_name, ws in tabbed_group_widgets.items():
            group_widget_content.append(dbc.Row(dbc.Col(html.H3(group_name, className='my-3 text-primary'))))
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

        # --- Individual Item Selector and Plotting Area (Right Column) ---

        # Use 'name' for the dropdown label if it exists, otherwise fall back.
        label_col = 'name' if 'name' in df.columns else 'imported_path_short'

        item_selector_dropdown = dcc.Dropdown(
            id='item-selector',
            options=[
                {'label': row[label_col], 'value': row['unique_id']}
                for row in df.select(['unique_id', label_col]).to_dicts()
            ],
            placeholder="Select an item to analyze...",
            clearable=True,
        )

        individual_item_panel = html.Div([
            html.H3("Individual Item Analysis", className="mt-3 text-primary"),
            html.P("Select an item from the dropdown to view its specific plots."),
            item_selector_dropdown,
            html.Div(id='individual-widgets-container', className='mt-4'),  # Container for individual plots
        ])

        # --- Data Stores ---
        stores = html.Div([
            dcc.Store(id='color-map-store'),
            dcc.Store(id='tb-process-store-tensorboard-embedding-projector', data={}),
            dcc.Store(id='individual-data-store'),  # Store for the selected single row data
        ])

        # --- Final Layout Assembly (with independent scrolling) ---
        # Set a fixed height for scrollable areas, subtracting the approx height of header/disclaimer
        scrollable_area_style = {
            'height': 'calc(100vh - 250px)',  # Adjust 250px based on actual header height
            'overflowY': 'auto',
            'padding': '15px'
        }

        return html.Div(
            dbc.Container(
                [
                    header,
                    disclaimer,
                    stores,
                    dbc.Row(
                        [
                            # Left Column for Group Widgets
                            dbc.Col(
                                html.Div([palette_row] + group_widget_content, style=scrollable_area_style),
                                width=8,
                            ),
                            # Right Column for Individual Analysis
                            dbc.Col(
                                html.Div(individual_item_panel, style=scrollable_area_style),
                                width=4,
                                style={'borderLeft': '1px solid #ddd'}
                            ),
                        ]
                    ),
                ],
                fluid=True,
                style={'maxWidth': '1800px', 'margin': '0 auto'},
            )
        )

    app.layout = serve_layout_closure

    # --- Callbacks ---

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
        """When an item is selected, filter the main df and store the single row data."""
        if selected_item_id is None:
            return None

        filtered_df = df.filter(pl.col("unique_id") == selected_item_id)
        return filtered_df.to_dicts()

    @app.callback(
        Output('individual-widgets-container', 'children'),
        Input('individual-data-store', 'data')
    )
    def update_individual_widgets_layout(item_data: List[Dict]):
        """Renders the layout for individual widgets, after cleaning the data."""
        if not item_data:
            return html.P("Select an item to view details.", className="text-muted mt-3")

        df_single_row = pl.from_dicts(item_data)

        # PERFORMANCE: Drop columns that are all null for this row before rendering.
        df_cleaned = df_single_row.select([s.name for s in df_single_row if not s.is_null().all()])

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
            return html.P("No applicable widgets for this item.", className="text-warning mt-3")

        return individual_widget_layouts

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