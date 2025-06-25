import importlib.util
from pathlib import Path
from typing import List, Dict

import dash_bootstrap_components as dbc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import polars as pl
from dash import Dash, html, dcc
from dash.dependencies import Input, Output

from pixel_patrol.core.project import Project
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


def load_widgets() -> List[PixelPatrolWidget]:
    """
    Recursively discover and load widget classes from Python files under `root`.
    Returns instances of classes inheriting from PixelPatrolWidget.
    """
    widgets: List[PixelPatrolWidget] = []
    for entry_point in importlib.metadata.entry_points().select(group='pixel_patrol.widgets'):
        widget_class = entry_point.load()
        widget_instance = widget_class()
        widgets.append(widget_instance)
    return widgets

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
    external_stylesheets = [dbc.themes.BOOTSTRAP]
    app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

    # Load widgets
    all_widgets = load_widgets()
    enabled_widgets = all_widgets

    def serve_layout_closure() -> html.Div:
        DEFAULT_WIDGET_WIDTH = 12

        palette_dropdown = dcc.Dropdown(
            id='palette-selector',
            options=[{'label': name, 'value': name} for name in sorted(plt.colormaps())],
            value=default_palette_name,
            clearable=False,
            style={'width': '250px'}
        )

        header = dbc.Row(
            dbc.Col(html.H1('Pixel Patrol', className='mt-3 mb-2'))
        )

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
                        html.P(
                            "Your feedback is welcome!", className="mb-0"
                        ),
                    ],
                    color="warning",
                    className="my-4"
                )
            )
        )

        palette_row = dbc.Row(
            dbc.Col(
                html.Div([html.Label('Color Palette:'), palette_dropdown]),
                className='mb-3'
            )
        )

        all_widget_content = []
        groups = organize_widgets_by_tab(enabled_widgets)

        for group_name, ws in groups.items():
            # Left-aligned group header
            all_widget_content.append(
                dbc.Row(
                    dbc.Col(html.H3(group_name, className='my-3 text-primary'))
                )
            )

            current_group_cols = []
            current_row_width = 0

            for w in ws:
                widget_width = getattr(w, 'width', DEFAULT_WIDGET_WIDTH)
                if current_row_width + widget_width > 12:
                    all_widget_content.append(dbc.Row(current_group_cols, className='g-4 p-3'))
                    current_group_cols = []
                    current_row_width = 0

                current_group_cols.append(dbc.Row(
                    dbc.Col(html.H4(w.name, className='my-3 text-primary'))
                ))
                current_group_cols.append(
                    dbc.Col(html.Div(w.layout()), width=widget_width, className='mb-3')
                )
                current_row_width += widget_width

            if current_group_cols:
                all_widget_content.append(dbc.Row(current_group_cols, className='g-4 p-3'))

        store = dcc.Store(id='color-map-store')
        tb_store = dcc.Store(id='tb-process-store-tensorboard-embedding-projector', data={})

        # Final layout with max width and centered
        return html.Div(
            dbc.Container(
                [header, disclaimer, palette_row, store, tb_store] + all_widget_content,
                style={'maxWidth': '1200px', 'margin': '0 auto'},
                fluid=True
            )
        )

    app.layout = serve_layout_closure

    # Register global callbacks (needs df to be accessible)
    @app.callback(
        Output('color-map-store', 'data'),
        Input('palette-selector', 'value')
    )
    def update_color_map(palette: str) -> Dict[str, str]:
        # Access df from the closure scope
        folders = df.select(pl.col('imported_path_short')).unique().to_series().to_list()
        cmap = cm.get_cmap(palette, len(folders))
        return {
            f: f"#{int(cmap(i)[0]*255):02x}{int(cmap(i)[1]*255):02x}{int(cmap(i)[2]*255):02x}"
            for i, f in enumerate(folders)
        }

    # Register widget callbacks (needs df to be accessible)
    for w in enabled_widgets:
        if hasattr(w, 'register_callbacks'):
            w.register_callbacks(app, df)

    return app