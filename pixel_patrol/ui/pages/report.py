import importlib.util
import inspect
import re
from pathlib import Path
from typing import List, Dict

import dash_bootstrap_components as dbc
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import polars as pl
from dash import html, dcc, Input, Output, State, callback_context

from pixel_patrol.report.dashboard_app import load_widgets
# Assuming your pixel_patrol modules are accessible
from pixel_patrol.report.widget import organize_widgets_by_tab
from pixel_patrol.report.widget_interface import PixelPatrolWidget

_df_cached: pl.DataFrame = pl.DataFrame()


def load_and_concat_parquets(
        paths: List[str]
) -> pl.DataFrame:
    """
    Read parquet files or directories and concatenate into a single DataFrame.
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

def create_report_layout(default_palette_name: str = 'tab10') -> html.Div:
    DEFAULT_WIDGET_WIDTH = 12

    palette_dropdown = dcc.Dropdown(
        id='palette-selector',
        options=[{'label': name, 'value': name} for name in sorted(plt.colormaps())],
        value=default_palette_name,
        clearable=False,
        style={'width': '250px'}
    )

    header = dbc.Row(
        dbc.Col(html.H1('Pixel Patrol Report', className='mt-3 mb-2'))
    )

    palette_row = dbc.Row(
        dbc.Col(
            html.Div([html.Label('Color Palette:'), palette_dropdown]),
            className='mb-3'
        )
    )

    all_widgets = load_widgets()
    all_widget_content = []
    groups = organize_widgets_by_tab(all_widgets)

    for group_name, ws in groups.items():
        all_widget_content.append(
            dbc.Row(
                dbc.Col(html.H3(group_name, className='my-3 text-primary'))
            )
        )

        current_group_cols = []
        current_row_width = 0

        for w in ws:
            # --- CHECK WIDGET REQUIREMENTS FIRST ---
            if should_display_widget(w, your_dataframe_columns):
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

        # Don't forget to add the last row if it's not empty
        if current_group_cols:
            all_widget_content.append(dbc.Row(current_group_cols, className='g-4 p-3'))

    # `data-df-store` to pass the path (still needed)
    data_df_store = dcc.Store(id='data-df-store')
    color_map_store = dcc.Store(id='color-map-store')
    tb_store = dcc.Store(id='tb-process-store-tensorboard-embedding-projector', data={})

    # Dummy Div for the callback output that updates _df_cached
    df_loader_output_div = html.Div(id='df-loader-output', style={'display': 'none'})

    return html.Div(
        dbc.Container(
            [header, palette_row, data_df_store, color_map_store, tb_store, df_loader_output_div] + all_widget_content,
            style={'maxWidth': '1200px', 'margin': '0 auto'},
            fluid=True
        )
    )


def register_report_callbacks(app):
    global _df_cached

    all_widgets = load_widgets()
    enabled_widgets = all_widgets

    # Callback to load the DataFrame from temp path and assign it to the GLOBAL _df_cached
    @app.callback(
        Output('df-loader-output', 'children'),  # This callback's output is still a dummy div
        Input('project-data-store', 'data'),  # <--- THIS IS THE MISSING TRIGGER!
        prevent_initial_call=False
    )
    def load_df_into_global_cache(project_info: Dict):
        global _df_cached
        if project_info and 'processed_data_path' in project_info:
            processed_data_path = Path(project_info['processed_data_path'])
            if processed_data_path.exists():
                print(f"DEBUG: Loading DataFrame from {processed_data_path} into global cache.")
                _df_cached = load_and_concat_parquets([str(processed_data_path)])
                print(f"DEBUG: Global DF shape: {_df_cached.shape}")
            else:
                print(f"DEBUG: Processed data file not found: {processed_data_path}. Global DF remains empty.")
                _df_cached = pl.DataFrame()
        else:
            print("DEBUG: No project info. Global DF remains empty.")
            _df_cached = pl.DataFrame()
        return None

    # Global callback for color map update
    @app.callback(
        Output('color-map-store', 'data'),
        Input('palette-selector', 'value'),
        # Ensure the DF loading callback has completed before this one potentially fires
        # by making its output a State.
        State('df-loader-output', 'children'),
    )
    def update_color_map(palette: str, _):
        global _df_cached

        df = _df_cached

        if df.is_empty():
            print("DEBUG: update_color_map: DF is empty, returning empty color_map.")
            return {}

        if 'imported_path_short' not in df.columns:
            df = df.with_columns(
                pl.col("imported_path").map_elements(
                    lambda x: Path(x).name if x is not None else "Unknown Folder",
                    return_dtype=pl.String
                ).alias("imported_path_short")
            )

        folders = df.select(pl.col('imported_path_short')).unique().to_series().to_list()
        cmap = cm.get_cmap(palette, len(folders))
        return {
            f: f"#{int(cmap(i)[0] * 255):02x}{int(cmap(i)[1] * 255):02x}{int(cmap(i)[2] * 255):02x}"
            for i, f in enumerate(folders)
        }

    # Register widget callbacks
    for w in enabled_widgets:
        if hasattr(w, 'register_callbacks'):
            print(f"DEBUG: Registering callbacks for {w.name} with _df_cached reference.")
            w.register_callbacks(app, _df_cached)
