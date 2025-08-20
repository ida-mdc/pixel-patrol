import os
import tempfile
from pathlib import Path

import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
from dash import html, dcc, Input, Output, State  # <--- No need for callback_context here

from pixel_patrol.api import create_project, add_paths, set_settings, process_images
from pixel_patrol.core.project import Settings

layout = dbc.Container([
    html.H2("Processing Configuration", className="my-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Dataset Base Path:"),
            dcc.Input(
                id="dataset-base-path-input",
                type="text",
                style={"width": "100%"},
                debounce=True
            ),
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Project Name:"),
            dcc.Input(
                id="project-name-input",
                type="text",
                value="My Pixel Patrol Project",
                style={"width": "100%"},
                debounce=True
            )
        ], width=6),
        dbc.Col([
            html.Label("File Extensions (comma-separated):"),
            dcc.Input(
                id="file-extensions-input",
                type="text",
                value="png,tiff,jpg,jpeg",
                style={"width": "100%"},
                debounce=True
            )
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Number of Example Images:"),
            dcc.Input(
                id="n-example-images-input",
                type="number",
                value=5,
                min=1,
                step=1,
                style={"width": "100%"},
                debounce=True
            )
        ], width=6),
        dbc.Col([
            html.Label("Colormap:"),
            dcc.Dropdown(
                id="colormap-dropdown",
                options=[{'label': cm_name, 'value': cm_name} for cm_name in sorted(plt.colormaps())],
                value="viridis",
                style={"width": "100%"},
                clearable=False
            )
        ], width=6),
    ]),
    html.Br(),
    dbc.Button("Run Processing", id="run-processing-button", className="me-2", n_clicks=0),
    html.Div(id="processing-status", className="mt-3"),
    dcc.Loading(
        id="loading-output",
        type="default",
        children=html.Div(id="loading-container")
    ),
    dcc.Store(id='project-data-store'),
    dcc.Location(id='url', refresh=False),
], fluid=True, style={'maxWidth': '800px', 'margin': '0 auto'})


# Define a function to register callbacks, which takes the Dash app instance
def register_callbacks(app):
    @app.callback(  # <--- CORRECTED: Using 'app.callback' directly
        Output("processing-status", "children"),
        Output("loading-container", "children"),
        Output("project-data-store", "data"),
        Output("url", "pathname"),
        Input("run-processing-button", "n_clicks"),
        State("dataset-base-path-input", "value"),
        State("project-name-input", "value"),
        State("file-extensions-input", "value"),
        State("n-example-images-input", "value"),
        State("colormap-dropdown", "value"),
        prevent_initial_call=True
    )
    def run_processing(
            n_clicks,
            dataset_base_path_str,
            project_name,
            file_extensions_str,
            n_example_images,
            colormap
    ):
        if n_clicks > 0:
            status_messages = []
            selected_file_extensions = set(ext.strip() for ext in file_extensions_str.split(',') if ext.strip())

            base_dataset_path = Path(dataset_base_path_str)
            if not base_dataset_path.is_dir():
                raise FileNotFoundError(f"Dataset base path not found: {base_dataset_path}")

            my_project = create_project(project_name, str(base_dataset_path))
            status_messages.append(f"Project '{project_name}' created at: {base_dataset_path}")

            initial_settings = Settings(
                selected_file_extensions=selected_file_extensions,
                cmap=colormap,
                n_example_images=n_example_images
            )
            set_settings(my_project, initial_settings)
            status_messages.append("Project settings applied.")

            process_images(my_project)
            status_messages.append("Processing complete (in memory).")

            temp_dir = tempfile.gettempdir()
            temp_parquet_file = Path(temp_dir) / f"pixel_patrol_temp_data_{os.getpid()}.parquet"

            my_project.images_df.write_parquet(temp_parquet_file)
            status_messages.append(f"DataFrame saved to temporary file: {temp_parquet_file}")

            project_info = {
                'processed_data_path': str(temp_parquet_file),
                'colormap': colormap
            }

            return html.Ul([html.Li(msg) for msg in status_messages]), None, project_info, "/report"
        return [], None, {}, "/processing"