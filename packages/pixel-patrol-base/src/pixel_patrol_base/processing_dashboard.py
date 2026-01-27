"""
Dash app for configuring and monitoring Pixel Patrol processing.

This app provides a web interface to:
1. Configure processing parameters (equivalent to CLI export command)
2. Monitor processing progress in real-time
3. Launch the report dashboard after processing completes
"""
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

import polars as pl
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate

from pixel_patrol_base import api
from pixel_patrol_base.core.project_settings import Settings
from pixel_patrol_base.report.factory import create_info_icon

import logging
from collections import deque

logger = logging.getLogger(__name__)

# Use the same assets folder as the report dashboard
ASSETS_DIR = (Path(__file__).parent / "report" / "assets").resolve()

# Update progress every 500 ms
PROGRESS_UPDATE_INTERVAL_MS = 500  

# Global state for processing progress
# Note: Only JSON-serializable types should be stored here
_processing_state = {
    "status": "idle",  # idle, running, completed, error
    "progress": 0,  # 0-100
    "current_file": "",
    "total_files": 0,
    "processed_files": 0,
    "message": "",
    "error": None,
    "output_zip": None,  # Store path as string, not Project object
}

_processing_lock = threading.Lock()

# ============================================================================
# WARNING CAPTURE FOR DASH UI
# ============================================================================

# Global queue to collect warnings during processing
_warning_queue = deque(maxlen=100)  # Keep last 100 warnings

class DashWarningHandler(logging.Handler):
    """Custom logging handler that captures warnings for display in Dash."""

    def emit(self, record):
        if record.levelno >= logging.WARNING:  # WARNING, ERROR, CRITICAL
            _warning_queue.append({
                "level": record.levelname,
                "message": record.getMessage(),
                "timestamp": record.created,
                "module": record.module
            })


def get_warnings():
    """Get all warnings from the queue."""
    return list(_warning_queue)


def clear_warnings():
    """Clear the warning queue."""
    _warning_queue.clear()

# ============================================================================

def get_processing_state() -> Dict[str, Any]:
    """Get current processing state (thread-safe)."""
    with _processing_lock:
        state = _processing_state.copy()
        # Remove any non-JSON-serializable objects (like Project instances)
        # Explicitly remove 'project' key if it exists
        if 'project' in state:
            del state['project']
        
        # Only keep JSON-serializable types: str, int, float, bool, None, dict, list
        cleaned_state = {}
        for key, value in state.items():
            # Skip 'project' key entirely
            if key == 'project':
                continue
            if value is None or isinstance(value, (str, int, float, bool)):
                cleaned_state[key] = value
            elif isinstance(value, (dict, list)):
                # Recursively check dict/list contents
                try:
                    import json
                    json.dumps(value)  # Test if it's JSON serializable
                    cleaned_state[key] = value
                except (TypeError, ValueError):
                    # Skip non-serializable values
                    logger.warning(f"Skipping non-serializable value for key '{key}': {type(value)}")
            else:
                # Skip other types (like Project objects)
                logger.warning(f"Skipping non-serializable value for key '{key}': {type(value)}")
        return cleaned_state


def update_processing_state(**kwargs):
    """Update processing state (thread-safe)."""
    with _processing_lock:
        # Explicitly remove 'project' key if present
        kwargs.pop('project', None)
        
        # Filter out any non-JSON-serializable objects before updating
        cleaned_kwargs = {}
        for key, value in kwargs.items():
            # Skip 'project' key entirely
            if key == 'project':
                continue
            if value is None or isinstance(value, (str, int, float, bool)):
                cleaned_kwargs[key] = value
            elif isinstance(value, (dict, list)):
                # Test if it's JSON serializable
                try:
                    import json
                    json.dumps(value)
                    cleaned_kwargs[key] = value
                except (TypeError, ValueError):
                    logger.warning(f"Skipping non-serializable value for key '{key}': {type(value)}")
            else:
                # Skip non-serializable types (like Project objects)
                logger.warning(f"Skipping non-serializable value for key '{key}': {type(value)}")
        _processing_state.update(cleaned_kwargs)
        # Ensure 'project' is never in the state
        _processing_state.pop('project', None)


def _get_available_loaders() -> List[Dict[str, Any]]:
    """Get list of available loaders with their names and supported extensions."""
    from pixel_patrol_base.plugin_registry import discover_plugins_from_entrypoints
    
    loaders = []
    loader_plugins = discover_plugins_from_entrypoints("pixel_patrol.loader_plugins")
    
    # Add "None" option
    loaders.append({
        "label": "None (basic file info only)",
        "value": "",
        "extensions": []
    })
    
    # Add discovered loaders
    for loader_class in loader_plugins:
        try:
            # Get supported extensions from the class
            extensions = getattr(loader_class, 'SUPPORTED_EXTENSIONS', set())
            loaders.append({
                "label": loader_class.NAME,
                "value": loader_class.NAME,
                "extensions": sorted(list(extensions)) if extensions else []
            })
        except Exception as e:
            logger.warning(f"Could not get extensions for loader {loader_class.NAME}: {e}")
            loaders.append({
                "label": loader_class.NAME,
                "value": loader_class.NAME,
                "extensions": []
            })
    
    return loaders


def create_processing_app() -> Dash:
    """Create and configure the processing dashboard app."""

    # Setup warning capture
    handler = DashWarningHandler()
    handler.setLevel(logging.WARNING)
    logging.getLogger("pixel_patrol_base").addHandler(handler)
    
    external_stylesheets = [
        dbc.themes.BOOTSTRAP,
        "https://codepen.io/chriddyp/pen/bWLwgP.css",
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css",
    ]

    app = Dash(
        __name__,
        external_stylesheets=external_stylesheets,
        suppress_callback_exceptions=True,
        assets_folder=str(ASSETS_DIR),
    )

    app.layout = _create_layout(app)

    # Register callbacks
    _register_callbacks(app)

    return app


def _create_layout(app: Dash) -> html.Div:
    """Create the main layout for the processing dashboard."""
    return html.Div(
        [
            dcc.Store(id="processing-state-store", data=get_processing_state()),
            dcc.Interval(
                id="progress-interval",
                interval=PROGRESS_UPDATE_INTERVAL_MS,  # When / how often to update progress
                n_intervals=0,
                disabled=True,
            ),
            dbc.Container(
                [
                    # Header with logo (same as report dashboard)
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Img(
                                            src=app.get_asset_url("prevalidation.png"),
                                            style={
                                                "height": "110px",
                                                "marginRight": "15px",
                                            },
                                        ),
                                        html.H1(
                                            "Pixel Patrol Processing", className="m-0"
                                        ),
                                    ],
                                    className="d-flex align-items-center justify-content-center",
                                ),
                                width=12,
                            ),
                        ],
                        className="my-3",
                    ),
                    # Configuration Form (centered)
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            "Processing Configuration",
                                                            width="auto",
                                                        ),
                                                        dbc.Col(
                                                            create_info_icon(
                                                                widget_id="processing-config-info",
                                                                help_text=(
                                                                    "**What happens when you start processing:**\n\n"
                                                                    "Pixel Patrol will scan for files **within the selected folders**:\n"
                                                                    "  - If **Paths** are provided, only those subfolders/paths (relative to the Base Directory) are processed.\n"
                                                                    "  - If **Paths** are empty, Pixel Patrol processes the Base Directory.\n\n"
                                                                    "If no Loader is specified - only basic file information is extracted (e.g., path, size, timestamps).\n\n"
                                                                    "If loader is specified - Only files matching the selected **File Extensions** are included (leave empty = all supported).\n"
                                                                    "And PixelPatrol tries to extract rich data from those files - e.g. for images metadata and image data is processed.\n\n"
                                                                    "When processing finishes, Pixel Patrol creates a **project ZIP** at the Output ZIP path. "
                                                                    "This ZIP contains everything needed to generate the Pixel Patrol report.\n\n"
                                                                    "You can open the report later from a **terminal** where Pixel Patrol is available (e.g., your venv is activated):\n\n"
                                                                    "`pixel-patrol report /path/to/project.zip`\n\n"
                                                                    "Or launch the report directly from this page once processing completes."
                                                                ),
                                                            ),
                                                            width="auto",
                                                        ),
                                                    ],
                                                    align="center",
                                                    justify="start",
                                                    className="g-2",
                                                ),
                                            ),
                                            dbc.CardBody(
                                                [
                                                    dbc.Form(
                                                        [
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label(
                                                                                "Base Directory *",
                                                                                html_for="base-directory",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="base-directory",
                                                                                type="text",
                                                                                placeholder="/path/to/dataset",
                                                                                required=True,
                                                                            ),
                                                                            dbc.FormText(
                                                                                "Path to root folder containing your dataset",
                                                                                color="secondary",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label(
                                                                                "Output ZIP Path *",
                                                                                html_for="output-zip",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="output-zip",
                                                                                type="text",
                                                                                placeholder="project.zip",
                                                                                required=True,
                                                                            ),
                                                                            dbc.FormText(
                                                                                "Path where the output ZIP file will be saved - e.g., /path/to/project.zip",
                                                                                color="secondary",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label(
                                                                                "Project Name",
                                                                                html_for="project-name",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="project-name",
                                                                                type="text",
                                                                                placeholder="Auto (uses base-directory name)",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label(
                                                                                "Loader",
                                                                                html_for="loader",
                                                                            ),
                                                                            dbc.Select(
                                                                                id="loader",
                                                                                options=[
                                                                                    {
                                                                                        "label": loader[
                                                                                            "label"
                                                                                        ],
                                                                                        "value": loader[
                                                                                            "value"
                                                                                        ],
                                                                                    }
                                                                                    for loader in _get_available_loaders()
                                                                                ],
                                                                                value=(
                                                                                    _get_available_loaders()[
                                                                                        1
                                                                                    ][
                                                                                        "value"
                                                                                    ]
                                                                                    if len(
                                                                                        _get_available_loaders()
                                                                                    )
                                                                                    > 1
                                                                                    else ""
                                                                                ),
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label(
                                                                                "Paths (experimental conditions)",
                                                                                html_for="paths",
                                                                            ),
                                                                            dbc.Textarea(
                                                                                id="paths",
                                                                                placeholder="path1, path2...",
                                                                                rows=3,
                                                                                style={
                                                                                    "resize": "vertical"
                                                                                },
                                                                            ),
                                                                            dbc.FormText(
                                                                                "Comma-separated. Subdirectories of Base Directory ((absolute or relative).",
                                                                                color="secondary",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label(
                                                                                "File Extensions",
                                                                                html_for="file-extensions",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="file-extensions",
                                                                                type="text",
                                                                                placeholder="Leave empty for all supported extensions",
                                                                            ),
                                                                            dbc.FormText(
                                                                                id="file-extensions-help",
                                                                                children="Comma-separated extensions (leave empty for all)",
                                                                                color="secondary",
                                                                            ),
                                                                        ],
                                                                        md=6,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label(
                                                                                "Flavor",
                                                                                html_for="flavor",
                                                                            ),
                                                                            dbc.Input(
                                                                                id="flavor",
                                                                                type="text",
                                                                                placeholder="Optional label",
                                                                            ),
                                                                        ],
                                                                        md=12,
                                                                    ),
                                                                ],
                                                                className="mb-3",
                                                            ),
                                                            html.Hr(),
                                                            dbc.Button(
                                                                "Start Processing",
                                                                id="start-processing-btn",
                                                                color="primary",
                                                                size="lg",
                                                                className="w-100",
                                                                disabled=False,
                                                            ),
                                                        ]
                                                    ),
                                                ]
                                            ),
                                        ],
                                        className="mb-4",
                                    ),
                                ],
                                md=8,
                                className="offset-md-2",  # Center the form
                            ),
                        ]
                    ),
                    # Progress Panel (centered, below form)
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardHeader("Processing Progress"),
                                            dbc.CardBody(
                                                [
                                                    html.Div(
                                                        id="progress-status",
                                                        children="Ready to start processing",
                                                    ),
                                                    html.Div(
                                                        id="progress-bar-container",
                                                        className="mt-3",
                                                    ),
                                                    html.Div(
                                                        id="progress-details",
                                                        className="mt-3",
                                                    ),
                                                    html.Div(
                                                        id="error-message",
                                                        className="mt-2",
                                                    ),
                                                    html.Div(
                                                        id="warnings-display",
                                                        className="mt-1",
                                                    ),
                                                    html.Div(
                                                        id="action-buttons",
                                                        className="mt-4",
                                                    ),
                                                ]
                                            ),
                                        ],
                                    ),
                                ],
                                md=8,
                                className="offset-md-2",  # Center the progress panel
                            ),
                        ]
                    ),
                ],
                fluid=True,
                className="py-4",
            ),
        ]
    )


def _register_callbacks(app: Dash):
    """Register all callbacks for the processing dashboard."""

    @app.callback(
        [
            Output("processing-state-store", "data"),
            Output("progress-interval", "disabled"),
            Output("start-processing-btn", "disabled"),
        ],
        [
            Input("start-processing-btn", "n_clicks"),
            Input("progress-interval", "n_intervals"),
        ],
        [
            State("base-directory", "value"),
            State("output-zip", "value"),
            State("project-name", "value"),
            State("loader", "value"),
            State("paths", "value"),
            State("file-extensions", "value"),
            State("flavor", "value"),
        ],
    )
    def update_processing_state_callback(
        n_clicks,
        _n_intervals,
        base_directory,
        output_zip,
        project_name,
        loader,
        paths,
        file_extensions,
        flavor,
    ):
        """Handle start button click and progress updates."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Start processing button clicked
        if trigger_id == "start-processing-btn" and n_clicks:
            state = get_processing_state()
            if state["status"] == "running":
                raise PreventUpdate

            # Validate required fields
            if not base_directory or not output_zip:
                update_processing_state(
                    status="error",
                    error="Base directory and output ZIP are required",
                )
                return get_processing_state(), True, False

            clear_warnings()

            # Clear old output_zip from state when starting new processing
            update_processing_state(
                status="running",
                progress=0,
                message="Starting processing...",
                error=None,
                output_zip=None,  # Clear old output_zip
            )
            
            # Start processing in background thread
            thread = threading.Thread(
                target=_run_processing,
                args=(
                    base_directory,
                    output_zip,
                    project_name,
                    loader or None,
                    paths,
                    file_extensions,
                    flavor or "",
                ),
                daemon=True,
            )
            thread.start()

            return get_processing_state(), False, True  # Enable interval, disable button

        # Progress interval update
        elif trigger_id == "progress-interval":
            state = get_processing_state()
            if state["status"] == "running":
                return state, False, True  # Keep interval running, button disabled
            elif state["status"] in ["completed", "error"]:
                return state, True, False  # Stop interval, enable button

        raise PreventUpdate

    @app.callback(
        [
            Output("progress-status", "children"),
            Output("progress-bar-container", "children"),
            Output("progress-details", "children"),
            Output("error-message", "children"),
            Output("action-buttons", "children"),
        ],
        [Input("processing-state-store", "data")],
    )
    def update_progress_display(state: Dict[str, Any]):
        """Update the progress display based on current state."""
        status = state.get("status", "idle")
        progress = state.get("progress", 0)
        message = state.get("message", "")
        error = state.get("error")
        current_file = state.get("current_file", "")
        processed_files = state.get("processed_files", 0)
        total_files = state.get("total_files", 0)
        output_zip = state.get("output_zip")

        # Status text
        if status == "idle":
            status_text = html.P("Ready to start processing", className="text-muted")
        elif status == "running":
            status_text = html.P(
                [html.Strong("Processing..."), html.Br(), message],
                className="text-primary",
            )
        elif status == "completed":
            status_text = html.P(
                [html.Strong("Processing completed!"), html.Br(), message],
                className="text-success",
            )
        elif status == "error":
            status_text = html.P(
                [html.Strong("Error occurred"), html.Br(), error or "Unknown error"],
                className="text-danger",
            )
        else:
            status_text = html.P("Unknown status", className="text-muted")

        # Progress bar
        if status == "running" or status == "completed":
            progress_bar = dbc.Progress(
                value=progress,
                label=f"{progress:.1f}%",
                color="primary" if status == "running" else "success",
                striped=True,
                animated=status == "running",
                className="mb-2",
            )
        else:
            progress_bar = html.Div()

        # Details
        details = []
        if current_file:
            details.append(html.P([html.Strong("Current file: "), current_file], className="small"))
        if total_files > 0:
            details.append(
                html.P(
                    [html.Strong(f"Progress: {processed_files}/{total_files} files")],
                    className="small",
                )
            )
        details_div = html.Div(details) if details else html.Div()

        # Error message
        error_div = (
            dbc.Alert(error, color="danger", className="mb-0")
            if error and status == "error"
            else html.Div()
        )

        # Action buttons
        action_buttons = []
        if status == "completed" and output_zip:
            action_buttons.append(
                dbc.Button(
                    "Launch Report Dashboard",
                    id="launch-report-btn",
                    color="success",
                    size="lg",
                    className="w-100",
                )
            )

        return (
            status_text,
            progress_bar,
            details_div,
            error_div,
            html.Div(action_buttons),
        )


    @app.callback(
        Output("warnings-display", "children"),
        Input("processing-state-store", "data")
    )
    def display_warnings(_state):
        """Display captured warnings and errors from processing."""
        warnings = get_warnings()

        if not warnings:
            return None

        warning_items = []
        for w in warnings[-10:]:  # Show last 10 warnings
            color = "danger" if w["level"] == "ERROR" else "warning"
            icon = "bi-x-circle-fill" if w["level"] == "ERROR" else "bi-exclamation-triangle-fill"

            warning_items.append(
                dbc.Alert(
                    [
                        html.I(className=f"{icon} me-2"),
                        html.Strong(f"{w['level']}: ", className="me-1"),
                        w["message"],
                    ],
                    color=color,
                    dismissable=True,
                    className="mb-2 py-2",
                    style={"fontSize": "0.9rem"}
                )
            )

        if warning_items:
            return html.Div([
                *warning_items
            ])

        return None

    @app.callback(
        Output("launch-report-btn", "n_clicks"),
        [Input("launch-report-btn", "n_clicks")],
        [State("processing-state-store", "data")],
        prevent_initial_call=True,
    )
    def launch_report(n_clicks, state):
        """Launch the report dashboard in a new thread."""
        if n_clicks:
            output_zip = state.get("output_zip")
            if output_zip:
                output_path = Path(output_zip)
                # Resolve to absolute path to ensure we use the correct file
                if not output_path.is_absolute():
                    output_path = output_path.resolve()
                
                if output_path.exists():
                    logger.info(f"Launching report from: {output_path}")
                    # Launch report in background thread with resolved path
                    thread = threading.Thread(
                        target=_launch_report_app,
                        args=(str(output_path),),
                        daemon=True,
                    )
                    thread.start()
                    update_processing_state(message=f"Report dashboard launching from {output_path.name}...")
                else:
                    logger.error(f"Output ZIP file does not exist: {output_path}")
                    update_processing_state(
                        status="error",
                        error=f"Output ZIP file not found: {output_path}",
                    )
            else:
                logger.error("No output ZIP path in state")
                update_processing_state(
                    status="error",
                    error="No output ZIP file available. Please run processing first.",
                )
        return 0  # Reset button click count

    @app.callback(
        [
            Output("file-extensions", "placeholder"),
            Output("file-extensions-help", "children"),
        ],
        [Input("loader", "value")],
    )
    def update_file_extensions_placeholder(loader_value: str):
        """Update file extensions placeholder based on selected loader."""
        loaders = _get_available_loaders()
        
        # Find the selected loader
        selected_loader = None
        for loader in loaders:
            if loader["value"] == loader_value:
                selected_loader = loader
                break
        
        if selected_loader and selected_loader["extensions"]:
            extensions_str = ", ".join(selected_loader["extensions"])
            placeholder = f"e.g., {extensions_str[:50]}{'...' if len(extensions_str) > 50 else ''}"
            help_text = [f"Leave empty for all supported extensions. Otherwise, comma-separated extensions:",
                         html.Br(),
                         f"Supported: {', '.join(selected_loader['extensions'][:10])}"
                         f"{'...' if len(selected_loader['extensions']) > 10 else ''}"]
        else:
            placeholder = "Leave empty for all supported extensions"
            help_text = "Comma-separated extensions (leave empty for all)"
        
        return placeholder, help_text


def _run_processing(
    base_directory: str,
    output_zip: str,
    project_name: Optional[str],
    loader: Optional[str],
    paths: Optional[str],
    file_extensions: Optional[str],
    flavor: str,
):
    """Run processing in background thread."""
    try:
        base_dir = Path(base_directory).resolve()
        if not base_dir.exists():
            update_processing_state(
                status="error",
                error=f"Base directory does not exist: {base_directory}",
            )
            return

        # Parse paths
        path_list = []
        if paths:
            path_list = [p.strip() for p in paths.split(",") if p.strip()]

        # Parse file extensions
        if file_extensions:
            extensions = {ext.strip().lstrip(".") for ext in file_extensions.split(",") if ext.strip()}
        else:
            extensions = "all"

        # Derive project name
        if not project_name:
            project_name = base_dir.name

        update_processing_state(
            status="running",
            progress=5,
            message="Creating project...",
        )

        # Create project
        project = api.create_project(project_name, str(base_dir), loader=loader)

        if path_list:
            api.add_paths(project, path_list)
        else:
            api.add_paths(project, base_dir)

        update_processing_state(
            status="running",
            progress=10,
            message="Setting project settings...",
        )

        # Set settings (using defaults for cmap and n_example_files)
        settings = Settings(
            cmap="rainbow",  # Default value
            n_example_files=9,  # Default value
            selected_file_extensions=extensions,
            pixel_patrol_flavor=flavor,
        )
        api.set_settings(project, settings)

        update_processing_state(
            status="running",
            progress=15,
            message="Processing files...",
        )

        # Get file count first for better progress tracking
        from pixel_patrol_base.core.file_system import walk_filesystem
        basic_df = walk_filesystem(
            project.get_paths() if project.get_paths() else [project.get_base_dir()],
            loader=project.get_loader(),
            accepted_extensions=extensions if isinstance(extensions, set) else "all"
        )
        
        total_files = 0
        if basic_df is not None and not basic_df.is_empty():
            total_files = basic_df.filter(pl.col("type") == "file").height
            update_processing_state(
                status="running",
                progress=20,
                message=f"Found {total_files} files. Processing...",
                total_files=total_files,
            )
        
        # Define progress callback that updates our processing state
        def progress_callback(current: int, total: int, current_file: Path) -> None:
            """Callback function to update processing state during file processing."""
            # File-level progress (20-85% range)
            progress_pct = 20 + int((current / total) * 65) if total > 0 else 20
            
            update_processing_state(
                status="running",
                progress=progress_pct,
                message=f"Processing file {current}/{total}...",
                processed_files=current,
                total_files=total,
                current_file=current_file.name if current_file else "",
            )
        
        # Use the existing processing method with progress callback
        api.process_files(project, progress_callback=progress_callback)
        
        # Update progress after processing
        if project.records_df is not None:
            actual_processed = project.records_df.height
            update_processing_state(
                status="running",
                progress=85,
                message=f"Processed {actual_processed} files. Exporting...",
                processed_files=actual_processed if total_files == 0 else total_files,
            )
        else:
            update_processing_state(
                status="running",
                progress=85,
                message="Processing complete. Exporting...",
            )

        update_processing_state(
            status="running",
            progress=90,
            message="Exporting project...",
        )

        # Export project
        output_path = Path(output_zip).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get modification time before export to verify it changes
        old_mtime = output_path.stat().st_mtime if output_path.exists() else 0
        
        api.export_project(project, output_path)
        
        # Verify the file was written and get its modification time
        if not output_path.exists():
            raise FileNotFoundError(f"Export failed: {output_path} does not exist after export")
        
        new_mtime = output_path.stat().st_mtime
        file_size = output_path.stat().st_size
        
        if new_mtime <= old_mtime and old_mtime > 0:
            logger.warning(f"ZIP file modification time did not change. Old: {old_mtime}, New: {new_mtime}")
        
        if file_size == 0:
            raise ValueError(f"Export failed: {output_path} is empty after export")

        # Update state with the resolved absolute path to ensure we use the correct file
        resolved_output_zip = str(output_path)
        update_processing_state(
            status="completed",
            progress=100,
            message=f"Project exported to {resolved_output_zip}",
            output_zip=resolved_output_zip,
        )
        
        logger.info(f"Processing completed. Output ZIP saved to: {resolved_output_zip}")
        logger.info(f"ZIP file size: {file_size} bytes, modified: {datetime.fromtimestamp(new_mtime)}")

    except Exception as e:
        logger.exception("Error during processing")
        update_processing_state(
            status="error",
            error=f"Processing failed: {str(e)}",
        )


def _launch_report_app(output_zip: str):
    """Launch the report dashboard app in a subprocess."""
    import subprocess
    import sys
    import socket
    import hashlib
    
    def is_port_in_use(port: int) -> bool:
        """Check if a port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return False
            except OSError:
                return True
    
    def find_free_port(start_port: int = 8050, max_attempts: int = 20) -> int:
        """Find a free port starting from start_port."""
        for i in range(max_attempts):
            port = start_port + i
            if not is_port_in_use(port):
                return port
        # If no free port found, return start_port anyway (will fail but at least we try)
        return start_port
    
    try:
        # Verify the ZIP file exists and get absolute path
        output_path = Path(output_zip)
        if not output_path.is_absolute():
            output_path = output_path.resolve()
        
        if not output_path.exists():
            raise FileNotFoundError(f"Output ZIP file does not exist: {output_path}")
        
        # Use a port based on the ZIP file hash to ensure consistency
        # This way, the same ZIP file always uses the same port (if available)
        # But different ZIP files can use different ports
        zip_hash = hashlib.md5(str(output_path).encode()).hexdigest()
        port_offset = int(zip_hash[:2], 16) % 100  # Use first 2 hex chars for port offset (0-255, mod 100 = 0-99)
        base_port = 8050
        report_port = base_port + port_offset
        
        # Find a free port starting from our calculated port
        report_port = find_free_port(report_port, max_attempts=10)
        
        logger.info(f"Launching report on port {report_port} from ZIP: {output_path}")
        logger.info(f"ZIP file size: {output_path.stat().st_size} bytes")
        logger.info(f"ZIP file modified: {datetime.fromtimestamp(output_path.stat().st_mtime)}")
        
        # Launch report in a separate process with absolute path
        # Use the absolute path to ensure we're using the correct file
        cmd = [
            sys.executable,
            "-m",
            "pixel_patrol_base.cli",
            "report",
            str(output_path),  # Use absolute path as string
            "--port",
            str(report_port),
        ]
        
        logger.info(f"Executing command: {' '.join(cmd)}")
        
        # Launch process - don't capture stdout/stderr so errors are visible
        process = subprocess.Popen(
            cmd,
            stdout=None,  # Let output go to console
            stderr=None,  # Let errors go to console
            cwd=None,
        )
        
        logger.info(f"Report process started with PID {process.pid} for ZIP: {output_path}")
        
        update_processing_state(
            message=f"Report dashboard launching on http://127.0.0.1:{report_port}/ from {output_path.name}",
        )
    except Exception as e:
        logger.exception(f"Error launching report app for {output_zip}")
        update_processing_state(
            status="error",
            error=f"Failed to launch report: {str(e)}",
        )
