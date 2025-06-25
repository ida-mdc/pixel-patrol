import subprocess
from typing import List, Dict
from dash import html, dcc, Input, Output, State, callback_context  # Import callback_context
import polars as pl
import polars.selectors as cs
import os
import requests
import time
import numpy as np
import tempfile
from pathlib import Path
from tensorboardX import SummaryWriter
from PIL import Image  # Ensure PIL is imported for sprite generation

# Define SPRITE_SIZE here or ensure it's imported
SPRITE_SIZE = 16


from pixel_patrol.report.widget_interface import PixelPatrolWidget
from pixel_patrol.report.widget_categories import WidgetCategories

# --- Helper Functions (unchanged, but ensure PIL.Image is handled correctly) ---
def _create_sprite_image(df: pl.DataFrame):
    """
    Creates a sprite image from thumbnails stored in a Polars DataFrame.
    Assumes 'thumbnail' column contains PIL Image objects or numpy arrays.
    """
    if "thumbnail" not in df.columns or df.get_column("thumbnail").is_empty():
        return None

    try:
        # Import PIL.Image here to ensure it's available only if used
        from PIL import Image

        image_list = df.get_column("thumbnail").to_list()
        processed_images = []
        for img_data in image_list:
            if img_data is None:
                continue
            if isinstance(img_data, Image.Image):
                img = img_data
            elif isinstance(img_data, np.ndarray):
                # Ensure it's 8-bit for PIL.Image.fromarray if it's float or other
                if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                    img_data = (img_data * 255).astype(np.uint8)  # Assuming float 0-1
                elif img_data.dtype != np.uint8:  # Convert other int types if needed
                    img_data = img_data.astype(np.uint8)
                img = Image.fromarray(img_data)
            else:
                continue
            processed_images.append(img.resize((SPRITE_SIZE, SPRITE_SIZE)))

        if not processed_images:
            return None

        num_images = len(processed_images)
        images_per_row = int(np.ceil(np.sqrt(num_images)))  # Square arrangement
        sprite_width = images_per_row * SPRITE_SIZE
        sprite_height = int(np.ceil(num_images / images_per_row)) * SPRITE_SIZE

        sprite_image = Image.new('RGB', (sprite_width, sprite_height))

        for i, img in enumerate(processed_images):
            row = i // images_per_row
            col = i % images_per_row
            sprite_image.paste(img, (col * SPRITE_SIZE, row * SPRITE_SIZE))

        return sprite_image
    except ImportError:
        print("PIL (Pillow) not installed. Cannot generate sprite image.")
        return None


def _generate_projector_checkpoint(
        embeddings: np.ndarray,
        meta_df: pl.DataFrame,  # Expect Polars DataFrame
        log_dir: Path,
):
    """Creates TensorBoard embedding files."""
    writer = SummaryWriter(logdir=str(log_dir))

    # Convert metadata DataFrame to Pandas, handling 'thumbnail' column as well
    # Metadata for TensorBoardX should be a Pandas DataFrame
    metadata_for_tb = meta_df.drop("thumbnail",
                                   strict=False).to_pandas()  # strict=False to avoid error if 'thumbnail' isn't there

    # Sanitize tabs in metadata content (prevents accidental column splits)
    sanitized_df = metadata_for_tb.astype(str).replace(r'\t', ' ', regex=True)

    # Convert to list of lists for metadata
    metadata = sanitized_df.values.tolist()

    # Generate and save sprite image
    sprite_np_array = None
    if "thumbnail" in meta_df.columns:
        sprite_pil_image = _create_sprite_image(meta_df)
        if sprite_pil_image:
            sprite_np_array = np.array(sprite_pil_image)
            # TensorBoardX expects N, H, W, C for label_img
            if len(sprite_np_array.shape) == 3 and sprite_np_array.shape[2] == 3:  # RGB
                pass  # Already correct
            elif len(sprite_np_array.shape) == 2:  # Grayscale, add channel dim
                sprite_np_array = np.expand_dims(sprite_np_array, axis=-1)
            elif len(sprite_np_array.shape) == 3 and sprite_np_array.shape[2] == 4:  # RGBA, convert to RGB
                from PIL import Image
                sprite_np_array = np.array(Image.fromarray(sprite_np_array).convert('RGB'))

    writer.add_embedding(
        mat=embeddings,
        metadata=metadata,
        metadata_header=sanitized_df.columns.to_list(),
        label_img=sprite_np_array,
        tag="pixel_patrol_embedding",
        global_step=0
    )
    writer.close()


def _launch_tensorboard_subprocess(logdir: Path, port: int):
    """Launches TensorBoard as a background subprocess."""
    logdir.mkdir(parents=True, exist_ok=True)

    cmd = ["tensorboard", f"--logdir={logdir}", f"--port={port}", "--bind_all"]
    env = os.environ.copy()
    env["GCS_READ_CACHE_MAX_SIZE_MB"] = "0"

    try:
        tb_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)

        for _ in range(30):  # Try for up to 6 seconds
            try:
                requests.get(f"http://127.0.0.1:{port}", timeout=1)
                return tb_process
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                time.sleep(0.2)
        print(f"TensorBoard did not start on port {port} within the expected time.")
        tb_process.terminate()
        return None
    except Exception as e:
        print(f"Error launching TensorBoard subprocess: {e}")
        return None


# --- Dash Widget Class ---

class EmbeddingProjectorWidget(PixelPatrolWidget):

    @property
    def tab(self) -> str:
        return WidgetCategories.VISUALIZATION.value

    @property
    def name(self) -> str:
        return "TensorBoard Embedding Projector"

    def required_columns(self) -> List[str]:
        """Returns required data column names."""
        return ["*", "imported_path", "name"]

    def layout(self) -> List:
        """
        Defines the layout of the Embedding Projector widget.
        The interactive controls are now directly in the layout.
        """
        return [
            html.Div(id="projector-intro", children=[
                html.P("The "), html.Strong("Embedding Projector"),
                html.Span(" allows you to explore high-dimensional data by reducing it to 2D or 3D using "),
                html.Strong("Principal Component Analysis (PCA)"), html.Span(" or "),
                html.Strong("t-SNE"), html.Span(". "),
                html.Span(
                    "Embeddings represent data as points in a high-dimensional space; closer points are more similar."),
                html.P("This tool helps visualize relationships, clusters, and patterns in large datasets."),
                html.P(id="projector-summary-info")  # To display numeric column count
            ]),
            # Interactive controls are now part of the initial layout
            html.Div([
                html.Label("TensorBoard Port:"),
                dcc.Input(id="tb-port-input", type="number", value=6006, min=1024, max=65535,
                          style={"marginLeft": "10px", "width": "100px"}),
                html.Button("🚀 Start TensorBoard", id="start-tb-button", n_clicks=0,
                            style={"marginLeft": "20px", "marginRight": "10px"}),
                html.Button("🛑 Stop TensorBoard", id="stop-tb-button", n_clicks=0),
            ], style={"marginTop": "20px"}),
            html.Div(id="projector-status", style={"marginTop": "10px"}),
            html.Div(id="projector-link-area", style={"marginTop": "10px"}),
        ]

    def register_callbacks(self, app, df_global: pl.DataFrame):

        # Store to hold the TensorBoard process object (if launched) and log directory
        # This ID must be added to the main app's layout (e.g., in app.py)
        tb_process_store_id = f"tb-process-store-{self.name.replace(' ', '-').lower()}"

        # --- Callback for managing TensorBoard process ---
        @app.callback(
            Output("projector-summary-info", "children"),
            Output("projector-status", "children"),
            Output("projector-link-area", "children"),
            Output("start-tb-button", "disabled"),
            Output("stop-tb-button", "disabled"),
            Output(tb_process_store_id, "data"),  # <-- ADD THIS LINE HERE AS THE 6TH OUTPUT
            Input("start-tb-button", "n_clicks"),
            Input("stop-tb-button", "n_clicks"),
            State("tb-port-input", "value"),
            State(tb_process_store_id, "data"),  # This State provides current store data
            prevent_initial_call=True
        )
        def manage_tensorboard(
                start_clicks: int,
                stop_clicks: int,
                port: int,
                tb_state: Dict  # tb_state holds the current data from the dcc.Store
        ):
            # Determine which input triggered the callback
            ctx = callback_context  # Use the imported callback_context
            triggered_id = ctx.triggered_id if ctx.triggered else None

            current_pid = tb_state.get('pid')
            current_log_dir_str = tb_state.get('log_dir')
            current_log_dir = Path(current_log_dir_str) if current_log_dir_str else None

            # Default states for outputs
            summary_info_text = html.P("")  # Will be updated dynamically
            status_message = html.Span("")
            projector_link_children = []
            start_button_disabled = False
            stop_button_disabled = True  # Initially stop is disabled if no process is known

            # Re-evaluate df_numeric for summary info each time (or pass as state)
            df_numeric = df_global.select(cs.by_dtype(pl.NUMERIC_DTYPES)).fill_null(0.0)  # Corrected Polars syntax
            if df_numeric.is_empty():
                summary_info_text = html.P(
                    "No numeric data found! Embedding visualization requires numerical features.")
                return summary_info_text, \
                    html.P("Cannot start TensorBoard: No numeric data.", className="text-danger"), \
                    [], True, True, \
                    tb_state  # Return current state even on error, or reset it as needed

            summary_info_text = html.P(
                f"✅ {df_numeric.shape[1]} numeric columns, "
                f"with {df_numeric.shape[0]} rows can be utilized to display the data in the Embedding Projector."
            )

            # Check if a process is already running and update button states/link
            if current_pid:
                # Check if the process is actually still alive
                try:
                    os.kill(current_pid, 0)  # Signal 0 doesn't kill, just checks existence
                    # Process is running
                    status_message = html.P(f"TensorBoard is running on port {port} (PID: {current_pid}).",
                                            className="text-info")
                    projector_link_children = [
                        html.A(
                            f"🔗 Open TensorBoard Projector on port {port}",
                            href=f"http://127.0.0.1:{port}/#projector",
                            target="_blank",
                            className="button button-primary"
                        )
                    ]
                    start_button_disabled = True
                    stop_button_disabled = False
                except OSError:
                    # Process is dead or doesn't exist, clear state
                    status_message = html.P("TensorBoard process was terminated externally or crashed.",
                                            className="text-warning")
                    tb_state['pid'] = None
                    tb_state['log_dir'] = None
                    start_button_disabled = False
                    stop_button_disabled = True

            if triggered_id == "stop-tb-button":
                if current_pid:
                    try:
                        os.kill(current_pid, 9)  # Send SIGKILL to ensure termination
                        if current_log_dir and current_log_dir.exists():
                            import shutil
                            shutil.rmtree(current_log_dir)  # Clean up log directory
                        status_message = html.P("TensorBoard stopped and logs cleared.", className="text-success")
                    except OSError as e:
                        status_message = html.P(f"Error stopping TensorBoard (PID {current_pid}): {e}",
                                                className="text-danger")
                    tb_state['pid'] = None
                    tb_state['log_dir'] = None
                    start_button_disabled = False
                    stop_button_disabled = True
                else:
                    status_message = html.P("TensorBoard is not running.", className="text-info")
                    start_button_disabled = False
                    stop_button_disabled = True  # Ensure stop is disabled if not running

            elif triggered_id == "start-tb-button":
                if current_pid:  # Should be caught by the initial check, but for safety
                    status_message = html.P(f"TensorBoard is already running on port {port}.", className="text-info")
                else:
                    status_message = html.P("Starting TensorBoard...", className="text-warning")
                    start_button_disabled = True  # Disable during startup
                    stop_button_disabled = True

                    # Generate embeddings and metadata
                    embeddings_array = df_numeric.to_numpy()

                    new_log_dir = Path(tempfile.mkdtemp(prefix="tb_log_"))

                    try:
                        _generate_projector_checkpoint(embeddings_array, df_global, new_log_dir)
                        tb_process = _launch_tensorboard_subprocess(new_log_dir, port)
                        if tb_process:
                            tb_state['pid'] = tb_process.pid
                            tb_state['log_dir'] = str(new_log_dir)
                            status_message = html.P(f"TensorBoard is running on port {port}!", className="text-success")
                            projector_link_children = [
                                html.A(
                                    f"🔗 Open TensorBoard Projector on port {port}",
                                    href=f"http://127.0.0.1:{port}/#projector",
                                    target="_blank",
                                    className="button button-primary"
                                )
                            ]
                            start_button_disabled = True
                            stop_button_disabled = False
                        else:
                            status_message = html.P("Failed to start TensorBoard.", className="text-danger")
                            tb_state['pid'] = None  # Clear state on failure
                            tb_state['log_dir'] = None
                            start_button_disabled = False  # Enable start again
                            stop_button_disabled = True
                    except Exception as e:
                        status_message = html.P(f"Error preparing or starting TensorBoard: {e}",
                                                className="text-danger")
                        tb_state['pid'] = None
                        tb_state['log_dir'] = None
                        start_button_disabled = False
                        stop_button_disabled = True

            # Return all outputs and the updated state for the store
            # The order must EXACTLY match the Outputs declared in the decorator
            return summary_info_text, status_message, projector_link_children, \
                start_button_disabled, stop_button_disabled, \
                tb_state  # <-- This is the 6th value for the dcc.Store Output

        # --- Initial Setup Callback (Runs once on page load) ---
        @app.callback(
            Output("projector-summary-info", "children", allow_duplicate=True),
            Output("projector-status", "children", allow_duplicate=True),
            Output("projector-link-area", "children", allow_duplicate=True),
            Output("start-tb-button", "disabled", allow_duplicate=True),
            Output("stop-tb-button", "disabled", allow_duplicate=True),
            Output(tb_process_store_id, "data", allow_duplicate=True),  # This is the crucial Output for the store
            Input(tb_process_store_id, "data"),  # This Input triggers on initial load due to `data` being set
            prevent_initial_call='initial_duplicate'
        )
        def initial_layout_setup(tb_state_initial: Dict):
            df_numeric = df_global.select(cs.by_dtype(pl.NUMERIC_DTYPES)).fill_null(0.0)  # Corrected Polars syntax

            summary_text = html.P(
                f"✅ {df_numeric.shape[1]} numeric columns, "
                f"with {df_numeric.shape[0]} rows can be utilized to display the data in the Embedding Projector."
            ) if not df_numeric.is_empty() else html.P(
                "No numeric data found! Embedding visualization requires numerical features.")

            status = html.P("TensorBoard not running.", className="text-info")
            link_area = []
            start_button_disabled = False
            stop_button_disabled = True

            # Check if a process was known from a previous session state
            current_pid_initial = tb_state_initial.get('pid')
            initial_port = tb_state_initial.get('port', 6006)

            if current_pid_initial:
                try:
                    os.kill(current_pid_initial, 0)
                    status = html.P(
                        f"TensorBoard seems to be running on port {initial_port} (PID: {current_pid_initial}).",
                        className="text-warning")
                    link_area = [
                        html.A(
                            f"🔗 Open TensorBoard Projector on port {initial_port}",
                            href=f"http://127.0.0.1:{initial_port}/#projector",
                            target="_blank",
                            className="button button-primary"
                        )
                    ]
                    start_button_disabled = True
                    stop_button_disabled = False
                except OSError:
                    # Process is dead, clear state and reset buttons
                    tb_state_initial = {'pid': None, 'log_dir': None}
                    status = html.P("Previous TensorBoard process found dead. State cleared.", className="text-warning")
                    start_button_disabled = False
                    stop_button_disabled = True

            # The order must EXACTLY match the Outputs declared in the decorator
            return summary_text, status, link_area, start_button_disabled, stop_button_disabled, tb_state_initial