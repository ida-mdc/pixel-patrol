import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import polars as pl
import polars.selectors as cs
import requests
from PIL import Image
from dash import html, dcc, Input, Output, State, callback_context
from tensorboardX import SummaryWriter

from pixel_patrol_base.report.widget_categories import WidgetCategories

SPRITE_SIZE = 16


def _create_sprite_image(df: pl.DataFrame):
    """
    Creates a sprite image from thumbnails stored in a Polars DataFrame.
    Assumes 'thumbnail' column contains PIL Image objects or numpy arrays.
    """
    if "thumbnail" not in df.columns or df.get_column("thumbnail").is_empty():
        return None

    image_list = df.get_column("thumbnail").to_list()
    processed_images = []
    for img_data in image_list:
        if img_data is None:
            continue
        if isinstance(img_data, Image.Image):
            img = img_data
        elif isinstance(img_data, np.ndarray):
            # normalize to uint8
            if img_data.dtype in (np.float32, np.float64):
                img_data = (np.clip(img_data, 0, 1) * 255).astype(np.uint8)
            elif img_data.dtype != np.uint8:
                img_data = img_data.astype(np.uint8)
            img = Image.fromarray(img_data)
        else:
            continue
        processed_images.append(img.resize((SPRITE_SIZE, SPRITE_SIZE)))

    if not processed_images:
        return None

    num_images = len(processed_images)
    images_per_row = int(np.ceil(np.sqrt(num_images)))  # Square grid
    sprite_width = images_per_row * SPRITE_SIZE
    sprite_height = int(np.ceil(num_images / images_per_row)) * SPRITE_SIZE

    sprite_image = Image.new("RGB", (sprite_width, sprite_height))
    for i, img in enumerate(processed_images):
        row = i // images_per_row
        col = i % images_per_row
        sprite_image.paste(img, (col * SPRITE_SIZE, row * SPRITE_SIZE))
    return sprite_image


def _generate_projector_checkpoint(
    embeddings: np.ndarray,
    meta_df: pl.DataFrame,
    log_dir: Path,
):
    """Creates TensorBoard embedding files."""
    writer = SummaryWriter(logdir=str(log_dir))

    # TensorBoardX expects pandas metadata; drop thumbnails if present
    metadata_for_tb = meta_df.drop("thumbnail", strict=False).to_pandas()
    sanitized_df = metadata_for_tb.astype(str).replace(r"\t", " ", regex=True)
    metadata = sanitized_df.values.tolist()

    # Optional sprite (thumbnail grid)
    sprite_np_array = None
    if "thumbnail" in meta_df.columns:
        sprite_pil_image = _create_sprite_image(meta_df)
        if sprite_pil_image:
            sprite_np_array = np.array(sprite_pil_image)
            if sprite_np_array.ndim == 2:
                sprite_np_array = np.expand_dims(sprite_np_array, axis=-1)
            elif sprite_np_array.ndim == 3 and sprite_np_array.shape[2] == 4:
                sprite_np_array = np.array(Image.fromarray(sprite_np_array).convert("RGB"))

    writer.add_embedding(
        mat=embeddings,
        metadata=metadata,
        metadata_header=list(sanitized_df.columns),
        label_img=sprite_np_array,
        tag="pixel_patrol_embedding",
        global_step=0,
    )
    writer.close()


def _launch_tensorboard_subprocess(logdir: Path, port: int):
    """Launch TensorBoard and wait briefly until it responds; return Popen or None."""
    logdir.mkdir(parents=True, exist_ok=True)
    cmd = ["tensorboard", f"--logdir={logdir}", f"--port={port}", "--bind_all"]
    env = os.environ.copy()
    env["GCS_READ_CACHE_MAX_SIZE_MB"] = "0"

    try:
        tb_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        for _ in range(30):  # up to ~6s
            try:
                requests.get(f"http://127.0.0.1:{port}", timeout=1)
                return tb_process
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                time.sleep(0.2)
        tb_process.terminate()
        return None
    except Exception:
        return None


class EmbeddingProjectorWidget:
    # ---- Declarative spec ----
    NAME: str = "TensorBoard Embedding Projector"
    TAB: str = WidgetCategories.VISUALIZATION.value
    REQUIRES: Set[str] = set()           # no specific columns required; uses numeric columns dynamically
    REQUIRES_PATTERNS = None

    # Component IDs
    INTRO_ID = "projector-intro"
    SUMMARY_ID = "projector-summary-info"
    STATUS_ID = "projector-status"
    LINK_ID = "projector-link-area"
    PORT_INPUT_ID = "tb-port-input"
    START_BTN_ID = "start-tb-button"
    STOP_BTN_ID = "stop-tb-button"
    # dcc.Store ID to preserve TB state; ensure a matching dcc.Store(id=STORE_ID, data={}) exists in app layout
    STORE_ID = "tb-process-store-embedding-projector"

    def layout(self) -> List:
        return [
            html.Div(
                id=self.INTRO_ID,
                children=[
                    html.P("The "),
                    html.Strong("Embedding Projector"),
                    html.Span(" allows you to explore high-dimensional data by reducing it to 2D or 3D using "),
                    html.Strong("Principal Component Analysis (PCA)"),
                    html.Span(" or "),
                    html.Strong("t-SNE"),
                    html.Span(". "),
                    html.Span(
                        "Embeddings represent data as points in a high-dimensional space; closer points are more similar."
                    ),
                    html.P("This tool helps visualize relationships, clusters, and patterns in large datasets."),
                    html.P(id=self.SUMMARY_ID),
                ],
            ),
            html.Div(
                [
                    html.Label("TensorBoard Port:"),
                    dcc.Input(
                        id=self.PORT_INPUT_ID, type="number", value=6006, min=1024, max=65535,
                        style={"marginLeft": "10px", "width": "100px"}
                    ),
                    html.Button("🚀 Start TensorBoard", id=self.START_BTN_ID, n_clicks=0,
                                style={"marginLeft": "20px", "marginRight": "10px"}),
                    html.Button("🛑 Stop TensorBoard", id=self.STOP_BTN_ID, n_clicks=0),
                ],
                style={"marginTop": "20px"},
            ),
            html.Div(id=self.STATUS_ID, style={"marginTop": "10px"}),
            html.Div(id=self.LINK_ID, style={"marginTop": "10px"}),
            # NOTE: Add dcc.Store(id=self.STORE_ID, data={}) to your app layout.
        ]

    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            Output(self.SUMMARY_ID, "children"),
            Output(self.STATUS_ID, "children"),
            Output(self.LINK_ID, "children"),
            Output(self.START_BTN_ID, "disabled"),
            Output(self.STOP_BTN_ID, "disabled"),
            Output(self.STORE_ID, "data"),
            Input(self.START_BTN_ID, "n_clicks"),
            Input(self.STOP_BTN_ID, "n_clicks"),
            State(self.PORT_INPUT_ID, "value"),
            State(self.STORE_ID, "data"),
            prevent_initial_call=True,
        )
        def manage_tensorboard(
            start_clicks: int,
            stop_clicks: int,
            port: int,
            tb_state: Dict,
        ):
            # defaults
            tb_state = tb_state or {}
            port = port or 6006

            ctx = callback_context
            triggered_id = ctx.triggered_id if ctx.triggered else None

            current_pid = tb_state.get("pid")
            current_log_dir_str = tb_state.get("log_dir")
            current_log_dir = Path(current_log_dir_str) if current_log_dir_str else None

            summary_info_text = html.P("")
            status_message = html.Span("")
            projector_link_children: List = []
            start_button_disabled = False
            stop_button_disabled = True

            # compute numeric feature matrix
            df_numeric = df_global.select(cs.by_dtype(pl.NUMERIC_DTYPES)).fill_null(0.0)
            if df_numeric.is_empty():
                summary_info_text = html.P(
                    "No numeric data found! Embedding visualization requires numerical features."
                )
                return (
                    summary_info_text,
                    html.P("Cannot start TensorBoard: No numeric data.", className="text-danger"),
                    [],
                    True,
                    True,
                    tb_state,
                )

            summary_info_text = html.P(
                f"✅ {df_numeric.shape[1]} numeric columns, "
                f"with {df_numeric.shape[0]} rows can be utilized to display the data in the Embedding Projector."
            )

            # if already running
            if current_pid:
                try:
                    os.kill(current_pid, 0)
                    status_message = html.P(
                        f"TensorBoard is running on port {port} (PID: {current_pid}).", className="text-info"
                    )
                    projector_link_children = [
                        html.A(
                            f"🔗 Open TensorBoard Projector on port {port}",
                            href=f"http://127.0.0.1:{port}/#projector",
                            target="_blank",
                            className="button button-primary",
                        )
                    ]
                    start_button_disabled = True
                    stop_button_disabled = False
                except OSError:
                    # dead -> clear
                    status_message = html.P(
                        "TensorBoard process was terminated externally or crashed.", className="text-warning"
                    )
                    tb_state["pid"] = None
                    tb_state["log_dir"] = None
                    start_button_disabled = False
                    stop_button_disabled = True

            if triggered_id == self.STOP_BTN_ID:
                if current_pid:
                    try:
                        os.kill(current_pid, 9)
                        if current_log_dir and current_log_dir.exists():
                            import shutil
                            shutil.rmtree(current_log_dir)
                        status_message = html.P("TensorBoard stopped and logs cleared.", className="text-success")
                    except OSError as e:
                        status_message = html.P(f"Error stopping TensorBoard (PID {current_pid}): {e}",
                                                className="text-danger")
                    tb_state["pid"] = None
                    tb_state["log_dir"] = None
                    start_button_disabled = False
                    stop_button_disabled = True
                else:
                    status_message = html.P("TensorBoard is not running.", className="text-info")
                    start_button_disabled = False
                    stop_button_disabled = True

            elif triggered_id == self.START_BTN_ID:
                if current_pid:
                    status_message = html.P(f"TensorBoard is already running on port {port}.", className="text-info")
                else:
                    status_message = html.P("Starting TensorBoard...", className="text-warning")
                    start_button_disabled = True
                    stop_button_disabled = True

                    embeddings_array = df_numeric.to_numpy()
                    new_log_dir = Path(tempfile.mkdtemp(prefix="tb_log_"))

                    try:
                        _generate_projector_checkpoint(embeddings_array, df_global, new_log_dir)
                        tb_process = _launch_tensorboard_subprocess(new_log_dir, port)
                        if tb_process:
                            tb_state["pid"] = tb_process.pid
                            tb_state["log_dir"] = str(new_log_dir)
                            tb_state["port"] = port
                            status_message = html.P(
                                f"TensorBoard is running on port {port}!", className="text-success"
                            )
                            projector_link_children = [
                                html.A(
                                    f"🔗 Open TensorBoard Projector on port {port}",
                                    href=f"http://127.0.0.1:{port}/#projector",
                                    target="_blank",
                                    className="button button-primary",
                                )
                            ]
                            start_button_disabled = True
                            stop_button_disabled = False
                        else:
                            status_message = html.P("Failed to start TensorBoard.", className="text-danger")
                            tb_state["pid"] = None
                            tb_state["log_dir"] = None
                            start_button_disabled = False
                            stop_button_disabled = True
                    except Exception as e:
                        status_message = html.P(f"Error preparing or starting TensorBoard: {e}",
                                                className="text-danger")
                        tb_state["pid"] = None
                        tb_state["log_dir"] = None
                        start_button_disabled = False
                        stop_button_disabled = True

            return (
                summary_info_text,
                status_message,
                projector_link_children,
                start_button_disabled,
                stop_button_disabled,
                tb_state,
            )

        # Initial state sync on page load
        @app.callback(
            Output(self.SUMMARY_ID, "children", allow_duplicate=True),
            Output(self.STATUS_ID, "children", allow_duplicate=True),
            Output(self.LINK_ID, "children", allow_duplicate=True),
            Output(self.START_BTN_ID, "disabled", allow_duplicate=True),
            Output(self.STOP_BTN_ID, "disabled", allow_duplicate=True),
            Output(self.STORE_ID, "data", allow_duplicate=True),
            Input(self.STORE_ID, "data"),
            prevent_initial_call="initial_duplicate",
        )
        def initial_layout_setup(tb_state_initial: Dict):
            tb_state_initial = tb_state_initial or {}
            df_numeric = df_global.select(cs.by_dtype(pl.NUMERIC_DTYPES)).fill_null(0.0)

            summary_text = (
                html.P(
                    f"✅ {df_numeric.shape[1]} numeric columns, "
                    f"with {df_numeric.shape[0]} rows can be utilized to display the data in the Embedding Projector."
                )
                if not df_numeric.is_empty()
                else html.P("No numeric data found! Embedding visualization requires numerical features.")
            )

            status = html.P("TensorBoard not running.", className="text-info")
            link_area: List = []
            start_button_disabled = False
            stop_button_disabled = True

            current_pid_initial = tb_state_initial.get("pid")
            initial_port = tb_state_initial.get("port", 6006)
            if current_pid_initial:
                try:
                    os.kill(current_pid_initial, 0)
                    status = html.P(
                        f"TensorBoard seems to be running on port {initial_port} (PID: {current_pid_initial}).",
                        className="text-warning",
                    )
                    link_area = [
                        html.A(
                            f"🔗 Open TensorBoard Projector on port {initial_port}",
                            href=f"http://127.0.0.1:{initial_port}/#projector",
                            target="_blank",
                            className="button button-primary",
                        )
                    ]
                    start_button_disabled = True
                    stop_button_disabled = False
                except OSError:
                    tb_state_initial = {"pid": None, "log_dir": None}
                    status = html.P("Previous TensorBoard process found dead. State cleared.", className="text-warning")
                    start_button_disabled = False
                    stop_button_disabled = True

            return (
                summary_text,
                status,
                link_area,
                start_button_disabled,
                stop_button_disabled,
                tb_state_initial,
            )
