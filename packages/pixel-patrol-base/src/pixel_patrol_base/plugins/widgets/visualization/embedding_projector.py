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

def _generate_projector_checkpoint(
        embeddings: np.ndarray,
        meta_df: pl.DataFrame,
        log_dir: Path,
):
    """Creates TensorBoard embedding files."""
    writer = SummaryWriter(logdir=str(log_dir))
    # print(f"DEBUG: _generate_projector_checkpoint received meta_df with shape: {meta_df.shape}")

    # Instead of creating a sprite, we create a stack of individual images
    # that tensorboardX will use to create its own sprite.
    images_for_tb = None
    if "thumbnail" in meta_df.columns:
        image_list = meta_df.get_column("thumbnail").to_list()
        processed_images = []

        for img_data in image_list:
            if img_data is None:
                processed_images.append(np.zeros((SPRITE_SIZE, SPRITE_SIZE, 3), dtype=np.uint8))
                continue

            if isinstance(img_data, list):
                img_data = np.array(img_data)

            if isinstance(img_data, Image.Image):
                img = img_data
            elif isinstance(img_data, np.ndarray):
                if img_data.size == 0:
                    processed_images.append(np.zeros((SPRITE_SIZE, SPRITE_SIZE, 3), dtype=np.uint8))
                    continue

                final_img_data = img_data
                if img_data.dtype == np.uint16:
                    # Correctly scale 16-bit data (0-65535) down to 8-bit (0-255).
                    # Integer division by 256 is an efficient way to do this.
                    final_img_data = (img_data // 256).astype(np.uint8)

                    # Some PNG loaders (like matplotlib) might produce floats from 0.0 to 1.0.
                elif img_data.dtype in [np.float32, np.float64]:
                    if img_data.max() <= 1.0:
                        final_img_data = (img_data * 255).astype(np.uint8)

                    # Ensure data is uint8 before creating the image.
                img = Image.fromarray(final_img_data.astype(np.uint8))

            else:
                processed_images.append(np.zeros((SPRITE_SIZE, SPRITE_SIZE, 3), dtype=np.uint8))
                continue
            resized_img_arr = np.array(img.resize((SPRITE_SIZE, SPRITE_SIZE)).convert("RGB"))
            processed_images.append(resized_img_arr)

        if processed_images:
            images_np = np.stack(processed_images)
            images_for_tb = images_np.transpose(0, 3, 1, 2)
            images_for_tb = images_for_tb.astype(float) / 255.

    # TensorBoardX expects pandas metadata; drop thumbnails if present
    metadata_for_tb = meta_df.drop("thumbnail", strict=False).to_pandas()
    sanitized_df = metadata_for_tb.astype(str).replace(r"[\n\r\t]", " ", regex=True)
    metadata = sanitized_df.values.tolist()

    writer.add_embedding(
        mat=embeddings,
        metadata=metadata,
        metadata_header=list(sanitized_df.columns),
        label_img=images_for_tb,
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
    STORE_ID = "tb-process-store-tensorboard-embedding-projector"

    def layout(self) -> List:
        return [
            html.Div(
                id=self.INTRO_ID,
                children=[
                    html.P("The Embedding Projector allows you to explore high-dimensional data by reducing it to 2D or 3D using "),
                    html.Strong("Principal Component Analysis (PCA)"),
                    html.Span(" or "),
                    html.Strong("t-SNE"),
                    html.Span(". "),
                    html.Span(
                        "Embeddings represent data as points in a high-dimensional space; closer points are more similar."
                    ),
                    html.P("This tool helps visualize relationships, clusters, and patterns in large datasets."),
                    html.Div(id=self.SUMMARY_ID),
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

    def prepare_data(self, df: pl.DataFrame):
        """
        Separates a DataFrame into embeddings and metadata based on data types and cardinality.

        Args:
            df: The input Polars DataFrame.
        Returns:
            A tuple containing:
            - np.ndarray: The numeric embedding vectors.
            - pl.DataFrame: The DataFrame containing only metadata columns.
        """
        # Identify columns that should be used for embeddings
        embedding_feature_cols = []
        skipped_cols = []
        for col in df.columns:
            if df[col].dtype.is_float():
                embedding_feature_cols.append(col)
            elif df[col].dtype.is_integer():
                embedding_feature_cols.append(col)
            elif df[col].dtype.is_nested() and col != "thumbnail":
                skipped_cols.append(col)

        if not embedding_feature_cols:
            # Fallback for when no clear embedding columns are found
            df_numeric = df.select(cs.by_dtype(pl.NUMERIC_DTYPES))
            embedding_feature_cols = df_numeric.columns

        # Create the embeddings array and the metadata DataFrame
        embeddings = df.select(embedding_feature_cols).fill_null(0.0).to_numpy()
        metadata_df = df.drop(embedding_feature_cols).drop(skipped_cols)

        return embeddings, metadata_df


    def register(self, app, df_global: pl.DataFrame):
        @app.callback(
            # Since this is the ONLY callback updating these, we remove 'allow_duplicate=True'
            Output(self.SUMMARY_ID, "children"),
            Output(self.STATUS_ID, "children"),
            Output(self.LINK_ID, "children"),
            Output(self.START_BTN_ID, "disabled"),
            Output(self.STOP_BTN_ID, "disabled"),
            Output(self.STORE_ID, "data"),
            # This callback now triggers on button clicks AND on page load (from the store)
            Input(self.START_BTN_ID, "n_clicks"),
            Input(self.STOP_BTN_ID, "n_clicks"),
            Input(self.STORE_ID, "data"),
            State(self.PORT_INPUT_ID, "value"),
            prevent_initial_call=True,
        )
        def manage_tensorboard(
                start_clicks: int,
                stop_clicks: int,
                tb_state: Dict,
                port: int,
        ):
            tb_state = tb_state or {}
            port = port or 6006
            ctx = callback_context
            triggered_id = ctx.triggered_id if ctx.triggered else None

            current_pid = tb_state.get("pid")
            current_log_dir_str = tb_state.get("log_dir")

            # --- Default UI component values ---
            summary_info_text = html.P("")
            status_message = html.Span("TensorBoard not running.", className="text-info")
            projector_link_children: List = []
            start_button_disabled = False
            stop_button_disabled = True

            if triggered_id is None or triggered_id == self.STORE_ID:
                df_numeric = df_global.select(cs.by_dtype(pl.NUMERIC_DTYPES))
                summary_info_text = html.P(
                    f"✅ Found {df_numeric.shape[1]} numeric columns in the {df_global.shape} DataFrame.")
                if current_pid:
                    try:
                        os.kill(current_pid, 0)
                        status_message = html.P(f"TensorBoard is running (PID: {current_pid}).", className="text-info")
                        projector_link_children = [
                            html.A("🔗 Open TensorBoard", href=f"http://127.0.0.1:{port}/#projector", target="_blank")]
                        start_button_disabled = True
                        stop_button_disabled = False
                    except OSError:
                        tb_state = {"pid": None, "log_dir": None}
                        status_message = html.P("Previous TensorBoard process found dead. State cleared.",
                                                className="text-warning")
                return summary_info_text, status_message, projector_link_children, start_button_disabled, stop_button_disabled, tb_state

            summary_info_text = html.P([
                f"Debug Info: Using DataFrame with shape {df_global.shape}. "
            ])

            # -- STOP BUTTON CLICKED --
            if triggered_id == self.STOP_BTN_ID:
                if current_pid:
                    try:
                        os.kill(current_pid, 9)
                        if current_log_dir_str and Path(current_log_dir_str).exists():
                            import shutil
                            shutil.rmtree(current_log_dir_str)
                        status_message = html.P("TensorBoard stopped.", className="text-success")
                    except OSError:
                        pass  # Process was already dead
                tb_state = {"pid": None, "log_dir": None}
                start_button_disabled = False
                stop_button_disabled = True

            # -- START BUTTON CLICKED --
            elif triggered_id == self.START_BTN_ID:
                embeddings_array, df_meta = self.prepare_data(df_global)
                summary_info_text.children.append(f" Processing {embeddings_array.shape[1]} features...")

                new_log_dir = Path(tempfile.mkdtemp(prefix="tb_log_"))
                _generate_projector_checkpoint(embeddings_array, df_meta, new_log_dir)
                tb_process = _launch_tensorboard_subprocess(new_log_dir, port)

                if tb_process:
                    tb_state = {"pid": tb_process.pid, "log_dir": str(new_log_dir), "port": port}
                    status_message = html.P(f"TensorBoard is running on port {port}!", className="text-success")
                    projector_link_children = [
                        html.A("🔗 Open TensorBoard", href=f"http://127.0.0.1:{port}/#projector", target="_blank")]
                    start_button_disabled = True
                    stop_button_disabled = False
                else:
                    status_message = html.P("Failed to start TensorBoard.", className="text-danger")
                    start_button_disabled = False
                    stop_button_disabled = True

            return summary_info_text, status_message, projector_link_children, start_button_disabled, stop_button_disabled, tb_state