import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple

import numpy as np
import polars as pl
import polars.selectors as cs
import requests
from PIL import Image
from dash import html, dcc, Input, Output, State, callback_context
from tensorboardX import SummaryWriter

from pixel_patrol_base.report.widget_categories import WidgetCategories
from pixel_patrol_base.report.base_widget import BaseReportWidget
from pixel_patrol_base.report.factory import show_no_data_message
from pixel_patrol_base.report.global_controls import (
    prepare_widget_data,
    GLOBAL_CONFIG_STORE_ID,
    FILTERED_INDICES_STORE_ID,
)

SPRITE_SIZE = 16

class EmbeddingProjectorWidget(BaseReportWidget):
    NAME: str = "TensorBoard Embedding Projector"
    TAB: str = WidgetCategories.VISUALIZATION.value
    REQUIRES: Set[str] = set()
    REQUIRES_PATTERNS = None

    CONTAINER_ID = "embedding-projector-container"
    SUMMARY_ID = "projector-summary-info"
    STATUS_ID = "projector-status"
    LINK_ID = "projector-link-area"
    PORT_INPUT_ID = "tb-port-input"
    START_BTN_ID = "start-tb-button"
    STOP_BTN_ID = "stop-tb-button"
    STORE_ID = "tb-process-store-tensorboard-embedding-projector"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._df: pl.DataFrame | None = None

    @property
    def help_text(self) -> str:
        return (
            "Launches the TensorBoard **Embedding Projector** for interactive exploration "
            "of high-dimensional data.\n\n"
            "- Embeddings are built from numeric columns of the (filtered) table.\n"
            "- TensorBoard can reduce them to 2D/3D using PCA, t-SNE, UMAP, etc.\n"
            "- Nearby points represent similar entries; clusters may reveal structure."
        )

    def get_content_layout(self) -> List:
        return [
            html.Div([
                html.Div(id=self.SUMMARY_ID),
                html.Div(
                    [
                        html.Label("TensorBoard Port:"),
                        dcc.Input(
                            id=self.PORT_INPUT_ID,
                            type="number",
                            value=6006,
                            min=1024,
                            max=65535,
                            style={"marginLeft": "10px", "width": "100px"},
                        ),
                        html.Button(
                            "🚀 Start TensorBoard",
                            id=self.START_BTN_ID,
                            n_clicks=0,
                            style={"marginLeft": "20px", "marginRight": "10px"},
                        ),
                        html.Button("🛑 Stop TensorBoard", id=self.STOP_BTN_ID, n_clicks=0),
                    ],
                    style={"marginTop": "20px"},
                ),
                html.Div(id=self.STATUS_ID, style={"marginTop": "10px"}),
                html.Div(id=self.LINK_ID, style={"marginTop": "10px"}),
            ], id=self.CONTAINER_ID)
        ]

    def register(self, app, df: pl.DataFrame) -> None:
        self._df = df

        app.callback(
            Output(self.SUMMARY_ID, "children"),
            Output(self.STATUS_ID, "children"),
            Output(self.LINK_ID, "children"),
            Output(self.START_BTN_ID, "disabled"),
            Output(self.STOP_BTN_ID, "disabled"),
            Output(self.STORE_ID, "data"),
            Input(self.START_BTN_ID, "n_clicks"),
            Input(self.STOP_BTN_ID, "n_clicks"),
            Input(self.STORE_ID, "data"),
            Input(FILTERED_INDICES_STORE_ID, "data"),
            Input(GLOBAL_CONFIG_STORE_ID, "data"),
            State(self.PORT_INPUT_ID, "value"),
            prevent_initial_call=True,
        )(self._manage_tensorboard)


    @staticmethod
    def _prepare_embeddings_and_meta(
        df: pl.DataFrame,
    ) -> Tuple[np.ndarray, pl.DataFrame]:
        """Separate DataFrame into embeddings and metadata."""
        embedding_feature_cols: List[str] = []
        skipped_cols: List[str] = []

        for col in df.columns:
            dtype = df[col].dtype
            if dtype.is_float() or dtype.is_integer():
                embedding_feature_cols.append(col)
            elif dtype.is_nested() and col != "thumbnail":
                skipped_cols.append(col)

        if not embedding_feature_cols:
            df_numeric = df.select(cs.by_dtype(pl.NUMERIC_DTYPES))
            embedding_feature_cols = df_numeric.columns

        embeddings = df.select(embedding_feature_cols).fill_null(0.0).to_numpy()
        metadata_df = df.drop(embedding_feature_cols).drop(skipped_cols)

        return embeddings, metadata_df

    @staticmethod
    def _summarize_numeric_columns(df: pl.DataFrame) -> html.P:
        df_numeric = df.select(cs.by_dtype(pl.NUMERIC_DTYPES))
        n_numeric = df_numeric.width
        return html.P(
            f"Numeric columns available for embeddings: {n_numeric} "
            f"(from {df.height} rows, {df.width} columns)."
        )

    @staticmethod
    def _kill_process_and_cleanup(pid: int | None, log_dir_str: str | None) -> None:
        if not pid:
            return
        try:
            os.kill(pid, 9)
        except OSError:
            pass
        if log_dir_str and Path(log_dir_str).exists():
            shutil.rmtree(log_dir_str, ignore_errors=True)

    def _manage_tensorboard(
        self,
        _start_clicks: int,
        _stop_clicks: int,
        tb_state: Dict | None,
        subset_indices: List[int] | None,
        global_config: Dict | None,
        port: int | None,
    ):

        tb_state = tb_state or {}
        port = port or 6006

        ctx = callback_context
        triggered_id = ctx.triggered_id if ctx.triggered else None

        df_filtered, _group_col, _resolved, _warning_msg, _order = prepare_widget_data(
            self._df,
            subset_indices,
            global_config or {},
            metric_base=None,
        )

        show_no_data_message()

        summary_info = self._summarize_numeric_columns(df_filtered)

        current_pid = tb_state.get("pid")
        current_log_dir_str = tb_state.get("log_dir")

        status_message = html.Span("TensorBoard not running.", className="text-info")
        link_children: List = []
        start_disabled = False
        stop_disabled = True

        if triggered_id is None or triggered_id == self.STORE_ID:
            if current_pid:
                try:
                    os.kill(current_pid, 0)
                    status_message = html.P(
                        f"TensorBoard is running (PID: {current_pid}).",
                        className="text-info",
                    )
                    link_children = [
                        html.A(
                            "🔗 Open TensorBoard",
                            href=f"http://127.0.0.1:{port}/#projector",
                            target="_blank",
                        )
                    ]
                    start_disabled = True
                    stop_disabled = False
                except OSError:
                    tb_state = {"pid": None, "log_dir": None, "port": None}
                    status_message = html.P(
                        "Previous TensorBoard process not found. State cleared.",
                        className="text-warning",
                    )

            return (
                summary_info,
                status_message,
                link_children,
                start_disabled,
                stop_disabled,
                tb_state,
            )

        if triggered_id == self.STOP_BTN_ID:
            self._kill_process_and_cleanup(current_pid, current_log_dir_str)
            tb_state = {"pid": None, "log_dir": None, "port": None}
            status_message = html.P(
                "TensorBoard stopped.", className="text-success"
            )
            start_disabled = False
            stop_disabled = True

        elif triggered_id == self.START_BTN_ID:
            embeddings_array, df_meta = self._prepare_embeddings_and_meta(df_filtered)

            if embeddings_array.size == 0:
                status_message = html.P(
                    "No numeric columns found to build embeddings "
                    "(after applying filters).",
                    className="text-warning",
                )
                tb_state = {"pid": None, "log_dir": None, "port": None}
                start_disabled = False
                stop_disabled = True
            else:
                new_log_dir = Path(tempfile.mkdtemp(prefix="tb_log_"))
                _generate_projector_checkpoint(embeddings_array, df_meta, new_log_dir)
                tb_process = _launch_tensorboard_subprocess(new_log_dir, port)

                if tb_process:
                    tb_state = {
                        "pid": tb_process.pid,
                        "log_dir": str(new_log_dir),
                        "port": port,
                    }
                    status_message = html.P(
                        f"TensorBoard is running on port {port}.",
                        className="text-success",
                    )
                    link_children = [
                        html.A(
                            "🔗 Open TensorBoard",
                            href=f"http://127.0.0.1:{port}/#projector",
                            target="_blank",
                        )
                    ]
                    start_disabled = True
                    stop_disabled = False
                else:
                    status_message = html.P(
                        "Failed to start TensorBoard.", className="text-danger"
                    )
                    self._kill_process_and_cleanup(None, str(new_log_dir))
                    tb_state = {"pid": None, "log_dir": None, "port": None}
                    start_disabled = False
                    stop_disabled = True

        return (
            summary_info,
            status_message,
            link_children,
            start_disabled,
            stop_disabled,
            tb_state,
        )

def _generate_projector_checkpoint(
    embeddings: np.ndarray,
    meta_df: pl.DataFrame,
    log_dir: Path,
) -> None:
    """Create TensorBoard embedding files."""
    writer = SummaryWriter(logdir=str(log_dir))

    images_for_tb = None
    if "thumbnail" in meta_df.columns:
        image_list = meta_df.get_column("thumbnail").to_list()
        processed_images = []

        for img_data in image_list:
            if img_data is None:
                processed_images.append(
                    np.zeros((SPRITE_SIZE, SPRITE_SIZE, 3), dtype=np.uint8)
                )
                continue

            if isinstance(img_data, list):
                img_data = np.array(img_data)

            if isinstance(img_data, Image.Image):
                img = img_data
            elif isinstance(img_data, np.ndarray):
                if img_data.size == 0:
                    processed_images.append(
                        np.zeros((SPRITE_SIZE, SPRITE_SIZE, 3), dtype=np.uint8)
                    )
                    continue

                final_img_data = img_data
                if img_data.dtype == np.uint16:
                    final_img_data = (img_data // 256).astype(np.uint8)
                elif img_data.dtype in (np.float32, np.float64):
                    if img_data.max() <= 1.0:
                        final_img_data = (img_data * 255).astype(np.uint8)

                img = Image.fromarray(final_img_data.astype(np.uint8))
            else:
                processed_images.append(
                    np.zeros((SPRITE_SIZE, SPRITE_SIZE, 3), dtype=np.uint8)
                )
                continue

            resized_img_arr = np.array(
                img.resize((SPRITE_SIZE, SPRITE_SIZE)).convert("RGB")
            )
            processed_images.append(resized_img_arr)

        if processed_images:
            images_np = np.stack(processed_images)
            images_for_tb = images_np.transpose(0, 3, 1, 2)
            images_for_tb = images_for_tb.astype(float) / 255.0

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


def _launch_tensorboard_subprocess(logdir: Path, port: int) -> subprocess.Popen | None:
    """Launch TensorBoard and wait briefly until it responds; return Popen or None."""
    logdir.mkdir(parents=True, exist_ok=True)
    cmd = ["tensorboard", f"--logdir={logdir}", f"--port={port}", "--bind_all"]
    env = os.environ.copy()
    env["GCS_READ_CACHE_MAX_SIZE_MB"] = "0"

    try:
        tb_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        for _ in range(30):  # up to ~6s
            try:
                requests.get(f"http://127.0.0.1:{port}", timeout=1)
                return tb_process
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                time.sleep(0.2)
        tb_process.terminate()
        return None
    except (OSError, FileNotFoundError):
        return None