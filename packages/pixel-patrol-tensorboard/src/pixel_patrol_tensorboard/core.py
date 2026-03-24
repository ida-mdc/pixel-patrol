import os
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import polars as pl
import polars.selectors as cs
import requests
from PIL import Image
from tensorboardX import SummaryWriter

SPRITE_SIZE = 16


def prepare_embeddings_and_meta(
    df: pl.DataFrame,
) -> Tuple[np.ndarray, pl.DataFrame]:
    """Separate DataFrame into embeddings (numeric cols) and metadata."""
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


def generate_projector_checkpoint(
    embeddings: np.ndarray,
    meta_df: pl.DataFrame,
    log_dir: Path,
) -> None:
    """Write TensorBoard embedding event files to *log_dir*."""
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


def launch_tensorboard_subprocess(logdir: Path, port: int) -> subprocess.Popen | None:
    """Launch TensorBoard and wait until it responds; return Popen or None on failure."""
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
        for _ in range(30):  # up to ~6 s
            try:
                requests.get(f"http://127.0.0.1:{port}", timeout=1)
                return tb_process
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                time.sleep(0.2)
        tb_process.terminate()
        return None
    except (OSError, FileNotFoundError):
        return None
