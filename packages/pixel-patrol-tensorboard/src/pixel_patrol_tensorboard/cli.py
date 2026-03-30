import shutil
import signal
import sys
import tempfile
from pathlib import Path

import click
import polars as pl

from pixel_patrol_tensorboard.core import (
    prepare_embeddings_and_meta,
    generate_projector_checkpoint,
    launch_tensorboard_subprocess,
)


@click.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--port", default=6006, show_default=True, help="TensorBoard port.")
def cli(source: str, port: int) -> None:
    """Launch TensorBoard Embedding Projector from a Pixel Patrol data file.

    SOURCE can be:

    \b
      - A Parquet file (.parquet)
      - An Arrow/IPC file (.arrow or .ipc)
      - A Pixel Patrol project export (.zip) — requires pixel-patrol-base
    """
    source_path = Path(source)

    click.echo(f"Loading data from {source_path} ...")

    if source_path.suffix == ".zip":
        try:
            from pixel_patrol_base.api import import_project
        except ImportError:
            raise click.ClickException(
                "Loading .zip projects requires pixel-patrol-base to be installed."
            )
        project = import_project(source_path)
        df = project.records_df
    elif source_path.suffix == ".parquet":
        df = pl.read_parquet(source_path)
    elif source_path.suffix in (".arrow", ".ipc"):
        df = pl.read_ipc(source_path)
    else:
        raise click.ClickException(
            f"Unsupported file type '{source_path.suffix}'. "
            "Use .parquet, .arrow/.ipc, or a .zip project export."
        )

    if df is None or df.is_empty():
        raise click.ClickException("No data found in the source file.")

    embeddings, meta_df = prepare_embeddings_and_meta(df)

    if embeddings.size == 0:
        raise click.ClickException(
            "No numeric columns found to build embeddings. "
            "Make sure the data has at least one numeric column."
        )

    n_cols = embeddings.shape[1] if embeddings.ndim > 1 else 1
    click.echo(
        f"Building embeddings from {n_cols} numeric column(s) "
        f"across {embeddings.shape[0]} rows."
    )

    log_dir = Path(tempfile.mkdtemp(prefix="tb_log_"))
    click.echo(f"Writing TensorBoard checkpoint to {log_dir} ...")
    generate_projector_checkpoint(embeddings, meta_df, log_dir)

    click.echo(f"Starting TensorBoard on port {port} ...")
    process = launch_tensorboard_subprocess(log_dir, port)

    if process is None:
        shutil.rmtree(log_dir, ignore_errors=True)
        raise click.ClickException(
            f"Failed to start TensorBoard on port {port}. "
            "Is TensorBoard installed and is the port free?"
        )

    url = f"http://127.0.0.1:{port}/#projector"
    click.echo(f"TensorBoard is running at {url}")
    click.echo("Press Ctrl+C to stop.")

    def _shutdown(signum, frame):
        click.echo("\nStopping TensorBoard ...")
        process.terminate()
        shutil.rmtree(log_dir, ignore_errors=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    process.wait()
    shutil.rmtree(log_dir, ignore_errors=True)
