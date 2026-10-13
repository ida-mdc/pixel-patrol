"""The user-facing args of `pixel-patrol process`: single source for the CLI, launcher and docs.

Order here is the order shown everywhere: basics first, advanced after.
"""

from dataclasses import dataclass
from pathlib import Path

import click

from pixel_patrol_base.config import (
    DEFAULT_MAX_IMAGES_PER_TASK,
    DEFAULT_MB_PER_WORKER,
    DEFAULT_PARQUET_ROW_GROUP_SIZE,
    DEFAULT_ROWS_PER_PART,
)


@dataclass(frozen=True)
class ProcessArg:
    name: str                     # python name: CLI parameter, launcher field, API argument
    help: str
    flags: tuple = ()             # default: --<name with dashes>
    metavar: str | None = None
    type: object = str
    multiple: bool = False
    flag: bool = False
    argument: bool = False        # given as a bare value, without a flag name
    required: bool = False
    default: object = None        # default shown in the launcher; the code resolves the real one

    @property
    def cli_flags(self) -> tuple:
        return self.flags or ("--" + self.name.replace("_", "-"),)

    @property
    def full_help(self) -> str:
        return self.help + (" Repeatable." if self.multiple else "")


PROCESS_ARGS = [
    # ── Basics ──
    ProcessArg("base_directory", argument=True, required=True,
               type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, path_type=Path),
               help="Root folder containing your dataset."),
    ProcessArg("output", flags=("--output", "-o"), metavar="PATH", required=True,
               type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True, path_type=Path),
               help="Path for the output parquet file (e.g., my_project.parquet)."),
    ProcessArg("loader", flags=("--loader", "-l"), metavar="NAME", help=(
        "Loader plugin to use (e.g. bioio, zarr, tifffile). The loader opens image files and extracts "
        "image metadata. Without a loader, only basic file info is collected (name, size, extension).")),
    ProcessArg("paths", flags=("--paths", "-p"), metavar="PATH", multiple=True, help=(
        "Subdirectory to process, relative to BASE_DIRECTORY. If specified, only files within those paths "
        "are processed and they become the default grouping when the interactive report opens. "
        "If omitted, all supported files under BASE_DIRECTORY are processed.")),
    ProcessArg("file_extensions", flags=("--file-extensions", "-e"), metavar="EXT", multiple=True, help=(
        "File extension to include (e.g. tif, nd2). "
        "Defaults to all extensions supported by the loader (every file if no loader is set).")),
    ProcessArg("name", metavar="TEXT",
               help="Name of the project. If not provided, derived from BASE_DIRECTORY."),
    ProcessArg("flavor", metavar="TEXT",
               help="Label shown next to the PixelPatrol title in the viewer."),
    ProcessArg("description", metavar="TEXT", help=(
        "Free-form description of the data or the project, shown below the title in the viewer and embedded in the parquet metadata.")),

    # ── What gets measured ──
    ProcessArg("processors_include", metavar="NAME", multiple=True, help=(
        "Run only these processors by ID (e.g. raster-basic, thumbnail). If set, the exclude list is ignored.")),
    ProcessArg("processors_exclude", metavar="NAME", multiple=True,
               help="Skip these processors by ID."),
    ProcessArg("slice_size", metavar="DIM=SIZE", multiple=True, help=(
        "Per-dimension granularity of statistics in the output table. Z=1 produces one set of statistics "
        "per Z slice; Z=5 groups every 5 slices. By default X and Y are full extent and all other dims "
        "step by 1. Use -1 for no slicing along that dimension. At least one dimension must have a slice size larger than 1.")),

    # ── Speed and memory ──
    ProcessArg("max_workers", metavar="N", type=int, help=(
        "Maximum number of parallel Dask workers (default: as many as fit in free RAM, at most the CPU count). "
        "Use 1 to disable parallelism. Ignored when connecting to an existing Dask cluster.")),
    ProcessArg("mb_per_worker", metavar="N", type=float, default=DEFAULT_MB_PER_WORKER, help=(
        f"RAM per worker in MB (default: {DEFAULT_MB_PER_WORKER:g}). More RAM per worker means fewer parallel "
        "workers. Images larger than 1/8 of this (headroom for in-memory expansion) are split into chunks; "
        "smaller images are batched up to that size. "
        "With an existing Dask cluster, defaults to the smallest worker's memory.")),
    ProcessArg("max_images_per_task", metavar="N", type=int, default=DEFAULT_MAX_IMAGES_PER_TASK, help=(
        f"Max images per batch task or sub-images per container task (default: {DEFAULT_MAX_IMAGES_PER_TASK}).")),
    ProcessArg("scheduler", metavar="URL", help=(
        "Connect to an existing Dask scheduler instead of spawning a local one (e.g. tcp://host:8786). "
        "Useful for HPC clusters.")),

    # ── Output tuning ──
    ProcessArg("rows_per_part", metavar="N", type=int, default=DEFAULT_ROWS_PER_PART, help=(
        "Number of rows buffered in memory before being flushed to a temporary file on disk "
        f"(default: {DEFAULT_ROWS_PER_PART}).")),
    ProcessArg("parquet_row_group_size", metavar="N", type=int, default=DEFAULT_PARQUET_ROW_GROUP_SIZE, help=(
        f"Rows per row group in the final parquet (default: {DEFAULT_PARQUET_ROW_GROUP_SIZE}). "
        "Smaller values speed up thumbnail sampling in the viewer.")),

    # ── Misc ──
    ProcessArg("log_file", flag=True,
               help="Write a debug log file alongside the output parquet (auto-named)."),
    ProcessArg("omit_base_dir", flag=True, help=(
        "Do not store the base directory in the parquet metadata. "
        "Useful when sharing tables without revealing local filesystem paths.")),
    ProcessArg("open_viewer", flags=("--view",), flag=True, help=(
        "Open the viewer immediately after processing completes, using default viewer settings. "
        "Equivalent to running `pixel-patrol view OUTPUT` afterwards.")),
]
