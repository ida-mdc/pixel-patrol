from pathlib import Path
from typing import Any, Dict, Iterator, List, Set, Tuple

import numpy as np
import pyarrow.parquet as pq

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.record import Record, record_from


class SkyPatchLoader:
    """Reads a tiny ``.parquet`` table as if it were a photo of a patch of sky.

    Each file holds a small grid of ``uint8`` columns - read column-by-column
    and stacked side by side, the table *is* the pixel grid (rows -> Y,
    columns -> X). Real microscopy formats carry instrument metadata
    (channel names, pixel sizes, acquisition stamps, ...) in a header the
    loader extracts; here the same role is played by two plain key/value
    pairs stashed in the parquet schema's metadata - "time_of_day" (when the
    patch was logged: dawn, day, dusk or night) and "cloud_cover" (clear or
    cloudy).
    """

    NAME = "sky-patch"

    SUPPORTED_EXTENSIONS: Set[str] = {"parquet"}
    FOLDER_EXTENSIONS:    Set[str] = set()
    CONTAINER_EXTENSIONS: Set[str] = set()

    OUTPUT_SCHEMA:          Dict[str, Any] = {"time_of_day": str, "cloud_cover": str}
    OUTPUT_SCHEMA_PATTERNS: List[tuple]    = []

    def is_folder_supported(self, path: Path) -> bool:
        return False

    def read_header(self, file_path: Path) -> FileInfo:
        meta = pq.ParquetFile(file_path).metadata
        return FileInfo(shape=(meta.num_rows, meta.num_columns), dtype=np.uint8, dim_order=("Y", "X"))

    def load(self, file_path: Path) -> Record:
        table = pq.read_table(file_path)

        # Each column is one pixel column (X); stacking them rebuilds the YX grid.
        columns = [table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names]
        pixels = np.column_stack(columns).astype(np.uint8)

        raw_meta = table.schema.metadata or {}
        log_entry = {k.decode(): v.decode() for k, v in raw_meta.items()}

        meta = {
            "time_of_day": log_entry.get("time_of_day", "unknown"),
            "cloud_cover": log_entry.get("cloud_cover", "unknown"),
            "dim_order":   "YX",
        }
        # kind="intensity" lets the standard pipeline treat the patch like any
        # other image: thumbnails, histograms and basic metrics all apply.
        return record_from(pixels, meta, kind="intensity")

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        raise NotImplementedError("sky-patch is not a container format")
