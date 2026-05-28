"""Configuration for Pixel Patrol processing runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Union

from pixel_patrol_base.config import DEFAULT_ROWS_PER_PART, DEFAULT_MAX_FILES_PER_TASK
from pixel_patrol_base.core.project_metadata import ProjectMetadata

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """All parameters that govern a pipeline run.

    Dask-specific fields vs. the original multiprocessing config:
      mb_per_task       — on-disk MB budget per batch task; also the threshold above
                          which a file is split into spatial memory-chunk tasks.
      leaf_block_shape  — per-dim leaf block sizes used by leaf processors.
                          Keys are dim names (e.g. "X", "Y", "Z", "C").
                          Value -1 means "never split this dim" (full extent always).
                          Positive N means "process in blocks of N along this dim".
                          None → default: X=-1, Y=-1, all other dims=1.

    Renamed vs. the old ProcessingConfig:
      processing_max_workers → max_workers   (prefix was redundant)
      records_flush_every_n  → rows_per_part (explicit about what it controls)
    """

    # ── Processor selection ──────────────────────────────────────────────────
    processors_included: Set[str] = field(default_factory=set)
    processors_excluded: Set[str] = field(default_factory=set)

    # ── File selection ───────────────────────────────────────────────────────
    selected_file_extensions: Union[Set[str], str] = "all"

    # ── Dask cluster ─────────────────────────────────────────────────────────
    max_workers: Optional[int] = None  # None → LocalCluster auto-detects CPU count

    # ── Task planning ────────────────────────────────────────────────────────
    mb_per_task:          float                   = 512.0
    max_files_per_task:   int                     = DEFAULT_MAX_FILES_PER_TASK
    leaf_block_shape:     Optional[Dict[str, int]] = None

    # ── Output ───────────────────────────────────────────────────────────────
    rows_per_part:        int            = DEFAULT_ROWS_PER_PART
    parquet_row_group_size: Optional[int] = None  # None → parquet_io default (2048)

    # ── Project metadata ─────────────────────────────────────────────────────
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)

    def __post_init__(self) -> None:
        if self.processors_included and self.processors_excluded:
            logger.warning(
                "ProcessingConfig: both processors_included and processors_excluded are set; "
                "processors_excluded will be ignored."
            )
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be a positive integer or None.")
        if self.mb_per_task <= 0:
            raise ValueError("mb_per_task must be positive.")
        if self.max_files_per_task < 1:
            raise ValueError("max_files_per_task must be a positive integer.")
        if self.leaf_block_shape is not None:
            for dim, sz in self.leaf_block_shape.items():
                if sz != -1 and sz < 1:
                    raise ValueError(f"leaf_block_shape['{dim}'] must be -1 or a positive integer.")
        if self.rows_per_part < 1:
            raise ValueError("rows_per_part must be a positive integer.")
