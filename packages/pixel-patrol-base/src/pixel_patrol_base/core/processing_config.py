"""Configuration for Pixel Patrol processing runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Union

from pixel_patrol_base.config import DEFAULT_ROWS_PER_PART, DEFAULT_MAX_IMAGES_PER_TASK
from pixel_patrol_base.core.project_metadata import ProjectMetadata

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:

    # ── Processor selection ──────────────────────────────────────────────────
    processors_included: Set[str] = field(default_factory=set)
    processors_excluded: Set[str] = field(default_factory=set)

    # ── File selection ───────────────────────────────────────────────────────
    selected_file_extensions: Union[Set[str], str] = "all"

    # ── Dask cluster ─────────────────────────────────────────────────────────
    max_workers: Optional[int] = 5  # TODO: None auto-detection gives unexpected thread/process split; fix in next version

    # ── Task planning ────────────────────────────────────────────────────────
    mb_per_task:          float                   = 512.0
    max_images_per_task:  int                     = DEFAULT_MAX_IMAGES_PER_TASK
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
        if self.max_images_per_task < 1:
            raise ValueError("max_images_per_task must be a positive integer.")
        if self.leaf_block_shape is not None:
            for dim, sz in self.leaf_block_shape.items():
                if sz != -1 and sz < 1:
                    raise ValueError(f"leaf_block_shape['{dim}'] must be -1 or a positive integer.")
        if self.rows_per_part < 1:
            raise ValueError("rows_per_part must be a positive integer.")
