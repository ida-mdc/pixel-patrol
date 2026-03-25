"""
Configuration for project metadata and file processing
"""

import logging
from dataclasses import dataclass, field
from typing import Set, Union, Optional
from pathlib import Path

from pixel_patrol_base.config import DEFAULT_RECORDS_FLUSH_EVERY_N

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for file processing operations."""
    # Slicing configuration: If False, slicing is disabled entirely (only full-image stats computed)
    slicing_enabled: bool = True
    # Dimensions to include in slicing (e.g., {"T", "C"}). If non-empty, only these dimensions are sliced.
    # If empty, slicing_dimensions_excluded is used instead.
    slicing_dimensions_included: Set[str] = field(default_factory=set)
    # Dimensions to exclude from slicing (e.g., {"X", "Y", "Z"}). Default: {"X", "Y"} are never sliced.
    # Only used if slicing_dimensions_included is empty.
    slicing_dimensions_excluded: Set[str] = field(default_factory=lambda: {"X", "Y"})
    processors_included: Set[str] = field(default_factory=set)
    processors_excluded: Set[str] = field(default_factory=set)

    # --- File selection ---
    # "all" or a set of extensions without dot, e.g. {"tif", "png"}
    selected_file_extensions: Union[Set[str], str] = 'all'

    # --- Run behaviour ---
    pixel_patrol_flavor: str = ""
    processing_max_workers: Optional[int] = None
    records_flush_every_n: int = DEFAULT_RECORDS_FLUSH_EVERY_N
    records_flush_dir: Optional[Path] = None


    def __post_init__(self):
        if not isinstance(self.slicing_enabled, bool):
            raise TypeError("slicing_enabled must be a bool.")

        if self.slicing_dimensions_included and self.slicing_dimensions_excluded != {"X", "Y"}:
            logger.warning(
                "ProcessingConfig: Both slicing_dimensions_included and slicing_dimensions_excluded "
                "are set. slicing_dimensions_excluded will be ignored."
            )

        if self.processors_included and self.processors_excluded:
            logger.warning(
                "ProcessingConfig: Both processors_included and processors_excluded are set. "
                "processors_excluded will be ignored."
            )

        if self.processing_max_workers is not None:
            if not isinstance(self.processing_max_workers, int) or self.processing_max_workers < 1:
                raise ValueError("processing_max_workers must be a positive integer or None.")

        if not isinstance(self.records_flush_every_n, int) or self.records_flush_every_n < 1:
            raise ValueError("records_flush_every_n must be a positive integer.")
