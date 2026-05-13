"""
Configuration for project metadata and file processing
"""

import logging
from dataclasses import dataclass, field
from typing import Set, Union, Optional

from pixel_patrol_base.config import DEFAULT_RECORDS_FLUSH_EVERY_N
from pixel_patrol_base.core.project_metadata import ProjectMetadata

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:

    processors_included: Set[str] = field(default_factory=set)
    processors_excluded: Set[str] = field(default_factory=set)

    # --- File selection ---
    # "all" or a set of extensions without dot, e.g. {"tif", "png"}
    selected_file_extensions: Union[Set[str], str] = 'all'

    # --- Run behaviour ---
    processing_max_workers: Optional[int] = None
    records_flush_every_n: Optional[int] = None
    parquet_row_group_size: Optional[int] = None  # None → use parquet_io default (2048)

    # --- Project Metadata ---
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)


    def __post_init__(self):

        if self.processors_included and self.processors_excluded:
            logger.warning(
                "ProcessingConfig: Both processors_included and processors_excluded are set. "
                "processors_excluded will be ignored."
            )

        if self.processing_max_workers is not None:
            if not isinstance(self.processing_max_workers, int) or self.processing_max_workers < 1:
                raise ValueError("processing_max_workers must be a positive integer or None.")

        if self.records_flush_every_n is None:
            self.records_flush_every_n = DEFAULT_RECORDS_FLUSH_EVERY_N

        if self.processing_max_workers is not None:
            if not isinstance(self.processing_max_workers, int) or self.processing_max_workers < 1:
                raise ValueError("processing_max_workers must be a positive integer or None.")

        if not isinstance(self.records_flush_every_n, int) or self.records_flush_every_n < 1:
            raise ValueError("records_flush_every_n must be a positive integer.")
