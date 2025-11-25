from pathlib import Path
from dataclasses import dataclass, field
from typing import Set, Union, Literal, Optional

from pixel_patrol_base.config import (
    DEFAULT_N_EXAMPLE_FILES,
    DEFAULT_PROCESSING_BATCH_SIZE,
    DEFAULT_RECORDS_FLUSH_EVERY_N,
)


@dataclass
class Settings: # TODO: change default values to not be hard coded
    cmap: str                                                   = "rainbow"
    n_example_files: int                                       = DEFAULT_N_EXAMPLE_FILES
    selected_file_extensions: Union[Set[str], Literal["all"]]   = field(default_factory=set)
    pixel_patrol_flavor: str                                    = "" # use this for indicating specific custom configurations of pixel patrol
    processing_max_workers: Optional[int]                       = None
    processing_batch_size: int                                  = DEFAULT_PROCESSING_BATCH_SIZE
    records_flush_every_n: int                                  = DEFAULT_RECORDS_FLUSH_EVERY_N
    records_flush_dir: Optional[Path]                           = None
