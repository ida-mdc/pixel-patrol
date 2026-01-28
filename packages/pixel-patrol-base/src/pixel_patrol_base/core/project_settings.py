from pathlib import Path
from dataclasses import dataclass, field
from typing import Set, Union, Literal, Optional

from pixel_patrol_base.config import (
    DEFAULT_N_EXAMPLE_FILES,
    DEFAULT_RECORDS_FLUSH_EVERY_N,
)


@dataclass
class Settings: # TODO: change default values to not be hard coded
    cmap: str                                                   = "rainbow"
    n_example_files: int                                        = DEFAULT_N_EXAMPLE_FILES
    selected_file_extensions: Union[Set[str], str]              = field(default_factory=set) # "all" or a set of extensions without dot, e.g. `{"tif", "png"}`
    pixel_patrol_flavor: str                                    = "" # use this for indicating specific custom configurations of pixel patrol
    processing_max_workers: Optional[int]                       = None
    records_flush_every_n: int                                  = DEFAULT_RECORDS_FLUSH_EVERY_N # rows kept in-memory before optional disk flush. Will default to half of dataset size if larger than that to ensure at least one flush.
    records_flush_dir: Optional[Path]                           = None
    # If True, skip already-processed images by reusing existing partial chunk files
    # in `records_flush_dir`. If False (default), any existing partial chunk files will
    # be cleared before processing to ensure a fresh run.
    resume: bool                                                = False
