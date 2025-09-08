from dataclasses import dataclass, field
from typing import Set, Union, Literal

from pixel_patrol_base.config import DEFAULT_N_EXAMPLE_FILES


@dataclass
class Settings: # TODO: change default values to not be hard coded
    cmap: str                                                   = "rainbow"
    n_example_files: int                                       = DEFAULT_N_EXAMPLE_FILES
    selected_file_extensions: Union[Set[str], Literal["all"]]   = field(default_factory=set)
    pixel_patrol_flavor: str                                    = "" # use this for indicating specific custom configurations of pixel patrol
