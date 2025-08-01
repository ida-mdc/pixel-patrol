from dataclasses import dataclass, field
from typing import Set, Union, Literal
from pixel_patrol.config import DEFAULT_N_EXAMPLE_IMAGES

@dataclass
class Settings: # TODO: change default values to not be hard coded
    cmap: str                                                   = "rainbow"
    n_example_images: int                                       = DEFAULT_N_EXAMPLE_IMAGES
    selected_file_extensions: Union[Set[str], Literal["all"]]   = field(default_factory=set)
