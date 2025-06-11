from dataclasses import dataclass, field
from typing import Set, Union, Literal

@dataclass
class Settings: # TODO: change default values to not be hard coded
    cmap: str                                                   = "rainbow"
    n_example_images: int                                       = 9
    selected_file_extensions: Union[Set[str], Literal["all"]]   = field(default_factory=set)
