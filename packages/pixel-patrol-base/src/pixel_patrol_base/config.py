from typing import Set

DEFAULT_PRESELECTED_FILE_EXTENSIONS: Set[str] = {
    # Bioimage Common Types
    "czi", "tif", "tiff", "nd2", "lif",
    # Additional Common Image Formats
    "jpg", "jpeg", "png", "bmp", # "gif" TODO: gif support is not implemented yet - produces errors.
    "zarr", "json"
}

SPRITE_SIZE = 64

STANDARD_DIM_ORDER = "TCZYXS"

RGB_WEIGHTS = [0.2989, 0.5870, 0.1140]
MIN_N_EXAMPLE_FILES: int = 1
MAX_N_EXAMPLE_FILES: int = 20
DEFAULT_N_EXAMPLE_FILES: int = 9
