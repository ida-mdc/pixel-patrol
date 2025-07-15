from typing import Set

DEFAULT_PRESELECTED_FILE_EXTENSIONS: Set[str] = {
    # Bioimage Common Types
    "zarr", "czi", "tif", "tiff", "nd2", "lsm", "hdf5", "lif",
    # Environmental Science
    "netcdf", "shapefile", "geojson",
    # Medical Images
    "dicom", "nii", "dcm",
    # Additional Common Image Formats
    "jpg", "jpeg", "png", "bmp", # "gif" TODO: gif support is not implemented yet - produces errors.
}

FOLDER_EXTENSIONS_AS_FILES = {
    ".zarr", ".ome.zarr", ".n5", ".imaris", ".napari", ".nd2folder"
}

MIN_N_EXAMPLE_IMAGES: int = 1
MAX_N_EXAMPLE_IMAGES: int = 20
DEFAULT_N_EXAMPLE_IMAGES: int = 9

STANDARD_DIM_ORDER = "TCZYXS"
