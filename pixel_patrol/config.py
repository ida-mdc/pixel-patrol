from typing import Set

DEFAULT_PRESELECTED_FILE_EXTENSIONS: Set[str] = {
    # Bioimage Common Types
    "zarr", "czi", "tif", "tiff", "nd2", "lsm", "hdf5",
    # Environmental Science
    "netcdf", "shapefile", "geojson",
    # Medical Images
    "dicom", "nii", "dcm",
    # Additional Common Image Formats
    "jpg", "jpeg", "png", "bmp", "gif"
}

FOLDER_EXTENSIONS_AS_FILES = {
    ".zarr", ".ome.zarr", ".n5", ".imaris", ".napari", ".nd2folder"
}

MIN_N_EXAMPLE_IMAGES: int = 1
MAX_N_EXAMPLE_IMAGES: int = 20
DEFAULT_N_EXAMPLE_IMAGES: int = 9