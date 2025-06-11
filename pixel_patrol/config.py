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