"""Shared output schema for raster-image loaders.

All loaders that read an N-D raster image (BioIO, tifffile, Zarr, LMDB, ...) emit
the same core set of image-metadata columns. Declaring that set in one place
keeps the loaders consistent and lets the schema catalog present it once as a
common "Raster Image Loaders" schema instead of repeating it per loader.

A loader adopts the common schema by spreading these into its class attributes
and adding only its own extras::

    from pixel_patrol_base.core.loader_schema import (
        RASTER_IMAGE_LOADER_SCHEMA, RASTER_IMAGE_LOADER_SCHEMA_PATTERNS,
    )

    OUTPUT_SCHEMA = {**RASTER_IMAGE_LOADER_SCHEMA, "zarr_attributes": dict}
    OUTPUT_SCHEMA_PATTERNS = list(RASTER_IMAGE_LOADER_SCHEMA_PATTERNS)

Each column's type and human description are declared together below (mirroring a
processor's ``RasterMetricSpec``); the schema catalog and the parquet save path
read the descriptions from here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import polars as pl

# Identifier/label used by the catalog to group loaders sharing this schema.
RASTER_IMAGE_SCHEMA_ID = "raster-image"
RASTER_IMAGE_SCHEMA_LABEL = "Raster Image Loaders"

# (column, dtype, description) - the exact columns every raster-image loader emits.
_RASTER_IMAGE_COLUMNS: List[Tuple[str, Any, str]] = [
    ("dim_order",     str,      "Axis order of the image, e.g. 'TCZYX'."),
    ("dim_names",     list,     "Human-readable names of the image axes."),
    ("n_images",      int,      "Number of sub-images in the source (>1 for container formats)."),
    ("num_pixels",    int,      "Total number of pixels in the image."),
    ("shape",         pl.Array, "Size of the image along each axis, in dim_order."),
    ("ndim",          int,      "Number of image dimensions."),
    ("channel_names", list,     "Names of the image channels, if available."),
    ("dtype",         str,      "Pixel data type of the source image (e.g. 'uint8', 'float32')."),
]

# (regex, dtype, description) - per-axis columns whose concrete names depend on the
# image's dimensions.
_RASTER_IMAGE_PATTERNS: List[Tuple[str, Any, str]] = [
    (r"^pixel_size_[A-Za-z]$", float, "Physical pixel size along the given axis, in the image's spatial unit."),
    (r"^[A-Za-z]_size$",       int,   "Extent (number of elements) of this row along the given axis."),
]

RASTER_IMAGE_LOADER_SCHEMA: Dict[str, Any] = {name: dtype for name, dtype, _ in _RASTER_IMAGE_COLUMNS}
RASTER_IMAGE_LOADER_SCHEMA_DESCRIPTIONS: Dict[str, str] = {name: desc for name, _, desc in _RASTER_IMAGE_COLUMNS}

RASTER_IMAGE_LOADER_SCHEMA_PATTERNS: List[Tuple[str, Any]] = [(pat, dtype) for pat, dtype, _ in _RASTER_IMAGE_PATTERNS]
RASTER_IMAGE_LOADER_SCHEMA_PATTERN_DESCRIPTIONS: List[Tuple[str, str]] = [(pat, desc) for pat, _, desc in _RASTER_IMAGE_PATTERNS]
