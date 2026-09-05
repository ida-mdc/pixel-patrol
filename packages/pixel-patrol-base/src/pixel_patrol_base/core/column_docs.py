"""Human-readable descriptions for the columns of a PixelPatrol table
that are *not* declared by a single loader or processor.

Two groups are covered here:

- ``BASE_COLUMN_DESCRIPTIONS`` - file-system columns (from ``core.file_system``)
  and pipeline-generated columns (from ``core.processing``: obs rows, sizes, ...).
- ``PATTERN_COLUMN_DESCRIPTIONS`` - regex-keyed descriptions for pipeline column
  families whose concrete names depend on the data (directory levels,
  per-dimension coordinates).

The shared raster-image loader columns are described where their types are
declared, in :mod:`pixel_patrol_base.core.loader_schema`.
:func:`base_column_description` also resolves those, so any table column not
owned by a specific plugin instance can be looked up by name here.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from pixel_patrol_base.core.loader_schema import (
    RASTER_IMAGE_LOADER_SCHEMA_DESCRIPTIONS,
    RASTER_IMAGE_LOADER_SCHEMA_PATTERN_DESCRIPTIONS,
)

# File-system columns produced by core.file_system, plus the pipeline-generated
# columns assembled in core.processing (obs rollup, per-row sizes, ...).
BASE_COLUMN_DESCRIPTIONS: Dict[str, str] = {
    # file-system / import columns
    "path":                "Path of the file (or folder), relative to the project base directory.",
    "name":                "File or folder name, including extension.",
    "type":                "Row kind: 'file', 'folder', or 'sub_file' (a sub-image inside a container).",
    "depth":               "Depth of the path below the imported base directory.",
    "size_bytes":          "Size on disk in bytes (aggregated for folders).",
    "file_extension":      "Lower-cased file extension without the leading dot.",
    "modification_date":   "Last modification timestamp of the file.",
    "imported_path":       "Path of the import base (the -p path or base directory) under which this file was found, relative to the project base directory ('.' when no -p paths were specified).",
    "imported_path_short": "Shortened, disambiguated label for the import base (only when several -p paths were used).",
    "common_base":         "Name of the longest path prefix shared by all import bases.",
    # pipeline-generated obs columns
    "obs_level":           "Aggregation level of the row: 0 is the whole-image summary; higher levels are per-dimension breakdowns.",
    "num_pixels":          "Number of pixels in this row's spatial extent (full image at obs_level=0, slice at higher levels).",
    "ndim":                "Number of image dimensions, derived from dim_order.",
}

BASE_COLUMN_DTYPES: Dict[str, str] = {
    "path":                "str",
    "name":                "str",
    "type":                "str",
    "depth":               "int",
    "size_bytes":          "int",
    "file_extension":      "str",
    "modification_date":   "datetime",
    "imported_path":       "str",
    "imported_path_short": "str",
    "common_base":         "str",
    "obs_level":           "int",
}

# Regex-keyed descriptions for pipeline column families whose exact names depend on
# the data. (Per-axis size / pixel-size families are described with the loader
# schema in loader_schema.py.)
PATTERN_COLUMN_DESCRIPTIONS: List[Tuple[str, str]] = [
    (r"^parent\d+$", "Name of one directory level of the containing directory, counted from the top: parent0 is the first directory below the base directory, parent1 the one below that; null when the file is not that deep."),
    (r"^dim_[A-Za-z]+$", "Coordinate of this row along the given axis (e.g. dim_z = Z index); null when the row spans the whole axis."),
    (r"^size_[A-Za-z]$", "Extent (number of elements) of this row along the given axis."),
]


def base_column_description(name: str) -> Optional[str]:
    """Return a description for any table column not owned by a specific plugin
    instance: base/pipeline columns and the shared raster-image loader columns.

    Exact matches take priority over pattern matches.
    """
    if name in BASE_COLUMN_DESCRIPTIONS:
        return BASE_COLUMN_DESCRIPTIONS[name]
    if name in RASTER_IMAGE_LOADER_SCHEMA_DESCRIPTIONS:
        return RASTER_IMAGE_LOADER_SCHEMA_DESCRIPTIONS[name]
    for pattern, desc in PATTERN_COLUMN_DESCRIPTIONS + RASTER_IMAGE_LOADER_SCHEMA_PATTERN_DESCRIPTIONS:
        if re.match(pattern, name):
            return desc
    return None
