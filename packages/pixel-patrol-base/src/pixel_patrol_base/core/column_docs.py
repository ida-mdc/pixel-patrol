"""Human-readable descriptions for the columns of a final Pixel Patrol report
that are *not* declared by a single loader or processor.

Two groups are covered here:

- ``BASE_COLUMN_DESCRIPTIONS`` - file-system columns (from ``core.file_system``)
  and pipeline-generated columns (from ``core.processing``: obs rows, sizes, ...).
- ``PATTERN_COLUMN_DESCRIPTIONS`` - regex-keyed descriptions for pipeline column
  families whose concrete names depend on the data (per-dimension coordinates).

The shared raster-image loader columns are described where their types are
declared, in :mod:`pixel_patrol_base.core.loader_schema`.
:func:`base_column_description` also resolves those, so any report column not
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
    "path":                "Absolute path of the file (or folder) on disk.",
    "name":                "File or folder name, including extension.",
    "type":                "Row kind: 'file', 'folder', or 'sub_file' (a sub-image inside a container).",
    "parent":              "Path of the containing directory.",
    "depth":               "Depth of the path below the imported base directory.",
    "size_bytes":          "Size on disk in bytes (aggregated for folders).",
    "size_readable":       "Human-readable size on disk (e.g. '2.4 MB').",
    "file_extension":      "Lower-cased file extension without the leading dot.",
    "modification_date":   "Last modification timestamp of the file.",
    "imported_path":       "The base directory under which this file was imported.",
    "imported_path_short": "Shortened, disambiguated label for the imported base directory (only when several bases were imported).",
    "common_base":         "Longest path prefix shared by all imported files.",
    "child_id":            "Stable identifier of a sub-image within a container file (null for plain files).",
    # pipeline-generated obs columns
    "obs_level":           "Aggregation level of the row: 0 is the whole-image summary; higher levels are per-dimension breakdowns.",
    "num_pixels":          "Number of pixels in this row's spatial extent (full image at obs_level=0, slice at higher levels).",
    "ndim":                "Number of image dimensions, derived from dim_order.",
}

# Regex-keyed descriptions for pipeline column families whose exact names depend on
# the data. (Per-axis size / pixel-size families are described with the loader
# schema in loader_schema.py.)
PATTERN_COLUMN_DESCRIPTIONS: List[Tuple[str, str]] = [
    (r"^dim_[A-Za-z]+$", "Coordinate of this row along the given axis (e.g. dim_z = Z index); null when the row spans the whole axis."),
]


def base_column_description(name: str) -> Optional[str]:
    """Return a description for any report column not owned by a specific plugin
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
