from pathlib import Path
import logging
from typing import Any, Dict, Iterator, List, Set, Tuple, Optional

import rasterio
import numpy as np

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.record import Record, record_from

logger = logging.getLogger(__name__)

class GeospatialLoader:
    """Read geospatial images/data using rasterio."""

    NAME = "geospatial"

    SUPPORTED_EXTENSIONS: Set[str] = {"tif", "tiff"}
    FOLDER_EXTENSIONS:    Set[str] = set()
    CONTAINER_EXTENSIONS: Set[str] = set()

    OUTPUT_SCHEMA:          Dict[str, Any] = {
        "crs_str": Optional[str],
        "crs_epsg": Optional[rasterio.crs.CRS],
        "latitude": Optional[float],
        "longitude": Optional[float],
    }
    OUTPUT_SCHEMA_PATTERNS: List[tuple]    = []

    def is_folder_supported(self, path: Path) -> bool:
        return False

    def read_header(self, file_path: Path) -> FileInfo:
        with rasterio.open(file_path) as ds:
            # ds.shape is just (height, width), but doesn't include the band count
            shape = ds.shape if ds.count == 1 else (ds.count,) + ds.shape

            dim_order = ("Y", "X") if ds.count == 1 else ("C", "Y", "X")
            if len(set(ds.dtypes)) > 1:
                error_msg = f"Different data types in one image is not supported! Consider creating several images per band! Found dtypes {set(ds.dtypes)}."
                raise NotImplementedError(error_msg)
            dtype = set(ds.dtypes).pop()
        logger.info(f"{shape=} {dtype=} {dim_order=}")
        fi = FileInfo(shape=shape, dtype=dtype, dim_order=dim_order)
        logger.info(fi)
        return fi

    def load(self, file_path: Path) -> Record:
        metadata: Dict[str, Any] = {}
        with rasterio.open(file_path) as img:
            if img.crs:
                metadata["crs_epsg"] = img.crs.to_epsg(confidence_threshold=70)
                metadata["crs_str"] = img.crs.to_string()
                lng, lat = img.lnglat()  # Geographic coordinates of the dataset’s center.
                metadata["latitude"] = lat
                metadata["longitude"] = lng
            else:
                metadata["crs_epsg"] = None
                metadata["crs_str"] = None
                metadata["latitude"] = None
                metadata["longitude"] = None

            pixels = img.read()
            pixels = np.nan_to_num(pixels)
        return record_from(pixels, metadata, kind="intensity")

    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        raise NotImplementedError("container format not support for geospatial")
