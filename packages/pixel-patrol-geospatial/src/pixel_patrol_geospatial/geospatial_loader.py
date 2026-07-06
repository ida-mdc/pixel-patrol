from pathlib import Path
import logging
from typing import Any, Dict, Iterator, List, Set, Tuple, Optional

import json
import rasterio
import rasterio.warp
import shapely
from shapely.geometry import Polygon
import numpy as np

from pixel_patrol_base.core.contracts import FileInfo
from pixel_patrol_base.core.record import Record, record_from

logger = logging.getLogger(__name__)

def _get_footprint_geojson(bounds: rasterio.coords.BoundingBox, img_crs) -> str:
    lat_lon_crs = rasterio.CRS.from_epsg(4326)
    coords = [
        (bounds.left,  bounds.bottom),
        (bounds.right, bounds.bottom),
        (bounds.right, bounds.top),
        (bounds.left,  bounds.top),
    ]
    geom = {"type": "Polygon", "coordinates": [coords]}
    geom_latlon = rasterio.warp.transform_geom(img_crs, lat_lon_crs, geom)
    ring = geom_latlon["coordinates"][0]
    ring_json = json.dumps(shapely.geometry.mapping(Polygon(ring[:-1])))
    return ring_json

def _get_nodata_value_rasterio(
        image_nodata_value: Optional,
        band_nodata_values: Optional[List]) -> Optional[int | float]:
    """Combine the two possible NoData fields of rasterio together."""
    nodata_values = list()
    if image_nodata_value: nodata_values.append(image_nodata_value)
    if band_nodata_values: nodata_values.extend(band_nodata_values)

    nodata_values = [v for v in nodata_values if v and ~np.isnan(v)]  # remove NaNs, because float("nan") != float("nan")
    if len(set(nodata_values)) > 1:
        msg = (
            "Multiple no data values found: '%s', which is currently not supported!\n"
            "If you encounter this and what to have it fixed: look at this issue: TODO" # todo
        )
        logger.warning(msg, nodata_values)
        return None
    elif len(set(nodata_values)) == 1:
        nodata_value = nodata_values[0]
        return nodata_value
    else:
        return None


class GeospatialLoader:
    """Read geospatial images/data using rasterio."""

    NAME = "geospatial"

    SUPPORTED_EXTENSIONS: Set[str] = {"tif", "tiff", "nc"}
    FOLDER_EXTENSIONS:    Set[str] = set()
    CONTAINER_EXTENSIONS: Set[str] = set()

    OUTPUT_SCHEMA:          Dict[str, Any] = {
        "crs_str": Optional[str],
        "crs_epsg": Optional[rasterio.crs.CRS],
        "latitude": Optional[float],
        "longitude": Optional[float],
        "dim_order": str,
        "dim_names": list,
        "n_images": int,
        "num_pixels": int,
        "shape": list,
        "ndim": int,
        "channel_names": list,  # could be list[str]
        "dtype": str,
        "footprint": Optional[str], # GeoJSON string, EPSG:4326; None only when crs is absent
        "nodata_value": Optional[int | float],
    }
    OUTPUT_SCHEMA_PATTERNS: List[tuple]    = []

    def is_folder_supported(self, path: Path) -> bool:
        return False

    def __get_shape_dtype_dim_order(self, ds: rasterio.io.DatasetReader) -> Tuple:
        # ds.shape is just (height, width), but doesn't include the band count
        shape = (ds.count,) + ds.shape
        dim_order = ("C", "Y", "X")
        if len(set(ds.dtypes)) > 1:
            error_msg = f"Different data types in one image is not supported! Consider creating several images per band! Found dtypes {set(ds.dtypes)}."
            raise NotImplementedError(error_msg)
        dtype = set(ds.dtypes).pop()
        return shape, dtype, dim_order

    def read_header(self, file_path: Path) -> FileInfo:
        with rasterio.open(file_path) as ds:
            shape, dtype, dim_order = self.__get_shape_dtype_dim_order(ds)
        fi = FileInfo(shape=shape, dtype=dtype, dim_order=dim_order)
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
                metadata["footprint"] = _get_footprint_geojson(img.bounds, img.crs)
            else:
                metadata["crs_epsg"] = None
                metadata["crs_str"] = None
                metadata["latitude"] = None
                metadata["longitude"] = None
                metadata["footprint"] = None

            shape, dtype, dim_order = self.__get_shape_dtype_dim_order(img)
            nodata_value = _get_nodata_value_rasterio(image_nodata_value=img.nodata, band_nodata_values=img.nodatavals)
            dim_order = "".join(dim_order)
            metadata["dim_order"] = dim_order
            metadata["shape"] = shape
            metadata["dtype"] = dtype
            metadata["dim_names"] = ["band", "height", "width"]
            metadata["n_images"] = 1
            metadata["num_pixels"] = np.prod(img.shape) * img.count
            metadata["ndim"] = len(shape)
            metadata["nodata_value"] = nodata_value

            pixels = img.read()
        return record_from(pixels, metadata, kind="intensity")


    def load_range(self, file_path: Path, start: int, stop: int) -> Iterator[Tuple[str, Record]]:
        raise NotImplementedError("container format not support for geospatial")
