import logging
from pathlib import Path
from typing import List, Optional, Set, Dict, Any

import rasterio
import rasterio.warp
import dask.array as da
import numpy as np

from pixel_patrol_base.core.record import Record, record_from
from pixel_patrol_base.core.contracts import PixelPatrolLoader

logger = logging.getLogger(__name__)

def _get_nodata_value(
        image_nodata_value: Optional,
        band_nodata_values: Optional[List]) -> Optional:
    nodata_values = list()
    if image_nodata_value: nodata_values.append(image_nodata_value)
    if band_nodata_values: nodata_values.extend(band_nodata_values)

    if len(set(nodata_values)) > 1:
        msg = (
            "Multiple no data values found: '%s', which is currently not supported!\n"
            "If you encounter this and what to have it fixed: look at this issue: TODO" # todo
        )
        logger.warning(msg, nodata_values)
        return None
    else:
        nodata_value = nodata_values[0]
        return nodata_value

def _get_datatype(dtypes: Optional) -> Optional:
    match dtypes:
        case None | [] | ():
            return None
        case list() | tuple() if len(set(dtypes)) == 1:
            return dtypes[0]
        case list():
            logger.warning(
                (
                    "Multiple data types found: %s, which is currently not supported!\n"
                    "Only images with one data type for all values are implemented.\n"
                    "If you encounter this and want it fixed, see issue: TODO" # todo
                ),
                dtypes,
            )
            return None
        case _:
            return dtypes


def _extract_metadata(img: rasterio.DatasetReader) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}

    metadata["band_count"] = img.count
    metadata["crs_epsg"] = img.crs.to_epsg(confidence_threshold=70)
    metadata["crs_str"] = img.crs.to_string()

    datatype = _get_datatype(img.dtypes)
    metadata["dtype"] = datatype

    nodata_value = _get_nodata_value(image_nodata_value=img.nodata, band_nodata_values=img.nodatavals)
    metadata["nodata_value"] = nodata_value

    if img.count == 1:
        metadata["dim_order"] = "YX"
        metadata["dim_names"] = ["Y", "X"]
        metadata["shape"] = [img.height, img.width]
    else:
        metadata["dim_order"] = "CYX"
        metadata["dim_names"] = ["C", "Y", "X"]
        metadata["shape"] = [img.count, img.height, img.width]
    metadata["num_pixels"] = np.prod(metadata["shape"])
    metadata["X_size"] = metadata["height"] = img.height
    metadata["Y_size"] = metadata["width"] = img.width


    lng, lat = img.lnglat()  # Geographic coordinates of the dataset’s center.
    metadata["latitude"] = lat
    metadata["longitude"] = lng

    bounds_img_crs = img.bounds  # eg. BoundingBox(left=395999.999999962, bottom=5931347.999999857, right=396060.799999962, top=5931408.799999856)
    lat_lon_crs = rasterio.CRS.from_epsg(4326)
    bounds_tuple_lat_lon = rasterio.warp.transform_bounds(img.crs, lat_lon_crs, *bounds_img_crs)
    # bounds_tuple_lat_lon:  (lower left x, lower left y, upper right x, upper right y)

    metadata["bbox_west"] = bounds_tuple_lat_lon[0]  # left
    metadata["bbox_south"] = bounds_tuple_lat_lon[1]  # bottom
    metadata["bbox_east"] = bounds_tuple_lat_lon[2]  # right
    metadata["bbox_north"] = bounds_tuple_lat_lon[3]  # top

    # TODO: pixel size in meters using img.res after (after converting to the local UTM grid)
    # TODO: scales if set
    # TODO: descriptions and color interpretations of each band if set

    dataset_mask = img.dataset_mask()  # np.ndarray, uint8. 0 = nodata, 255 = valid data.
    invalid_data_count = (dataset_mask == 0).sum()
    invalid_data_percentage = (invalid_data_count / dataset_mask.size) * 100
    metadata["rasterio_valid_data_count"] = dataset_mask.size
    metadata["rasterio_invalid_data_count"] = invalid_data_count
    metadata["rasterio_invalid_data_percentage"] = invalid_data_percentage

    return metadata


class GeoImageLoader(PixelPatrolLoader):
    NAME = "geospatial"
    OUTPUT_SCHEMA = {"latitude": float,
                     "longitude": float,
                     "band_count": int,
                     "crs_epsg": rasterio.crs.CRS,
                     "crs_str": str,
                     "nodata_value": Optional[float],
                     "bbox_north": float, "bbox_south": float, "bbox_east": float, "bbox_west": float,
                     "dim_order": str,
                     "dim_names": List[str],
                     "shape": List[int],
                     "num_pixels": int,
                     "X_size": int,
                     "Y_size": int,
                     "width": int, "height": int,
                     "dtype": Optional,
                     "rasterio_valid_data_count": int,
                     "rasterio_invalid_data_count": int,
                     "rasterio_invalid_data_percentage": float,
                     }
    OUTPUT_SCHEMA_PATTERNS = [(rf"^nodata_c\d+$", float)]

    SUPPORTED_EXTENSIONS: Set[str] = {"tif"}

    def load(self, source: str) -> Record:
        p = Path(source)

        with rasterio.open(p) as ds:
            data = ds.read()
            meta = _extract_metadata(img=ds)
            data = da.squeeze(data)
            data = da.from_array(data)

        record = record_from(data, meta, kind="intensity")
        return record
