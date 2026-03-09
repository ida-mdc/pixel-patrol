import re
from pathlib import Path
from typing import List, Optional, Set, Dict, Any

import rasterio
import dask.array as da
import numpy as np
import time

from pixel_patrol_base.core.record import Record, record_from


def _extract_metadata(img: rasterio.DatasetReader) -> Dict[str, Any]:
    """
    Extract metadata from a rasterio DatasetReader into a flat dict.
    """
    metadata: Dict[str, Any] = {}
    
    metadata["bounds"] = str(img.bounds)
    metadata["band_count"] = img.count

    metadata["crs_epsg"] = img.crs.to_epsg(confidence_threshold=70)
    metadata["crs"] = img.crs.to_string()
    # TODO: descriptions and color interpreations of each band if set
    
    if img.dtypes:
        # multiple, identical dtypes
        if isinstance(img.dtypes, list) and len(set(img.dtypes)) == 1:
            metadata["dtypes"] = img.dtypes[0]
        else:
            metadata["dtypes"] = img.dtypes
    
    # TODO nodata can be a single value or a value per band
    if img.nodata:
        metadata["nodata"] = img.nodata
    if img.nodatavals:
        metadata["nodatavals"] = img.nodatavals
    
    if img.count == 1:
        metadata["dim_order"] = "YX"
        metadata["dim_names"] = ["Y", "X"]
        metadata["shape"] = img.height, img.width
    else:
        metadata["dim_order"] = "CYX"
        metadata["dim_names"] = ["C", "Y", "X"]
        metadata["shape"] = img.count, img.height, img.width
    metadata["num_pixels"] = np.prod(metadata["shape"])
    metadata["X_size"] = metadata["height"] = img.height
    metadata["Y_size"] = metadata["width"] = img.width
    
    
    lng, lat = img.lnglat()
    metadata["latitude"] = lat
    metadata["longitude"] = lng
    
    # TODO: pixel size in meters using img.res after (after converting to the local UTM grid)
    # TODO: scales if set
    
    # TODO: add bounding box coordinates
    # TODO: valid data/pixels?
    

    return metadata


class GeoImageLoader:
    NAME = "geospatial"

    SUPPORTED_EXTENSIONS: Set[str] = {"tif"}

    def load(self, source: str) -> Record:
        p = Path(source)

        with rasterio.open(p) as ds:
            data = ds.read()            
            meta = _extract_metadata(img=ds)
            # TODO: don't replace NaN with zero, but instead properly support 
            #       missing data
            data = np.nan_to_num(data, nan=0)
            data = da.squeeze(data)

        record = record_from(data, meta, kind="intensity")
        return record
