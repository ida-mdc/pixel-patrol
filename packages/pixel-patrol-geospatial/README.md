# PixelPatrol Geospatial Package (`pixel-patrol-geospatial`)

Extension for **PixelPatrol** that adds support for geospatial raster data.

- **Loader** for GeoTIFF (`.tif`, `.tiff`) and NetCDF (`.nc`) files via [rasterio](https://rasterio.readthedocs.io/) — extracts CRS, bounding-box footprint, centre coordinates, nodata value, resolution, and physical units.
- **NoData processor** that counts pixels matching the declared nodata sentinel.
- **Viewer widgets**: an interactive map showing image locations and footprints, and bar charts of nodata value frequency and nodata pixel percentage.

See the main [pixel-patrol documentation](https://github.com/ida-mdc/pixel-patrol/) for usage.
