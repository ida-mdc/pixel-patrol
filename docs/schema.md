# Table Schema

Every PixelPatrol table is a single `.parquet` file. Its columns are assembled
from the registered **loaders** (image metadata), **processors** (metrics), and a
set of pipeline-generated columns; the **viewer widgets** then consume those
columns to draw the interactive report.

The same per-column descriptions below are embedded into each table's parquet
field metadata, so produced files are self-describing.

Two companion resources are generated directly from the installed plugins, so
they always match the build:

- **[Interactive table map](assets/schema.html)** — follow any *producer → column
  → widget* path. Click a node for its description and connections; hover to trace
  its links.
- **[Full catalog as JSON](assets/schema.json)** — the machine-readable schema and
  plugin catalog. Also available from the CLI:

  ```bash
  pixel-patrol schema              # writes schema.json in the current directory
  pixel-patrol schema -o my/path/schema.json  # custom output path
  pixel-patrol schema --print      # print to stdout (e.g. for piping)
  ```

## All table columns

Every column that can appear in a table, with the source that produces it. The
table is generated from the installed plugins.

<!-- BEGIN GENERATED COLUMNS (docs/gen_schema_docs.py) -->
| Column | Type | Description | Source | Created by | Package |
| --- | --- | --- | --- | --- | --- |
| `common_base` | `str` | Name of the longest path prefix shared by all import bases. | File system | base processing | pixel-patrol-base |
| `depth` | `int` | Depth of the path below the imported base directory. | File system | base processing | pixel-patrol-base |
| `file_extension` | `str` | Lower-cased file extension without the leading dot. | File system | base processing | pixel-patrol-base |
| `imported_path` | `str` | Path of the import base (the -p path or base directory) under which this file was found, relative to the project base directory ('.' when no -p paths were specified). | File system | base processing | pixel-patrol-base |
| `imported_path_short` | `str` | Shortened, disambiguated label for the import base (only when several -p paths were used). | File system | base processing | pixel-patrol-base |
| `modification_date` | `datetime` | Last modification timestamp of the file. | File system | base processing | pixel-patrol-base |
| `name` | `str` | File or folder name, including extension. | File system | base processing | pixel-patrol-base |
| `parent` | `str` | Path of the containing directory, relative to the project base directory ('.' for files directly in the base directory). | File system | base processing | pixel-patrol-base |
| `path` | `str` | Path of the file (or folder), relative to the project base directory. | File system | base processing | pixel-patrol-base |
| `size_bytes` | `int` | Size on disk in bytes (aggregated for folders). | File system | base processing | pixel-patrol-base |
| `type` | `str` | Row kind: 'file', 'folder', or 'sub_file' (a sub-image inside a container). | File system | base processing | pixel-patrol-base |
| `channel_names` | `list` | Names of the image channels, if available. | Loaders | aqqua_lmdb, bioio, tifffile, video, zarr | pixel-patrol-aqqua, pixel-patrol-loader-bio, pixel-patrol-loader-video |
| `child_id` | `string` | Identifier of a sub-image within a container file (null for single-image files). | Loaders | aqqua_lmdb, bioio, tifffile, video, zarr | pixel-patrol-aqqua, pixel-patrol-loader-bio, pixel-patrol-loader-video |
| `codec` | `string` |  | Loaders | video | pixel-patrol-loader-video |
| `crs_epsg` | `Optional` | EPSG code of the CRS; None if not EPSG-registered. | Loaders | geospatial | pixel-patrol-geospatial |
| `crs_str` | `Optional` | Coordinate reference system as a WKT string. | Loaders | geospatial | pixel-patrol-geospatial |
| `dim_names` | `list` | Human-readable names of the image axes. | Loaders | aqqua_lmdb, bioio, geospatial, tifffile, video, zarr | pixel-patrol-aqqua, pixel-patrol-geospatial, pixel-patrol-loader-bio, pixel-patrol-loader-video |
| `dim_order` | `string` | Axis order of the image, e.g. 'TCZYX'. | Loaders | aqqua_lmdb, bioio, geospatial, tifffile, video, zarr | pixel-patrol-aqqua, pixel-patrol-geospatial, pixel-patrol-loader-bio, pixel-patrol-loader-video |
| `dtype` | `string` | Pixel data type of the source image (e.g. 'uint8', 'float32'). | Loaders | aqqua_lmdb, bioio, geospatial, tifffile, video, zarr | pixel-patrol-aqqua, pixel-patrol-geospatial, pixel-patrol-loader-bio, pixel-patrol-loader-video |
| `duration_seconds` | `float` |  | Loaders | video | pixel-patrol-loader-video |
| `footprint` | `Optional` | Bounding-box footprint as a GeoJSON polygon in EPSG:4326. | Loaders | geospatial | pixel-patrol-geospatial |
| `fps` | `float` |  | Loaders | video | pixel-patrol-loader-video |
| `latitude` | `Optional` | Centroid latitude in WGS-84 degrees. | Loaders | geospatial | pixel-patrol-geospatial |
| `longitude` | `Optional` | Centroid longitude in WGS-84 degrees. | Loaders | geospatial | pixel-patrol-geospatial |
| `n_channels` | `int` |  | Loaders | video | pixel-patrol-loader-video |
| `n_frames` | `int` |  | Loaders | video | pixel-patrol-loader-video |
| `n_images` | `int` | Number of sub-images in the source (>1 for container formats). | Loaders | aqqua_lmdb, bioio, geospatial, tifffile, video, zarr | pixel-patrol-aqqua, pixel-patrol-geospatial, pixel-patrol-loader-bio, pixel-patrol-loader-video |
| `nodata_value` | `Union` | Declared nodata/fill value from the file metadata. | Loaders | geospatial | pixel-patrol-geospatial |
| `pixel_size_<axis>` | `float` | Physical pixel size along the given axis, in the image's spatial unit. | Loaders | aqqua_lmdb, bioio, tifffile, video, zarr | pixel-patrol-aqqua, pixel-patrol-loader-bio, pixel-patrol-loader-video |
| `shape` | `list` | Raster dimensions as [bands, height, width]. | Loaders | geospatial | pixel-patrol-geospatial |
| `zarr_attributes` | `dict` | Raw key-value attributes stored in the Zarr/OME-Zarr group metadata. | Loaders | zarr | pixel-patrol-loader-bio |
| `ndim` | `int` | Number of image dimensions, derived from dim_order. | Pipeline | base processing | pixel-patrol-base |
| `num_pixels` | `int` | Number of pixels in this row's spatial extent (full image at obs_level=0, slice at higher levels). | Pipeline | base processing | pixel-patrol-base |
| `size_<axis>` | `int` | Extent (number of elements) of this row along the given axis. | Pipeline | base processing | pixel-patrol-base |
| `dim_<axis>` | `int` | Coordinate of this row along the given axis (e.g. dim_z = Z index); null when the row spans the whole axis. | Aggregation | base processing | pixel-patrol-base |
| `obs_level` | `int` | Aggregation level of the row: 0 is the whole-image summary; higher levels are per-dimension breakdowns. | Aggregation | base processing | pixel-patrol-base |
| `bright_clipping_fraction` | `float32` | Fraction of pixels at the dtype's maximum representable value (integer types only; NaN for float). Detects sensor saturation at the dtype ceiling. | raster-quality | raster-quality | pixel-patrol-base |
| `dark_clipping_fraction` | `float32` | Fraction of pixels at the dtype's minimum representable value (integer types only; NaN for float). Indicates underexposure, background, or nodata. | raster-quality | raster-quality | pixel-patrol-base |
| `finite_pixel_count` | `uint64` | Number of finite (non-NaN/Inf) pixels contributing to the statistics. | raster-basic | raster-basic | pixel-patrol-base |
| `histogram_counts` | `array` | Per-bin pixel counts over the histogram value range (fixed number of bins; integer data uses one-level-wide bins). | raster-histogram | raster-histogram | pixel-patrol-base |
| `histogram_max` | `float32` | Upper bound of the histogram's value range. | raster-histogram | raster-histogram | pixel-patrol-base |
| `histogram_min` | `float32` | Lower bound of the histogram's value range. | raster-histogram | raster-histogram | pixel-patrol-base |
| `histogram_nan_count` | `uint64` | Number of NaN pixels excluded from the histogram. | raster-histogram | raster-histogram | pixel-patrol-base |
| `laplacian_variance` | `float32` | High-frequency content measure. Low values indicate blur or heavy compression. High values are ambiguous (noise, quantization artifacts, and genuine sharpness all produce high scores). Inflated by clipping boundaries. | raster-quality | raster-quality | pixel-patrol-base |
| `max_intensity` | `float32` | Maximum pixel intensity over the covered extent (ignoring NaNs). | raster-basic | raster-basic | pixel-patrol-base |
| `mean_intensity` | `float32` | Pixel-count-weighted mean intensity over the covered extent. | raster-basic | raster-basic | pixel-patrol-base |
| `min_intensity` | `float32` | Minimum pixel intensity over the covered extent (ignoring NaNs). | raster-basic | raster-basic | pixel-patrol-base |
| `nodata_count` | `int` | Number of pixels equal to the declared nodata value. | nodata-statistics | nodata-statistics | pixel-patrol-geospatial |
| `spectral_slope` | `float32` | Log-log slope of the radially averaged power spectrum (mid-frequency band). Blur steepens the slope (more negative); noise flattens it (toward zero). The only metric that maps blur and noise to opposite directions. | raster-quality | raster-quality | pixel-patrol-base |
| `std_intensity` | `float32` | Pooled standard deviation of intensity over the covered extent. | raster-basic | raster-basic | pixel-patrol-base |
| `thumbnail` | `bytes` | Raw RGBA bytes of the assembled thumbnail sprite (fixed sprite size). | thumbnail | thumbnail | pixel-patrol-base |
| `thumbnail_dtype` | `string` | Original pixel dtype of the source image the thumbnail was built from. | thumbnail | thumbnail | pixel-patrol-base |
| `thumbnail_norm_max` | `float` | Upper intensity bound used to normalize the thumbnail. | thumbnail | thumbnail | pixel-patrol-base |
| `thumbnail_norm_min` | `float` | Lower intensity bound used to normalize the thumbnail. | thumbnail | thumbnail | pixel-patrol-base |
<!-- END GENERATED COLUMNS -->
