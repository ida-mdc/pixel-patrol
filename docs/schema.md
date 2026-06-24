# Report Schema

Every Pixel Patrol report is a single `.parquet` file. Its columns are assembled
from the registered **loaders** (image metadata), **processors** (metrics), and a
set of pipeline-generated columns; the **viewer widgets** then consume those
columns to draw the report.

The same per-column descriptions below are embedded into each report's parquet
field metadata, so produced files are self-describing.

Two companion resources are generated directly from the installed plugins, so
they always match the build:

- **[Interactive report map](assets/schema.html)** — follow any *producer → column
  → widget* path. Click a node for its description and connections; hover to trace
  its links.
- **[Full catalog as JSON](assets/schema.json)** — the machine-readable schema and
  plugin catalog. Also available from the CLI:

  ```bash
  pixel-patrol schema -o schema.json
  ```

## All report columns

Every column that can appear in a report, with the source that produces it. The
table is generated from the installed plugins.

<!-- BEGIN GENERATED COLUMNS (docs/gen_schema_docs.py) -->
| Column | Type | Source | Created by | Package | Description |
| --- | --- | --- | --- | --- | --- |
| `common_base` |  | File system | base processing | pixel-patrol-base | Longest path prefix shared by all imported files. |
| `depth` |  | File system | base processing | pixel-patrol-base | Depth of the path below the imported base directory. |
| `file_extension` |  | File system | base processing | pixel-patrol-base | Lower-cased file extension without the leading dot. |
| `imported_path` |  | File system | base processing | pixel-patrol-base | The base directory under which this file was imported. |
| `imported_path_short` |  | File system | base processing | pixel-patrol-base | Shortened, disambiguated label for the imported base directory (only when several bases were imported). |
| `modification_date` |  | File system | base processing | pixel-patrol-base | Last modification timestamp of the file. |
| `name` |  | File system | base processing | pixel-patrol-base | File or folder name, including extension. |
| `parent` |  | File system | base processing | pixel-patrol-base | Path of the containing directory. |
| `path` |  | File system | base processing | pixel-patrol-base | Absolute path of the file (or folder) on disk. |
| `size_bytes` |  | File system | base processing | pixel-patrol-base | Size on disk in bytes (aggregated for folders). |
| `size_readable` |  | File system | base processing | pixel-patrol-base | Human-readable size on disk (e.g. '2.4 MB'). |
| `type` |  | File system | base processing | pixel-patrol-base | Row kind: 'file', 'folder', or 'sub_file' (a sub-image inside a container). |
| `channel_names` | `list` | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio | Names of the image channels, if available. |
| `child_id` | `string` | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio | Identifier of a sub-image within a container file (null for single-image files). |
| `dim_names` | `list` | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio | Human-readable names of the image axes. |
| `dim_order` | `string` | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio | Axis order of the image, e.g. 'TCZYX'. |
| `dtype` | `string` | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio | Pixel data type of the source image (e.g. 'uint8', 'float32'). |
| `n_images` | `int` | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio | Number of sub-images in the source (>1 for container formats). |
| `pixel_size_<axis>` | `float` | Loaders | bioio, tifffile | pixel-patrol-loader-bio | Physical pixel size along the given axis, in the image's spatial unit. |
| `shape` | `array` | Loaders | bioio, tifffile | pixel-patrol-loader-bio | Size of the image along each axis, in dim_order. |
| `zarr_attributes` | `dict` | Loaders | zarr | pixel-patrol-loader-bio |  |
| `<axis>_size` | `int` | Pipeline | base processing | pixel-patrol-base | Extent (number of elements) of this row along the given axis. |
| `ndim` | `int` | Pipeline | base processing | pixel-patrol-base | Number of image dimensions, derived from dim_order. |
| `num_pixels` | `int` | Pipeline | base processing | pixel-patrol-base | Number of pixels in this row's spatial extent (full image at obs_level=0, slice at higher levels). |
| `dim_<axis>` | `int` | Aggregation | base processing | pixel-patrol-base | Coordinate of this row along the given axis (e.g. dim_z = Z index); null when the row spans the whole axis. |
| `obs_level` |  | Aggregation | base processing | pixel-patrol-base | Aggregation level of the row: 0 is the whole-image summary; higher levels are per-dimension breakdowns. |
| `blocking_index` | `float32` | raster-compression | raster-compression | pixel-patrol-image |  |
| `finite_pixel_count` | `uint64` | raster-basic | raster-basic | pixel-patrol-base | Number of finite (non-NaN/Inf) pixels contributing to the statistics. |
| `histogram_counts` | `array` | raster-histogram | raster-histogram | pixel-patrol-base | Per-bin pixel counts over the histogram value range (fixed number of bins). |
| `histogram_max` | `float32` | raster-histogram | raster-histogram | pixel-patrol-base | Upper bound of the histogram's value range. |
| `histogram_min` | `float32` | raster-histogram | raster-histogram | pixel-patrol-base | Lower bound of the histogram's value range. |
| `histogram_nan_count` | `uint64` | raster-histogram | raster-histogram | pixel-patrol-base | Number of NaN pixels excluded from the histogram. |
| `laplacian_variance` | `float32` | raster-quality | raster-quality | pixel-patrol-image |  |
| `max_intensity` | `float32` | raster-basic | raster-basic | pixel-patrol-base | Maximum pixel intensity over the covered extent (ignoring NaNs). |
| `mean_intensity` | `float32` | raster-basic | raster-basic | pixel-patrol-base | Pixel-count-weighted mean intensity over the covered extent. |
| `michelson_contrast` | `float32` | raster-quality | raster-quality | pixel-patrol-image |  |
| `min_intensity` | `float32` | raster-basic | raster-basic | pixel-patrol-base | Minimum pixel intensity over the covered extent (ignoring NaNs). |
| `mscn_variance` | `float32` | raster-quality | raster-quality | pixel-patrol-image |  |
| `ringing_index` | `float32` | raster-compression | raster-compression | pixel-patrol-image |  |
| `std_intensity` | `float32` | raster-basic | raster-basic | pixel-patrol-base | Pooled standard deviation of intensity over the covered extent. |
| `texture_heterogeneity` | `float32` | raster-quality | raster-quality | pixel-patrol-image |  |
| `thumbnail` | `bytes` | thumbnail | thumbnail | pixel-patrol-base | Raw RGBA bytes of the assembled thumbnail sprite (fixed sprite size). |
| `thumbnail_dtype` | `string` | thumbnail | thumbnail | pixel-patrol-base | Original pixel dtype of the source image the thumbnail was built from. |
| `thumbnail_norm_max` | `float` | thumbnail | thumbnail | pixel-patrol-base | Upper intensity bound used to normalize the thumbnail. |
| `thumbnail_norm_min` | `float` | thumbnail | thumbnail | pixel-patrol-base | Lower intensity bound used to normalize the thumbnail. |
<!-- END GENERATED COLUMNS -->
