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
  pixel-patrol schema              # writes schema.json in the current directory
  pixel-patrol schema -o my/path/schema.json  # custom output path
  pixel-patrol schema --print      # print to stdout (e.g. for piping)
  ```

## All report columns

Every column that can appear in a report, with the source that produces it. The
table is generated from the installed plugins.

<!-- BEGIN GENERATED COLUMNS (docs/gen_schema_docs.py) -->
| Column | Type | Description | Source | Created by | Package |
| --- | --- | --- | --- | --- | --- |
| `common_base` | `str` | Longest path prefix shared by all imported files. | File system | base processing | pixel-patrol-base |
| `depth` | `int` | Depth of the path below the imported base directory. | File system | base processing | pixel-patrol-base |
| `file_extension` | `str` | Lower-cased file extension without the leading dot. | File system | base processing | pixel-patrol-base |
| `imported_path` | `str` | The base directory under which this file was imported. | File system | base processing | pixel-patrol-base |
| `imported_path_short` | `str` | Shortened, disambiguated label for the imported base directory (only when several bases were imported). | File system | base processing | pixel-patrol-base |
| `modification_date` | `datetime` | Last modification timestamp of the file. | File system | base processing | pixel-patrol-base |
| `name` | `str` | File or folder name, including extension. | File system | base processing | pixel-patrol-base |
| `parent` | `str` | Path of the containing directory. | File system | base processing | pixel-patrol-base |
| `path` | `str` | Absolute path of the file (or folder) on disk. | File system | base processing | pixel-patrol-base |
| `size_bytes` | `int` | Size on disk in bytes (aggregated for folders). | File system | base processing | pixel-patrol-base |
| `size_readable` | `str` | Human-readable size on disk (e.g. '2.4 MB'). | File system | base processing | pixel-patrol-base |
| `type` | `str` | Row kind: 'file', 'folder', or 'sub_file' (a sub-image inside a container). | File system | base processing | pixel-patrol-base |
| `channel_names` | `list` | Names of the image channels, if available. | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio |
| `child_id` | `string` | Identifier of a sub-image within a container file (null for single-image files). | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio |
| `dim_names` | `list` | Human-readable names of the image axes. | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio |
| `dim_order` | `string` | Axis order of the image, e.g. 'TCZYX'. | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio |
| `dtype` | `string` | Pixel data type of the source image (e.g. 'uint8', 'float32'). | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio |
| `n_images` | `int` | Number of sub-images in the source (>1 for container formats). | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio |
| `pixel_size_<axis>` | `float` | Physical pixel size along the given axis, in the image's spatial unit. | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio |
| `shape` | `array` | Size of the image along each axis, in dim_order. | Loaders | bioio, tifffile, zarr | pixel-patrol-loader-bio |
| `zarr_attributes` | `dict` | Raw key-value attributes stored in the Zarr/OME-Zarr group metadata. | Loaders | zarr | pixel-patrol-loader-bio |
| `ndim` | `int` | Number of image dimensions, derived from dim_order. | Pipeline | base processing | pixel-patrol-base |
| `num_pixels` | `int` | Number of pixels in this row's spatial extent (full image at obs_level=0, slice at higher levels). | Pipeline | base processing | pixel-patrol-base |
| `size_<axis>` | `int` | Extent (number of elements) of this row along the given axis. | Pipeline | base processing | pixel-patrol-base |
| `dim_<axis>` | `int` | Coordinate of this row along the given axis (e.g. dim_z = Z index); null when the row spans the whole axis. | Aggregation | base processing | pixel-patrol-base |
| `obs_level` | `int` | Aggregation level of the row: 0 is the whole-image summary; higher levels are per-dimension breakdowns. | Aggregation | base processing | pixel-patrol-base |
| `blocking_index` | `float32` | Strength of block-boundary discontinuities, indicating JPEG-style blocking artifacts. | raster-compression | raster-compression | pixel-patrol-image |
| `finite_pixel_count` | `uint64` | Number of finite (non-NaN/Inf) pixels contributing to the statistics. | raster-basic | raster-basic | pixel-patrol-base |
| `histogram_counts` | `array` | Per-bin pixel counts over the histogram value range (fixed number of bins). | raster-histogram | raster-histogram | pixel-patrol-base |
| `histogram_max` | `float32` | Upper bound of the histogram's value range. | raster-histogram | raster-histogram | pixel-patrol-base |
| `histogram_min` | `float32` | Lower bound of the histogram's value range. | raster-histogram | raster-histogram | pixel-patrol-base |
| `histogram_nan_count` | `uint64` | Number of NaN pixels excluded from the histogram. | raster-histogram | raster-histogram | pixel-patrol-base |
| `laplacian_variance` | `float32` | Variance of the Laplacian; a focus/sharpness measure (higher = sharper). | raster-quality | raster-quality | pixel-patrol-image |
| `max_intensity` | `float32` | Maximum pixel intensity over the covered extent (ignoring NaNs). | raster-basic | raster-basic | pixel-patrol-base |
| `mean_intensity` | `float32` | Pixel-count-weighted mean intensity over the covered extent. | raster-basic | raster-basic | pixel-patrol-base |
| `michelson_contrast` | `float32` | Michelson contrast: (max - min) / (max + min) of intensities. | raster-quality | raster-quality | pixel-patrol-image |
| `min_intensity` | `float32` | Minimum pixel intensity over the covered extent (ignoring NaNs). | raster-basic | raster-basic | pixel-patrol-base |
| `mscn_variance` | `float32` | Variance of mean-subtracted contrast-normalized (MSCN) coefficients; a no-reference naturalness/quality cue. | raster-quality | raster-quality | pixel-patrol-image |
| `ringing_index` | `float32` | Strength of ringing artifacts near high-contrast edges. | raster-compression | raster-compression | pixel-patrol-image |
| `std_intensity` | `float32` | Pooled standard deviation of intensity over the covered extent. | raster-basic | raster-basic | pixel-patrol-base |
| `texture_heterogeneity` | `float32` | Local texture heterogeneity of the image. | raster-quality | raster-quality | pixel-patrol-image |
| `thumbnail` | `bytes` | Raw RGBA bytes of the assembled thumbnail sprite (fixed sprite size). | thumbnail | thumbnail | pixel-patrol-base |
| `thumbnail_dtype` | `string` | Original pixel dtype of the source image the thumbnail was built from. | thumbnail | thumbnail | pixel-patrol-base |
| `thumbnail_norm_max` | `float` | Upper intensity bound used to normalize the thumbnail. | thumbnail | thumbnail | pixel-patrol-base |
| `thumbnail_norm_min` | `float` | Lower intensity bound used to normalize the thumbnail. | thumbnail | thumbnail | pixel-patrol-base |
<!-- END GENERATED COLUMNS -->
