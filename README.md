# PixelPatrol: Scientific Image Dataset Pre-validation Tool

PixelPatrol is an early-version tool designed for the systematic pre-validation of scientific image datasets. It helps researchers proactively assess their data before engaging in computationally intensive analysis, ensuring the quality and integrity of datasets for reliable downstream analysis.

![Overview of the PixelPatrol dashboard, showing interactive data exploration.](readme_assets/overview.png)
*PixelPatrol's main dashboard provides an intuitive interface for dataset exploration.*

## Features

* **Dataset-wide Visualization and Interactive Exploration**
* **Detailed Statistical Summaries**: Generates plots and distributions covering image dimensions.
* **Early Identification of Issues**: Helps in finding outliers and identifying potential issues, discrepancies, or unexpected characteristics, including those related to metadata and acquisition parameters.
* **Comparison Across Experimental Conditions**
* **Dashboard Report**: Interactive reports are served as a web application using Dash.

### Coming soon:

* **GUI**: A user-friendly graphical interface for easier interaction.
* **User-Configurable**: Tailor checks to specific needs and datasets.
* **Big data support**: Efficiently handle large datasets with optimized data processing.

## Installation

PixelPatrol is published on PyPI. 
https://pypi.org/project/pixel-patrol/  

We recommend installing it using `uv` for a fast and efficient installation (you can install `uv` as described [here](https://docs.astral.sh/uv/getting-started/installation/)):
```bash
uv pip install pixel-patrol
```

You can install and run the command line tool of `pixel-patrol` in one call using this command:
```bash
uvx pixel-patrol
```

## Getting Started

Please see example scripts in the `examples` directory for detailed usage.  
To run the tool on an example dataset, you can run the `examples/create_process_report_w_example_data.py` script.

## Example visualizations

* Visualize the distribution of image sizes within your dataset.*
        ![Plot showing the distribution of image sizes.](readme_assets/size_plot.png)
* A mosaic view can quickly highlight inconsistencies across images.*
        ![Mosaic view of images, highlighting potential discrepancies.](readme_assets/mosiac.png)
* Many additional plots and distributions are available.*
        ![Statistical plots showing image dimensions and distributions.](readme_assets/example_stats_plot.png)


## Command-Line Interface

The CLI operates in two main steps: `export` (process data to a ZIP file) and `report` (view a report from a ZIP file).

Use `--help` for detailed command options:

```bash
pixel-patrol --help
pixel-patrol export --help
pixel-patrol report --help
```

#### Exporting a report

The `export` command processes your image dataset, applies settings, and saves the project data to a ZIP file.

**Syntax:**

```bash
pixel-patrol export <BASE_DIRECTORY> [OPTIONS]
```

  * **`BASE_DIRECTORY`**: The required path to your base image data directory.

**Options:**

  * `-o, --output-zip <PATH>` **(Required)**: The path and filename for the output ZIP archive (e.g., `my_project.zip`).
  * `--name` **(Optional)**. A name for your project. If not provided, it will be automatically derived from the name of the `BASE_DIRECTORY`.
  * `-p, --paths <PATH>` **(Multiple, Optional)**: Paths to include in the project, **relative to the `BASE_DIRECTORY`**. Can be specified multiple times.
  * `--cmap <COLORMAP>` (Default: `rainbow`): The colormap to use for report visualizations (e.g., `viridis`, `plasma`, `rainbow`).
  * `-e, --file-extension <EXT>` **(Multiple, Optional)**: File extensions to include (e.g., `png`, `jpg`, `tiff`). Can be specified multiple times. If not specified, all supported extensions will be used.

**Examples:**

1.  **Process a dataset, derive project name, auto-discover paths, and save:**

    ```bash
    pixel-patrol export /path/to/my_image_dataset -o output_report.zip --cmap viridis -e png -e jpg
    ```

    *(This will name the project "my\_image\_dataset" and include all direct subfolders within it.)*

2.  **Process with a custom project name and specific relative paths:**

    ```bash
    pixel-patrol export /data/my_photos --name "Vacation 2024" -o vacation_report.zip -p good_shots -p blurry_ones
    ```

    *(Only `good_shots` and `blurry_ones` subfolders relative to `/data/my_photos` will be processed.)*

#### Showing a report

Loads an exported `pixel-patrol` ZIP file and displays the interactive report in your browser.

**Syntax:** `pixel-patrol report <INPUT_ZIP> [OPTIONS]`

  * `INPUT_ZIP`: Path to the exported ZIP file.

**Options:**

  * `--port <INTEGER>` (Default: `8050`): The port number on which the Dash report server will run.

**Example:**

```bash
pixel-patrol report my_report.zip
```
