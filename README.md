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
