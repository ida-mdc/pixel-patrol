# PixelPatrol: Scientific Image Dataset Pre-validation Tool

PixelPatrol is an early-version tool designed for the systematic pre-validation of scientific image datasets. It helps researchers proactively assess their data before engaging in computationally intensive analysis, ensuring the quality and integrity of datasets for reliable downstream analysis.

![Overview of the PixelPatrol dashboard, showing interactive data exploration.](readme_assets/overview.png)
*PixelPatrol's main dashboard provides an interface for dataset exploration.*

## Features

* **Dataset-wide Visualization and Interactive Exploration**
* **Detailed Statistical Summaries**: Generates plots and distributions covering image dimensions.
* **Early Identification of Issues**: Helps in finding outliers and identifying potential issues, discrepancies, or unexpected characteristics, including those related to metadata and acquisition parameters.
* **Comparison Across Experimental Conditions**
* **Dashboard Report**: Interactive reports are served as a web application using Dash.

### Coming soon:

* **GUI**: A user-friendly graphical interface for easier project generation.
* **User-Configurable**: Tailor checks to specific needs and datasets.
* **Big data support**: Efficiently handle large datasets with optimized data processing.

## Installation

PixelPatrol and its add-on packages are published on PyPI: https://pypi.org/project/pixel-patrol/

### 1. Install `uv` (recommended)

`uv` provides fast virtualenv management and dependency resolution. Install it once and reuse it for all workflows.

* **macOS / Linux:**
  ```bash
  curl -Ls https://astral.sh/uv/install.sh | sh
  ```

* **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

After installation, restart your shell (if needed) and verify it works:

```bash
uv --version
```

If you prefer an alternative installation method, consult the official guide: https://docs.astral.sh/uv/getting-started/installation/

### 2. Install PixelPatrol

Before installing the package, activate a clean virtual environment so its dependencies stay isolated from other projects. Create one with your preferred tool:

```bash
# macOS / Linux (uv)
uv venv pixel-patrol-env
source pixel-patrol-env/bin/activate

# Windows PowerShell
uv venv pixel-patrol-env
pixel-patrol-env\\Scripts\\Activate.ps1
```

#### Option A - Default - Full `pixel-patrol` Bundle

This is the quickest path to running Pixel Patrol with everything ready to go. Install it and you get the CLI plus the standard widgets, processors, and loaders.

Works the same on macOS, Windows (PowerShell), and Linux terminals:

```bash
uv pip install pixel-patrol
pixel-patrol --help
```

The first command downloads the latest release and adds `pixel-patrol` to your PATH; the second command confirms it’s ready.

#### Option B — Build your own stack (`pixel-patrol-base` + add-ons)

Advanced users may prefer to assemble only the components they need:

```bash
uv pip install pixel-patrol-base
```

Add functionality by layering optional packages:

* `pixel-patrol-image` – extra processors and widgets for image analysis.
* `pixel-patrol-loader-bio` – Adds the loaders Bioio and Zarr.

You can also add your own packages to add loaders, processors, and widgets to PixelPatrol.   
See `examples/minimal-extension` for a minimal template.

## Getting Started

1. Install PixelPatrol (Instructions are in the previous section).
2. Have all the files you would like to inspect under a common root directory.
3. If you want to compare conditions - place files of each condition under a separate subdirectory within the root.
4. Run Pixel Patrol via the CLI (see [Command-Line Interface](#command-line-interface)) or use the Python API demonstrated in [API Use](#api-use).
5. Explore the interactive dashboard in your browser.

## Example visualizations

* Visualize the distribution of image sizes within your dataset.*
        ![Plot showing the distribution of image sizes.](readme_assets/size_plot.png)
* A mosaic view can quickly highlight inconsistencies across images.*
        ![Mosaic view of images, highlighting potential discrepancies.](readme_assets/mosiac.png)
* Many additional plots and distributions are available.*
        ![Statistical plots showing image dimensions and distributions.](readme_assets/example_stats_plot.png)


## Command-Line Interface

With the CLI you can use all of pixel-patrol Python API building blocks by calling two commands one after the other.       
1. First run `pixel-patrol export` to create a pixel-patrol project and saving it as a ZIP file.   
2. Then pass that ZIP to `pixel-patrol report` when you want to explore the generated report in the dashboard.

### Common commands

```bash
pixel-patrol --help
pixel-patrol export --help
pixel-patrol report --help
```

### `pixel-patrol export`

Processes a directory tree, applies the selected loader and settings, and saves a portable ZIP archive.

```bash
pixel-patrol export <BASE_DIRECTORY> -o <OUTPUT_ZIP> [OPTIONS]
```

Key options:

* `BASE_DIRECTORY` – the root folder that contains your dataset. Use an absolute path or a path relative to your current working directory.
* `-o, --output-zip PATH` **(required)** – where to store the generated pixel-patrol project zip.
* `--name TEXT` – give your pixel-patrol project a name (defaults to the folder name).
* `-p, --paths PATH` – Optional. Subdirectories or absolute paths to treat as experimental conditions; use multiple `-p` flags for multiple paths. When you pass a relative path it is resolved against `BASE_DIRECTORY`. If omitted, everything under `BASE_DIRECTORY` is processed as a single condition.
* `-l, --loader TEXT` – Optional but recommended. Loader plug-in (e.g. `bioio`, `zarr`). If omitted pixel-patrol only shows basic file info.  
* `-e, --file-extension EXT` – Optional. One or more file extensions to include (meaning filter for). When unspecified the loader’s supported extensions (or `all` for no loader) are used.
* `--cmap NAME` – Optional Matplotlib colormap for visualizations (`rainbow` by default).
* `--flavor TEXT` – optional label shown next to the Pixel Patrol title inside the report.

Example (BioIO loader, two conditions to compare - by specifying the path to their directories, only processing file extensions tif and png:

```bash
pixel-patrol export examples/datasets/bioio -o examples/out/test_project.zip \
  --loader bioio --name "test_project" -p tifs -p pngs \
  -e tif -e png --cmap viridis
```

### `pixel-patrol report`

Launches the Dash dashboard from a previously exported project ZIP file. The command prints the URL and attempts to open the browser automatically.

```bash
pixel-patrol report <REPORT_ZIP> [--port 8050]
```

If the default port is unavailable, supply `--port 8051` (or any free port). The command can be rerun at any time; the ZIP file is never modified.  
Always run `export` before `report`; the exported ZIP is the on-disk representation of a Pixel Patrol project.

### Troubleshooting

* The CLI validates loader names at runtime; if you see `Unknown loader`, ensure the corresponding plug-in package is installed and available in the active environment.

## API Use

The `examples/` directory demonstrates how to use pixel-patrol API and for advanced users also how to extend pixel-patrol (loaders, processors, and widgets) by creating a package.

* `examples/01_quickstart.py` – end-to-end walkthrough using the base API. Process the bundled sample data and launch the dashboard:
  ```bash
  uv run examples/01_quickstart.py
  ```
  The script highlights each API step (create project → add paths → configure settings → process → show → export/import).
  Feel free to adapt the scripts to your datasets and needed settings.  

* `examples/02_example_plankton_bioio.py` – downloads an open plankton dataset (≈200MB), processes it with the BioIO loader, and exports a ready-to-share report. Run it with:
  ```bash
  uv run examples/02_example_plankton_bioio.py
  ```

* `examples/minimal-extension/` – For people who want to extend pixel-patrol, it offers an example minimal plug-in package that registers a custom loader (`markdown-diary`), processor, and widgets.   
Use this as a starting point for your own plug-ins: update the `pyproject.toml` metadata (name, version, entry points) to match your project, replace the `MARKDOWN_DIARY` identifiers with your loader ID, and adjust the processor/widget code to emit the fields you care about. Entry points must be registered under `pixel_patrol.loader_plugins`, `pixel_patrol.processor_plugins`, or `pixel_patrol.widget_plugins` so Pixel Patrol can discover them automatically.
