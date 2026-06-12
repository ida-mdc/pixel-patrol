# <img src="https://raw.githubusercontent.com/ida-mdc/pixel-patrol/main/packages/pixel-patrol-base/src/pixel_patrol_base/launch_assets/prevalidation.png" width="80">  Pixel Patrol

### Image Dataset Quality Control and Exploration

<img src="https://raw.githubusercontent.com/ida-mdc/pixel-patrol/main/packages/pixel-patrol/readme_assets/HI_logo.jpg" width="80">

**[Documentation](https://ida-mdc.github.io/pixel-patrol/docs/) | [Tutorials](https://ida-mdc.github.io/pixel-patrol/docs/tutorials/) | [Example Report](https://ida-mdc.github.io/pixel-patrol/viewer/?data=../example.parquet) | [PyPI](https://pypi.org/project/pixel-patrol/) | [Viewer](https://ida-mdc.github.io/pixel-patrol/viewer/)**

Image datasets are rarely as clean or consistent as they appear. Pixel Patrol scans your images and generates a shareable, interactive browser report - file and image metadata, pixel statistics, quality metrics, and per-dimension slice statistics. Get immediate results, compare conditions, catch outliers, verify batch consistency, and get the full picture before you use your dataset.

<img src="https://raw.githubusercontent.com/ida-mdc/pixel-patrol/main/packages/pixel-patrol/readme_assets/overview.png" width="">

*The interactive viewer - filter, group, and explore your dataset.*

---

## Installation

Requires Python 3.11+. We recommend [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
uv pip install pixel-patrol
```

Or with pip:

```bash
pip install pixel-patrol
```

For a modular install (core only + selected add-ons):

```bash
uv pip install pixel-patrol-base
uv pip install pixel-patrol-loader-bio   # BioIO, Zarr, Tifffile loaders
uv pip install pixel-patrol-image        # image quality metrics and extra widgets
```

---

## Quickstart

**1. Process your dataset:**

```bash
pixel-patrol process path/to/images/ -o report.parquet --loader bioio
```

For datasets with experimental conditions:

```bash
pixel-patrol process path/to/images/ -o report.parquet --loader bioio \
  -p condition_a -p condition_b
```

**2. Explore in the viewer:**

```bash
pixel-patrol view report.parquet
```

**Or use the processing dashboard** for a visual interface:

```bash
pixel-patrol launch
```

---

## Python API

```python
from pixel_patrol_base import api

project = api.create_project("my-project", base_dir="path/to/images/", loader="bioio")
api.add_paths(project, ["condition_a", "condition_b"])  # optional
api.process_files(project)
api.view(project)
```

---

## Example visualizations

![Plot showing the distribution of image sizes.](https://raw.githubusercontent.com/ida-mdc/pixel-patrol/main/packages/pixel-patrol/readme_assets/size_plot.png)

*File size distribution across the dataset.*

![Mosaic view of images, highlighting potential discrepancies.](https://raw.githubusercontent.com/ida-mdc/pixel-patrol/main/packages/pixel-patrol/readme_assets/mosiac.png)

*Image mosaic - sort by any metric to surface outliers visually.*

![Statistical plots showing image dimensions and distributions.](https://raw.githubusercontent.com/ida-mdc/pixel-patrol/main/packages/pixel-patrol/readme_assets/example_stats_plot.png)

*Dimension size distributions and statistics.*

---

## Sharing a report

Send the `.parquet` file and open it in the [hosted viewer](https://ida-mdc.github.io/pixel-patrol/viewer/) - no installation needed. Or build a self-contained static viewer:

```bash
pixel-patrol build-viewer-html -o viewer.html
```

> **Note:** The static viewer may not load very large parquet files (e.g. 5 GB+). Use `pixel-patrol view` for large reports.

---

## Extending Pixel Patrol

Pixel Patrol is designed to be extended with custom loaders, processors, and viewer widgets as standalone Python packages. See `examples/minimal-extension/` for a working template, and the [Extensions](https://ida-mdc.github.io/pixel-patrol/docs/extensions/) documentation.

---

## Full documentation

[ida-mdc.github.io/pixel-patrol/docs/](https://ida-mdc.github.io/pixel-patrol/docs/)
