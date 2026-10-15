# Installation

Choose one: the one-click app (no terminal needed), or an install from the terminal (Python 3.12+).

---

## Easiest: one-click app

Download the **PixelPatrol Launcher** - a single clickable file for Windows, macOS, or Linux. On first run it sets up a managed Python environment at `~/.pixel-patrol/`, installs PixelPatrol, and opens the app in your browser. No Python required.

[🪟 Windows](https://github.com/ida-mdc/pixel-patrol/releases/latest/download/pixel-patrol-launcher-windows.exe){ .md-button .md-button--primary }
[🍎 macOS](https://github.com/ida-mdc/pixel-patrol/releases/latest/download/pixel-patrol-launcher-macos){ .md-button .md-button--primary }
[🐧 Linux](https://github.com/ida-mdc/pixel-patrol/releases/latest/download/pixel-patrol-launcher-linux){ .md-button .md-button--primary }

!!! note "First time running it?"
    - **Windows**: if SmartScreen shows a warning, click **More info** → **Run anyway**.
    - **macOS**: right-click the file → **Open** → confirm **Open** when macOS warns about an unidentified developer.
    - **Linux**: make it executable first - `chmod +x pixel-patrol-launcher-linux`, then double-click it (or run it from a terminal).

---

## Terminal: install with uv

We recommend [uv](https://docs.astral.sh/uv/) for fast, reproducible installs. It handles virtual environments and dependency resolution cleanly.

**Install uv** (once):

```bash
# macOS / Linux:
curl -Ls https://astral.sh/uv/install.sh | sh

# Windows:
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Create a virtual environment and install:**

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
uv pip install pixel-patrol
```

Verify:

```bash
pixel-patrol --version
```

---

## Modular install

PixelPatrol is split into focused packages so you only install what you need. The core package (`pixel-patrol-base`) provides the CLI, viewer, and processing framework. Loaders and extra processors are optional add-ons.

```bash
uv pip install pixel-patrol-base           # core, includes image quality metrics
uv pip install pixel-patrol-loader-bio     # BioIO + Zarr + Tifffile + H5 loaders (CZI, ND2, LIF, TIFF, ...)
uv pip install pixel-patrol-loader-medical # NIfTI and DICOM loaders
uv pip install pixel-patrol-geospatial     # GeoTIFF and NetCDF loader, nodata processor
```

Without a loader, only basic file metadata (name, size, extension) is collected and no image data is read. Install a loader package and choose one of its loaders with `--loader` to also get image metadata, pixel statistics, and thumbnails.

---

## Packages overview

| Package | What it adds |
|---|---|
| `pixel-patrol` | Full bundle: base, loader-bio, loader-medical, and geospatial in one install |
| `pixel-patrol-base` | Core framework, CLI, viewer, image quality metrics |
| `pixel-patrol-loader-bio` | BioIO, Zarr, Tifffile, and H5 loaders for scientific image formats |
| `pixel-patrol-loader-medical` | NIfTI and DICOM loaders |
| `pixel-patrol-geospatial` | GeoTIFF and NetCDF loader, nodata processor |

Additional packages are available in the [GitHub repository](https://github.com/ida-mdc/pixel-patrol) - for example `pixel-patrol-aqqua` for AQQUA datasets. You can also extend PixelPatrol by creating your own packages; see [Extensions](extensions.md).
