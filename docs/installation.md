# Installation

Pixel Patrol requires Python 3.11 or higher.

---

## Recommended: install with uv

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
pixel-patrol --help
```

---

## Quick install with pip

!!! warning
    Installing without a virtual environment is not recommended - it can conflict with other packages in your Python installation. If in doubt, use the uv method above.

```bash
pip install pixel-patrol
```

---

## Modular install

Pixel Patrol is split into focused packages so you only install what you need. The core package (`pixel-patrol-base`) provides the CLI, viewer, and processing framework. Loaders and extra processors are optional add-ons.

```bash
uv pip install pixel-patrol-base           # core only
uv pip install pixel-patrol-loader-bio     # BioIO + Zarr + Tifffile loaders (CZI, ND2, LIF, TIFF, ...)
uv pip install pixel-patrol-image          # image quality metrics and extra viewer widgets
```

`pixel-patrol-base` alone collects basic file metadata (name, size, extension) without reading image data. Add `pixel-patrol-loader-bio` to extract image metadata, pixel statistics, and thumbnails.

---

## Packages overview

| Package | What it adds |
|---|---|
| `pixel-patrol` | Full bundle - everything below in one install |
| `pixel-patrol-base` | Core framework, CLI, viewer |
| `pixel-patrol-loader-bio` | BioIO, Zarr, and Tifffile loaders for scientific image formats |
| `pixel-patrol-image` | Image quality metrics (blur, contrast, noise) and extra viewer widgets |

Additional packages are available in the [GitHub repository](https://github.com/ida-mdc/pixel-patrol) - for example `pixel-patrol-aqqua` for AQQUA datasets. You can also extend Pixel Patrol by creating your own packages; see [Extensions](extensions.md).

---

## Prefer a one-click app?

If you'd rather skip the command line entirely, download the **Pixel Patrol Launcher** - a single clickable file for Windows, macOS, or Linux. On first run it sets up a managed Python environment at `~/.pixel-patrol/`, installs Pixel Patrol, and opens the app in your browser. No Python required.

[🪟 Windows](https://github.com/ida-mdc/pixel-patrol/releases/latest/download/pixel-patrol-launcher-windows.exe){ .md-button .md-button--primary }
[🍎 macOS](https://github.com/ida-mdc/pixel-patrol/releases/latest/download/pixel-patrol-launcher-macos){ .md-button .md-button--primary }
[🐧 Linux](https://github.com/ida-mdc/pixel-patrol/releases/latest/download/pixel-patrol-launcher-linux){ .md-button .md-button--primary }
