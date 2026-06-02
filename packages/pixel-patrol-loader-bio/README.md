# PixelPatrol Bio-Image Loader Extension (`pixel-patrol-loader-bio`)

This is an extension for **PixelPatrol** that enables loading of **advanced, multi-dimensional life-science imaging formats** via [BioIO](https://github.com/bioio-devs/bioio).

If you work with microscopy, high-content screening, or proprietary instrument formats, this is the loader you need.

## Supported formats

- `.czi` — Zeiss CZI
- `.lif` — Leica LIF
- `.nd2` — Nikon ND2
- `.tif`, `.tiff`, `.ome.tif` — TIFF via BioIO (for TIFF-only use, `pixel-patrol-loader-tifffile` is lighter)
- `.jpg`, `.jpeg`, `.png`, `.bmp` — common raster formats

## Installation

```bash
uv pip install pixel-patrol-loader-bio
```

Or as part of the full stack:

```bash
uv pip install pixel-patrol
```

## Getting Started

Please see the `pixel-patrol` documentation for usage instructions.  
https://github.com/ida-mdc/pixel-patrol/
