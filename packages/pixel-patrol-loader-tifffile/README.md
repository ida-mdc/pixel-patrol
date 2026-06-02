# PixelPatrol TIFF Loader Extension (`pixel-patrol-loader-tifffile`)

This is a lightweight extension for **PixelPatrol** that loads **TIFF and OME-TIFF** files using [tifffile](https://github.com/cgohlke/tifffile) directly — no bioio dependency required.

## Supported formats

- `.tif`, `.tiff` — standard TIFF (ImageJ, plain)
- `.ome.tif` — OME-TIFF (multi-series, physical pixel sizes, channel names)

## Installation

```bash
uv pip install pixel-patrol-loader-tifffile
```

Or as part of the full stack:

```bash
uv pip install pixel-patrol
```

## Getting Started

Please see the `pixel-patrol` documentation for usage instructions.  
https://github.com/ida-mdc/pixel-patrol/
