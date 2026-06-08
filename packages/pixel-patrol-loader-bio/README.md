# PixelPatrol Bio-Image Loader Extension (`pixel-patrol-loader-bio`)

This extension provides three loaders for multi-dimensional life-science imaging formats.

## Loaders

### `bioio` - BioIO (CZI, LIF, ND2, TIFF, rasters)
Uses the [BioIO](https://github.com/bioio-devs/bioio) library for broad format support.
- `.czi` - Zeiss CZI
- `.lif` - Leica LIF
- `.nd2` - Nikon ND2
- `.tif`, `.tiff`, `.ome.tif` - TIFF via BioIO
- `.jpg`, `.jpeg`, `.png`, `.bmp` - common raster formats
- `.zarr`, `.ome.zarr` - zarr stores via BioIO

### `tifffile` - TiffFile (TIFF / OME-TIFF)
Direct TIFF loading via [tifffile](https://github.com/cgohlke/tifffile) with lazy Zarr-backed access.
- `.tif`, `.tiff`, `.ome.tif` - including multi-series OME-TIFF
- Lighter than `bioio` for TIFF-only datasets

### `zarr` - Zarr (zarr / OME-Zarr)
Native zarr store loading with OME-NGFF axis support.
- `.zarr`, `.ome.zarr`

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
