# PixelPatrol Bio-Image Loader Extension (`pixel-patrol-loader-bio`)

This is an extension for **PixelPatrol** that enables pre-validation for **advanced, multi-dimensional life-science imaging data**.

If you work with microscopy, high-content screening, or other complex bio-imaging formats, this is probably the plugin you need for your image data to be loaded.

## 🔬 Why You Need This Extension

The `pixel-patrol-loader-bio` extension adds essential compatibility by integrating BioIO and zarr formats.

## 🚀 Installation

### Recommended (Full) Installation

For the easiest start, we recommend installing the main `pixel-patrol` package. This automatically includes the base functionality (`pixel-patrol-base`), this bio-loader extension, and the basic image analysis plugins (`pixel-patrol-image`):

```bash
uv pip install pixel-patrol
```

### Minimal Installation (Loader Only)

If you only want this specific loader plugin and the core PixelPatrol functionality, you can install just this package. This is useful if you plan to manage other extensions yourself:

```bash
uv pip install pixel-patrol-loader-bio
```

## Getting Started

Please look at the documentation of `pixel-patrol` for usage instructions.  
https://github.com/ida-mdc/pixel-patrol/