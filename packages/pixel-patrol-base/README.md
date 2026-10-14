### PixelPatrol Base (pixel-patrol-base)

The core of PixelPatrol: API, CLI, processing framework, viewer, and the plugin system other packages (like pixel-patrol-loader-bio) build on. It includes file-system statistics, thumbnails, and basic image statistics and quality metrics.

#### Who should install this?

Most users should install the full package, which includes base and the common loaders:

```
uv pip install pixel-patrol
```

Install base directly only if you are building an extension (loader, processor, or widget) or want a minimal setup:

```
uv pip install pixel-patrol-base
```

Documentation: https://pixelpatrol.app/docs/latest/
