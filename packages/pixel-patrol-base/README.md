### PixelPatrol Base (pixel-patrol-base)

The Core Engine and Framework for the PixelPatrol Ecosystem.

This package is the essential, non-optional foundation of PixelPatrol. It provides the command-line interface (CLI), the interactive report structure, and the plugin system that allows other packages (like pixel-patrol-loader-bio or pixel-patrol-image) to integrate seamlessly.

It also includes the Processing Framework: It defines the official system for reading files, processing data, and generating the final feature sets.
It includes more universal plug-ins such as basic file-system statistics, as well as plug-ins that don't require additional library installation such as thumbnail generation and standard statistical analysis.

#### 👥 Who Should Install This?

Most end-users should install the main, consolidated package:
```
uv pip install pixel-patrol
```

The main package automatically installs pixel-patrol-base along with the most common loaders and analysis tools.

You should install pixel-patrol-base directly only if you are:

* A Developer: You are building your own custom extension (a new loader, processor, or widget) and need the core API.
* A Power User: You want a minimal installation and plan to manually install other official extensions.

##### 🚀 Installation (Minimal)
To install just the core framework:
```
uv pip install pixel-patrol-base
```

📖 Usage and Documentation  
For instructions please refer to the main documentation:  
https://github.com/ida-mdc/pixel-patrol/