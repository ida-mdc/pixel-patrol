# pixel-patrol-tomography

PixelPatrol extension for tomography data in MRC format - e.g. soft X-ray tilt series (`.st`) and their
reconstructions (`.rec`, `.mrc`). Based on the the [`mrcfile`](https://github.com/ccpem/mrcfile).

* supported file extensions: `mrc`, `mrcs`, `st`, `ali`, `preali`, `rec`, `map`
* dimension orders: either `AYX` for image stacks/tilt series, `ZYX` for volumes or `YX` for single images.
* pixel sizes are by convention in Angstrom
* not supported:
  * complex data (real+imaginary parts)
  * non-standard axis mapping (array order is different to real order)
  * 4D volumes