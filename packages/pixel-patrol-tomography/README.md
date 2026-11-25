# pixel-patrol-tomography

PixelPatrol extension for tomography data in MRC format - e.g. soft X-ray tilt series (`.st`) and their
reconstructions (`.rec`, `.mrc`). Based on the the [`mrcfile`](https://github.com/ccpem/mrcfile).

* supported file extensions: `mrc`, `mrcs`, `st`, `ali`, `preali`, `rec`, `map`
* dimension orders: either `AYX` for image stacks/tilt series, `ZYX` for volumes or `YX` for single images.
* pixel sizes are by convention in Angstrom
* processor `axis-profile`: adds `x_profile_mean` and `y_profile_mean` (mean intensity per column/row, averaged over
  all other dimensions) to the global row of every image, e.g. to spot dead sensor rows or columns at the borders.
  It works on any image with X and Y, not only MRC files.
* not supported:
  * complex data (real+imaginary parts)
  * non-standard axis mapping (array order is different to real order)
  * 4D volumes