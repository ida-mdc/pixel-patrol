/**
 * Exhibit: shared canvas rendering of the images behind the metrics. Lives in
 * viewer/src (bundled) so the point inspector and any plugin draw them
 * identically rather than each carrying its own decode.
 */

export const SPRITE = 64;

/** Draw RGBA (len ≥ SPRITE²·4) or grayscale (len SPRITE²) thumbnail bytes at 0,0. */
export function drawThumbnailRGBA(ctx2d, pixels) {
  const iData = ctx2d.createImageData(SPRITE, SPRITE);
  const d = iData.data;
  const isRGBA = pixels.length >= SPRITE * SPRITE * 4;
  for (let i = 0; i < SPRITE * SPRITE; i++) {
    const o = i * 4;
    if (isRGBA) {
      const a = pixels[i * 4 + 3] ?? 255;
      if (a < 128) { d[o] = d[o + 1] = d[o + 2] = d[o + 3] = 0; }
      else { d[o] = pixels[i * 4]; d[o + 1] = pixels[i * 4 + 1]; d[o + 2] = pixels[i * 4 + 2]; d[o + 3] = 255; }
    } else {
      const v = pixels[i] ?? 0;
      d[o] = d[o + 1] = d[o + 2] = v; d[o + 3] = 255;
    }
  }
  ctx2d.putImageData(iData, 0, 0);
}
