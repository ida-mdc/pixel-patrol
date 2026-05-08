"""
Shared NumPy kernels for joint statistics between two co-registered channel arrays.

Used by ChannelColocalizationProcessor and (in future) a bleed-through processor.
All functions operate on pre-tiled arrays of shape
(n_planes, n_tiles_y, n_tiles_x, tile_h, tile_w) and reduce over axes (-2, -1).
"""

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Joint statistics — compute once, derive multiple metrics from the result
# ---------------------------------------------------------------------------

class JointTileStats:
    """Per-tile joint statistics for a channel pair.

    All fields have shape (n_planes, n_tiles_y, n_tiles_x).
    """
    __slots__ = ("mu1", "mu2", "std1", "std2", "cov")

    def __init__(self, mu1, mu2, std1, std2, cov):
        self.mu1 = mu1
        self.mu2 = mu2
        self.std1 = std1
        self.std2 = std2
        self.cov = cov


def joint_stats_tile(
    c1: np.ndarray,
    c2: np.ndarray,
    axes: Tuple[int, int] = (-2, -1),
) -> JointTileStats:
    """Per-tile means, stds, and covariance for two channel arrays.

    c1, c2 must have identical shape (..., tile_h, tile_w).
    """
    with np.errstate(all="ignore"):
        mu1 = np.nanmean(c1, axis=axes)
        mu2 = np.nanmean(c2, axis=axes)
        d1 = c1 - np.nanmean(c1, axis=axes, keepdims=True)
        d2 = c2 - np.nanmean(c2, axis=axes, keepdims=True)
        cov = np.nanmean(d1 * d2, axis=axes)
        std1 = np.nanstd(c1, axis=axes)
        std2 = np.nanstd(c2, axis=axes)
    return JointTileStats(mu1, mu2, std1, std2, cov)


# ---------------------------------------------------------------------------
# Pearson's r — primary co-localisation metric
# ---------------------------------------------------------------------------

def pearson_r_from_stats(stats: JointTileStats) -> np.ndarray:
    """Pearson's correlation coefficient from pre-computed joint statistics.

    Invariant to both additive offsets and multiplicative scaling of either channel
    independently: r(aX + b, cY + d) = sign(ac) · r(X, Y).

    Returns values in [−1, 1].  NaN for tiles where either channel is flat (std = 0).

    Ref: Manders, E. M. M., Verbeek, F. J., & Aten, J. A. (1993). Measurement of
    co-localization of objects in dual-colour confocal images. Journal of Microscopy,
    169(3), 375–382.
    Also: Pearson, K. (1895). Notes on regression and inheritance in the case of two
    parents. Proceedings of the Royal Society of London, 58, 240–242.
    """
    with np.errstate(all="ignore"):
        denom = stats.std1 * stats.std2
        return np.where(denom > 0, stats.cov / denom, np.nan)


# ---------------------------------------------------------------------------
# SSIM — structural similarity decomposed into three components
# ---------------------------------------------------------------------------

def ssim_from_stats(
    stats: JointTileStats,
    dynamic_range: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SSIM luminance, contrast, structure, and composite score per tile.

    dynamic_range: per-tile combined dynamic range, shape (n_planes, n_ty, n_tx).
    Stability constants C1 and C2 are set relative to dynamic_range following
    Wang et al. (2004): C1 = (0.01·L)², C2 = (0.03·L)².

    Returns: (luminance, contrast, structure, ssim)
    All outputs have the same shape as the fields of stats.

    Interpretation for cross-channel comparison:
      structure  ≈ Pearson's r (co-localisation; invariant to offset and scale)
      contrast   — whether channels have proportional intensity spread
      luminance  — whether channels have similar mean intensity; NOT a quality
                   indicator for fluorescence since different fluorophores have
                   inherently different brightness (more relevant for bleed-through)
      ssim       — composite; dominated by structure for typical fluorescence images

    Ref: Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    Transactions on Image Processing, 13(4), 600–612.
    """
    L = np.maximum(dynamic_range, 1e-8)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    C3 = C2 / 2.0

    mu1, mu2 = stats.mu1, stats.mu2
    std1, std2, cov = stats.std1, stats.std2, stats.cov
    var1, var2 = std1 ** 2, std2 ** 2

    with np.errstate(all="ignore"):
        luminance = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1)
        contrast  = (2 * std1 * std2 + C2) / (var1 + var2 + C2)
        structure = (cov + C3) / (std1 * std2 + C3)
        ssim      = luminance * contrast * structure

    return luminance, contrast, structure, ssim
