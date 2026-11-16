import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LUMA_WEIGHTS = (0.299, 0.587, 0.114)
LIT_FRACTION = 0.05
EIGHT_BIT = 255.0
SQRT3 = np.float32(math.sqrt(3))

SUM_KEYS = ("n", "n_lit", "luma", "luma2", "sat", "hue_x", "hue_y", "chroma")


def value_ceiling(dtype: np.dtype, chunk: np.ndarray) -> float:
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0 if float(np.nanmax(chunk)) <= 1.0 else EIGHT_BIT


def colour_planes(chunk: np.ndarray, dim_order: str, colour_axis: str) -> Optional[np.ndarray]:
    """The chunk as (planes, Y, X, 3), or None when it does not hold all three colour samples."""
    if colour_axis not in dim_order or "Y" not in dim_order or "X" not in dim_order:
        return None
    axes = [dim_order.index(a) for a in ("Y", "X", colour_axis)]
    lead = [i for i in range(chunk.ndim) if i not in axes]
    moved = chunk.transpose(lead + axes)
    if moved.shape[-1] < 3:
        return None
    return moved.reshape(-1, *moved.shape[-3:-1], moved.shape[-1])[..., :3]


def empty_sums() -> Dict:
    return {k: 0.0 for k in SUM_KEYS}


def luma_of(raw: Sequence[np.ndarray]) -> np.ndarray:
    luma = np.multiply(raw[0], LUMA_WEIGHTS[0], dtype=np.float32)
    for channel, weight in zip(raw[1:], LUMA_WEIGHTS[1:]):
        luma += np.multiply(channel, weight, dtype=np.float32)
    return luma


def lit_saturation_sum(raw: Sequence[np.ndarray], lit: np.ndarray) -> float:
    """Sum of HSV saturation over the lit pixels, worked out in the data type of the image so that only one float plane exists."""
    peak = np.maximum(np.maximum(raw[0], raw[1]), raw[2])
    spread = peak - np.minimum(np.minimum(raw[0], raw[1]), raw[2])
    saturation = np.zeros(peak.shape, dtype=np.float32)
    np.divide(spread, peak, out=saturation, where=lit & (peak > 0))
    return float(saturation.sum(dtype=np.float64))


def hue_sums(raw: Sequence[np.ndarray]) -> Tuple[float, float, float]:
    """Sums of the colour direction (x, y) and of its length over a plane, three float planes at a time."""
    hue_x = raw[0].astype(np.float32)
    hue_x *= 2
    hue_x -= raw[1]
    hue_x -= raw[2]
    hue_y = raw[1].astype(np.float32)
    hue_y -= raw[2]
    hue_y *= SQRT3
    chroma = np.hypot(hue_x, hue_y)
    return tuple(float(plane.sum(dtype=np.float64)) for plane in (hue_x, hue_y, chroma))


def add_plane(sums: Dict, plane: np.ndarray, ceiling: float) -> None:
    """Add one (Y, X, 3) colour plane to the running sums.

    The chunk is whatever the pipeline's memory budget allows, so a plane is processed whole, like the intensity
    processor does. Luma, saturation and hue are worked out one after the other from the raw samples, so the peak
    stays near four times the raw plane for 8-bit data, inside the pipeline's eightfold allowance for processors.
    """
    raw = [plane[..., i] for i in range(3)]
    luma = luma_of(raw)
    lit = luma >= LIT_FRACTION * ceiling
    sums["n"] += luma.size
    sums["n_lit"] += int(lit.sum())
    sums["luma"] += float(luma.sum(dtype=np.float64))
    sums["luma2"] += float(np.einsum("ij,ij->", luma, luma, dtype=np.float64))
    del luma
    sums["sat"] += lit_saturation_sum(raw, lit)
    del lit
    for key, total in zip(("hue_x", "hue_y", "chroma"), hue_sums(raw)):
        sums[key] += total


def chunk_sums(planes: np.ndarray, ceiling: float) -> Dict:
    sums = empty_sums()
    for plane in planes:
        add_plane(sums, plane, ceiling)
    return sums


def merge_sums(parts: Sequence[Dict]) -> Dict:
    merged = empty_sums()
    for part in parts:
        for key in SUM_KEYS:
            merged[key] += part[key]
    return merged


def ratio(top: float, bottom: float) -> Optional[float]:
    return top / bottom if bottom > 0 else None


def hue_coherence(sums: Dict) -> Optional[float]:
    """Length of the average colour direction over the average colour strength: 1 when all coloured pixels share a hue, 0 when the hues cancel."""
    return ratio(math.hypot(sums["hue_x"], sums["hue_y"]), sums["chroma"])


def colour_metrics(sums: Dict) -> Dict[str, Optional[float]]:
    n = sums["n"]
    if n == 0:
        return {}
    luma_mean = sums["luma"] / n
    return {
        "rgb_luma_std": math.sqrt(max(0.0, sums["luma2"] / n - luma_mean ** 2)),
        "rgb_saturation_mean": sums["sat"] / sums["n_lit"] if sums["n_lit"] else None,
        "rgb_hue_coherence": hue_coherence(sums),
    }


def combine(parts: List[Dict]) -> Dict[str, Optional[float]]:
    return colour_metrics(merge_sums(parts)) if parts else {}
