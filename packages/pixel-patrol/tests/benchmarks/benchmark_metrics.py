"""
Metric computation benchmark - cost per processor group.

Uses a fixed in-memory plane as a single leaf chunk; no slicing involved.

Run: python benchmark_metrics.py
"""
import time
from pathlib import Path
from typing import Dict

import numpy as np

from pixel_patrol_base.core.record import record_from
from pixel_patrol_image.plugins.processors.raster_image_processor import (
    CompressionMetricsProcessor,
    QualityMetricsProcessor,
)
from pixel_patrol_base.plugins.processors.raster_processor import (
    BasicMetricsProcessor,
    HistogramProcessor,
)

WARMUP      = 3
MEASURE     = 10
PLANE_SHAPE = (512, 512)
OUT_DIR     = Path(__file__).parent / "results"

_PROCESSOR_GROUPS = [
    BasicMetricsProcessor(),
    QualityMetricsProcessor(),
    HistogramProcessor(),
    CompressionMetricsProcessor(),
]


def _prepare():
    plane = np.random.RandomState(42).randint(0, 65536, size=PLANE_SHAPE, dtype=np.uint16)
    arr   = plane.astype(np.float32)
    return record_from(arr, {"dim_order": "YX"})


def _time_processor(proc, record) -> tuple[float, float]:
    for _ in range(WARMUP):
        proc.run_chunk(record)
    samples = []
    for _ in range(MEASURE):
        t0 = time.perf_counter()
        proc.run_chunk(record)
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(samples)), float(np.std(samples))


def _time_all(procs, record) -> tuple[float, float]:
    for _ in range(WARMUP):
        for proc in procs:
            proc.run_chunk(record)
    samples = []
    for _ in range(MEASURE):
        t0 = time.perf_counter()
        for proc in procs:
            proc.run_chunk(record)
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(samples)), float(np.std(samples))


def run() -> Dict[str, tuple[float, float]]:
    record = _prepare()
    print(f"Chunk {PLANE_SHAPE}")

    groups: Dict[str, tuple[float, float]] = {}
    print(f"\n{'group':<28}  {'mean ms':>8}  {'std ms':>8}")
    for proc in _PROCESSOR_GROUPS:
        label = proc.NAME.replace("raster-", "")
        mean, std = _time_processor(proc, record)
        groups[label] = (mean, std)
        print(f"  {label:<26}  {mean:>8.2f}  {std:>8.2f}")

    mean, std = _time_all(_PROCESSOR_GROUPS, record)
    groups["all"] = (mean, std)
    print(f"  {'all':<26}  {mean:>8.2f}  {std:>8.2f}")

    return groups


def plot(groups: Dict[str, tuple[float, float]]) -> None:
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(groups) * 0.5)))
    fig.suptitle(f"Metric cost - chunk {PLANE_SHAPE}")

    ordered = sorted((k for k in groups if k != "all"), key=lambda k: groups[k][0])
    ordered = ["all"] + ordered
    means  = [groups[k][0] for k in ordered]
    stds   = [groups[k][1] for k in ordered]
    colors = ["tab:orange" if k == "all" else "tab:blue" for k in ordered]
    bars = ax.barh(ordered, means, xerr=stds, color=colors,
                   error_kw={"capsize": 4, "elinewidth": 1.2, "ecolor": "black"})
    ax.bar_label(bars, labels=[f"{m:.2f}" for m in means], padding=6, fontsize=8)
    ax.set_xlabel("ms per call")
    ax.set_title("Raster processors")
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "metric_results.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot → {out}")


if __name__ == "__main__":
    groups = run()
    plot(groups)
