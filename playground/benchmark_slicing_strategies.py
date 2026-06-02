"""
Standalone slicing strategy benchmark — no pixel-patrol imports.

Compares approaches to applying a 2D metric across all planes of a 5D
(T, C, Z, H, W) array.  The metric (std over XY) requires a real spatial
reduction, so per-plane Python dispatch overhead is visible.

Strategies
----------
loop              explicit np.ndindex loop over all (t, c, z) combinations
np.vectorize      np.vectorize with gufunc signature "(h,w)->()"
np.frompyfunc     np.frompyfunc — creates a real numpy ufunc (not just a wrapper)
np.apply_over_axes np.apply_over_axes reducing the last two axes
da.apply_gufunc   da.apply_gufunc with vectorize=True
da.map_blocks     da.map_blocks with numpy reduction inside each block
numpy_direct      np.std(arr, axis=(-2,-1)) — no per-plane Python at all

Run: python benchmark_slicing_strategies.py
"""
import time
from pathlib import Path
from typing import Callable, Dict, List

import dask.array as da
import numpy as np

WARMUP  = 2
MEASURE = 8
OUT_DIR = Path(__file__).parent

# (name, t, c, z, h, w)
CONFIGS: List[tuple] = [
    # Vary plane count (T×C×Z), fixed 256×256 spatial
    ("tcz_1",    1, 1,  1, 256, 256),
    ("tcz_4",    2, 2,  1, 256, 256),
    ("tcz_24",   2, 3,  4, 256, 256),
    ("tcz_96",   4, 4,  6, 256, 256),
    ("tcz_240",  4, 5, 12, 256, 256),
    # Vary spatial size, fixed T=2 C=3 Z=4
    ("s64",      2, 3,  4,  64,  64),
    ("s128",     2, 3,  4, 128, 128),
    ("s256",     2, 3,  4, 256, 256),
    ("s512",     2, 3,  4, 512, 512),
    ("s1024",    2, 3,  4, 1024, 1024),
]


def _metric_2d(plane: np.ndarray) -> float:
    return float(np.std(plane))


def strategy_loop(arr: np.ndarray) -> np.ndarray:
    out = np.empty(arr.shape[:-2])
    for idx in np.ndindex(arr.shape[:-2]):
        out[idx] = _metric_2d(arr[idx])
    return out


def strategy_vectorize(arr: np.ndarray) -> np.ndarray:
    return np.vectorize(_metric_2d, signature="(h,w)->()")(arr)


def strategy_frompyfunc(arr: np.ndarray) -> np.ndarray:
    # np.frompyfunc creates a genuine numpy ufunc object from a Python callable.
    # It broadcasts over array elements, so we hand it an object array whose
    # elements are the individual 2D planes.
    batch_shape = arr.shape[:-2]
    obj = np.empty(batch_shape, dtype=object)
    for idx in np.ndindex(batch_shape):
        obj[idx] = arr[idx]
    ufunc = np.frompyfunc(_metric_2d, 1, 1)
    return np.array(ufunc(obj), dtype=float)


def strategy_apply_over_axes(arr: np.ndarray) -> np.ndarray:
    return np.apply_over_axes(np.std, arr, [-2, -1]).reshape(arr.shape[:-2])


def strategy_da_apply_gufunc(arr: np.ndarray) -> np.ndarray:
    chunks = (*[1] * (arr.ndim - 2), arr.shape[-2], arr.shape[-1])
    darr = da.from_array(arr, chunks=chunks)
    result = da.apply_gufunc(
        _metric_2d,
        "(h,w)->()",
        darr,
        axes=[(-2, -1), ()],
        output_dtypes=float,
        vectorize=True,
    )
    return np.asarray(result.compute(scheduler="synchronous"))


def strategy_da_map_blocks(arr: np.ndarray) -> np.ndarray:
    chunks = (*[1] * (arr.ndim - 2), arr.shape[-2], arr.shape[-1])
    darr = da.from_array(arr, chunks=chunks)
    result = da.map_blocks(
        lambda b: np.std(b, axis=(-2, -1), keepdims=True),
        darr,
        dtype=float,
        chunks=(*[1] * (arr.ndim - 2), 1, 1),
    )
    return np.asarray(result.compute(scheduler="synchronous")).reshape(arr.shape[:-2])


def strategy_numpy_direct(arr: np.ndarray) -> np.ndarray:
    return np.std(arr, axis=(-2, -1))


def strategy_loop_batched(arr: np.ndarray) -> np.ndarray:
    # Process K planes per numpy call to amortise Python loop overhead
    # while keeping the working set cache-friendly.
    # K chosen so each batch is ~4 MB (adjust to taste).
    plane_bytes = arr.shape[-2] * arr.shape[-1] * arr.itemsize
    k = max(1, (4 * 1024 * 1024) // plane_bytes)
    flat = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    out = np.empty(flat.shape[0])
    for start in range(0, flat.shape[0], k):
        batch = flat[start : start + k]
        out[start : start + k] = np.std(batch, axis=(-2, -1))
    return out.reshape(arr.shape[:-2])


STRATEGIES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "loop":               strategy_loop,
    "loop_batched":       strategy_loop_batched,
    "numpy_direct":       strategy_numpy_direct,
    "np.vectorize":       strategy_vectorize,
    "np.frompyfunc":      strategy_frompyfunc,
    "np.apply_over_axes": strategy_apply_over_axes,
    "da.apply_gufunc":    strategy_da_apply_gufunc,
    "da.map_blocks":      strategy_da_map_blocks,
}


def _time(fn: Callable, arr: np.ndarray) -> tuple[float, float]:
    for _ in range(WARMUP):
        fn(arr)
    samples = []
    for _ in range(MEASURE):
        t0 = time.perf_counter()
        fn(arr)
        samples.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(samples)), float(np.std(samples))


def run() -> Dict[str, Dict[str, tuple[float, float]]]:
    results: Dict[str, Dict[str, tuple[float, float]]] = {s: {} for s in STRATEGIES}
    col_w = 16
    header = f"{'config':<12}" + "".join(f"  {s:>{col_w}}" for s in STRATEGIES)
    print(header)
    for name, t, c, z, h, w in CONFIGS:
        arr = np.random.RandomState(42).randint(0, 256, size=(t, c, z, h, w), dtype=np.uint8)
        row = f"{name:<12}"
        for sname, fn in STRATEGIES.items():
            mean, std = _time(fn, arr)
            results[sname][name] = (mean, std)
            cell = f"{mean:.1f}±{std:.1f}"
            row += f"  {cell:>{col_w}}"
        print(row)
    return results


def plot(results: Dict[str, Dict[str, tuple[float, float]]]) -> None:
    import matplotlib.pyplot as plt

    plane_configs = [n for n, *_ in CONFIGS if n.startswith("tcz")]
    size_configs  = [n for n, *_ in CONFIGS if n.startswith("s")]
    strategy_names = list(STRATEGIES)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, (ax_planes, ax_size) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Slicing strategy comparison — std over XY, 5D array (T, C, Z, H, W)")

    def _plot_group(ax, config_names, x_labels, xlabel):
        x = np.arange(len(config_names))
        w = 0.8 / len(strategy_names)
        for i, (sname, color) in enumerate(zip(strategy_names, colors)):
            means = [results[sname][c][0] for c in config_names]
            stds  = [results[sname][c][1] for c in config_names]
            ax.bar(x + i * w, means, w, yerr=stds, label=sname, color=color,
                   error_kw={"capsize": 3, "elinewidth": 1})
        ax.set_xticks(x + w * (len(strategy_names) - 1) / 2)
        ax.set_xticklabels(x_labels, rotation=15, ha="right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ms per call")
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    _plot_group(ax_planes, plane_configs,
                [f"T×C×Z={t*c*z}" for n, t, c, z, h, w in CONFIGS if n.startswith("tcz")],
                "Plane count (256×256 spatial)")
    _plot_group(ax_size, size_configs,
                [f"{h}×{w}" for n, t, c, z, h, w in CONFIGS if n.startswith("s")],
                "Spatial size (T=2 C=3 Z=4)")

    fig.tight_layout()
    out = OUT_DIR / "benchmark_slicing_strategies.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nPlot → {out}")


if __name__ == "__main__":
    results = run()
    plot(results)
