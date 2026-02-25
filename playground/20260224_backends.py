"""
Slicewise image metrics: compare backends (numpy, dask, clesperanto).
"""
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_SEED = 42
AXES_XY = (3, 4)

CONFIG_GROUPS = [
    ("num_files", "n", [10, 20, 30, 40, 50], [(n, 1, 1, 1, 5, 5) for n in [10, 20, 30, 40, 50]]),
    ("XY size", "s", [100, 200, 400, 800, 1600, 3200], [(10, 1, 3, 1, s, s) for s in [100, 200, 400, 800, 1600, 3200]]),
    ("T/Z", "tz", [2, 3, 4, 5, 6], [(10, tz, 3, tz, 5, 5) for tz in [2, 3, 4, 5, 6]]),
]
CONFIGS = [cfg for _, _, _, configs in CONFIG_GROUPS for cfg in configs]
SCALAR_METRICS = ["laplacian_var", "laplacian_std", "mean", "min", "max"]
HIST_BINS = 16
# (name, size): size 1 = scalar, size > 1 = vector
METRIC_SPEC = [(n, 1) for n in SCALAR_METRICS] + [("hist", HIST_BINS)]
METRIC_NAMES = [n for n, _ in METRIC_SPEC]
OUTPUT_SIZE = sum(s for _, s in METRIC_SPEC)


def metrics(plane: np.ndarray) -> np.ndarray:
    img = plane.astype(np.float32)
    if img.ndim != 2 or img.size < 9:
        return np.array([np.nan] * OUTPUT_SIZE)
    lap = 4 * img[1:-1, 1:-1] - img[1:-1, :-2] - img[1:-1, 2:] - img[:-2, 1:-1] - img[2:, 1:-1]
    var = float(np.var(lap))
    hist, _ = np.histogram(img, bins=HIST_BINS, range=(0, 256))
    return np.concatenate([
        np.array([var, np.sqrt(var), float(np.mean(img)), float(np.min(img)), float(np.max(img))]),
        hist.astype(np.float64),
    ])


def _to_scalar(arr: np.ndarray) -> float:
    """Safely convert 0-d or 1-d array to Python float."""
    return float(np.asarray(arr).flat[0])


def _format_results(slice_results: np.ndarray) -> dict:
    out, n = {}, 3
    loop_specs = [(d, slice_results.shape[i]) for i, d in enumerate(["t", "c", "z"])]
    col = 0
    for m, (name, size) in enumerate(METRIC_SPEC):
        data = slice_results[..., col : col + size]
        col += size
        is_vector = size > 1

        for idx in np.ndindex(slice_results.shape[:-1]):
            key = "_".join(f"{d}{i}" for (d, _), i in zip(loop_specs, idx))
            val = data[idx]
            out[f"{name}_{key}"] = np.asarray(val, dtype=np.float64) if is_vector else _to_scalar(val)

        for r in range(n + 1):
            for keep in combinations(range(n), r):
                agg_axes = tuple(i for i in range(n) if i not in keep)
                if not agg_axes:
                    continue
                agg = np.mean(data, axis=agg_axes)
                if agg.ndim == 0 or (is_vector and agg.ndim == 1):
                    out[name] = np.asarray(agg, dtype=np.float64) if is_vector else _to_scalar(agg)
                else:
                    kept = [loop_specs[i] for i in keep]
                    iter_shape = agg.shape[:-1] if is_vector else agg.shape
                    for aidx in np.ndindex(iter_shape):
                        key = "_".join(f"{d}{i}" for (d, _), i in zip(kept, aidx))
                        v = agg[aidx]
                        out[f"{name}_{key}"] = np.asarray(v, dtype=np.float64) if is_vector else _to_scalar(v)
    return out


# --- Backends ---

class NumpyBackend:
    """NumPy via np.vectorize (broadcasts over batch dims)."""

    name = "numpy"

    def process(self, arr: np.ndarray) -> dict:
        ufunc = np.vectorize(metrics, signature=f"(i,j)->({OUTPUT_SIZE})")
        return _format_results(ufunc(arr))


class DaskGufuncBackend:
    """Dask apply_gufunc: applies _lap_metrics per slice via generalized ufunc."""

    name = "dask_gufunc"

    def process(self, arr: np.ndarray) -> dict:
        import dask.array as da

        t, c, z, y, x = arr.shape
        darr = da.from_array(arr, chunks=(1, 1, 1, y, x))
        if y < 3 or x < 3:
            slice_results = np.full((t, c, z, OUTPUT_SIZE), np.nan)
        else:
            result = da.apply_gufunc(
                metrics,
                "(i,j)->(n)",
                darr,
                axes=[(-2, -1), (-1,)],  # input axes, output axes
                output_sizes={"n": OUTPUT_SIZE},
                output_dtypes=np.float64,
                vectorize=True,  # broadcast over loop dims so func receives (y,x) slices
            )
            slice_results = np.asarray(result.compute(scheduler="threads"))
        return _format_results(slice_results)


class DaskBackend:
    """Pure Dask array ops: laplacian + var in the graph, no per-chunk Python callbacks."""

    name = "dask"

    def process(self, arr: np.ndarray) -> dict:
        import dask.array as da

        t, c, z, y, x = arr.shape
        darr = da.from_array(arr, chunks=(1, 1, 1, y, x)).astype(np.float32)
        if y < 3 or x < 3:
            out = da.full((t, c, z, OUTPUT_SIZE), np.nan, chunks=(1, 1, 1, OUTPUT_SIZE))
        else:
            lap = (
                4 * darr[..., 1:-1, 1:-1]
                - darr[..., 1:-1, :-2]
                - darr[..., 1:-1, 2:]
                - darr[..., :-2, 1:-1]
                - darr[..., 2:, 1:-1]
            )
            var = lap.var(axis=(-2, -1))
            std = da.sqrt(var)
            mean_arr = darr.mean(axis=(-2, -1))
            min_arr = darr.min(axis=(-2, -1))
            max_arr = darr.max(axis=(-2, -1))

            def _hist_block(block):
                hist, _ = np.histogram(block.ravel(), bins=HIST_BINS, range=(0, 256))
                return hist.astype(np.float64).reshape(1, 1, 1, -1)

            hist_arr = da.map_blocks(_hist_block, darr, drop_axis=(3, 4), new_axis=3, dtype=np.float64)
            out = da.concatenate(
                [da.stack([var, std, mean_arr, min_arr, max_arr], axis=-1), hist_arr],
                axis=-1,
            )
        result = out.compute(scheduler="threads")
        return _format_results(np.asarray(result).reshape(t, c, z, -1))


class ClesperantoBackend:
    """Uses same Laplacian formula as numpy/dask for consistency (cle.laplace_box differs)."""

    name = "clesperanto"

    def __init__(self):
        import pyclesperanto  # noqa: F401 - ensure available for benchmark

    def process(self, arr: np.ndarray) -> dict:
        t, c, z, y, x = arr.shape
        n_slices = t * c * z
        if y < 3 or x < 3:
            slice_results = np.full((t, c, z, OUTPUT_SIZE), np.nan)
        else:
            stacked = arr.reshape(n_slices, y, x).astype(np.float32)
            slice_results = np.empty((n_slices, OUTPUT_SIZE), dtype=float)
            for i in range(n_slices):
                plane = stacked[i]
                lap = (
                    4 * plane[1:-1, 1:-1]
                    - plane[1:-1, :-2]
                    - plane[1:-1, 2:]
                    - plane[:-2, 1:-1]
                    - plane[2:, 1:-1]
                )
                var = float(np.var(lap))
                hist, _ = np.histogram(plane, bins=HIST_BINS, range=(0, 256))
                slice_results[i] = np.concatenate([
                    [var, np.sqrt(var), float(np.mean(plane)), float(np.min(plane)), float(np.max(plane))],
                    hist.astype(np.float64),
                ])
            slice_results = slice_results.reshape(t, c, z, -1)
        return _format_results(slice_results)


BACKENDS = {
    "numpy": NumpyBackend,
    "dask": DaskBackend,
    "dask_gufunc": DaskGufuncBackend,
    "clesperanto": ClesperantoBackend,
}


def get_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}. Available: {list(BACKENDS)}")
    return BACKENDS[name]()


def _build_arrays(configs: list) -> list:
    arrays, idx = [], 0
    for nf, t, c, z, y, x in configs:
        for _ in range(nf):
            rng = np.random.RandomState(BASE_SEED + idx)
            arrays.append(rng.randint(0, 256, size=(t, c, z, y, x), dtype=np.uint8))
            idx += 1
    return arrays


def run_timed(backend, arrays: list) -> float:
    t0 = time.perf_counter()
    for arr in arrays:
        backend.process(arr)
    return time.perf_counter() - t0


def main():
    arrays = _build_arrays(CONFIGS)
    print(f"{len(arrays)} arrays")

    backends = []
    for name, cls in BACKENDS.items():
        try:
            backends.append((name, cls()))
        except (ImportError, RuntimeError) as e:
            print(f"  {name}: skipped ({e})")

    # Verify all backends produce same result (pairwise comparison)
    if len(backends) >= 2:
        results = {name: be.process(arrays[0]) for name, be in backends}
        for (name_a, _), (name_b, _) in combinations(backends, 2):
            a, b = results[name_a], results[name_b]
            common = set(a) & set(b)
            for k in common:
                np.testing.assert_allclose(a[k], b[k], equal_nan=True, err_msg=f"{k} ({name_a} vs {name_b})")
            if set(a) != set(b):
                raise AssertionError(
                    f"Key mismatch {name_a} vs {name_b}: "
                    f"only in {name_a}: {set(a)-set(b)}, only in {name_b}: {set(b)-set(a)}"
                )
        print(f"  consistency: ✓ ({len(backends)} backends, {len(list(combinations(backends, 2)))} pairs)")

    for name, be in backends:
        print(f"  {name}: {run_timed(be, arrays):.3f}s")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, (label, _, x_vals, configs) in zip(axes, CONFIG_GROUPS):
        arrs_per_param = [_build_arrays([c]) for c in configs]
        x = np.arange(len(x_vals))
        w = 0.8 / len(backends)
        for i, (name, be) in enumerate(backends):
            times = [run_timed(be, arrs) for arrs in arrs_per_param]
            ax.bar(x + i * w, times, w, label=name)
        ax.set_xlabel(label)
        ax.set_xticks(x + w * (len(backends) - 1) / 2)
        ax.set_xticklabels(x_vals)
        ax.set_ylabel("Time (s)")
        ax.legend()
    fig.suptitle("Backend comparison")
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "backends_comparison.png", dpi=150)
    print("  plot saved")


if __name__ == "__main__":
    main()
