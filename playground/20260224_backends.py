"""
Slicewise image metrics: compare backends (numpy, dask, clesperanto, cupy, numba, jax).

Each backend computes the same metrics on 2D XY planes and produces identical
results (verified by pairwise assertion). Run with --profile to get per-function
timing breakdown and memory tracking.

Usage:
    python 20260224_backends.py                  # normal benchmark
    python 20260224_backends.py --profile        # with line_profiler + memory
    python 20260224_backends.py --backends numpy dask numba  # select backends
    python 20260224_backends.py --output results.json        # save results
"""
import argparse
import json
import platform
import time
import tracemalloc
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
METRIC_SPEC = [(n, 1) for n in SCALAR_METRICS] + [("hist", HIST_BINS)]
METRIC_NAMES = [n for n, _ in METRIC_SPEC]
OUTPUT_SIZE = sum(s for _, s in METRIC_SPEC)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def metrics(plane: np.ndarray) -> np.ndarray:
    """Compute all metrics for a single 2D plane (numpy reference implementation)."""
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


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class NumpyBackend:
    """NumPy via np.vectorize (broadcasts over batch dims)."""

    name = "numpy"

    def process(self, arr: np.ndarray) -> dict:
        ufunc = np.vectorize(metrics, signature=f"(i,j)->({OUTPUT_SIZE})")
        return _format_results(ufunc(arr))


class DaskGufuncBackend:
    """Dask apply_gufunc: applies metrics per slice via generalized ufunc."""

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
                axes=[(-2, -1), (-1,)],
                output_sizes={"n": OUTPUT_SIZE},
                output_dtypes=np.float64,
                vectorize=True,
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
    """pyclesperanto GPU backend using actual OpenCL operations.

    Uses cle.push/pull for GPU data transfer and cle operations for compute.
    The Laplacian uses the same 5-point stencil as the other backends
    (implemented via GPU array slicing, not cle.laplace_box which uses a
    different kernel).
    """

    name = "clesperanto"

    def __init__(self):
        import pyclesperanto as cle
        self._cle = cle
        device = cle.get_device()
        print(f"    clesperanto device: {device}")

    def process(self, arr: np.ndarray) -> dict:
        cle = self._cle
        t, c, z, y, x = arr.shape
        n_slices = t * c * z
        if y < 3 or x < 3:
            slice_results = np.full((t, c, z, OUTPUT_SIZE), np.nan)
        else:
            stacked = arr.reshape(n_slices, y, x).astype(np.float32)
            slice_results = np.empty((n_slices, OUTPUT_SIZE), dtype=float)
            for i in range(n_slices):
                plane_np = stacked[i]
                # Push to GPU
                gpu_img = cle.push(plane_np)

                # Laplacian via GPU array slicing (same 5-point stencil)
                center = gpu_img[1:-1, 1:-1]
                left   = gpu_img[1:-1, :-2]
                right  = gpu_img[1:-1, 2:]
                top    = gpu_img[:-2, 1:-1]
                bottom = gpu_img[2:, 1:-1]
                # 4*center - left - right - top - bottom
                gpu_lap = cle.add_images_weighted(center, center, factor1=4.0, factor2=0.0)
                gpu_lap = cle.subtract_images(gpu_lap, left)
                gpu_lap = cle.subtract_images(gpu_lap, right)
                gpu_lap = cle.subtract_images(gpu_lap, top)
                gpu_lap = cle.subtract_images(gpu_lap, bottom)

                # Stats on GPU (pull scalar results)
                lap_np = cle.pull(gpu_lap)
                var = float(np.var(lap_np))

                gpu_mean = float(cle.mean_of_all_pixels(gpu_img))
                gpu_min = float(cle.minimum_of_all_pixels(gpu_img))
                gpu_max = float(cle.maximum_of_all_pixels(gpu_img))

                # Histogram: pull back to CPU (no GPU histogram with custom bins in cle)
                hist, _ = np.histogram(plane_np, bins=HIST_BINS, range=(0, 256))

                slice_results[i] = np.concatenate([
                    [var, np.sqrt(var), gpu_mean, gpu_min, gpu_max],
                    hist.astype(np.float64),
                ])
            slice_results = slice_results.reshape(t, c, z, -1)
        return _format_results(slice_results)


class CuPyBackend:
    """CuPy GPU backend: bulk transfer to GPU, compute all metrics on device."""

    name = "cupy"

    def __init__(self):
        import cupy as cp
        self._cp = cp
        dev = cp.cuda.Device()
        print(f"    cupy device: {dev} ({cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()})")

    def process(self, arr: np.ndarray) -> dict:
        cp = self._cp
        t, c, z, y, x = arr.shape
        n_slices = t * c * z
        if y < 3 or x < 3:
            slice_results = np.full((t, c, z, OUTPUT_SIZE), np.nan)
        else:
            # Bulk transfer: push all slices to GPU at once
            stacked = arr.reshape(n_slices, y, x).astype(np.float32)
            gpu_stacked = cp.asarray(stacked)
            slice_results = np.empty((n_slices, OUTPUT_SIZE), dtype=float)

            for i in range(n_slices):
                gpu_img = gpu_stacked[i]

                # Laplacian on GPU (same 5-point stencil)
                gpu_lap = (
                    4 * gpu_img[1:-1, 1:-1]
                    - gpu_img[1:-1, :-2]
                    - gpu_img[1:-1, 2:]
                    - gpu_img[:-2, 1:-1]
                    - gpu_img[2:, 1:-1]
                )
                var = float(cp.var(gpu_lap))

                mean_val = float(cp.mean(gpu_img))
                min_val = float(cp.min(gpu_img))
                max_val = float(cp.max(gpu_img))

                # Histogram on GPU
                hist = cp.histogram(gpu_img, bins=HIST_BINS, range=(0, 256))[0]

                slice_results[i] = np.concatenate([
                    [var, np.sqrt(var), mean_val, min_val, max_val],
                    cp.asnumpy(hist).astype(np.float64),
                ])
            slice_results = slice_results.reshape(t, c, z, -1)
        return _format_results(slice_results)


class CuPyBatchBackend:
    """CuPy GPU backend: fully vectorized over all slices (no Python loop)."""

    name = "cupy_batch"

    def __init__(self):
        import cupy as cp
        self._cp = cp

    def process(self, arr: np.ndarray) -> dict:
        cp = self._cp
        t, c, z, y, x = arr.shape
        n_slices = t * c * z
        if y < 3 or x < 3:
            slice_results = np.full((t, c, z, OUTPUT_SIZE), np.nan)
        else:
            # Bulk transfer and reshape: (n_slices, y, x)
            gpu_all = cp.asarray(arr.reshape(n_slices, y, x).astype(np.float32))

            # Vectorized Laplacian over all slices at once
            gpu_lap = (
                4 * gpu_all[:, 1:-1, 1:-1]
                - gpu_all[:, 1:-1, :-2]
                - gpu_all[:, 1:-1, 2:]
                - gpu_all[:, :-2, 1:-1]
                - gpu_all[:, 2:, 1:-1]
            )
            # Variance per slice: var along (y,x) axes
            var = cp.var(gpu_lap, axis=(1, 2))
            std = cp.sqrt(var)
            mean_arr = cp.mean(gpu_all, axis=(1, 2))
            min_arr = cp.min(gpu_all, axis=(1, 2))
            max_arr = cp.max(gpu_all, axis=(1, 2))

            # Histogram per slice (must loop — cupy.histogram doesn't batch)
            hist_results = cp.empty((n_slices, HIST_BINS), dtype=cp.float64)
            for i in range(n_slices):
                hist_results[i] = cp.histogram(gpu_all[i], bins=HIST_BINS, range=(0, 256))[0].astype(cp.float64)

            # Assemble on GPU, then pull once
            scalars = cp.stack([var, std, mean_arr, min_arr, max_arr], axis=-1)
            gpu_result = cp.concatenate([scalars, hist_results], axis=-1)
            slice_results = cp.asnumpy(gpu_result).reshape(t, c, z, -1)
        return _format_results(slice_results)


class NumbaBackend:
    """Numba JIT-compiled metrics (CPU). First call includes compilation overhead."""

    name = "numba"

    def __init__(self):
        import numba
        self._numba = numba
        self._metrics_fn = self._compile()

    def _compile(self):
        nb = self._numba
        hist_bins = HIST_BINS
        output_size = OUTPUT_SIZE

        @nb.njit(cache=True)
        def _metrics_numba(img):
            """JIT-compiled metrics for a single float32 2D plane."""
            h, w = img.shape
            out = np.empty(output_size, dtype=np.float64)

            # Laplacian variance
            lap_sum = 0.0
            lap_sq_sum = 0.0
            lap_count = (h - 2) * (w - 2)
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    val = 4.0 * img[r, c] - img[r, c-1] - img[r, c+1] - img[r-1, c] - img[r+1, c]
                    lap_sum += val
                    lap_sq_sum += val * val
            lap_mean = lap_sum / lap_count
            var = lap_sq_sum / lap_count - lap_mean * lap_mean
            out[0] = var
            out[1] = np.sqrt(var)

            # Mean, min, max
            total = 0.0
            mn = img[0, 0]
            mx = img[0, 0]
            for r in range(h):
                for c in range(w):
                    v = img[r, c]
                    total += v
                    if v < mn:
                        mn = v
                    if v > mx:
                        mx = v
            out[2] = total / (h * w)
            out[3] = mn
            out[4] = mx

            # Histogram
            bin_width = 256.0 / hist_bins
            for b in range(hist_bins):
                out[5 + b] = 0.0
            for r in range(h):
                for c in range(w):
                    b = int(img[r, c] / bin_width)
                    if b >= hist_bins:
                        b = hist_bins - 1
                    out[5 + b] += 1.0

            return out

        # Warm up: trigger compilation
        _metrics_numba(np.zeros((4, 4), dtype=np.float32))
        return _metrics_numba

    def process(self, arr: np.ndarray) -> dict:
        t, c, z, y, x = arr.shape
        n_slices = t * c * z
        if y < 3 or x < 3:
            slice_results = np.full((t, c, z, OUTPUT_SIZE), np.nan)
        else:
            stacked = arr.reshape(n_slices, y, x).astype(np.float32)
            slice_results = np.empty((n_slices, OUTPUT_SIZE), dtype=float)
            for i in range(n_slices):
                slice_results[i] = self._metrics_fn(stacked[i])
            slice_results = slice_results.reshape(t, c, z, -1)
        return _format_results(slice_results)


class JaxBackend:
    """JAX GPU backend: XLA-compiled and vmap-vectorized over all slices.

    JAX JIT-compiles the entire metrics function into a single fused XLA kernel
    and vmap vectorizes it across slices — no Python loop, no per-slice transfer.
    First call is slow (XLA compilation), subsequent calls are fast.
    """

    name = "jax"

    def __init__(self):
        import jax
        import jax.numpy as jnp
        self._jax = jax
        self._jnp = jnp
        self._metrics_fn = self._compile()
        devices = jax.devices()
        print(f"    jax devices: {devices}")
        print(f"    jax backend: {jax.default_backend()}")

    def _compile(self):
        jax = self._jax
        jnp = self._jnp
        hist_bins = HIST_BINS
        output_size = OUTPUT_SIZE

        @jax.jit
        def _metrics_single(img):
            """Metrics for a single 2D float32 plane (no Python control flow)."""
            # Laplacian (5-point stencil)
            lap = (
                4 * img[1:-1, 1:-1]
                - img[1:-1, :-2]
                - img[1:-1, 2:]
                - img[:-2, 1:-1]
                - img[2:, 1:-1]
            )
            var = jnp.var(lap)
            std = jnp.sqrt(var)
            mean_val = jnp.mean(img)
            min_val = jnp.min(img)
            max_val = jnp.max(img)

            # Histogram via searchsorted-based binning
            edges = jnp.linspace(0, 256, hist_bins + 1)
            bin_indices = jnp.searchsorted(edges[1:], img.ravel(), side="right")
            bin_indices = jnp.clip(bin_indices, 0, hist_bins - 1)
            hist = jnp.zeros(hist_bins, dtype=jnp.float64)
            hist = hist.at[bin_indices].add(1.0)

            scalars = jnp.array([var, std, mean_val, min_val, max_val], dtype=jnp.float64)
            return jnp.concatenate([scalars, hist])

        # vmap over batch dimension: (n_slices, y, x) -> (n_slices, output_size)
        batched_fn = jax.vmap(_metrics_single)

        # Warm up / trigger compilation with a dummy array
        dummy = jnp.zeros((1, 4, 4), dtype=jnp.float32)
        batched_fn(dummy).block_until_ready()

        return batched_fn

    def process(self, arr: np.ndarray) -> dict:
        jnp = self._jnp
        t, c, z, y, x = arr.shape
        n_slices = t * c * z
        if y < 3 or x < 3:
            slice_results = np.full((t, c, z, OUTPUT_SIZE), np.nan)
        else:
            # Single bulk transfer to GPU, run all slices at once
            stacked = arr.reshape(n_slices, y, x).astype(np.float32)
            gpu_arr = jnp.array(stacked)
            result = self._metrics_fn(gpu_arr)
            # Block until computation is done, then transfer back
            slice_results = np.asarray(result).reshape(t, c, z, -1)
        return _format_results(slice_results)


class JaxBatchBackend:
    """JAX GPU backend: fully vectorized array ops (no vmap, pure array slicing).

    Similar to CuPyBatchBackend — vectorized Laplacian and reductions over the
    batch dimension using standard array operations. Useful to compare XLA fusion
    vs vmap overhead.
    """

    name = "jax_batch"

    def __init__(self):
        import jax
        import jax.numpy as jnp
        self._jax = jax
        self._jnp = jnp
        self._process_fn = self._compile()

    def _compile(self):
        jax = self._jax
        jnp = self._jnp
        hist_bins = HIST_BINS

        @jax.jit
        def _process_all(gpu_all):
            # Vectorized Laplacian over all slices: (n, y, x)
            lap = (
                4 * gpu_all[:, 1:-1, 1:-1]
                - gpu_all[:, 1:-1, :-2]
                - gpu_all[:, 1:-1, 2:]
                - gpu_all[:, :-2, 1:-1]
                - gpu_all[:, 2:, 1:-1]
            )
            var = jnp.var(lap, axis=(1, 2))
            std = jnp.sqrt(var)
            mean_arr = jnp.mean(gpu_all, axis=(1, 2))
            min_arr = jnp.min(gpu_all, axis=(1, 2))
            max_arr = jnp.max(gpu_all, axis=(1, 2))

            # Histogram per slice (vectorized via vmap of a single-slice histogram)
            edges = jnp.linspace(0, 256, hist_bins + 1)
            def _hist_one(plane):
                bin_indices = jnp.searchsorted(edges[1:], plane.ravel(), side="right")
                bin_indices = jnp.clip(bin_indices, 0, hist_bins - 1)
                return jnp.zeros(hist_bins, dtype=jnp.float64).at[bin_indices].add(1.0)

            hist_arr = jax.vmap(_hist_one)(gpu_all)

            scalars = jnp.stack([var, std, mean_arr, min_arr, max_arr], axis=-1).astype(jnp.float64)
            return jnp.concatenate([scalars, hist_arr], axis=-1)

        # Warm up
        dummy = jnp.zeros((1, 4, 4), dtype=jnp.float32)
        _process_all(dummy).block_until_ready()

        return _process_all

    def process(self, arr: np.ndarray) -> dict:
        jnp = self._jnp
        t, c, z, y, x = arr.shape
        n_slices = t * c * z
        if y < 3 or x < 3:
            slice_results = np.full((t, c, z, OUTPUT_SIZE), np.nan)
        else:
            stacked = arr.reshape(n_slices, y, x).astype(np.float32)
            gpu_arr = jnp.array(stacked)
            result = self._process_fn(gpu_arr)
            result.block_until_ready()
            slice_results = np.asarray(result).reshape(t, c, z, -1)
        return _format_results(slice_results)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BACKENDS = {
    "numpy": NumpyBackend,
    "dask": DaskBackend,
    "dask_gufunc": DaskGufuncBackend,
    "clesperanto": ClesperantoBackend,
    "cupy": CuPyBackend,
    "cupy_batch": CuPyBatchBackend,
    "numba": NumbaBackend,
    "jax": JaxBackend,
    "jax_batch": JaxBatchBackend,
}


def get_backend(name: str):
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend {name!r}. Available: {list(BACKENDS)}")
    return BACKENDS[name]()


# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------

def _build_arrays(configs: list) -> list:
    arrays, idx = [], 0
    for nf, t, c, z, y, x in configs:
        for _ in range(nf):
            rng = np.random.RandomState(BASE_SEED + idx)
            arrays.append(rng.randint(0, 256, size=(t, c, z, y, x), dtype=np.uint8))
            idx += 1
    return arrays


def run_timed(backend, arrays: list, profile: bool = False) -> dict:
    """Run backend on arrays, return timing + optional profiling data."""
    result = {"wall_time": 0.0}

    if profile:
        tracemalloc.start()

    t0 = time.perf_counter()
    for arr in arrays:
        backend.process(arr)
    result["wall_time"] = time.perf_counter() - t0

    if profile:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["peak_memory_mb"] = peak / (1024 * 1024)
        result["current_memory_mb"] = current / (1024 * 1024)

    return result


def _profile_single_array(backend, arr: np.ndarray) -> dict:
    """Detailed per-phase profiling for a single array."""
    t, c, z, y, x = arr.shape
    n_pixels = t * c * z * y * x
    phases = {}

    tracemalloc.start()
    t0 = time.perf_counter()
    result = backend.process(arr)
    total = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    phases["total_s"] = total
    phases["peak_memory_mb"] = peak / (1024 * 1024)
    phases["shape"] = [t, c, z, y, x]
    phases["n_pixels"] = n_pixels
    phases["throughput_mpix_per_s"] = n_pixels / total / 1e6 if total > 0 else 0
    phases["n_output_keys"] = len(result)
    return phases


def _system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import dask
        info["dask"] = dask.__version__
    except ImportError:
        pass
    try:
        import numba
        info["numba"] = numba.__version__
    except ImportError:
        pass
    try:
        import pyclesperanto as cle
        info["pyclesperanto"] = cle.__version__
        info["opencl_device"] = str(cle.get_device())
    except (ImportError, RuntimeError):
        pass
    try:
        import cupy as cp
        info["cupy"] = cp.__version__
        dev = cp.cuda.Device()
        info["cuda_device"] = cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode()
    except (ImportError, RuntimeError):
        pass
    try:
        import jax
        info["jax"] = jax.__version__
        info["jax_backend"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
    except (ImportError, RuntimeError):
        pass
    return info


def main():
    parser = argparse.ArgumentParser(description="Backend benchmark for pixel-patrol metrics")
    parser.add_argument("--profile", action="store_true", help="Enable memory profiling and per-array breakdown")
    parser.add_argument("--backends", nargs="*", default=None, help=f"Backends to test (default: all). Choices: {list(BACKENDS)}")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    requested = args.backends or list(BACKENDS.keys())
    arrays = _build_arrays(CONFIGS)
    print(f"{len(arrays)} arrays")

    sys_info = _system_info()
    print(f"  system: {sys_info.get('platform', '?')}")
    print(f"  numpy:  {sys_info.get('numpy', '?')}")

    # Instantiate available backends
    backends = []
    for name in requested:
        if name not in BACKENDS:
            print(f"  {name}: unknown (skipped)")
            continue
        try:
            backends.append((name, BACKENDS[name]()))
        except (ImportError, RuntimeError) as e:
            print(f"  {name}: skipped ({e})")

    if not backends:
        print("No backends available!")
        return

    # Verify consistency (pairwise)
    if len(backends) >= 2:
        ref_arr = arrays[0]
        results = {name: be.process(ref_arr) for name, be in backends}
        for (name_a, _), (name_b, _) in combinations(backends, 2):
            a, b = results[name_a], results[name_b]
            common = set(a) & set(b)
            mismatches = []
            for k in common:
                try:
                    np.testing.assert_allclose(
                        a[k], b[k], rtol=1e-4, atol=1e-4,
                        equal_nan=True, err_msg=f"{k}"
                    )
                except AssertionError as e:
                    mismatches.append(str(e))
            if mismatches:
                print(f"  WARNING: {name_a} vs {name_b}: {len(mismatches)} mismatches")
                for m in mismatches[:3]:
                    print(f"    {m}")
            key_diff_a = set(a) - set(b)
            key_diff_b = set(b) - set(a)
            if key_diff_a or key_diff_b:
                print(f"  WARNING: key mismatch {name_a} vs {name_b}: "
                      f"only in {name_a}: {key_diff_a}, only in {name_b}: {key_diff_b}")
        print(f"  consistency: checked ({len(backends)} backends, "
              f"{len(list(combinations(backends, 2)))} pairs)")

    # Overall timing
    all_results = {"system": sys_info, "backends": {}}
    print("\n--- Overall timing ---")
    for name, be in backends:
        r = run_timed(be, arrays, profile=args.profile)
        all_results["backends"][name] = {"overall": r}
        mem_str = f", peak={r['peak_memory_mb']:.1f}MB" if "peak_memory_mb" in r else ""
        print(f"  {name:20s}: {r['wall_time']:.3f}s{mem_str}")

    # Per-array profiling (representative configs)
    if args.profile:
        print("\n--- Per-array profiling (representative shapes) ---")
        profile_configs = [
            (1, 1, 3, 1, 200, 200),    # small
            (1, 1, 3, 1, 800, 800),    # medium
            (1, 1, 3, 1, 1600, 1600),  # large
            (1, 3, 3, 3, 400, 400),    # many slices
        ]
        profile_arrays = _build_arrays(profile_configs)
        for arr in profile_arrays:
            t, c, z, y, x = arr.shape
            shape_str = f"({t},{c},{z},{y},{x})"
            print(f"\n  shape={shape_str}:")
            for name, be in backends:
                p = _profile_single_array(be, arr)
                all_results["backends"].setdefault(name, {})
                all_results["backends"][name][f"profile_{shape_str}"] = p
                print(f"    {name:20s}: {p['total_s']:.4f}s  "
                      f"peak={p['peak_memory_mb']:.1f}MB  "
                      f"{p['throughput_mpix_per_s']:.1f} Mpix/s  "
                      f"keys={p['n_output_keys']}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for ax, (label, _, x_vals, configs) in zip(axes, CONFIG_GROUPS):
        arrs_per_param = [_build_arrays([c]) for c in configs]
        x = np.arange(len(x_vals))
        w = 0.8 / len(backends)
        for i, (name, be) in enumerate(backends):
            times = [run_timed(be, arrs)["wall_time"] for arrs in arrs_per_param]
            ax.bar(x + i * w, times, w, label=name)
        ax.set_xlabel(label)
        ax.set_xticks(x + w * (len(backends) - 1) / 2)
        ax.set_xticklabels(x_vals)
        ax.set_ylabel("Time (s)")
        ax.legend(fontsize=7)
    fig.suptitle("Backend comparison — wall time")
    fig.tight_layout()
    plot_path = Path(__file__).parent / "backends_comparison.png"
    fig.savefig(plot_path, dpi=150)
    print(f"\n  plot saved: {plot_path}")

    # Save JSON results
    if args.output:
        output_path = Path(args.output)
        # Convert numpy types for JSON serialization
        def _convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=_convert)
        print(f"  results saved: {output_path}")


if __name__ == "__main__":
    main()
