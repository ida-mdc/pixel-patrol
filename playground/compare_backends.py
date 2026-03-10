"""
Compare benchmark results from different backends/devices.

Reads all JSON files in the results/ folder and produces a combined comparison
plot with:
  - Overall wall time (bar chart)
  - Per-shape time breakdown (grouped bars)
  - Per-shape throughput (Mpix/s)
  - Peak memory usage

Usage:
    python compare_backends.py                           # default: results/ folder
    python compare_backends.py --results-dir my_results  # custom folder
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path) -> dict[str, dict]:
    """Load all JSON result files. Returns {label: data}."""
    results = {}
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        # Use filename stem as the label (e.g. "cupy_batch", "pyclesperanto_amd")
        label = p.stem
        results[label] = data
    return results


def _device_info(data: dict) -> str:
    """Extract a short device description from system info."""
    sys = data.get("system", {})
    if "cuda_device" in sys:
        return sys["cuda_device"]
    if "opencl_device" in sys:
        return sys["opencl_device"]
    return sys.get("processor", "")


def plot_comparison(results: dict[str, dict], output_path: Path):
    labels = list(results.keys())
    n = len(labels)

    # Collect profile keys common to all results
    profile_keys = None
    for label, data in results.items():
        backend_name = list(data["backends"].keys())[0]
        bdata = data["backends"][backend_name]
        pkeys = [k for k in bdata if k.startswith("profile_")]
        if profile_keys is None:
            profile_keys = pkeys
        else:
            profile_keys = [k for k in profile_keys if k in pkeys]

    # Sort profile keys by total pixel count (n_pixels) for intuitive ordering
    if profile_keys:
        first_label = labels[0]
        first_backend = list(results[first_label]["backends"].keys())[0]
        first_bdata = results[first_label]["backends"][first_backend]
        profile_keys.sort(key=lambda k: first_bdata[k].get("n_pixels", 0))

    has_profiles = bool(profile_keys)
    n_profiles = len(profile_keys) if has_profiles else 0

    # Layout: row 0 = overall + memory, row 1 = per-shape time + throughput (shared scale)
    #         row 2 = per-shape time (individual axes), row 3 = per-shape throughput (individual axes)
    if has_profiles:
        n_rows = 4
        fig = plt.figure(figsize=(max(14, 3 * n_profiles), 18))
        gs = fig.add_gridspec(n_rows, max(n_profiles, 2), hspace=0.45, wspace=0.35)
        half = max(n_profiles, 2) // 2
        ax_overall = fig.add_subplot(gs[0, :half])
        ax_memory = fig.add_subplot(gs[0, half:])
        ax_time = fig.add_subplot(gs[1, :half])
        ax_throughput = fig.add_subplot(gs[1, half:])
        axes_time_ind = [fig.add_subplot(gs[2, j]) for j in range(n_profiles)]
        axes_tp_ind = [fig.add_subplot(gs[3, j]) for j in range(n_profiles)]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_overall, ax_memory = axes[0], axes[1]

    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 10)))

    # --- Overall wall time ---
    wall_times = []
    for label in labels:
        data = results[label]
        backend_name = list(data["backends"].keys())[0]
        wall_times.append(data["backends"][backend_name]["overall"]["wall_time"])

    x = np.arange(n)
    bars = ax_overall.bar(x, wall_times, color=colors[:n])
    ax_overall.set_xticks(x)
    ax_overall.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax_overall.set_ylabel("Wall time (s)")
    ax_overall.set_title("Overall wall time")
    for bar, t in zip(bars, wall_times):
        ax_overall.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{t:.1f}s", ha="center", va="bottom", fontsize=7)

    # --- Peak memory ---
    peak_mems = []
    for label in labels:
        data = results[label]
        backend_name = list(data["backends"].keys())[0]
        overall = data["backends"][backend_name]["overall"]
        peak_mems.append(overall.get("peak_memory_mb", 0))

    bars = ax_memory.bar(x, peak_mems, color=colors[:n])
    ax_memory.set_xticks(x)
    ax_memory.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax_memory.set_ylabel("Peak memory (MB)")
    ax_memory.set_title("Peak memory usage")
    for bar, m in zip(bars, peak_mems):
        ax_memory.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f"{m:.0f}", ha="center", va="bottom", fontsize=7)

    # --- Per-shape breakdown (if profiles exist) ---
    if has_profiles:
        # Build labels showing shape and total pixels (e.g. "(1,3,1,200,200)\n0.1 Mpix")
        first_bdata = results[labels[0]]["backends"][list(results[labels[0]]["backends"].keys())[0]]
        shape_labels = []
        for k in profile_keys:
            shape_str = k.replace("profile_", "")
            n_pix = first_bdata[k].get("n_pixels", 0)
            shape_labels.append(f"{shape_str}\n{n_pix/1e6:.1f} Mpix")
        x_shapes = np.arange(n_profiles)
        w = 0.8 / n

        for i, label in enumerate(labels):
            data = results[label]
            backend_name = list(data["backends"].keys())[0]
            bdata = data["backends"][backend_name]

            times = [bdata[k]["total_s"] for k in profile_keys]
            throughputs = [bdata[k]["throughput_mpix_per_s"] for k in profile_keys]

            ax_time.bar(x_shapes + i * w, times, w, label=label, color=colors[i])
            ax_throughput.bar(x_shapes + i * w, throughputs, w, label=label, color=colors[i])

        ax_time.set_xticks(x_shapes + w * (n - 1) / 2)
        ax_time.set_xticklabels(shape_labels, fontsize=7)
        ax_time.set_ylabel("Time (s)")
        ax_time.set_title("Per-shape processing time")
        ax_time.legend(fontsize=7, loc="upper left")

        ax_throughput.set_xticks(x_shapes + w * (n - 1) / 2)
        ax_throughput.set_xticklabels(shape_labels, fontsize=7)
        ax_throughput.set_ylabel("Throughput (Mpix/s)")
        ax_throughput.set_title("Per-shape throughput")
        ax_throughput.legend(fontsize=7, loc="upper left")

        # --- Per-shape individual axes (auto-scaled per shape) ---
        x_backends = np.arange(n)
        for j, k in enumerate(profile_keys):
            times_j = []
            tp_j = []
            for label in labels:
                data = results[label]
                backend_name = list(data["backends"].keys())[0]
                bdata = data["backends"][backend_name]
                times_j.append(bdata[k]["total_s"])
                tp_j.append(bdata[k]["throughput_mpix_per_s"])

            # Time axis
            ax_t = axes_time_ind[j]
            bars = ax_t.bar(x_backends, times_j, color=colors[:n])
            ax_t.set_xticks(x_backends)
            ax_t.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
            ax_t.set_title(shape_labels[j], fontsize=7)
            ax_t.set_ylabel("Time (s)" if j == 0 else "", fontsize=7)
            ax_t.tick_params(axis="y", labelsize=6)
            for bar, t in zip(bars, times_j):
                ax_t.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                          f"{t:.3g}s", ha="center", va="bottom", fontsize=6)

            # Throughput axis
            ax_tp = axes_tp_ind[j]
            bars = ax_tp.bar(x_backends, tp_j, color=colors[:n])
            ax_tp.set_xticks(x_backends)
            ax_tp.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
            ax_tp.set_title(shape_labels[j], fontsize=7)
            ax_tp.set_ylabel("Mpix/s" if j == 0 else "", fontsize=7)
            ax_tp.tick_params(axis="y", labelsize=6)
            for bar, tp in zip(bars, tp_j):
                ax_tp.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           f"{tp:.1f}", ha="center", va="bottom", fontsize=6)

        # Row labels for the individual-axis rows
        fig.text(0.02, 0.38, "Per-shape time (individual scale)", fontsize=9,
                 fontweight="bold", rotation=90, va="center")
        fig.text(0.02, 0.14, "Per-shape throughput (individual scale)", fontsize=9,
                 fontweight="bold", rotation=90, va="center")

    # Subtitle with device info
    info_lines = []
    for label in labels:
        dev = _device_info(results[label])
        if dev:
            info_lines.append(f"{label}: {dev}")
    subtitle = " | ".join(info_lines) if info_lines else ""
    fig.suptitle("Backend / Device Comparison", fontsize=13, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.97, subtitle, ha="center", fontsize=7, style="italic")

    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved: {output_path}")


def print_summary(results: dict[str, dict]):
    print(f"\n{'Label':<30s} {'Wall time':>10s} {'Peak mem':>10s} {'Device'}")
    print("-" * 80)
    for label, data in results.items():
        backend_name = list(data["backends"].keys())[0]
        overall = data["backends"][backend_name]["overall"]
        wt = overall["wall_time"]
        pm = overall.get("peak_memory_mb", 0)
        dev = _device_info(data)
        print(f"{label:<30s} {wt:>9.2f}s {pm:>9.1f}MB  {dev}")


def main():
    parser = argparse.ArgumentParser(description="Compare backend benchmark results")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing result JSON files (default: results/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output plot path (default: backends_comparison.png)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else Path(__file__).parent / "results"
    output_path = Path(args.output) if args.output else Path(__file__).parent / "backends_comparison.png"

    results = load_results(results_dir)
    if not results:
        print(f"No JSON files found in {results_dir}")
        return

    print(f"Loaded {len(results)} result files from {results_dir}")
    print_summary(results)
    plot_comparison(results, output_path)


if __name__ == "__main__":
    main()
