#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import polars as pl


# ----------------------------
# Detection
# ----------------------------

def _is_e2e(df: pl.DataFrame) -> bool:
    return {"processing_time_sec", "import_time_sec", "widget_time_sec"}.issubset(set(df.columns))


def _is_processors(df: pl.DataFrame) -> bool:
    return {"wall_time_sec", "memory_mb"}.issubset(set(df.columns))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No CSV found at {path}")
    return pl.read_csv(path)


def _github_step_summary(md: str) -> None:
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if not p:
        return
    md = md.replace("![](", "Image in artifacts: (")
    with open(p, "a") as f:
        f.write(md)


# ----------------------------
# Formatting
# ----------------------------

def _dims_str(row: dict) -> str:
    keys = ["num_files", "t", "c", "z", "y", "x"]
    parts = []
    for k in keys:
        if k in row and row[k] is not None:
            parts.append(f"{k}={row[k]}")
    return ", ".join(parts)


def _fmt_mean_std(row: dict, base_col: str) -> str:
    m = row.get(f"{base_col}_mean")
    s = row.get(f"{base_col}_std")
    if m is None:
        return "N/A"
    s = 0.0 if s is None else float(s)
    return f"{float(m):.3f} ± {s:.3f}"


def _fmt_delta(curr: Optional[float], ref: Optional[float]) -> str:
    if curr is None or ref is None:
        return "n/a"
    if ref == 0:
        return f"{curr - ref:+.3f} (n/a%)"
    pct = (curr - ref) / ref * 100.0
    return f"{curr - ref:+.3f} ({pct:+.1f}%)"


# ----------------------------
# Metrics
# ----------------------------

def _metrics_for_kind(df: pl.DataFrame, kind: str, processor: Optional[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    returns:
      - headline metrics: label -> base col
      - proc cols: list of base proc_*_sec columns (filtered if processor provided)
    """
    if kind == "e2e":
        return (
            {
                "Processing (s)": "processing_time_sec",
                "Import (s)": "import_time_sec",
                "Widget (s)": "widget_time_sec",
            },
            [],
        )

    # processors
    proc_cols = [c for c in df.columns if c.startswith("proc_") and c.endswith("_sec")]
    if processor:
        wanted = f"proc_{processor}_sec"
        proc_cols = [c for c in proc_cols if c == wanted]

    return (
        {
            "Wall (s)": "wall_time_sec",
            "Memory (MB)": "memory_mb",
        },
        proc_cols,
    )


# ----------------------------
# Aggregation
# ----------------------------

def _aggregate(df: pl.DataFrame, base_metric_cols: List[str]) -> pl.DataFrame:
    group_cols = [c for c in ["test_name", "branch", "num_files", "t", "c", "z", "y", "x"] if c in df.columns]

    aggs = []
    for c in base_metric_cols:
        aggs += [
            pl.col(c).cast(pl.Float64).mean().alias(f"{c}_mean"),
            pl.col(c).cast(pl.Float64).std().alias(f"{c}_std"),
        ]

    return df.group_by(group_cols).agg(aggs).sort(["test_name", "branch"])


def _pick_ref_branch(agg: pl.DataFrame, ref_branch: Optional[str]) -> str:
    branches = [b for b in agg.select(pl.col("branch").unique()).to_series().to_list() if b not in (None, "")]
    if not branches:
        return ""
    if ref_branch:
        if ref_branch not in branches:
            raise ValueError(f"--ref-branch '{ref_branch}' not found in branches: {sorted(branches)}")
        return ref_branch
    return "main" if "main" in branches else sorted(branches)[0]


# ----------------------------
# Markdown rendering
# ----------------------------

def _render_branch_table(
    agg: pl.DataFrame,
    metrics: Dict[str, str],
    ref_branch: str,
) -> str:
    tests = sorted(agg.select(pl.col("test_name").unique()).to_series().to_list())
    branches = sorted([b for b in agg.select(pl.col("branch").unique()).to_series().to_list() if b not in (None, "")])

    has_ref = ref_branch in branches and ref_branch != ""

    md: List[str] = []
    md.append("## Summary by test\n\n")
    if branches:
        md.append(f"- Branches: {', '.join(branches)}\n")
    if has_ref:
        md.append(f"- Δ definition: `branch - {ref_branch}` (e.g. `Δ = current-main` if branch is `current`)\n")
    md.append("\n")

    header = ["Test", "Dims", "Branch"]
    for label in metrics.keys():
        header.append(label)
        if has_ref:
            header.append(f"Δ ({label}) = branch-{ref_branch}")
    md.append("| " + " | ".join(header) + " |\n")
    md.append("| " + " | ".join([":---"] * len(header)) + " |\n")

    for t in tests:
        sub = agg.filter(pl.col("test_name") == t)
        if sub.is_empty():
            continue

        first = sub.row(0, named=True)
        dims = _dims_str(first)

        ref_row = None
        if has_ref:
            rsub = sub.filter(pl.col("branch") == ref_branch)
            if not rsub.is_empty():
                ref_row = rsub.row(0, named=True)

        for row in sub.iter_rows(named=True):
            br = str(row.get("branch", ""))
            line = [t, dims, br]

            for _label, col in metrics.items():
                line.append(_fmt_mean_std(row, col))
                if has_ref:
                    curr = row.get(f"{col}_mean")
                    ref = None if ref_row is None else ref_row.get(f"{col}_mean")
                    line.append(_fmt_delta(None if curr is None else float(curr), None if ref is None else float(ref)))

            md.append("| " + " | ".join(line) + " |\n")

    md.append("\n")
    return "".join(md)


def _render_processors_table(
    agg: pl.DataFrame,
    proc_cols: List[str],
    ref_branch: str,
) -> str:
    if not proc_cols:
        return ""

    branches = sorted([b for b in agg.select(pl.col("branch").unique()).to_series().to_list() if b not in (None, "")])
    has_ref = ref_branch in branches and ref_branch != ""

    # Keep ALL processors (requested). This can be wide.
    headers = [c.removeprefix("proc_").removesuffix("_sec") for c in proc_cols]

    md: List[str] = []
    md.append("## Processor times\n\n")
    if has_ref:
        md.append(f"- Δ definition: `branch - {ref_branch}`\n\n")

    # Table of means
    md.append("| Test | Branch | " + " | ".join(headers) + " |\n")
    md.append("| :--- | :--- | " + " | ".join([":---"] * len(headers)) + " |\n")

    for t in sorted(agg.select(pl.col("test_name").unique()).to_series().to_list()):
        sub = agg.filter(pl.col("test_name") == t).sort("branch")
        for row in sub.iter_rows(named=True):
            line = [t, str(row.get("branch", ""))]
            for c in proc_cols:
                v = row.get(f"{c}_mean")
                line.append(f"{float(v):.3f}" if v is not None else "")
            md.append("| " + " | ".join(line) + " |\n")

    md.append("\n")

    # Optional delta table (means) if ref exists
    if has_ref:
        md.append(f"### Processor deltas (mean seconds): Δ = branch-{ref_branch}\n\n")
        md.append("| Test | Branch | " + " | ".join(headers) + " |\n")
        md.append("| :--- | :--- | " + " | ".join([":---"] * len(headers)) + " |\n")

        for t in sorted(agg.select(pl.col("test_name").unique()).to_series().to_list()):
            sub = agg.filter(pl.col("test_name") == t)
            rsub = sub.filter(pl.col("branch") == ref_branch)
            if rsub.is_empty():
                continue
            ref_row = rsub.row(0, named=True)

            for row in sub.sort("branch").iter_rows(named=True):
                br = str(row.get("branch", ""))
                if br == ref_branch:
                    continue
                line = [t, br]
                for c in proc_cols:
                    curr = row.get(f"{c}_mean")
                    ref = ref_row.get(f"{c}_mean")
                    if curr is None or ref is None:
                        line.append("n/a")
                    else:
                        line.append(f"{float(curr) - float(ref):+.3f}")
                md.append("| " + " | ".join(line) + " |\n")

        md.append("\n")

    return "".join(md)


# ----------------------------
# Plotting (optional)
# ----------------------------


def _has_full_scaling_tests(agg: pl.DataFrame) -> bool:
    if "test_name" not in agg.columns:
        return False
    names = agg.select(pl.col("test_name").unique()).to_series().to_list()
    return any(str(t).startswith("scaling_xy_") or str(t).startswith("scaling_tz_") for t in names)


def _plot_scaling_curve(
    agg: pl.DataFrame,
    metric_label: str,
    base_col: str,
    out_path: Path,
    scaling_kind: str,  # "xy" or "tz"
) -> Optional[str]:
    """
    Plot mean(metric) vs scaling parameter (xy_size or tz_size), one line per branch.
    Only uses tests named scaling_{scaling_kind}_<N>.
    """
    mcol = f"{base_col}_mean"
    if "test_name" not in agg.columns or "branch" not in agg.columns or mcol not in agg.columns:
        return None

    prefix = f"scaling_{scaling_kind}_"
    sub = agg.filter(pl.col("test_name").cast(pl.Utf8).str.starts_with(prefix))
    if sub.is_empty():
        return None

    # Extract numeric size from test_name
    sub = sub.with_columns(
        pl.col("test_name")
        .cast(pl.Utf8)
        .str.replace(prefix, "")
        .cast(pl.Int64, strict=False)
        .alias("scale_n")
    ).filter(pl.col("scale_n").is_not_null())

    if sub.is_empty():
        return None

    branches = sorted([b for b in sub.select(pl.col("branch").unique()).to_series().to_list() if b not in (None, "")])
    if not branches:
        return None

    # Sort x axis by scale_n
    xs = sorted(sub.select(pl.col("scale_n").unique()).to_series().to_list())
    if not xs:
        return None

    plt.figure(figsize=(8, 4))
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    for br in branches:
        ys: List[Optional[float]] = []
        for n in xs:
            row = sub.filter((pl.col("branch") == br) & (pl.col("scale_n") == n))
            if row.is_empty():
                ys.append(None)
            else:
                ys.append(float(row.select(pl.col(mcol)).item()))
        plt.plot(xs, ys, marker="o", linewidth=2, label=str(br))

    plt.xlabel("XY size (pixels)" if scaling_kind == "xy" else "Sliced dims size (T=Z)")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} vs scaling_{scaling_kind}", fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path.name


def _plot_metric_by_test(
    agg: pl.DataFrame,
    metric_label: str,
    base_col: str,
    out_path: Path,
) -> Optional[str]:
    mcol = f"{base_col}_mean"
    if mcol not in agg.columns:
        return None

    tests = sorted(agg.select(pl.col("test_name").unique()).to_series().to_list())
    branches = sorted([b for b in agg.select(pl.col("branch").unique()).to_series().to_list() if b not in (None, "")])
    if not tests or not branches:
        return None

    plt.figure(figsize=(10, 4))
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)

    x = list(range(len(tests)))
    for br in branches:
        ys: List[Optional[float]] = []
        for t in tests:
            sub = agg.filter((pl.col("test_name") == t) & (pl.col("branch") == br))
            if sub.is_empty():
                ys.append(None)
            else:
                ys.append(float(sub.select(pl.col(mcol)).item()))
        plt.plot(x, ys, marker="o", linewidth=2, label=str(br))

    plt.xticks(x, tests, rotation=30, ha="right")
    plt.ylabel(metric_label)
    plt.title(metric_label, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path.name


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate markdown report from a benchmark CSV (E2E or Processors)")
    ap.add_argument("--csv", type=Path, required=True, help="Path to e2e_results.csv or processor_results.csv")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: CSV parent)")
    ap.add_argument("--ref-branch", type=str, default=None, help="Reference branch for deltas (default: main if present)")
    ap.add_argument("--figures", action="store_true", help="Generate charts (default: off)")
    ap.add_argument(
        "--processor",
        type=str,
        default=None,
        help="Processors only: select a specific processor NAME (e.g. HistogramProcessor). Default: all.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or args.csv.parent
    assets_dir = out_dir / "assets"
    report_path = out_dir / "REPORT.md"
    _ensure_dir(out_dir)
    if args.figures:
        _ensure_dir(assets_dir)

    df = _read_csv(args.csv)

    if _is_e2e(df):
        kind = "e2e"
        title = "E2E Performance Benchmarks"
    elif _is_processors(df):
        kind = "processors"
        title = "Processor Performance Benchmarks"
    else:
        raise ValueError(f"Couldn't detect benchmark type from columns in {args.csv}")

    headline_metrics, proc_cols = _metrics_for_kind(df, kind, args.processor)

    # aggregate all needed base cols
    base_metric_cols = list(headline_metrics.values()) + proc_cols
    agg = _aggregate(df, base_metric_cols)

    ref_branch = _pick_ref_branch(agg, args.ref_branch)
    branches = sorted([b for b in agg.select(pl.col("branch").unique()).to_series().to_list() if b not in (None, "")])

    parts: List[str] = []
    parts.append(f"# 📊 {title}\n\n")
    parts.append(f"- Data: `{args.csv}`\n")
    if branches:
        parts.append(f"- Branches: {', '.join(branches)}\n")
        parts.append(f"- Reference branch: `{ref_branch}`\n")
        parts.append(f"- Δ definition: `branch - {ref_branch}`\n")
    parts.append(f"- Figures: `{'on' if args.figures else 'off'}`\n")
    if kind == "processors":
        parts.append(f"- Processor filter: `{args.processor or 'all'}`\n")
    parts.append("\n")

    # Figures: headline metrics + (processors) one plot per processor column
    if args.figures:
        parts.append("## Figures\n\n")

        # --- Scaling plots (ONLY when full-mode scaling tests exist) ---
        if _has_full_scaling_tests(agg):
            parts.append("### Scaling curves (full mode)\n\n")

            for label, col in headline_metrics.items():
                png_xy = _plot_scaling_curve(
                    agg,
                    label,
                    col,
                    assets_dir / f"{kind}_scaling_xy_{col}.png",
                    scaling_kind="xy",
                )
                if png_xy:
                    parts.append(f"![]({assets_dir.name}/{png_xy})\n\n")

                png_tz = _plot_scaling_curve(
                    agg,
                    label,
                    col,
                    assets_dir / f"{kind}_scaling_tz_{col}.png",
                    scaling_kind="tz",
                )
                if png_tz:
                    parts.append(f"![]({assets_dir.name}/{png_tz})\n\n")

        # --- Existing per-test plots ---
        for label, col in headline_metrics.items():
            png = _plot_metric_by_test(
                agg, label, col, assets_dir / f"{kind}_{col}.png"
            )
            if png:
                parts.append(f"![]({assets_dir.name}/{png})\n\n")

        if kind == "processors":
            for c in proc_cols:
                pname = c.removeprefix("proc_").removesuffix("_sec")
                png = _plot_metric_by_test(
                    agg, f"{pname} (s)", c, assets_dir / f"{kind}_{c}.png"
                )
                if png:
                    parts.append(f"![]({assets_dir.name}/{png})\n\n")

    # Tables
    parts.append(_render_branch_table(agg, headline_metrics, ref_branch))
    if kind == "processors":
        parts.append(_render_processors_table(agg, proc_cols, ref_branch))

    md = "".join(parts)
    report_path.write_text(md)
    print(f"Report generated: {report_path}")
    if args.figures:
        print(f"Figures saved under: {assets_dir}")

    _github_step_summary(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
