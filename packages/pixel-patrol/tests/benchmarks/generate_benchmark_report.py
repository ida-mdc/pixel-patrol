import os

import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
BENCHMARK_DIR = Path(__file__).parent
DATA_PATH = BENCHMARK_DIR / "benchmark_results" / "benchmark_results.csv"
ASSETS_DIR = BENCHMARK_DIR / "benchmark_results" / "assets"
REPORT_PATH = BENCHMARK_DIR / "benchmark_results" / "REPORT.md"

ASSETS_DIR.mkdir(exist_ok=True)


def load_and_aggregate():
    """Loads raw CSV and aggregates iterations (Mean +/- StdDev)."""
    if not DATA_PATH.exists():
        print(f"No data found at {DATA_PATH}! Run benchmarks first.")
        return None

    df = pl.read_csv(DATA_PATH)

    time_cols = ["processing_time", "import_time", "widget_time"]

    # Aggregation: Group by ALL potential varying parameters
    agg_df = (
        df.group_by(["test_name", "num_files", "t_size", "c_size", "z_size", "y_size", "x_size"])
        .agg([
                 pl.col(c).mean().alias(f"{c}_mean") for c in time_cols
             ] + [
                 pl.col(c).std().alias(f"{c}_std") for c in time_cols
             ])
    )
    return agg_df


def get_scenario_description(df, test_name, varying_col, coupled_cols=None):
    """
    Dynamically generates a description string showing ALL varying columns.
    """
    subset = df.filter(pl.col("test_name") == test_name)
    if subset.is_empty():
        return "No data available."

    # 1. Identify all varying columns (Primary + Coupled)
    varying_group = [varying_col]
    if coupled_cols:
        varying_group.extend(coupled_cols)

    # 2. Identify Constants
    param_cols = ["num_files", "t_size", "c_size", "z_size", "y_size", "x_size"]
    constants = []

    def fmt_name(c):
        """Helper to format column names nicely."""
        if c == "num_files": return "Files"
        return c.replace("_size", "").upper()

    for col in param_cols:
        # Skip columns that are intentionally varying
        if col in varying_group:
            continue

        unique_vals = subset[col].unique().to_list()
        if not unique_vals: continue

        if len(unique_vals) == 1:
            val = unique_vals[0]
            constants.append(f"{fmt_name(col)}={val}")
        else:
            # If it accidentally varied, show range
            min_v, max_v = min(unique_vals), max(unique_vals)
            constants.append(f"{fmt_name(col)}=[{min_v}-{max_v}]")

    # 3. Format the Varying Range description
    # We assume coupled columns move in lockstep, so we take the range from the primary
    var_vals = subset[varying_col].unique().sort()
    if len(var_vals) > 0:
        range_str = f"{var_vals[0]} to {var_vals[-1]}"
    else:
        range_str = "Unknown"

    # Create label: "T & Z" or "X & Y"
    varying_labels = " & ".join([fmt_name(c) for c in varying_group])

    return f"**Varying:** {varying_labels} ({range_str}).<br>**Constants:** {', '.join(constants)}"


def plot_metrics(df, test_name, x_col, x_label, metrics_map, filename, custom_title, title_suffix):
    """
    Generates a plot for a specific subset of metrics.
    custom_title: Explicit string for the chart title (e.g. "Scaling TZ")
    """
    subset = df.filter(pl.col("test_name") == test_name).to_pandas()

    if subset.empty:
        print(f"Warning: No data found for test '{test_name}'")
        return None

    # SORT by x-axis
    subset = subset.sort_values(by=x_col)

    plt.figure(figsize=(8, 5))
    plt.grid(True, linestyle='--', alpha=0.7)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for (label, col_prefix), color in zip(metrics_map.items(), colors):
        mean_col = f"{col_prefix}_mean"
        std_col = f"{col_prefix}_std"

        if mean_col not in subset.columns: continue

        plt.errorbar(
            subset[x_col],
            subset[mean_col],
            yerr=subset[std_col],
            marker='o',
            capsize=5,
            label=label,
            color=color,
            linewidth=2
        )

    # Use the explicit custom title passed from main()
    plt.title(f"{custom_title}: {title_suffix}", fontsize=12, fontweight='bold')
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel("Time (seconds)", fontsize=10)
    plt.legend()

    out_path = ASSETS_DIR / filename
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Generated plot: {filename}")
    return filename


def format_table(df, test_name, key_col):
    """Creates a markdown table with all metrics."""
    subset = df.filter(pl.col("test_name") == test_name).sort(key_col)

    md = "| Input Size | Processing (s) | Import (s) | Widget Load (s) |\n"
    md += "| :--- | :--- | :--- | :--- |\n"

    for row in subset.iter_rows(named=True):
        def fmt(prefix):
            mean_val = row.get(f'{prefix}_mean')
            std_val = row.get(f'{prefix}_std')
            if mean_val is None: return "N/A"
            std_str = f"{std_val:.3f}" if std_val is not None else "0.000"
            return f"{mean_val:.3f} ± {std_str}"

        key = row[key_col]
        md += f"| {key} | {fmt('processing_time')} | {fmt('import_time')} | {fmt('widget_time')} |\n"

    return md


def write_github_summary(report_content):
    """Writes the markdown content to the GitHub Job Summary."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        # We strip out the image links because GitHub Summaries
        # cannot display images that aren't hosted publicly.
        clean_content = report_content.replace("![]", "Image available in Artifacts: ")

        with open(summary_file, "a") as f:
            f.write(clean_content)
        print("Written to GitHub Job Summary")


def main():
    df = load_and_aggregate()
    if df is None: return

    group_heavy = {"Processing": "processing_time"}
    group_interactive = {
        "Import Project": "import_time",
        "Widget Load": "widget_time"
    }

    # --- 1. Scaling by File Count ---
    desc_files = get_scenario_description(df, "scaling_files", "num_files")

    # We pass "Scaling Files" explicitly
    f_proc = plot_metrics(df, "scaling_files", "num_files", "Number of Files",
                          group_heavy, "files_proc.png", "Scaling Files", "Processing Time")
    f_int = plot_metrics(df, "scaling_files", "num_files", "Number of Files",
                         group_interactive, "files_interactive.png", "Scaling Files", "Interactive Latency")
    t_files = format_table(df, "scaling_files", "num_files")

    # --- 2. Scaling by XY ---
    desc_xy = get_scenario_description(df, "scaling_xy", "y_size", coupled_cols=["x_size"])

    # We pass "Scaling XY" explicitly
    x_proc = plot_metrics(df, "scaling_xy", "y_size", "XY Size (px)",
                          group_heavy, "xy_proc.png", "Scaling XY", "Processing Time")
    x_int = plot_metrics(df, "scaling_xy", "y_size", "XY Size (px)",
                         group_interactive, "xy_interactive.png", "Scaling XY", "Interactive Latency")
    t_xy = format_table(df, "scaling_xy", "y_size")

    # --- 3. Scaling by TZ ---
    desc_tz = get_scenario_description(df, "scaling_tz", "t_size", coupled_cols=["z_size"])

    # We pass "Scaling TZ" explicitly
    t_proc = plot_metrics(df, "scaling_tz", "t_size", "Stack Size (T/Z)",
                          group_heavy, "tz_proc.png", "Scaling TZ", "Processing Time")
    t_int = plot_metrics(df, "scaling_tz", "t_size", "Stack Size (T/Z)",
                         group_interactive, "tz_interactive.png", "Scaling TZ", "Interactive Latency")
    t_tz = format_table(df, "scaling_tz", "t_size")

    # --- Generate Markdown ---
    report = f"""# 📊 Performance Benchmarks

*Automated report generated from `pixel-patrol`.*

## 1. Scaling by File Count
{desc_files}

| Processing (Batch) | Interactive (User) |
| :---: | :---: |
| ![]({ASSETS_DIR.name}/{f_proc}) | ![]({ASSETS_DIR.name}/{f_int}) |

{t_files}

---

## 2. Scaling by Resolution (XY)
{desc_xy}

| Processing (Batch) | Interactive (User) |
| :---: | :---: |
| ![]({ASSETS_DIR.name}/{x_proc}) | ![]({ASSETS_DIR.name}/{x_int}) |

{t_xy}

---

## 3. Scaling by Stack Depth (T/Z)
{desc_tz}

| Processing (Batch) | Interactive (User) |
| :---: | :---: |
| ![]({ASSETS_DIR.name}/{t_proc}) | ![]({ASSETS_DIR.name}/{t_int}) |

{t_tz}
"""

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"\nReport generated: {REPORT_PATH}")

    write_github_summary(report)



if __name__ == "__main__":
    main()