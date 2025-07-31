# generate_report.py
import argparse
import itertools
import json
from pathlib import Path

import polars as pl
import statsmodels.stats.multitest as smm
from scipy.stats import mannwhitneyu


def process_data_for_report(df: pl.DataFrame) -> dict:
    """
    Performs all Python-based data processing and returns a dictionary of results
    ready for JSON serialization.
    """
    groups = df["imported_path_short"].unique().sort().to_list()
    if not groups:
        return {"table_data": [], "plot_data": []}

    # 1. Partition columns into those for plotting vs. a simple table
    numeric_cols = [c for c in df.columns if df[c].dtype in pl.NUMERIC_DTYPES and "intensity" in c]
    cols_to_plot = []
    no_variance_data = []
    for col in numeric_cols:
        series = df.get_column(col).drop_nulls()
        if series.n_unique() == 1 and len(series) > 0:
            no_variance_data.append({"Metric": col.replace('_', ' ').title(), "Value": f"{series[0]:.4f}"})
        elif series.n_unique() > 1:
            cols_to_plot.append(col)

    # 2. Prepare data for violin plots, including statistical tests
    all_plot_data = []
    for col_name in cols_to_plot:
        # Extract the raw data needed for plotting
        plot_data = {
            "metric": col_name,
            "groups_data": {
                g: df.filter(pl.col("imported_path_short") == g).get_column(col_name).to_list()
                for g in groups
            },
            "stats": []
        }

        # Perform statistical tests in Python
        if len(groups) > 1:
            comparisons = list(itertools.combinations(groups, 2))
            p_values = [
                mannwhitneyu(plot_data["groups_data"][g1], plot_data["groups_data"][g2]).pvalue
                for g1, g2 in comparisons
            ]
            if p_values:
                _, pvals_corrected, _, _ = smm.multipletests(p_values, alpha=0.05, method="bonferroni")
                for i, (g1, g2) in enumerate(comparisons):
                    p_corr = pvals_corrected[i]
                    sig = "***" if p_corr < 0.001 else "**" if p_corr < 0.01 else "*" if p_corr < 0.05 else "ns"
                    plot_data["stats"].append({"pair": [g1, g2], "sig": sig})
        all_plot_data.append(plot_data)

    return {"table_data": no_variance_data, "plot_data": all_plot_data}


def create_html_report(report_data: dict, output_path: str):
    """Generates the final HTML file with embedded data and JavaScript renderer."""
    # Serialize the Python dictionary to a JSON string
    json_data = json.dumps(report_data)

    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Statistical Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: sans-serif; margin: 2em; }}
        .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 2em; }}
        table {{ border-collapse: collapse; width: 50%; margin-top: 1em; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Pixel Value Statistics Report</h1>
    <div id="report-container"></div>

    <script>
        // --- EMBEDDED DATA FROM PYTHON ---
        const REPORT_DATA = {json_data};

        // --- JAVASCRIPT RENDERER ---
        document.addEventListener("DOMContentLoaded", function() {{
            const container = document.getElementById('report-container');

            // 1. Render the table for metrics with no variance
            const tableData = REPORT_DATA.table_data;
            if (tableData.length > 0) {{
                const table = document.createElement('table');
                table.innerHTML = `
                    <caption>Metrics with No Variance</caption>
                    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                    <tbody>
                        ${{tableData.map(row => `<tr><td>${{row.Metric}}</td><td>${{row.Value}}</td></tr>`).join('')}}
                    </tbody>`;
                container.appendChild(table);
                container.appendChild(document.createElement('hr'));
            }}

            // 2. Render the grid of violin plots
            const plotGrid = document.createElement('div');
            plotGrid.className = 'plot-grid';
            container.appendChild(plotGrid);

            REPORT_DATA.plot_data.forEach(plotInfo => {{
                const plotDiv = document.createElement('div');
                plotGrid.appendChild(plotDiv);

                const traces = Object.entries(plotInfo.groups_data).map(([groupName, y_data]) => ({{
                    y: y_data,
                    name: groupName,
                    type: 'violin',
                    box: {{ visible: true }},
                    meanline: {{ visible: true }},
                    points: 'all'
                }}));

                const layout = {{
                    title: `Distribution of ${{plotInfo.metric.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}`,
                    showlegend: true,
                    legend: {{ orientation: "h", y: -0.2 }},
                    shapes: [],
                    annotations: []
                }};

                // Add statistical annotations from pre-computed results
                const groups = Object.keys(plotInfo.groups_data);
                const y_max = Math.max(...[].concat(...Object.values(plotInfo.groups_data)));
                plotInfo.stats.forEach(stat => {{
                    // Basic annotation positioning (can be improved)
                    const [g1, g2] = stat.pair;
                    const x0 = groups.indexOf(g1);
                    const x1 = groups.indexOf(g2);
                    if (x0 === -1 || x1 === -1 || Math.abs(x0-x1) > 1) return; // Only show for adjacent for now

                    const bracket_y = y_max * 1.1;
                    layout.shapes.push({{ type: 'line', x0: x0, y0: bracket_y, x1: x1, y1: bracket_y, line: {{ color: 'black' }} }});
                    layout.annotations.push({{ x: (x0+x1)/2, y: bracket_y*1.05, text: stat.sig, showarrow: false }});
                }});

                Plotly.newPlot(plotDiv, traces, layout, {{ responsive: true }});
            }});
        }});
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"Report successfully generated at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a static HTML report from a Parquet file.")
    parser.add_argument("input_file", type=str, help="Path to the input Parquet file.")
    parser.add_argument("-o", "--output", type=str, default="report.html", help="Path to the output HTML file.")
    args = parser.parse_args()

    df = pl.read_parquet("http://0.0.0.0:8080/examples/exported_projects/images_df.parquet")
    report_data = process_data_for_report(df)
    create_html_report(report_data, args.output)


if __name__ == "__main__":
    main()