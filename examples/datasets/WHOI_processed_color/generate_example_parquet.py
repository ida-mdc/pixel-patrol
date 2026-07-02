"""
Generate docs/example.parquet from the WHOI_processed_color dataset.

Run from the repo root:
    uv run python examples/datasets/WHOI_processed_color/generate_example_parquet.py
"""

from pathlib import Path
from pixel_patrol_base import api

BASE_DIR    = Path(__file__).parent.resolve()
OUTPUT_PATH = Path(__file__).parents[3] / "docs" / "example.parquet"
CONDITIONS  = ["condition1_org", "condition2_bl", "condition3_comp", "condition4_nois"]

def main():
    project = api.create_project(
        "WHOI Plankton",
        base_dir=BASE_DIR,
        loader="bioio",
        output_path=OUTPUT_PATH,
    )
    api.add_paths(project, CONDITIONS)
    api.process_files(
        project,
        max_workers=5,
        flavor="Example Report",
        description=(
            "40 plankton images from the WHOI IFCB dataset, across 4 conditions "
            "(original, blurred, compressed, noisy). Colors are artificially added "
            "to demonstrate multi-channel (S) statistics."
        ),
    )
    print(f"Written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
