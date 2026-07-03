"""Run Pixel Patrol directly on remote images listed in an S3 manifest.

Dataset: cpg0036-EU-OS-bioactives (EU-OPENSCREEN bioactives Cell Painting),
hosted anonymously on the AWS Cell Painting Gallery. Images are never
downloaded as a set - each channel TIFF is streamed from S3 on demand.
Analysis code for this dataset: https://github.com/schmiedc/EU-OS_bioactives

How it works:
  * The manifest is a Cell Painting load_data.csv with one row per field-of-view
    and a URL_* column per channel (URL_OrigDNA, URL_OrigMito, ...).
  * ManifestSource turns each URL_* cell into one image to process and attaches
    the row's Metadata_* columns (Plate, Well, Site) to each record.
  * The 'bioio' loader opens the s3:// TIFFs via fsspec (anonymously).

Requires the pixel-patrol-loader-bio package and s3fs (an fsspec S3 backend).
"""
from pathlib import Path
import logging

from pixel_patrol_base import api
from pixel_patrol_base.plugins.sources.manifest_source import ManifestSource

logging.basicConfig(level=logging.INFO)

# One plate's manifest on the public Cell Painting Gallery (anonymous access).
MANIFEST_URL = (
    "s3://cellpainting-gallery/cpg0036-EU-OS-bioactives/FMP/workspace/"
    "load_data_csv/2021_09_03_Batch1_HepG2/B1001_R1/load_data.csv"
)

# A full plate manifest is ~3,456 rows x 4 channels (~13k images). Sample the
# first few rows so the example runs in a couple of minutes; set to None to
# process the whole plate.
SAMPLE_ROWS = 4


def main():
    output_dir = Path("out")
    output_dir.mkdir(parents=True, exist_ok=True)

    # An explicit ManifestSource lets us cap the number of rows for the demo.
    # (Without a source= override, a .csv path is auto-routed to a ManifestSource
    #  that reads every row.)
    source = ManifestSource(max_rows=SAMPLE_ROWS)

    # base_dir is only used for the local output parquet - the images are remote.
    project = api.create_project(
        "cytodata-eu-os-bioactives",
        base_dir=output_dir,
        loader="bioio",
        output_path=output_dir / "cytodata.parquet",
        source=source,
    )
    api.add_paths(project, MANIFEST_URL)

    # max_workers=1: remote reads are currently only stable single-worker.
    api.process_files(
        project,
        max_workers=1,
        description=(
            "cpg0036-EU-OS-bioactives (FMP, HepG2) streamed from the AWS Cell "
            "Painting Gallery via an S3 load_data.csv manifest."
        ),
    )

    api.view(project)


if __name__ == "__main__":
    main()
