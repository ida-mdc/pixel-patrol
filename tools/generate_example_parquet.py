"""
Generate docs/example.parquet from the BBBC006 example dataset (downloaded and cached).

Run from the repo root:
    uv run python tools/generate_example_parquet.py [--cache-dir DIR]
"""

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path

from pixel_patrol_base import api

URL = "https://syncandshare.desy.de/index.php/s/zQjryd2qmZG4Gc7/download"
SHA256 = "e9de0f79efa4cd908ad72ffa79b9704069f1242e2a716b0423a11384422eaa5b"
ARCHIVE = "bbbc006_stacks.tar.gz"
TOP_DIR = "bbbc006_stacks"
CONDITIONS = ["far_below_focus", "below_focus", "near_focus", "far_above_focus"]
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "docs" / "example.parquet"

DESCRIPTION = (
    "Human U2OS cells (Hoechst, phalloidin), 20x, in-focus and out-of-focus z-planes. "
    "Source: BBBC006 (https://bbbc.broadinstitute.org/BBBC006), Anne Carpenter, "
    "Broad Institute, CC0 1.0. Modified: z-planes of one site stacked into ZCYX OME-TIFFs, subset of sites and planes."
)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_dataset(cache_dir: Path) -> Path:
    root = cache_dir / SHA256[:12]
    extracted = root / TOP_DIR
    if extracted.is_dir() and all((extracted / c).is_dir() for c in CONDITIONS):
        return extracted
    root.mkdir(parents=True, exist_ok=True)
    archive = root / ARCHIVE
    if not archive.exists() or sha256_of(archive) != SHA256:
        urllib.request.urlretrieve(URL, archive)
    if sha256_of(archive) != SHA256:
        raise RuntimeError(f"Checksum mismatch for {URL}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root, filter="data")
    return extracted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "pixel-patrol" / "example")
    args = parser.parse_args()

    project = api.create_project(
        "U2OS focus z-stacks (BBBC006)",
        base_dir=get_dataset(args.cache_dir),
        loader="bioio",
        output_path=OUTPUT_PATH,
    )
    api.add_paths(project, CONDITIONS)
    api.process_files(
        project,
        max_workers=5,
        flavor="Example Report",
        description=DESCRIPTION,
        omit_base_dir=True,
    )
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
