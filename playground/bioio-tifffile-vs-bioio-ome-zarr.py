# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "bioio==3.0.0",
#   "bioio-tifffile==1.3.0",
#   "bioio-ome-zarr==2.3.0",
#   "tifffile==2025.9.20",
# ]
# ///
from pathlib import Path
from urllib.request import urlopen

from bioio import BioImage

FIJI_URL = "https://samples.fiji.sc/FakeTracks.tif"

def download(url: str, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as r, open(dst, "wb") as f:
        f.write(r.read())
    return dst

def main():
    path = Path.cwd() / "FakeTracks.tif"
    if not path.exists():
        print(f"Downloading sample TIFF from {FIJI_URL} …")
        download(FIJI_URL, path)

    print("bioio + plugins imported successfully.")
    img = BioImage(str(path))
    img.dask_data.compute()

if __name__ == "__main__":
    main()
