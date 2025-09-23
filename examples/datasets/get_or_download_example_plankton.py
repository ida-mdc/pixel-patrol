from pathlib import Path
import requests, tarfile, os

PLANKTON_URL = "https://syncandshare.desy.de/index.php/s/R5GLtp9DBSy68R9/download?path=/&files=plankton_processed.tar.gz"
ARCHIVE_NAME = "plankton_processed.tar.gz"
EXTRACTED_TOP = "plankton_processed"

def get_or_download_example_plankton(target_dir: Path) -> Path:
    """
    Ensures example data exists under target_dir (e.g., datasets/plankton).
    Returns the path to datasets/plankton/plankton_processed
    which contains the condition subfolders.
    """
    target_dir = Path(target_dir)
    extracted = target_dir / EXTRACTED_TOP
    if extracted.is_dir() and any(extracted.iterdir()):
        return extracted

    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / ARCHIVE_NAME

    # download if not cached
    if not archive_path.exists():
        with requests.get(PLANKTON_URL, stream=True) as r:
            r.raise_for_status()
            with open(archive_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    # extract
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir)

    # optional: clean up archive
    try:
        os.remove(archive_path)
    except OSError:
        pass

    return extracted
