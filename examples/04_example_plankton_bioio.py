from pathlib import Path
from datasets.get_or_download_example_plankton import get_or_download_example_plankton
from pixel_patrol_base import api

import logging
logging.basicConfig(level=logging.INFO)

def main():
    base_path = get_or_download_example_plankton(target_dir=Path("datasets")).resolve()
    output_path = Path("out/plankton.parquet")
    paths = [p.name for p in base_path.iterdir() if p.is_dir() and not p.name.startswith('.')]

    project = api.create_project("plankton", base_path, loader="bioio", output_path=output_path)
    api.add_paths(project, paths)

    api.process_files(project, processing_max_workers=20, authors="pixel-patrol-team")

    api.show_report(project)



if __name__ == "__main__":
    main()
