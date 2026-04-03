from pathlib import Path
from datasets.get_or_download_example_plankton import get_or_download_example_plankton
from pixel_patrol_base import api
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.project_metadata import ProjectMetadata

import logging
logging.basicConfig(level=logging.INFO)

def main():
    base_path = get_or_download_example_plankton(target_dir=Path("datasets")).resolve()
    output_path = Path("out/plankton.parquet")
    paths = [p.name for p in base_path.iterdir() if p.is_dir() and not p.name.startswith('.')]

    project = api.create_project("plankton", base_path, loader="bioio", output_path=output_path)
    api.add_paths(project, paths)

    processing_config = ProcessingConfig(
        processing_max_workers=20,
        metadata=ProjectMetadata(authors="pixel-patrol-team"),
    )

    api.process_files(project, processing_config=processing_config)

    api.show_report(project)



if __name__ == "__main__":
    main()
