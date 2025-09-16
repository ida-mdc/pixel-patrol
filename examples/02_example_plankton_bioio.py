from pathlib import Path

from datasets.get_or_download_example_plankton import get_or_download_example_plankton
from pixel_patrol_base.api import (
    create_project,
    add_paths,
    set_settings,
    process_files,
    show_report,
    export_project,
    import_project,
)
from pixel_patrol_base.core.project_settings import Settings

import logging
logging.basicConfig(level=logging.INFO)

base_path = get_or_download_example_plankton(target_dir=Path("datasets")).resolve()

paths = [p.name for p in base_path.iterdir() if p.is_dir() and not p.name.startswith('.')]

settings = Settings(selected_file_extensions={"png","tif","tiff","jpg","jpeg"})

my_project = create_project("Plankton Project", base_path, loader="bioio")
my_project = add_paths(my_project, paths)
my_project = set_settings(my_project, settings)
my_project = process_files(my_project)

# Open http://127.0.0.1:8050/ in your browser
show_report(my_project)

# Optional: export & import to out/
zip_path = Path("out/plankton_project.zip")
zip_path.parent.mkdir(parents=True, exist_ok=True)
export_project(my_project, zip_path)
imported = import_project(zip_path)