from pixel_patrol_base.api import create_project, set_settings, process_images, show_report
from pixel_patrol_base.core.project_settings import Settings

# 1. Create a project using the BioIO loader
project = create_project(
    name="my_demo_bioio_project",
    base_dir="/home/ella/work/pixel-patrol/packages/pixel-patrol/tests/data",   # replace with your dataset
    loader="bioio"                            # loader NAME defined in bioio_loader.py
)

# 2. Configure settings (e.g., accept all supported extensions)
project = set_settings(project, Settings(selected_file_extensions="all"))

# 3. Run processing (loader + processors)
project = process_images(project)

# 4. Launch interactive dashboard report
show_report(project)