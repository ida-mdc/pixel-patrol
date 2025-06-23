from pixel_patrol.core.project import Project
from pixel_patrol.core.project_settings import Settings

import logging

# Configure root logger for basic console output
# This is a basic setup; a more advanced application might allow custom handlers
# and different levels for different modules.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# You can also define a specific logger for the top-level package
logger = logging.getLogger(__name__)
logger.info("Pixel Patrol package initialized.")
