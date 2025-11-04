from pixel_patrol_base.core.project import Project
from pixel_patrol_base.core.project_settings import Settings

import logging
import random
import numpy as np
import dask

# TODO: probably most of this should be moved to a setup file and called.
# We're configuring runtime environment. If an app imports pixel-patrol-base, we're forcing those settings.

# Configure root logger for basic console output
# This is a basic setup; a more advanced application might allow custom handlers
# and different levels for different modules.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# You can also define a specific logger for the top-level package
logger = logging.getLogger(__name__)
logger.info("Pixel Patrol package initialized.")

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
dask.config.set({"random.seed": 42})
