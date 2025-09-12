import logging
import time
from pathlib import Path
from typing import Dict, List

from pixel_patrol_base.core.contracts import PixelPatrolLoader, PixelPatrolProcessor
from pixel_patrol_base.core.specs import is_artifact_matching_processor

logger = logging.getLogger(__name__)

def get_all_artifact_properties(file_path: Path, loader: PixelPatrolLoader, processors: List[PixelPatrolProcessor]) -> Dict:
    start_total_time = time.monotonic()

    if not file_path.exists():
        logger.warning(f"File not found: '{file_path}'. Cannot extract metadata.")
        return {}

    extracted_properties = {}

    logger.info(f"Attempting to load '{file_path}' with loader: {loader.NAME}")

    start_load_time = time.monotonic()
    try:
        art = loader.load(str(file_path))
        metadata = dict(art.meta)
    except Exception as e:
        logger.info(f"Loader '{loader.NAME}' failed with exception, skipping: {e}")
        return {}

    load_duration = time.monotonic() - start_load_time
    logger.info(f"Loading with '{loader.NAME}' took {load_duration:.4f} seconds.")

    # Always process using Artifact; processors opt-in via INPUT spec
    extracted_properties.update(metadata)
    for P in processors:
        if not is_artifact_matching_processor(art, P.INPUT):
            continue
        out = P.run(art)
        if isinstance(out, dict):
            extracted_properties.update(out)
        else:
            art = out  # chainable: processors may transform the artifact
            extracted_properties.update(art.meta)

    total_duration = time.monotonic() - start_total_time
    logger.info(f"Successfully loaded and processed '{file_path}'. Total time: {total_duration} seconds.")
    return extracted_properties

