import re
from typing import List
from pixel_patrol.core.loader_interface import PixelPatrolLoader

def get_loader_requirements_as_patterns(loader: PixelPatrolLoader) -> List[str]:
    """
    Consolidates a loader's static and dynamic column specifications into
    a single list of regex patterns.

    Args:
        loader: An instance of a PixelPatrolLoader.

    Returns:
        A list of regex strings representing all required columns.
    """
    # 1. Get exact keys from the static specification.
    # We anchor them with '^' and '$' to ensure an exact match.
    exact_keys_as_patterns = [
        f"^{re.escape(key)}$" for key in loader.get_specification().keys()
    ]

    # 2. Get the regex patterns from the dynamic specification.
    dynamic_patterns = [
        pattern_tuple[0] for pattern_tuple in loader.get_dynamic_specification_patterns()
    ]

    # 3. Combine and return the lists.
    return exact_keys_as_patterns + dynamic_patterns