import re
from abc import abstractmethod
from typing import List, Protocol, Dict, Any, Tuple


class SpecProvider(Protocol):

    @abstractmethod
    def get_specification(self) -> Dict[str, Any]:
        return {}

    def get_dynamic_specification_patterns(self) -> List[Tuple[str, Any]]:
        """
        Returns a list of (regex_pattern, polars_data_type) tuples for dynamic columns.
        """
        return []

def get_requirements_as_patterns(component: SpecProvider) -> List[str]:
    """
    Consolidates a component's (loader or processor) static and dynamic
    column specifications into a single list of regex patterns.

    Args:
        component: An instance of a loader or processor.

    Returns:
        A list of regex strings representing all required columns.
    """
    # 1. Get exact keys from the static specification.
    exact_keys_as_patterns = [
        f"^{re.escape(key)}$" for key in component.get_specification().keys()
    ]

    # 2. Get the regex patterns from the dynamic specification.
    dynamic_patterns = [
        pattern_tuple[0] for pattern_tuple in component.get_dynamic_specification_patterns()
    ]

    return exact_keys_as_patterns + dynamic_patterns


def get_dynamic_patterns(component: SpecProvider) -> List[str]:
    dynamic_patterns = [
        pattern_tuple[0] for pattern_tuple in component.get_dynamic_specification_patterns()
    ]
    return dynamic_patterns