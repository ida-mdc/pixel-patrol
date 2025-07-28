from abc import ABC, abstractmethod
from typing import Any, Tuple, List

import dask.array as da


class PixelPatrolProcessor(ABC):

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def description(self) -> str:
        return "MISSING DESCRIPTION"

    @abstractmethod
    def process(self, data: da.array, dim_order: str) -> dict:
        pass

    @abstractmethod
    def get_specification(self) -> dict:
        return {}

    def get_dynamic_specification_patterns(self) -> List[Tuple[str, Any]]:
        """
        Returns a list of (regex_pattern, polars_data_type) tuples for dynamic columns.
        """
        return []
