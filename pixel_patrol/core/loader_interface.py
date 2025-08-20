from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Any, List

import dask.array as da


class PixelPatrolLoader(ABC):

    @staticmethod
    @abstractmethod
    def id() -> str:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def description(self) -> str:
        return ""

    @property
    def reads_only_metadata(self) -> bool:
        return False

    @abstractmethod
    def read_metadata(self, path: Path) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def read_metadata_and_data(self, path: Path, filtered_extensions) -> Tuple[dict, da.array]:
        raise NotImplementedError()

    def get_supported_file_extensions(self):
        return []

    @abstractmethod
    def get_specification(self) -> dict:
        return {}

    def get_dynamic_specification_patterns(self) -> List[Tuple[str, Any]]:
        """
        Returns a list of (regex_pattern, polars_data_type) tuples for dynamic columns.
        """
        return []
