from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Any, Dict, List

import dask.array as da


class PixelPatrolLoader(ABC):

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def reads_only_metadata(self) -> bool:
        return False

    @abstractmethod
    def read_metadata(self, path: Path) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def read_metadata_and_data(self, path: Path) -> Tuple[dict, da.array]:
        raise NotImplementedError()
