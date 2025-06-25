from abc import ABC, abstractmethod
from typing import List
import polars as pl


class PixelPatrolWidget(ABC):
    @property
    @abstractmethod
    def tab(self) -> str:
        """Return the name of the tab this widget belongs to."""
        pass

    @property
    def name(self) -> str:
        return type(self).__name__

    def required_columns(self) -> List[str]:
        """Returns required data column names"""
        return []

    def uses_example_images(self) -> bool:
        return False

    def summary(self, data_frame: pl.DataFrame):
        """Renders summary"""
        pass


    @abstractmethod
    def layout(self) -> List:
        """Return Dash components (charts, inputs, descriptions)."""
        pass

    def register_callbacks(self, app, df: pl.DataFrame):
        """Attach Dash callbacks, capturing `df` in the closure."""
        return