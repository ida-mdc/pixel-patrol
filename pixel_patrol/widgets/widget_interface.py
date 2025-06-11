from abc import ABC, abstractmethod
from typing import List
import polars as pl


class PixelPatrolWidget(ABC):
    @property
    @abstractmethod
    def category(self) -> str:
        """Return the name of the tab this widget belongs to."""
        pass

    @property
    def name(self) -> str:
        return type(self).__name__

    def required_columns(self) -> List[str]:
        """Returns required data column names"""
        return []

    def summary(self, data_frame: pl.DataFrame):
        """Renders summary"""
        pass
    #
    # def options(self, data_frame: pl.DataFrame) -> dict:
    #     """Should display any streamlit input objects like buttons or sliders to determine options of the widget interactively. Returns the options."""
    #     return {}

    # @abstractmethod
    # def render(self, data_frame: pl.DataFrame, *nargs):
    #     """Execute the widget's functionality without any streamlit input objects such as buttons etc.."""
    #     pass

    # def uses_example_images(self) -> bool:
    #     return False