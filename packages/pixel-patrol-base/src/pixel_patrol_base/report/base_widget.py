from abc import ABC, abstractmethod
from typing import List, Set, Optional, Iterable, Union, Pattern
import polars as pl
from dash.development.base_component import Component

from pixel_patrol_base.report.factory import create_widget_card


class BaseReportWidget(ABC):
    """
    Abstract Base Class for all Report Widgets.

    It implements the 'layout' method of the PixelPatrolWidget protocol
    by wrapping the subclass's content in a standardized Card with a Title
    and an optional Info/Help popover.
    """

    # --- Protocol Requirements (Subclasses MUST override these) ---
    NAME: str = "Unnamed Widget"
    TAB: str = "Other"
    REQUIRES: Set[str] = set()
    REQUIRES_PATTERNS: Optional[Iterable[Union[str, Pattern]]] = None

    # --- New Optional Properties ---
    @property
    def help_text(self) -> Optional[str]:
        """
        Override this property in subclasses to provide text for the
        (i) info popover in the widget header.
        """
        return None

    @property
    def widget_id_prefix(self) -> str:
        """
        Generates a unique-ish ID string based on the class name.
        Useful for generating ID strings for components.
        """
        return self.__class__.__name__.lower()

    # --- The Template Method ---
    def layout(self) -> List[Component]:
        """
        The Standard Layout Entry Point required by PixelPatrolWidget.

        DO NOT OVERRIDE THIS in your widgets.
        Override `get_content_layout` instead.
        """
        # 1. Get the specific content from the subclass
        inner_content = self.get_content_layout()

        # 2. Wrap it in the standard visual shell
        card = create_widget_card(
            title=self.NAME,
            content=inner_content,
            widget_id=self.widget_id_prefix,
            help_text=self.help_text
        )

        # Return as a list because Dash callbacks often expect list-of-components
        return [card]

    @abstractmethod
    def get_content_layout(self) -> List[Component]:
        """
        Subclasses implement this to return ONLY their specific graphs/controls.
        Do not include titles, cards, or outer containers here.
        """
        pass

    @abstractmethod
    def register(self, app, df_global: pl.DataFrame):
        """Standard register method for callbacks."""
        pass