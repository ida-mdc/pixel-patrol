import importlib.util
from collections import defaultdict, OrderedDict
from typing import List

from pixel_patrol.report.widget_categories import WidgetCategories
from pixel_patrol.report.widget_interface import PixelPatrolWidget


def organize_widgets_by_tab(widgets):
    """Organize widgets based on their designated tabs."""
    tabbed_widgets = defaultdict(list)
    for widget in widgets:
        tabbed_widgets[widget.tab].append(widget)
    default_tab_values = {tab.value for tab in WidgetCategories}
    ordered_keys = [tab.value for tab in WidgetCategories if tab.value in tabbed_widgets]
    extra_keys = [tab for tab in tabbed_widgets if tab not in default_tab_values]
    all_keys = ordered_keys + extra_keys
    return OrderedDict((tab, tabbed_widgets[tab]) for tab in all_keys)
