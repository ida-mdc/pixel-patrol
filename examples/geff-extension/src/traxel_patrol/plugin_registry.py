from traxel_patrol.loaders.geff_loader import GeffLoader
from traxel_patrol.loaders.traccuracy_loader import TraccuracyLoader
from traxel_patrol.widgets.geff_dynamic_metrics_widget import DynamicGeffMetricsWidget
from traxel_patrol.widgets.geff_summary_widget import GeffSummaryWidget
from traxel_patrol.widgets.traccuracy_summary_widget import TraccuracySummaryWidget


def register_loader_plugins():
    return [
        GeffLoader,
        TraccuracyLoader,
    ]

def register_widget_plugins():
    return [
        GeffSummaryWidget,
        DynamicGeffMetricsWidget,
        TraccuracySummaryWidget,
    ]
