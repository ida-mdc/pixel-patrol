import importlib
from typing import Type, Union, List

from pixel_patrol_base.core.contracts import PixelPatrolLoader, PixelPatrolProcessor, PixelPatrolWidget
from pixel_patrol_base.plugins.processors.basic_stats_processor import BasicStatsProcessor
from pixel_patrol_base.plugins.processors.histogram_processor import HistogramProcessor
from pixel_patrol_base.plugins.processors.thumbnail_processor import ThumbnailProcessor
from pixel_patrol_base.plugins.widgets.dataset_stats.dataset_histograms import DatasetHistogramWidget
from pixel_patrol_base.plugins.widgets.dataset_stats.dataset_stats import DatasetStatsWidget
from pixel_patrol_base.plugins.widgets.dataset_stats.dynamic_dataset_metrics import DynamicStatsWidget
from pixel_patrol_base.plugins.widgets.file_stats.file_stats import FileStatisticsWidget
from pixel_patrol_base.plugins.widgets.metadata.data_type import DataTypeWidget
from pixel_patrol_base.plugins.widgets.metadata.dim_order import DimOrderWidget
from pixel_patrol_base.plugins.widgets.metadata.dim_size import DimSizeWidget
from pixel_patrol_base.plugins.widgets.summary.dataframe import DataFrameWidget
from pixel_patrol_base.plugins.widgets.summary.file_summary import FileSummaryWidget
from pixel_patrol_base.plugins.widgets.summary.sunburst import FileSunburstWidget
from pixel_patrol_base.plugins.widgets.visualization.embedding_projector import EmbeddingProjectorWidget
from pixel_patrol_base.plugins.widgets.visualization.image_mosaik import ImageMosaikWidget

PixelPluginClass = Union[Type[PixelPatrolLoader], Type[PixelPatrolProcessor], Type[PixelPatrolWidget]]

def discover_loader(loader_id: str) -> PixelPatrolLoader:
    plugins = discover_plugins_from_entrypoints("pixel_patrol.loader_plugins")
    print("Discovered loader plugins: ", ", ".join([plugin.NAME for plugin in plugins]))
    for loader_plugin in plugins:
        if loader_plugin.NAME == loader_id:
            return loader_plugin()
    raise RuntimeError(f"Could not find loader plugin `{loader_id}` in discovered loader plugins: {[plugin.NAME for plugin in plugins]}")

def discover_processor_plugins() -> List[PixelPatrolProcessor]:
    plugins = discover_plugins_from_entrypoints("pixel_patrol.processor_plugins")
    initialized_plugins = [plugin() for plugin in plugins]
    print("Discovered processor plugins: ", ", ".join([plugin.NAME for plugin in initialized_plugins]))
    return initialized_plugins

def discover_widget_plugins() -> List[PixelPatrolWidget]:
    plugins = discover_plugins_from_entrypoints("pixel_patrol.widget_plugins")
    initialized_plugins = [plugin() for plugin in plugins]
    print("Discovered widget plugins: ", ", ".join([plugin.NAME for plugin in initialized_plugins]))
    return initialized_plugins


def discover_plugins_from_entrypoints(plugins_id) -> List[PixelPluginClass]:
    res: List[PixelPluginClass] = []
    entry_points = importlib.metadata.entry_points(group=plugins_id)
    for ep in entry_points:
        try:
            registration_func = ep.load()
            components = registration_func()
            res.extend(components)
        except Exception as e:
            print(f"Could not load plugin '{ep.name}': {e}")
    return res


def register_processor_plugins():
    return [
        BasicStatsProcessor,
        ThumbnailProcessor,
        HistogramProcessor,
    ]

def register_widget_plugins():
    return [
        FileStatisticsWidget,
        EmbeddingProjectorWidget,
        FileSummaryWidget,
        DataFrameWidget,
        FileSunburstWidget,

        DataTypeWidget,
        DimOrderWidget,
        DimSizeWidget,
        ImageMosaikWidget,
        DatasetStatsWidget,
        DynamicStatsWidget,
        DatasetHistogramWidget,
    ]
