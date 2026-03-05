import importlib
import logging
from typing import Type, Union, List

from pixel_patrol_base.core.contracts import PixelPatrolLoader, PixelPatrolProcessor, PixelPatrolWidget
from pixel_patrol_base.plugins.processors.basic_stats_processor import BasicStatsProcessor
from pixel_patrol_base.plugins.processors.histogram_processor import HistogramProcessor
from pixel_patrol_base.plugins.processors.thumbnail_processor import ThumbnailProcessor
from pixel_patrol_base.plugins.widgets.dataset_stats.dataset_histograms import DatasetHistogramWidget
from pixel_patrol_base.plugins.widgets.dataset_stats.dataset_stats import DatasetStatsWidget
from pixel_patrol_base.plugins.widgets.dataset_stats.dataset_stats_across_dims import DatasetStatsAcrossDimensionsWidget
from pixel_patrol_base.plugins.widgets.file_stats.file_stats import FileStatisticsWidget
from pixel_patrol_base.plugins.widgets.metadata.data_type import DataTypeWidget
from pixel_patrol_base.plugins.widgets.metadata.dim_order import DimOrderWidget
from pixel_patrol_base.plugins.widgets.metadata.dim_size import DimSizeWidget
from pixel_patrol_base.plugins.widgets.summary.dataframe import DataFrameWidget
from pixel_patrol_base.plugins.widgets.summary.file_summary import FileSummaryWidget
from pixel_patrol_base.plugins.widgets.summary.sunburst import FileSunburstWidget
from pixel_patrol_base.plugins.widgets.visualization.embedding_projector import EmbeddingProjectorWidget
from pixel_patrol_base.plugins.widgets.visualization.image_mosaik import ImageMosaikWidget

logger = logging.getLogger(__name__)

PixelPluginClass = Union[Type[PixelPatrolLoader], Type[PixelPatrolProcessor], Type[PixelPatrolWidget]]

# Simple in-process caches so we don't repeatedly hit entry points or spam logs
_CACHED_LOADER_PLUGINS: list[PixelPatrolLoader] | None = None
_CACHED_PROCESSOR_PLUGINS: list[PixelPatrolProcessor] | None = None
_CACHED_WIDGET_PLUGINS: list[PixelPatrolWidget] | None = None

def discover_loader(loader_id: str) -> PixelPatrolLoader:
    global _CACHED_LOADER_PLUGINS
    if _CACHED_LOADER_PLUGINS is None:
        plugin_classes = discover_plugins_from_entrypoints("pixel_patrol.loader_plugins")
        _CACHED_LOADER_PLUGINS = [plugin_class() for plugin_class in plugin_classes]
        logger.info(
            "Discovered loader plugins: %s",
            ", ".join([plugin.NAME for plugin in _CACHED_LOADER_PLUGINS]),
        )
    plugins = _CACHED_LOADER_PLUGINS
    for loader_plugin in plugins:
        if loader_plugin.NAME == loader_id:
            # loader_plugin is already an instantiated loader; just return it.
            return loader_plugin
    raise RuntimeError(
        f'Could not find loader plugin "{loader_id}" in discovered loader plugins: '
        f'{[plugin.NAME for plugin in plugins]}'
    )

def discover_processor_plugins() -> List[PixelPatrolProcessor]:
    global _CACHED_PROCESSOR_PLUGINS
    if _CACHED_PROCESSOR_PLUGINS is None:
        plugin_classes = discover_plugins_from_entrypoints("pixel_patrol.processor_plugins")
        _CACHED_PROCESSOR_PLUGINS = [plugin_class() for plugin_class in plugin_classes]
        logger.info(
            "Discovered processor plugins: %s",
            ", ".join([plugin.NAME for plugin in _CACHED_PROCESSOR_PLUGINS]),
        )
    # Return a shallow copy so callers can't mutate our cache
    return list(_CACHED_PROCESSOR_PLUGINS)

def discover_widget_plugins() -> List[PixelPatrolWidget]:
    global _CACHED_WIDGET_PLUGINS
    if _CACHED_WIDGET_PLUGINS is None:
        plugin_classes = discover_plugins_from_entrypoints("pixel_patrol.widget_plugins")
        _CACHED_WIDGET_PLUGINS = [plugin_class() for plugin_class in plugin_classes]
        logger.info(
            "Discovered widget plugins: %s",
            ", ".join([plugin.NAME for plugin in _CACHED_WIDGET_PLUGINS]),
        )
    # Return a shallow copy so callers can't mutate our cache
    return list(_CACHED_WIDGET_PLUGINS)


def discover_plugins_from_entrypoints(plugins_id) -> List[PixelPluginClass]:
    res: List[PixelPluginClass] = []
    entry_points = importlib.metadata.entry_points(group=plugins_id)
    for ep in entry_points:
        try:
            registration_func = ep.load()
            components = registration_func()
            res.extend(components)
        except Exception as e:
            logger.error(f"Could not load plugin '{ep.name}': {e}")
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
        DatasetStatsAcrossDimensionsWidget,
        DatasetHistogramWidget,
    ]
