import importlib
from typing import Type, Union, List

from pixel_patrol.core.loader_interface import PixelPatrolLoader
from pixel_patrol.core.loaders.bioio_loader import BioIoLoader
from pixel_patrol.core.loaders.zarr_loader import ZarrLoader
from pixel_patrol.core.processor_interface import PixelPatrolProcessor
from pixel_patrol.core.processors.basic_stats_processor import BasicStatsProcessor
from pixel_patrol.core.processors.histogram_processor import HistogramProcessor
from pixel_patrol.core.processors.quality_metrics_processor import QualityMetricsProcessor
from pixel_patrol.core.processors.thumbnail_processor import ThumbnailProcessor
from pixel_patrol.report.widgets.dataset_stats.dataset_stats import DatasetStatsWidget
from pixel_patrol.report.widgets.dataset_stats.dynamic_dataset_metrics import DynamicStatsWidget
from pixel_patrol.report.widgets.dataset_stats.dynamic_quality_metrics import DynamicQualityMetricsWidget
from pixel_patrol.report.widgets.dataset_stats.image_quality import ImageQualityWidget
from pixel_patrol.report.widgets.file_stats.file_stats import FileStatisticsWidget
from pixel_patrol.report.widgets.metadata.data_type import DataTypeWidget
from pixel_patrol.report.widgets.metadata.dim_order import DimOrderWidget
from pixel_patrol.report.widgets.metadata.dim_size import DimSizeWidget
from pixel_patrol.report.widgets.summary.dataframe import DataFrameWidget
from pixel_patrol.report.widgets.summary.file_summary import FileSummaryWidget
from pixel_patrol.report.widgets.summary.sunburst import FileSunburstWidget
from pixel_patrol.report.widgets.visualization.embedding_projector import EmbeddingProjectorWidget
from pixel_patrol.report.widgets.visualization.image_mosaik import ImageMosaikWidget
from pixel_patrol.report.widget_interface import PixelPatrolWidget
from pixel_patrol.report.widgets.dataset_stats.dynamic_histograms import SlicedHistogramsWidget

PixelPluginClass = Union[Type[PixelPatrolLoader], Type[PixelPatrolProcessor], Type[PixelPatrolWidget]]

def discover_loader(loader_id: str) -> PixelPatrolLoader:
    plugins = discover_plugins_from_entrypoints("pixel_patrol.loader_plugins")
    print("Discovered loader plugins: ", ", ".join([plugin.id() for plugin in plugins]))
    for loader_plugin in plugins:
        if loader_plugin.id() == loader_id:
            return loader_plugin()
    raise RuntimeError(f"Could not find loader plugin `{loader_id}` in discovered loader plugins: {[plugin.id() for plugin in plugins]}")

def discover_processor_plugins() -> List[PixelPatrolProcessor]:
    plugins = discover_plugins_from_entrypoints("pixel_patrol.processor_plugins")
    initialized_plugins = [plugin() for plugin in plugins]
    print("Discovered processor plugins: ", ", ".join([plugin.name for plugin in initialized_plugins]))
    return initialized_plugins

def discover_widget_plugins() -> List[PixelPatrolWidget]:
    plugins = discover_plugins_from_entrypoints("pixel_patrol.widget_plugins")
    initialized_plugins = [plugin() for plugin in plugins]
    print("Discovered widget plugins: ", ", ".join([plugin.name for plugin in initialized_plugins]))
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

def register_loader_plugins():
    return [
        BioIoLoader,
        ZarrLoader,
    ]

def register_processor_plugins():
    return [
        BasicStatsProcessor,
        QualityMetricsProcessor,
        ThumbnailProcessor,
        HistogramProcessor,
    ]

def register_widget_plugins():
    return [
        FileStatisticsWidget,
        DataTypeWidget,
        DimOrderWidget,
        DimSizeWidget,
        ImageMosaikWidget,
        DatasetStatsWidget,
        EmbeddingProjectorWidget,
        ImageQualityWidget,
        FileSummaryWidget,
        DataFrameWidget,
        FileSunburstWidget,
        DynamicStatsWidget,
        DynamicQualityMetricsWidget,
        SlicedHistogramsWidget,
    ]
