import importlib
from typing import Union, Type, Dict, List

from pixel_patrol.core.loader_interface import PixelPatrolLoader
from pixel_patrol.core.loaders.bioio_loader import BioIoLoader
from pixel_patrol.core.loaders.geff_loader import GeffLoader
from pixel_patrol.core.loaders.traccuracy_loader import TraccuracyLoader
from pixel_patrol.core.loaders.zarr_loader import ZarrLoader
from pixel_patrol.core.processor_interface import PixelPatrolProcessor
from pixel_patrol.core.processors.basic_stats_processor import BasicStatsProcessor
from pixel_patrol.core.processors.quality_metrics_processor import QualityMetricsProcessor
from pixel_patrol.core.processors.thumbnail_processor import ThumbnailProcessor
from pixel_patrol.report.widget_interface import PixelPatrolWidget
from pixel_patrol.report.widgets.dataset_stats.dataset_stats import DatasetStatsWidget
from pixel_patrol.report.widgets.dataset_stats.dynamic_dataset_metrics import DynamicStatsWidget
from pixel_patrol.report.widgets.dataset_stats.dynamic_quality_metrics import DynamicQualityMetricsWidget
from pixel_patrol.report.widgets.dataset_stats.image_quality import ImageQualityWidget
from pixel_patrol.report.widgets.file_stats.file_stats import FileStatisticsWidget
from pixel_patrol.report.widgets.geff.geff_dynamic_metrics import GeffDynamicMetricsWidget
from pixel_patrol.report.widgets.metadata.data_type import DataTypeWidget
from pixel_patrol.report.widgets.metadata.dim_order import DimOrderWidget
from pixel_patrol.report.widgets.metadata.dim_size import DimSizeWidget
from pixel_patrol.report.widgets.summary.dataframe import DataFrameWidget
from pixel_patrol.report.widgets.summary.file_summary import FileSummaryWidget
from pixel_patrol.report.widgets.geff.traccuracy_summary import TraccuracySummaryWidget
from pixel_patrol.report.widgets.geff.geff_summary import GeffSummaryWidget
from pixel_patrol.report.widgets.summary.sunburst import FileSunburstWidget
from pixel_patrol.report.widgets.visualization.embedding_projector import EmbeddingProjectorWidget
from pixel_patrol.report.widgets.visualization.image_mosaik import ImageMosaikWidget

ComponentClass = Union[Type[PixelPatrolLoader], Type[PixelPatrolProcessor]]
RegistryProcessing = Dict[str, List[ComponentClass]]
RegistryReport = Dict[str, List[PixelPatrolWidget]]


def discover_processing_plugins() -> RegistryProcessing:
    """
    Discovers all components from the 'pixel_patrol.plugins' entry point
    and organizes them into a central registry.
    """
    registry: RegistryProcessing = {
        "loaders": [],
        "processors": [],
    }

    update_registry_from_entrypoints("pixel_patrol.processing_plugins", registry)
    print("Discovered processing plugins: ", registry["loaders"], " ", registry["processors"])

    return registry

def discover_report_plugins() -> RegistryReport:
    """
    Discovers all components from the 'pixel_patrol.plugins' entry point
    and organizes them into a central registry.
    """
    print("Discovering Pixel Patrol report plugins...")
    registry: RegistryReport = {
        "group_widgets": [],
        "individual_widgets": [],
    }

    update_registry_from_entrypoints("pixel_patrol.report_plugins", registry)

    return registry


def update_registry_from_entrypoints(plugins_id, registry):
    entry_points = importlib.metadata.entry_points(group=plugins_id)
    for ep in entry_points:
        try:
            registration_func = ep.load()
            components = registration_func()
            for component_type, items in components.items():
                if component_type in registry:
                    registry[component_type].extend(i() for i in items)
        except Exception as e:
            print(f"Could not load plugin '{ep.name}': {e}")


def register_processing_plugins():
    return {
        "loaders": [
            BioIoLoader,
            ZarrLoader,
            GeffLoader,
            TraccuracyLoader,
        ],
        "processors": [
            BasicStatsProcessor,
            QualityMetricsProcessor,
            ThumbnailProcessor,
        ],
    }

def register_report_plugins():
    """
    Register all built-in loaders, processors, and widgets for Pixel Patrol.
    This function is discovered through the 'pixel_patrol.plugins' entry point.
    """
    return {
        "group_widgets": [
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
            TraccuracySummaryWidget,
            GeffSummaryWidget,
            FileSunburstWidget,
            DynamicStatsWidget,
            DynamicQualityMetricsWidget,
            GeffDynamicMetricsWidget
        ],
        "individual_widgets": [
        ]
    }