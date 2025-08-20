from pixel_patrol.plugins.loaders.bioio_loader import BioIoLoader
from pixel_patrol.plugins.loaders.zarr_loader import ZarrLoader
from pixel_patrol.plugins.processors.basic_stats_processor import BasicStatsProcessor
from pixel_patrol.plugins.processors.histogram_processor import HistogramProcessor
from pixel_patrol.plugins.processors.quality_metrics_processor import QualityMetricsProcessor
from pixel_patrol.plugins.processors.thumbnail_processor import ThumbnailProcessor
from pixel_patrol.plugins.widgets.dataset_stats.dataset_stats import DatasetStatsWidget
from pixel_patrol.plugins.widgets.dataset_stats.dynamic_dataset_metrics import DynamicStatsWidget
from pixel_patrol.plugins.widgets.dataset_stats.dynamic_histograms import SlicedHistogramsWidget
from pixel_patrol.plugins.widgets.dataset_stats.dynamic_quality_metrics import DynamicQualityMetricsWidget
from pixel_patrol.plugins.widgets.dataset_stats.image_quality import ImageQualityWidget
from pixel_patrol.plugins.widgets.metadata.data_type import DataTypeWidget
from pixel_patrol.plugins.widgets.metadata.dim_order import DimOrderWidget
from pixel_patrol.plugins.widgets.metadata.dim_size import DimSizeWidget
from pixel_patrol.plugins.widgets.visualization.image_mosaik import ImageMosaikWidget


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
        DataTypeWidget,
        DimOrderWidget,
        DimSizeWidget,
        ImageMosaikWidget,
        DatasetStatsWidget,
        ImageQualityWidget,
        DynamicStatsWidget,
        DynamicQualityMetricsWidget,
        SlicedHistogramsWidget,
    ]
