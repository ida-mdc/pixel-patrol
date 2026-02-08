"""
Tests for Settings configuration features:
- Slicing configuration (enabled/disabled, included/excluded dimensions)
- Processor selection (included/excluded)
- Widget selection (included/excluded)
"""

import pytest
import numpy as np
import dask.array as da
from pixel_patrol_base.core.project_settings import Settings

from pixel_patrol_base.core.record import record_from
from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.report_config import ReportConfig
from pixel_patrol_base.plugins.processors.basic_stats_processor import BasicStatsProcessor
from pixel_patrol_base.utils.array_utils import (
    calculate_sliced_stats,
    set_slicing_config,
    get_slicing_config,
)
from pixel_patrol_base.plugin_registry import discover_processor_plugins, discover_widget_plugins


class TestSlicingConfiguration:
    """Test slicing configuration features."""

    def test_slicing_disabled(self):
        """Test that disabling slicing only computes full-image stats."""
        # Create a TCYX image: 2 time, 2 channels, 3x3 spatial
        data = np.random.rand(2, 2, 3, 3).astype(np.float32) * 100
        dask_data = da.from_array(data, chunks=(1, 1, 3, 3))
        
        record = record_from(dask_data, {"dim_order": "TCYX"})
        processor = BasicStatsProcessor()
        
        # Set slicing disabled
        processing_config = ProcessingConfig(slicing_enabled=False)
        # Create a temporary object for thread-local storage
        class _TempSlicingConfig:
            def __init__(self, config):
                self.slicing_enabled = config.slicing_enabled
                self.slicing_dimensions_included = config.slicing_dimensions_included
                self.slicing_dimensions_excluded = config.slicing_dimensions_excluded
        set_slicing_config(_TempSlicingConfig(processing_config))
        
        result = processor.run(record)
        
        # Should have full-image stats
        assert "mean_intensity" in result
        
        # Should NOT have per-slice stats
        assert "mean_intensity_t0" not in result
        assert "mean_intensity_t1" not in result
        assert "mean_intensity_c0" not in result
        assert "mean_intensity_c1" not in result
        
        # Cleanup
        set_slicing_config(None)

    def test_slicing_dimensions_included(self):
        """Test slicing only specific dimensions using included."""
        # Create a TCZYX image: 2 time, 2 channels, 2 z-slices, 3x3 spatial
        data = np.random.rand(2, 2, 2, 3, 3).astype(np.float32) * 100
        dask_data = da.from_array(data, chunks=(1, 1, 1, 3, 3))
        
        record = record_from(dask_data, {"dim_order": "TCZYX"})
        processor = BasicStatsProcessor()
        
        # Set to only slice T dimension
        processing_config = ProcessingConfig(slicing_dimensions_included={"T"})
        class _TempSlicingConfig:
            def __init__(self, config):
                self.slicing_enabled = config.slicing_enabled
                self.slicing_dimensions_included = config.slicing_dimensions_included
                self.slicing_dimensions_excluded = config.slicing_dimensions_excluded
        set_slicing_config(_TempSlicingConfig(processing_config))
        
        result = processor.run(record)
        
        # Should have per-time-slice stats
        assert "mean_intensity_t0" in result
        assert "mean_intensity_t1" in result
        
        # Should NOT have per-channel or per-z-slice stats
        assert "mean_intensity_c0" not in result
        assert "mean_intensity_c1" not in result
        assert "mean_intensity_z0" not in result
        assert "mean_intensity_z1" not in result
        
        # Should have full-image stats
        assert "mean_intensity" in result
        
        # Cleanup
        set_slicing_config(None)

    def test_slicing_dimensions_excluded(self):
        """Test excluding specific dimensions from slicing."""
        # Create a TCZYX image: 2 time, 2 channels, 2 z-slices, 3x3 spatial
        data = np.random.rand(2, 2, 2, 3, 3).astype(np.float32) * 100
        dask_data = da.from_array(data, chunks=(1, 1, 1, 3, 3))
        
        record = record_from(dask_data, {"dim_order": "TCZYX"})
        processor = BasicStatsProcessor()
        
        # Exclude Z dimension (should slice T and C, but not Z)
        processing_config = ProcessingConfig(slicing_dimensions_excluded={"X", "Y", "Z"})
        class _TempSlicingConfig:
            def __init__(self, config):
                self.slicing_enabled = config.slicing_enabled
                self.slicing_dimensions_included = config.slicing_dimensions_included
                self.slicing_dimensions_excluded = config.slicing_dimensions_excluded
        set_slicing_config(_TempSlicingConfig(processing_config))
        
        result = processor.run(record)
        
        # Should have per-time-slice stats
        assert "mean_intensity_t0" in result
        assert "mean_intensity_t1" in result
        
        # Should have per-channel stats
        assert "mean_intensity_c0" in result
        assert "mean_intensity_c1" in result
        
        # Should NOT have per-z-slice stats (Z was excluded)
        assert "mean_intensity_z0" not in result
        assert "mean_intensity_z1" not in result
        
        # Cleanup
        set_slicing_config(None)

    def test_slicing_default_behavior(self):
        """Test that default behavior slices all dimensions except X and Y."""
        # Create a TCZYX image: 2 time, 2 channels, 2 z-slices, 3x3 spatial
        data = np.random.rand(2, 2, 2, 3, 3).astype(np.float32) * 100
        dask_data = da.from_array(data, chunks=(1, 1, 1, 3, 3))
        
        record = record_from(dask_data, {"dim_order": "TCZYX"})
        processor = BasicStatsProcessor()
        
        # Use default settings (no config set)
        set_slicing_config(None)
        
        result = processor.run(record)
        
        # Should have per-dimension stats for T, C, Z (but not X, Y)
        assert "mean_intensity_t0" in result
        assert "mean_intensity_c0" in result
        assert "mean_intensity_z0" in result
        
        # Cleanup
        set_slicing_config(None)

    def test_slicing_included_takes_precedence(self):
        """Test that included takes precedence over excluded when both are set."""
        # Create a TCZYX image: 2 time, 2 channels, 2 z-slices, 3x3 spatial
        data = np.random.rand(2, 2, 2, 3, 3).astype(np.float32) * 100
        dask_data = da.from_array(data, chunks=(1, 1, 1, 3, 3))
        
        record = record_from(dask_data, {"dim_order": "TCZYX"})
        processor = BasicStatsProcessor()
        
        # Set both included and excluded - included should take precedence
        processing_config = ProcessingConfig(
            slicing_dimensions_included={"T"},
            slicing_dimensions_excluded={"C"}  # This should be ignored
        )
        class _TempSlicingConfig:
            def __init__(self, config):
                self.slicing_enabled = config.slicing_enabled
                self.slicing_dimensions_included = config.slicing_dimensions_included
                self.slicing_dimensions_excluded = config.slicing_dimensions_excluded
        set_slicing_config(_TempSlicingConfig(processing_config))
        
        result = processor.run(record)
        
        # Should only have T slices (included takes precedence)
        assert "mean_intensity_t0" in result
        assert "mean_intensity_t1" in result
        
        # Should NOT have C slices (even though C is in excluded, included takes precedence)
        assert "mean_intensity_c0" not in result
        assert "mean_intensity_c1" not in result
        
        # Cleanup
        set_slicing_config(None)


class TestProcessorSelection:
    """Test processor selection features."""

    def test_processors_included(self):
        """Test including only specific processors."""
        processing_config = ProcessingConfig(processors_included={"BasicStatsProcessor"})
        
        # Get all discovered processors
        all_processors = discover_processor_plugins()
        
        # Filter processors based on processing_config
        processors_included = processing_config.processors_included
        processors_excluded = processing_config.processors_excluded
        
        if processors_included:
            filtered = [p for p in all_processors if p.__class__.__name__ in processors_included]
        elif processors_excluded:
            filtered = [p for p in all_processors if p.__class__.__name__ not in processors_excluded]
        else:
            filtered = all_processors
        
        assert len(filtered) == 1
        assert filtered[0].__class__.__name__ in processors_included

    def test_processors_excluded(self):
        """Test excluding specific processors."""
        processing_config = ProcessingConfig(processors_excluded={"HistogramProcessor"})
        
        # Get all discovered processors
        all_processors = discover_processor_plugins()
        
        # Filter processors based on processing_config
        processors_included = processing_config.processors_included
        processors_excluded = processing_config.processors_excluded
        
        if processors_included:
            filtered = [p for p in all_processors if p.__class__.__name__ in processors_included]
        elif processors_excluded:
            filtered = [p for p in all_processors if p.__class__.__name__ not in processors_excluded]
        else:
            filtered = all_processors
        
        filtered_names = {p.__class__.__name__ for p in filtered}
        assert "HistogramProcessor" not in filtered_names
        assert "BasicStatsProcessor" in filtered_names

    def test_processors_default_all(self):
        """Test that default behavior uses all processors."""
        processing_config = ProcessingConfig()  # Default config
        
        # Get all discovered processors
        all_processors = discover_processor_plugins()
        
        # Filter processors based on processing_config
        processors_included = processing_config.processors_included
        processors_excluded = processing_config.processors_excluded
        
        if processors_included:
            filtered = [p for p in all_processors if p.__class__.__name__ in processors_included]
        elif processors_excluded:
            filtered = [p for p in all_processors if p.__class__.__name__ not in processors_excluded]
        else:
            filtered = all_processors
        
        # Should have all processors
        assert len(filtered) == len(all_processors)

    def test_processors_included_takes_precedence(self):
        """Test that included takes precedence over excluded."""
        processing_config = ProcessingConfig(
            processors_included={"BasicStatsProcessor"},
            processors_excluded={"BasicStatsProcessor"}  # This should be ignored
        )
        
        # Get all discovered processors
        all_processors = discover_processor_plugins()
        
        # Filter processors based on processing_config
        processors_included = processing_config.processors_included
        processors_excluded = processing_config.processors_excluded
        
        if processors_included:
            filtered = [p for p in all_processors if p.__class__.__name__ in processors_included]
        elif processors_excluded:
            filtered = [p for p in all_processors if p.__class__.__name__ not in processors_excluded]
        else:
            filtered = all_processors
        
        assert len(filtered) == 1
        assert filtered[0].__class__.__name__ in processors_included


class TestWidgetSelection:
    """Test widget selection features."""

    def test_widgets_included(self):
        """Test including only specific widgets."""
        all_widgets = discover_widget_plugins()
        if not all_widgets:
            pytest.skip("No widgets discovered")
        
        widget_names = {w.__class__.__name__ for w in all_widgets}
        test_widgets = list(widget_names)[:2]
        
        report_config = ReportConfig(widgets_included=set(test_widgets))
        
        if report_config.widgets_included:
            filtered = [w for w in all_widgets if w.__class__.__name__ in report_config.widgets_included]
        elif report_config.widgets_excluded:
            filtered = [w for w in all_widgets if w.__class__.__name__ not in report_config.widgets_excluded]
        else:
            filtered = all_widgets
        
        assert len(filtered) == len(test_widgets)
        filtered_names = {w.__class__.__name__ for w in filtered}
        assert filtered_names == set(test_widgets)

    def test_widgets_excluded(self):
        """Test excluding specific widgets."""
        all_widgets = discover_widget_plugins()
        if not all_widgets:
            pytest.skip("No widgets discovered")
        
        widget_names = {w.__class__.__name__ for w in all_widgets}
        widget_to_exclude = list(widget_names)[0]
        
        report_config = ReportConfig(widgets_excluded={widget_to_exclude})
        
        if report_config.widgets_included:
            filtered = [w for w in all_widgets if w.__class__.__name__ in report_config.widgets_included]
        elif report_config.widgets_excluded:
            filtered = [w for w in all_widgets if w.__class__.__name__ not in report_config.widgets_excluded]
        else:
            filtered = all_widgets
        
        filtered_names = {w.__class__.__name__ for w in filtered}
        assert widget_to_exclude not in filtered_names
        assert len(filtered) == len(all_widgets) - 1

    def test_widgets_default_all(self):
        """Test that default behavior uses all widgets."""
        report_config = ReportConfig()
        
        all_widgets = discover_widget_plugins()
        
        if report_config.widgets_included:
            filtered = [w for w in all_widgets if w.__class__.__name__ in report_config.widgets_included]
        elif report_config.widgets_excluded:
            filtered = [w for w in all_widgets if w.__class__.__name__ not in report_config.widgets_excluded]
        else:
            filtered = all_widgets
        
        assert len(filtered) == len(all_widgets)

    def test_widgets_included_takes_precedence(self):
        """Test that included takes precedence over excluded."""
        all_widgets = discover_widget_plugins()
        if not all_widgets:
            pytest.skip("No widgets discovered")
        
        widget_names = {w.__class__.__name__ for w in all_widgets}
        test_widget = list(widget_names)[0]
        
        report_config = ReportConfig(
            widgets_included={test_widget},
            widgets_excluded={test_widget}  # This should be ignored
        )
        
        if report_config.widgets_included:
            filtered = [w for w in all_widgets if w.__class__.__name__ in report_config.widgets_included]
        elif report_config.widgets_excluded:
            filtered = [w for w in all_widgets if w.__class__.__name__ not in report_config.widgets_excluded]
        else:
            filtered = all_widgets
        
        assert len(filtered) == 1
        assert filtered[0].__class__.__name__ == test_widget


class TestCombinedConfiguration:
    """Test combinations of configuration options."""

    def test_combined_configuration(self):
        """Test combining multiple configuration options."""
        all_widgets = discover_widget_plugins()
        widget_name = list({w.__class__.__name__ for w in all_widgets})[0] if all_widgets else "TestWidget"
        
        processing_config = ProcessingConfig(
            slicing_enabled=False,
            slicing_dimensions_included={"T"},
            processors_excluded={"HistogramProcessor"}
        )
        
        report_config = ReportConfig(
            widgets_excluded={widget_name}
        )
        
        assert processing_config.slicing_enabled is False
        assert processing_config.slicing_dimensions_included == {"T"}
        assert processing_config.processors_excluded == {"HistogramProcessor"}
        assert report_config.widgets_excluded == {widget_name}

    def test_configuration_defaults(self):
        """Test that ProcessingConfig and ReportConfig have correct defaults."""
        processing_config = ProcessingConfig()
        report_config = ReportConfig()
        
        assert processing_config.slicing_enabled is True
        assert processing_config.slicing_dimensions_included == set()
        assert processing_config.slicing_dimensions_excluded == {"X", "Y"}
        assert processing_config.processors_included == set()
        assert processing_config.processors_excluded == set()
        
        assert report_config.widgets_included == set()
        assert report_config.widgets_excluded == set()
