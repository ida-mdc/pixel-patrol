"""
Tests for configuration features:
- Processor selection (included/excluded)
- Widget selection (included/excluded)
"""

import pytest

from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.report_config import ReportConfig
from pixel_patrol_base.plugin_registry import discover_processor_plugins, discover_widget_plugins


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
            processors_excluded={"HistogramProcessor"}
        )

        report_config = ReportConfig(
            widgets_excluded={widget_name}
        )

        assert processing_config.processors_excluded == {"HistogramProcessor"}
        assert report_config.widgets_excluded == {widget_name}

    def test_configuration_defaults(self):
        """Test that ProcessingConfig and ReportConfig have correct defaults."""
        processing_config = ProcessingConfig()
        report_config = ReportConfig()

        assert processing_config.processors_included == set()
        assert processing_config.processors_excluded == set()

        assert report_config.widgets_included == set()
        assert report_config.widgets_excluded == set()