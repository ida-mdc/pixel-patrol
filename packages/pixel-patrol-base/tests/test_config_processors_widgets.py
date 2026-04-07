"""
Tests for configuration features:
- Processor selection (included/excluded)
- Widget selection (included/excluded)
"""

import pytest

from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.core.report_config import ReportConfig
from pixel_patrol_base.report.dashboard_app import _filter_widgets
from pixel_patrol_base.plugin_registry import discover_processor_plugins, discover_widget_plugins


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_processor_filter(all_processors, processing_config):
    """Mirror the processor-filtering logic used in production."""
    if processing_config.processors_included:
        return [p for p in all_processors if p.__class__.__name__ in processing_config.processors_included]
    if processing_config.processors_excluded:
        return [p for p in all_processors if p.__class__.__name__ not in processing_config.processors_excluded]
    return all_processors


# ---------------------------------------------------------------------------
# Processor selection
# ---------------------------------------------------------------------------

class TestProcessorSelection:
    """Test processor selection features."""

    def test_processors_included(self):
        """Only the requested processor should survive the filter."""
        config = ProcessingConfig(processors_included={"BasicStatsProcessor"})
        filtered = _apply_processor_filter(discover_processor_plugins(), config)

        assert len(filtered) == 1
        assert filtered[0].__class__.__name__ == "BasicStatsProcessor"

    def test_processors_excluded(self):
        """The excluded processor must not appear; others must survive."""
        config = ProcessingConfig(processors_excluded={"HistogramProcessor"})
        filtered = _apply_processor_filter(discover_processor_plugins(), config)

        filtered_names = {p.__class__.__name__ for p in filtered}
        assert "HistogramProcessor" not in filtered_names
        assert "BasicStatsProcessor" in filtered_names

    def test_processors_default_all(self):
        """Default config must return all processors unchanged."""
        all_processors = discover_processor_plugins()
        filtered = _apply_processor_filter(all_processors, ProcessingConfig())

        assert len(filtered) == len(all_processors)

    def test_processors_included_takes_precedence(self):
        """When both sets are given, included wins and excluded is ignored."""
        config = ProcessingConfig(
            processors_included={"BasicStatsProcessor"},
            processors_excluded={"BasicStatsProcessor"},
        )
        filtered = _apply_processor_filter(discover_processor_plugins(), config)

        assert len(filtered) == 1
        assert filtered[0].__class__.__name__ == "BasicStatsProcessor"


# ---------------------------------------------------------------------------
# Widget selection
# ---------------------------------------------------------------------------

class TestWidgetSelection:
    """Test widget selection features via _filter_widgets."""

    def test_widgets_included(self):
        """Only the requested widgets should survive the filter."""
        all_widgets = discover_widget_plugins()
        if not all_widgets:
            pytest.skip("No widgets discovered")

        test_names = {w.NAME for w in all_widgets[:2]}
        filtered = _filter_widgets(all_widgets, ReportConfig(widgets_included=test_names))

        assert {w.NAME for w in filtered} == test_names

    def test_widgets_excluded(self):
        """The excluded widget must not appear; count must drop by exactly one."""
        all_widgets = discover_widget_plugins()
        if not all_widgets:
            pytest.skip("No widgets discovered")

        name_to_exclude = all_widgets[0].NAME
        filtered = _filter_widgets(all_widgets, ReportConfig(widgets_excluded={name_to_exclude}))

        assert name_to_exclude not in {w.NAME for w in filtered}
        assert len(filtered) == len(all_widgets) - 1

    def test_widgets_excluded_invariant(self):
        """Invariant: _filter_widgets must never return a widget that was excluded."""
        all_widgets = discover_widget_plugins()
        if not all_widgets:
            pytest.skip("No widgets discovered")

        name_to_exclude = all_widgets[0].NAME
        filtered = _filter_widgets(all_widgets, ReportConfig(widgets_excluded={name_to_exclude}))

        assert name_to_exclude not in {w.NAME for w in filtered}, (
            f"Widget '{name_to_exclude}' should have been excluded but is still present"
        )

    def test_widgets_default_all(self):
        """Default config must return all widgets unchanged."""
        all_widgets = discover_widget_plugins()
        filtered = _filter_widgets(all_widgets, ReportConfig())

        assert len(filtered) == len(all_widgets)

    def test_widgets_included_takes_precedence(self):
        """When both sets are given, included wins and excluded is ignored."""
        all_widgets = discover_widget_plugins()
        if not all_widgets:
            pytest.skip("No widgets discovered")

        name = all_widgets[0].NAME
        filtered = _filter_widgets(
            all_widgets,
            ReportConfig(widgets_included={name}, widgets_excluded={name}),
        )

        assert len(filtered) == 1
        assert filtered[0].NAME == name


# ---------------------------------------------------------------------------
# Combined configuration
# ---------------------------------------------------------------------------

class TestCombinedConfiguration:
    """Test combinations of configuration options."""

    def test_combined_configuration(self):
        """Processor and widget exclusions can coexist independently."""
        all_widgets = discover_widget_plugins()
        widget_name = all_widgets[0].NAME if all_widgets else "TestWidget"

        processing_config = ProcessingConfig(processors_excluded={"HistogramProcessor"})
        report_config = ReportConfig(widgets_excluded={widget_name})

        assert processing_config.processors_excluded == {"HistogramProcessor"}
        assert report_config.widgets_excluded == {widget_name}

    def test_configuration_defaults(self):
        """Both config objects start with empty include/exclude sets."""
        processing_config = ProcessingConfig()
        report_config = ReportConfig()

        assert processing_config.processors_included == set()
        assert processing_config.processors_excluded == set()
        assert report_config.widgets_included == set()
        assert report_config.widgets_excluded == set()