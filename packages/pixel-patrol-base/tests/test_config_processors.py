"""
Tests for processor selection configuration (included/excluded).

Uses processor NAME values ("basic-stats", "histogram", "thumbnail") which is
what the production filtering code in processing.py matches against, not class names.
"""

from pixel_patrol_base.core.processing_config import ProcessingConfig
from pixel_patrol_base.plugin_registry import discover_processor_plugins


def _apply_processor_filter(processors, config):
    """Exact copy of the inline filtering logic from processing.py."""
    if config.processors_included:
        return [p for p in processors if p.NAME in config.processors_included]
    if config.processors_excluded:
        return [p for p in processors if p.NAME not in config.processors_excluded]
    return processors


class TestProcessorSelection:

    def test_processors_included(self):
        """Only the processor with the matching NAME survives the filter."""
        config = ProcessingConfig(processors_included={"basic-stats"})
        filtered = _apply_processor_filter(discover_processor_plugins(), config)

        assert len(filtered) == 1
        assert filtered[0].NAME == "basic-stats"

    def test_processors_excluded(self):
        """The excluded processor must not appear; others must survive."""
        config = ProcessingConfig(processors_excluded={"histogram"})
        filtered = _apply_processor_filter(discover_processor_plugins(), config)

        names = {p.NAME for p in filtered}
        assert "histogram" not in names
        assert "basic-stats" in names

    def test_processors_default_all(self):
        """Default config returns all processors unchanged."""
        all_processors = discover_processor_plugins()
        filtered = _apply_processor_filter(all_processors, ProcessingConfig())

        assert len(filtered) == len(all_processors)

    def test_processors_included_takes_precedence(self):
        """When both sets are given, included wins and excluded is ignored."""
        config = ProcessingConfig(
            processors_included={"basic-stats"},
            processors_excluded={"basic-stats"},
        )
        filtered = _apply_processor_filter(discover_processor_plugins(), config)

        assert len(filtered) == 1
        assert filtered[0].NAME == "basic-stats"

    def test_configuration_defaults(self):
        """ProcessingConfig starts with empty include/exclude sets."""
        config = ProcessingConfig()

        assert config.processors_included == set()
        assert config.processors_excluded == set()