import pytest
from pixel_patrol_base.core.report_config import ReportConfig


# --- Defaults ---

def test_defaults():
    config = ReportConfig()
    assert config.cmap == "rainbow"
    assert config.widgets_included == set()
    assert config.widgets_excluded == set()
    assert config.group_col is None
    assert config.filter is None
    assert config.dimensions is None
    assert config.is_show_significance is False


# --- cmap ---

def test_cmap_valid():
    config = ReportConfig(cmap="viridis")
    assert config.cmap == "viridis"


def test_cmap_invalid_falls_back_to_default(caplog):
    with caplog.at_level("WARNING"):
        config = ReportConfig(cmap="not_a_real_colormap")
    assert config.cmap == "rainbow"
    assert "Invalid colormap" in caplog.text
    assert "falling back to 'rainbow'" in caplog.text


# --- widgets_included / excluded conflict ---

def test_widgets_included_only():
    config = ReportConfig(widgets_included={"HistogramWidget"})
    assert config.widgets_included == {"HistogramWidget"}


def test_widgets_excluded_only():
    config = ReportConfig(widgets_excluded={"EmbeddingProjectorWidget"})
    assert config.widgets_excluded == {"EmbeddingProjectorWidget"}


def test_widgets_both_set_warns(caplog):
    with caplog.at_level("WARNING"):
        ReportConfig(
            widgets_included={"HistogramWidget"},
            widgets_excluded={"EmbeddingProjectorWidget"},
        )
    assert "widgets_excluded will be ignored" in caplog.text


# --- to_dict ---

def test_to_dict_empty():
    d = ReportConfig().to_dict()
    assert d == {"cmap": "rainbow"}


def test_to_dict_with_all_fields():
    config = ReportConfig(
        cmap="plasma",
        group_col="size_readable",
        filter={"file_extension": {"op": "in", "value": "tif, png"}},
        dimensions={"T": "0", "C": "1"},
        is_show_significance=True,
    )
    d = config.to_dict()
    assert d["cmap"] == "plasma"
    assert d["group_col"] == "size_readable"
    assert d["filter"] == {"file_extension": {"op": "in", "value": "tif, png"}}
    assert d["dimensions"] == {"T": "0", "C": "1"}
    assert d["is_show_significance"] is True


def test_to_dict_omits_false_significance():
    d = ReportConfig(is_show_significance=False).to_dict()
    assert "is_show_significance" not in d


def test_to_dict_omits_none_fields():
    d = ReportConfig().to_dict()
    assert "group_col" not in d
    assert "filter" not in d
    assert "dimensions" not in d


# --- from_dict ---

def test_from_dict_none_returns_defaults():
    config = ReportConfig.from_dict(None)
    assert config.cmap == "rainbow"
    assert config.group_col is None
    assert config.filter is None


def test_from_dict_full():
    d = {
        "cmap": "plasma",
        "group_col": "size_readable",
        "filter": {"file_extension": {"op": "in", "value": "tif"}},
        "dimensions": {"C": "0"},
        "is_show_significance": True,
    }
    config = ReportConfig.from_dict(d)
    assert config.cmap == "plasma"
    assert config.group_col == "size_readable"
    assert config.filter == {"file_extension": {"op": "in", "value": "tif"}}
    assert config.dimensions == {"C": "0"}
    assert config.is_show_significance is True


def test_from_dict_missing_cmap_uses_default():
    config = ReportConfig.from_dict({"group_col": "size_readable"})
    assert config.cmap == "rainbow"


def test_from_dict_kwargs_override():
    config = ReportConfig.from_dict(
        {"group_col": "size_readable"},
        widgets_excluded={"EmbeddingProjectorWidget"},
    )
    assert config.group_col == "size_readable"
    assert config.widgets_excluded == {"EmbeddingProjectorWidget"}


# --- round-trip ---

def test_to_dict_from_dict_roundtrip():
    original = ReportConfig(
        cmap="inferno",
        group_col="file_extension",
        filter={"file_extension": {"op": "eq", "value": "tif"}},
        dimensions={"Z": "2"},
        is_show_significance=True,
    )
    restored = ReportConfig.from_dict(original.to_dict())
    assert restored.cmap == original.cmap
    assert restored.group_col == original.group_col
    assert restored.filter == original.filter
    assert restored.dimensions == original.dimensions
    assert restored.is_show_significance == original.is_show_significance