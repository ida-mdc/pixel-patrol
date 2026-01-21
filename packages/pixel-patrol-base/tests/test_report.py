import numpy as np
import polars as pl
import pytest

from pixel_patrol_base.report import data_utils as du
from pixel_patrol_base.report import global_controls as gc
from pixel_patrol_base.report import dashboard_app as da
from pixel_patrol_base.report.base_widget import BaseReportWidget
import pixel_patrol_base.report.base_widget as base_widget


# -----------------------
# data_utils.py
# -----------------------

def test_get_sortable_columns_filters_numeric_and_excludes_dim_slices():
    df = pl.DataFrame(
        {
            # typical metadata
            "unique_id": [1001, 1002],
            "path": ["/data/runA/img_001.tif", "/data/runA/img_002.tif"],
            "file_extension": [".tif", ".tif"],
            "size_bytes": [1_234_567, 2_345_678],  # numeric, should be sortable

            # metric bases
            "mean_intensity": [123.4, 98.7],  # numeric, base => sortable
            "mean_intensity_t0": [120.0, 90.0],  # dim slice => excluded
            "max_intensity_s2": [255, 254],  # dim slice => excluded
            "std_intensity_c0_z0": [5.1, 6.2],  # dim slice => excluded

            # non-numeric and non-scalar
            "type": ["file", "file"],  # string
            "tags": [["qc_ok"], ["qc_fail"]],  # list => excluded
        }
    )

    cols = du.get_sortable_columns(df)
    assert "mean_intensity" in cols
    assert "size_bytes" in cols

    # dim-sliced / dim-composed metrics should not be offered as sortable
    assert "mean_intensity_t0" not in cols
    assert "max_intensity_s2" not in cols
    assert "std_intensity_c0_z0" not in cols

    # non-numeric / list
    assert "type" not in cols
    assert "tags" not in cols


def test_get_all_available_dimensions_only_returns_dims_with_more_than_one_index():
    df = pl.DataFrame(
        {
            "unique_id": [1, 2],
            "report_group": ["GroupA", "GroupB"],

            # dims that exist in your tables (t/c/z/s)
            "mean_intensity_t0": [10.0, 11.0],
            "mean_intensity_t1": [12.0, 13.0],  # -> t has multiple indices => included
            "mean_intensity_s0": [1.0, 2.0],
            "mean_intensity_s1": [3.0, 4.0],
            "mean_intensity_s2": [5.0, 6.0],  # -> s has multiple indices => included

            "max_intensity_c0": [255, 254],  # only one c index => should be ignored
            "min_intensity_z0": [0, 1],  # only one z index => should be ignored

            # not matching regex (upper-case dim letter)
            "std_intensity_T0": [1.0, 1.0],
        }
    )

    dims = du.get_all_available_dimensions(df)
    assert dims == {"t": ["0", "1"], "s": ["0", "1", "2"]}


def test_get_dim_aware_column_prefers_best_dim_specific_match():
    cols = [
        "mean_intensity",  # base
        "mean_intensity_t0",  # t slice
        "mean_intensity_s1",  # s slice
        "mean_intensity_t0_s1",  # combined slices
        "mean_intensity_t1_s1",
    ]

    # No dims: base column if present
    assert du.get_dim_aware_column(cols, "mean_intensity", {}) == "mean_intensity"

    # With one dim: pick matching dim slice
    assert du.get_dim_aware_column(cols, "mean_intensity", {"s": "1"}) == "mean_intensity_s1"
    assert du.get_dim_aware_column(cols, "mean_intensity", {"t": "0"}) == "mean_intensity_t0"

    # With multiple dims: pick combined match
    assert du.get_dim_aware_column(cols, "mean_intensity", {"t": "0", "s": "1"}) == "mean_intensity_t0_s1"

    # No match for requested dim
    assert du.get_dim_aware_column(cols, "mean_intensity", {"z": "9"}) is None


def test_format_selection_title_and_parse_metric_dimension_column():
    assert du.format_selection_title({}) is None
    assert du.format_selection_title({"t": "0", "s": "2"}) == "Filter: S=2, T=0"

    supported = ["mean_intensity", "max_intensity", "min_intensity", "std_intensity"]

    assert du.parse_metric_dimension_column("mean_intensity_t0_s2", supported) == (
        "mean_intensity",
        {"t": 0, "s": 2},
    )
    assert du.parse_metric_dimension_column("max_intensity", supported) == ("max_intensity", {})
    assert du.parse_metric_dimension_column("unknown_metric_t0", supported) is None
    assert du.parse_metric_dimension_column("mean_intensity__bogus", supported) is None


def test_compute_histogram_edges_none_minmax():
    counts = np.ones(5)
    edges, centers, widths = du.compute_histogram_edges(counts, None, None)
    assert edges.shape == (6,)
    assert np.allclose(edges, np.arange(6, dtype=float))
    assert centers.shape == (5,)
    assert widths.shape == (5,)


def test_rebin_histogram_preserves_probability_mass():
    # a toy but realistic "counts over bins" case
    counts = np.array([10, 10, 20, 0], dtype=float)
    src_edges = np.array([0, 64, 128, 192, 256], dtype=float)
    tgt_edges = np.array([0, 128, 256], dtype=float)  # merge 2 bins into 1
    reb = du.rebin_histogram(counts, src_edges, tgt_edges)
    assert reb.shape == (2,)
    assert reb.sum() == pytest.approx(1.0, abs=1e-9)


def test_aggregate_histograms_by_group_shape_mode():
    df = pl.DataFrame(
        {
            "report_group": ["GroupA", "GroupA", "GroupB"],
            "hist": [
                [1.0] * 256,
                [0.0] * 255 + [10.0],
                [2.0] * 256,
            ],
            "min_value": [0.0, 0.0, 0.0],
            "max_value": [255.0, 255.0, 255.0],
        }
    )

    out = du.aggregate_histograms_by_group(
        df=df,
        group_col="report_group",
        hist_col="hist",
        min_col="min_value",
        max_col="max_value",
        mode="shape",
    )

    assert set(out.keys()) == {"GroupA", "GroupB"}
    x, y = out["GroupA"]
    assert x.shape == (256,)
    assert y.shape == (256,)
    assert y.sum() == pytest.approx(1.0, abs=1e-6)


# -----------------------
# global_controls.py
# -----------------------

def _base_df_for_global_controls():
    """
    A minimal-but-plausible "report table" shape:
    - file-ish metadata
    - grouping column used by report
    - a filterable categorical column
    - intensity metric base + dim-sliced variants

    Data layout:
    Row 0: unique_id=100, ext=".tif", report_group="GroupA", mean_intensity_t0=120.0, mean_intensity_t1=None,  s0=1.0,  s1=None, s2=None
    Row 1: unique_id=101, ext=".tif", report_group="GroupA", mean_intensity_t0=None,  mean_intensity_t1=125.0, s0=None, s1=3.0,  s2=None
    Row 2: unique_id=102, ext=".tif", report_group="GroupA", mean_intensity_t0=110.0, mean_intensity_t1=None,  s0=None, s1=None, s2=5.0
    Row 3: unique_id=103, ext=".png", report_group="GroupB", mean_intensity_t0=None,  mean_intensity_t1=None,  s0=2.0,  s1=None, s2=None
    Row 4: unique_id=104, ext=".tif", report_group="GroupB", mean_intensity_t0=90.0,  mean_intensity_t1=95.0,  s0=None, s1=None, s2=6.0
    """
    return pl.DataFrame(
        {
            "unique_id": [100, 101, 102, 103, 104],
            "path": [
                "/data/runA/img_001.tif",
                "/data/runA/img_002.tif",
                "/data/runB/img_003.tif",
                "/data/runB/img_004.tif",
                "/data/runB/img_005.tif",
            ],
            "file_extension": [".tif", ".tif", ".tif", ".png", ".tif"],
            "type": ["file", "file", "file", "file", "file"],
            "report_group": ["GroupA", "GroupA", "GroupA", "GroupB", "GroupB"],

            # metric slices: mimic sparse availability by dim
            "mean_intensity_t0": [120.0, None, 110.0, None, 90.0],
            "mean_intensity_t1": [None, 125.0, None, None, 95.0],

            # another dim family for testing "any dimension column non-null"
            "mean_intensity_s0": [1.0, None, None, 2.0, None],
            "mean_intensity_s1": [None, 3.0, None, None, None],
            "mean_intensity_s2": [None, None, 5.0, None, 6.0],
        }
    )


def test_compute_filtered_row_positions_none_when_no_filters_or_dims():
    """Test that compute_filtered_row_positions returns None when no filters are active."""
    df = _base_df_for_global_controls()

    # Empty filter and dimensions should return None (meaning "use all rows")
    assert gc.compute_filtered_row_positions(df, {"filter": {}, "dimensions": {}}) is None
    assert gc.compute_filtered_row_positions(df, None) is None


def test_compute_filtered_row_positions_with_all_dimension_returns_empty():
    """
    Test that {"t": "All"} returns empty list (not None).

    This is the current implementation behavior: when dimensions dict is non-empty
    but all values are "All", the dimension tokens are empty, so no columns match,
    resulting in an empty result rather than None.

    Note: This might be arguable behavior - one could expect {"t": "All"} to be
    equivalent to no dimension filter. But this test documents the current behavior.
    """
    df = _base_df_for_global_controls()

    # "All" dimension value results in empty list (dimensions dict is non-empty but
    # no actual dimension filter is applied, so no columns match)
    result = gc.compute_filtered_row_positions(df, {"filter": {}, "dimensions": {"t": "All"}})
    assert result == []


def test_compute_filtered_row_positions_with_value_filter_only():
    """Test filtering by value (file_extension) without dimension filters."""
    df = _base_df_for_global_controls()

    cfg = {
        "filter": {"file_extension": [".tif"]},
        "dimensions": {},
    }

    # .tif rows are at positions 0, 1, 2, 4 (row 3 has .png)
    positions = gc.compute_filtered_row_positions(df, cfg)
    assert positions == [0, 1, 2, 4]

    # Verify by checking the actual unique_ids at those positions
    filtered_ids = df[positions]["unique_id"].to_list()
    assert filtered_ids == [100, 101, 102, 104]


def test_compute_filtered_row_positions_with_dimension_filter():
    """Test filtering by dimension (t=0) which requires non-null in _t0 columns."""
    df = _base_df_for_global_controls()

    cfg = {
        "filter": {},
        "dimensions": {"t": "0"},
    }

    # t=0 requires any column containing "_t0" to be non-null
    # mean_intensity_t0 values: [120.0, None, 110.0, None, 90.0]
    # Non-null at positions: 0, 2, 4
    positions = gc.compute_filtered_row_positions(df, cfg)
    assert positions == [0, 2, 4]

    # Verify the actual data
    filtered_df = df[positions]
    assert filtered_df["unique_id"].to_list() == [100, 102, 104]
    # All these rows should have non-null mean_intensity_t0
    assert all(v is not None for v in filtered_df["mean_intensity_t0"].to_list())


def test_compute_filtered_row_positions_with_value_and_dimension_filter():
    """Test combining value filter and dimension filter."""
    df = _base_df_for_global_controls()

    # Filter for .png AND t=0
    cfg = {
        "filter": {"file_extension": [".png"]},
        "dimensions": {"t": "0"},
    }

    # .png rows: only row 3 (unique_id=103)
    # t=0 non-null rows: 0, 2, 4
    # Intersection: empty (row 3 has mean_intensity_t0=None)
    positions = gc.compute_filtered_row_positions(df, cfg)
    assert positions == []

    # Now test .tif AND t=0 - should give rows where both conditions are met
    cfg2 = {
        "filter": {"file_extension": [".tif"]},
        "dimensions": {"t": "0"},
    }
    # .tif rows: 0, 1, 2, 4
    # t=0 non-null rows: 0, 2, 4
    # Intersection: 0, 2, 4 (row 1 excluded because mean_intensity_t0 is None)
    positions2 = gc.compute_filtered_row_positions(df, cfg2)
    assert positions2 == [0, 2, 4]


def test_compute_filtered_row_positions_dimension_not_found_returns_empty():
    """Test that a non-existent dimension filter returns empty list."""
    df = _base_df_for_global_controls()

    cfg = {
        "filter": {},
        "dimensions": {"z": "9"},  # no _z9 columns exist
    }

    # When dimensions are specified but no matching columns found, return empty
    positions = gc.compute_filtered_row_positions(df, cfg)
    assert positions == []


def test_compute_filtered_row_positions_with_s_dimension():
    """Test filtering by s dimension."""
    df = _base_df_for_global_controls()

    cfg = {
        "filter": {},
        "dimensions": {"s": "1"},
    }

    # s=1 requires any column containing "_s1" to be non-null
    # mean_intensity_s1 values: [None, 3.0, None, None, None]
    # Non-null only at position 1
    positions = gc.compute_filtered_row_positions(df, cfg)
    assert positions == [1]

    # Verify the actual value
    assert df[1]["mean_intensity_s1"].item() == 3.0


def test_compute_filtered_row_positions_with_dict_filter_spec():
    """Test filtering with dict-style filter spec (op + value)."""
    df = _base_df_for_global_controls()

    # Test "contains" operator
    cfg = {
        "filter": {"path": {"op": "contains", "value": "runA"}},
        "dimensions": {},
    }

    # runA paths are rows 0, 1 (unique_ids 100, 101)
    positions = gc.compute_filtered_row_positions(df, cfg)
    assert positions == [0, 1]

    # Test "not_contains" operator
    cfg2 = {
        "filter": {"path": {"op": "not_contains", "value": "runA"}},
        "dimensions": {},
    }
    # runB paths are rows 2, 3, 4
    positions2 = gc.compute_filtered_row_positions(df, cfg2)
    assert positions2 == [2, 3, 4]


def test_prepare_widget_data_resolves_dim_aware_metric_and_warns_when_missing():
    """Test prepare_widget_data with dimension-aware metric resolution."""
    df = _base_df_for_global_controls()

    # dims t=0, metric_base mean_intensity => resolve to mean_intensity_t0 and drop nulls
    df_f, group_col, resolved, warn, group_order = gc.prepare_widget_data(
        df=df,
        subset_row_positions=None,
        global_config={"group_col": "report_group", "filter": {}, "dimensions": {"t": "0"}},
        metric_base="mean_intensity",
    )

    # Resolved column should be the dim-specific one
    assert resolved == "mean_intensity_t0"
    # No warning when column exists and has data
    assert warn is None
    # group_col should contain "report_group" (may be prefixed by ensure_discrete_grouping)
    assert "report_group" in group_col
    # Rows with non-null mean_intensity_t0: unique_ids 100, 102, 104
    assert set(df_f["unique_id"].to_list()) == {100, 102, 104}
    # Verify we can access the resolved column in the filtered df
    assert resolved in df_f.columns
    assert all(v is not None for v in df_f[resolved].to_list())


def test_prepare_widget_data_warns_when_dimension_not_found():
    """Test prepare_widget_data generates warning when dim-specific column doesn't exist."""
    df = _base_df_for_global_controls()

    # dims t=9 => no mean_intensity_t9 column exists
    df_f, group_col, resolved, warn, group_order = gc.prepare_widget_data(
        df=df,
        subset_row_positions=None,
        global_config={"group_col": "report_group", "filter": {}, "dimensions": {"t": "9"}},
        metric_base="mean_intensity",
    )

    # Should return empty df
    assert df_f.height == 0
    # Resolved column should be None
    assert resolved is None
    # Warning should be set and mention the metric and dimension
    assert warn is not None
    assert "mean_intensity" in warn
    assert "t=9" in warn


def test_prepare_widget_data_with_subset_row_positions():
    """Test prepare_widget_data when subset_row_positions is provided."""
    df = _base_df_for_global_controls()

    # Only use rows at positions 0, 2 (unique_ids 100, 102)
    df_f, group_col, resolved, warn, group_order = gc.prepare_widget_data(
        df=df,
        subset_row_positions=[0, 2],
        global_config={"group_col": "report_group", "filter": {}, "dimensions": {"t": "0"}},
        metric_base="mean_intensity",
    )

    assert resolved == "mean_intensity_t0"
    # Both rows 0 and 2 have non-null mean_intensity_t0 (120.0 and 110.0)
    assert set(df_f["unique_id"].to_list()) == {100, 102}
    # Verify the actual values
    values = sorted(df_f["mean_intensity_t0"].to_list())
    assert values == [110.0, 120.0]


def test_prepare_widget_data_subset_filters_before_metric_null_check():
    """Test that subset_row_positions is applied before metric null filtering."""
    df = _base_df_for_global_controls()

    # Row 1 (unique_id=101) has mean_intensity_t0=None
    # If we only include row 1, the result should be empty after null filtering
    df_f, group_col, resolved, warn, group_order = gc.prepare_widget_data(
        df=df,
        subset_row_positions=[1],  # Only row 1, which has t0=None
        global_config={"group_col": "report_group", "filter": {}, "dimensions": {"t": "0"}},
        metric_base="mean_intensity",
    )

    # The resolved column exists, but no rows pass the null filter
    assert resolved == "mean_intensity_t0"
    assert df_f.height == 0
    assert warn is not None  # Warning about no data matching


def test_prepare_widget_data_empty_subset_returns_empty_df():
    """Test prepare_widget_data with empty subset_row_positions."""
    df = _base_df_for_global_controls()

    df_f, group_col, resolved, warn, group_order = gc.prepare_widget_data(
        df=df,
        subset_row_positions=[],
        global_config={"group_col": "report_group", "filter": {}, "dimensions": {}},
        metric_base="mean_intensity",
    )

    assert df_f.is_empty()
    assert warn is not None
    assert "No data" in warn


def test_prepare_widget_data_no_metric_base_skips_column_resolution():
    """Test prepare_widget_data without metric_base doesn't try to resolve columns."""
    df = _base_df_for_global_controls()

    df_f, group_col, resolved, warn, group_order = gc.prepare_widget_data(
        df=df,
        subset_row_positions=None,
        global_config={"group_col": "report_group", "filter": {}, "dimensions": {}},
        metric_base=None,  # No metric base
    )

    # Should return all rows since no metric filtering
    assert df_f.height == 5
    assert resolved is None
    assert warn is None


def test_apply_global_row_filters_and_grouping_filters_then_resolves_group_col():
    """Test apply_global_row_filters_and_grouping with filters."""
    df = _base_df_for_global_controls()

    df2, group_col = gc.apply_global_row_filters_and_grouping(
        df,
        {
            "group_col": "report_group",
            "filter": {"file_extension": [".png"]},
            "dimensions": {},
        },
    )

    # group_col should be the original column name (not prefixed - this function is for exports)
    assert group_col == "report_group"
    # Only row 3 has .png extension (unique_id 103)
    assert df2["unique_id"].to_list() == [103]
    # Verify the filtered row's data
    assert df2["file_extension"].to_list() == [".png"]
    assert df2["report_group"].to_list() == ["GroupB"]


def test_apply_global_row_filters_and_grouping_no_filters_returns_full_df():
    """Test that no filters returns the full dataframe."""
    df = _base_df_for_global_controls()

    df2, group_col = gc.apply_global_row_filters_and_grouping(
        df,
        {"group_col": "report_group", "filter": {}, "dimensions": {}},
    )

    assert df2.height == df.height
    assert df2["unique_id"].to_list() == df["unique_id"].to_list()


def test_apply_global_row_filters_and_grouping_fallback_when_group_col_missing():
    """Test that group_col falls back correctly when specified column doesn't exist."""
    df = _base_df_for_global_controls()

    df2, group_col = gc.apply_global_row_filters_and_grouping(
        df,
        {
            "group_col": "nonexistent_column",
            "filter": {},
            "dimensions": {},
        },
    )

    # Should fall back to a column that exists in the df
    assert group_col in df.columns
    # The dataframe should still be returned (no filter applied)
    assert df2.height == df.height


def test_resolve_group_column_returns_valid_column():
    """Test resolve_group_column returns a valid column from the dataframe."""
    df = _base_df_for_global_controls()

    # With valid column
    group_col = gc.resolve_group_column(df, {"group_col": "report_group"})
    assert group_col == "report_group"

    # With valid column that's not report_group
    group_col2 = gc.resolve_group_column(df, {"group_col": "file_extension"})
    assert group_col2 == "file_extension"

    # With nonexistent column, should fall back to something valid
    group_col3 = gc.resolve_group_column(df, {"group_col": "nonexistent"})
    assert group_col3 in df.columns


def test_is_group_col_accepted_rejects_float_and_dimension_pattern():
    """Test is_group_col_accepted logic for various column types."""
    df = pl.DataFrame({
        "good_col": ["A", "B", "C"],
        "float_col": [1.1, 2.2, 3.3],
        "int_col": [1, 2, 3],
        "metric_t0": ["X", "Y", "Z"],  # ends with dimension pattern _t0
        "metric_t0_s1": ["X", "Y", "Z"],  # ends with dimension pattern _t0_s1
    })

    # String column with low cardinality - should be accepted
    assert gc.is_group_col_accepted(df, "good_col") is True

    # Integer column with low cardinality - should be accepted
    assert gc.is_group_col_accepted(df, "int_col") is True

    # Float column - should be rejected
    assert gc.is_group_col_accepted(df, "float_col") is False

    # Dimension-patterned columns - should be rejected
    assert gc.is_group_col_accepted(df, "metric_t0") is False
    assert gc.is_group_col_accepted(df, "metric_t0_s1") is False

    # Nonexistent column - should be rejected
    assert gc.is_group_col_accepted(df, "nonexistent") is False


def test_init_global_config_validates_and_sanitizes():
    """Test init_global_config returns sanitized config."""
    df = _base_df_for_global_controls()

    cfg = gc.init_global_config(df, {
        "group_col": "report_group",
        "filter": {"file_extension": [".tif"]},
        "dimensions": {"t": "0"},
    })

    assert cfg["group_col"] == "report_group"
    assert "file_extension" in cfg["filter"]
    # The filter value should be preserved
    assert cfg["filter"]["file_extension"] == [".tif"]
    assert cfg["dimensions"] == {"t": "0"}


def test_init_global_config_removes_invalid_filter_columns():
    """Test that init_global_config removes filters on non-existent columns."""
    df = _base_df_for_global_controls()

    cfg = gc.init_global_config(df, {
        "group_col": "report_group",
        "filter": {
            "nonexistent_col": [".tif"],
            "file_extension": [".png"],  # This one exists
        },
        "dimensions": {},
    })

    # The invalid filter should be removed
    assert "nonexistent_col" not in cfg["filter"]
    # The valid filter should remain
    assert "file_extension" in cfg["filter"]


def test_init_global_config_rejects_invalid_group_col():
    """Test that init_global_config rejects invalid group columns."""
    df = _base_df_for_global_controls()

    # Float column should be rejected as group_col
    cfg = gc.init_global_config(df, {
        "group_col": "mean_intensity_t0",  # This is a float column
        "filter": {},
        "dimensions": {},
    })

    # Should fall back to None (which means use default)
    assert cfg["group_col"] is None


# -----------------------
# dashboard_app.py (non-integration parts)
# -----------------------

def test_should_display_widget_requires_and_patterns_with_project_like_columns():
    class Widget:
        NAME = "IntensityWidget"
        REQUIRES = {"unique_id", "report_group"}
        REQUIRES_PATTERNS = [r"^(mean_intensity|min_intensity|max_intensity|std_intensity)(?:_.*)?$"]

    cols_ok = [
        "unique_id",
        "report_group",
        "path",
        "mean_intensity_t0",
        "max_intensity_s2",
    ]
    assert da.should_display_widget(Widget(), cols_ok) is True

    cols_missing_required = ["report_group", "mean_intensity_t0"]
    assert da.should_display_widget(Widget(), cols_missing_required) is False

    cols_missing_pattern = ["unique_id", "report_group", "path", "file_extension"]
    assert da.should_display_widget(Widget(), cols_missing_pattern) is False


# -----------------------
# base_widget.py
# -----------------------

def test_base_report_widget_layout_wraps_content(monkeypatch):
    calls = {}

    def fake_create_widget_card(title, content, widget_id, help_text=None):
        calls["title"] = title
        calls["content"] = content
        calls["widget_id"] = widget_id
        calls["help_text"] = help_text
        return {"card": True}

    monkeypatch.setattr(base_widget, "create_widget_card", fake_create_widget_card)

    class MeanIntensityWidget(BaseReportWidget):
        NAME = "Mean intensity"
        REQUIRES = {"unique_id", "report_group"}
        REQUIRES_PATTERNS = [r"^mean_intensity(?:_.*)?$"]

        @property
        def help_text(self):
            return "Distribution of mean intensity across selected dims."

        def get_content_layout(self):
            return ["plot"]

        def register(self, app, df_global):
            pass

    w = MeanIntensityWidget()
    out = w.layout()

    assert out == [{"card": True}]
    assert calls["title"] == "Mean intensity"
    assert calls["content"] == ["plot"]
    assert calls["widget_id"] == "meanintensitywidget"
    assert calls["help_text"] == "Distribution of mean intensity across selected dims."