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
            "mean_intensity": [123.4, 98.7],        # numeric, base => sortable
            "mean_intensity_t0": [120.0, 90.0],     # dim slice => excluded
            "max_intensity_s2": [255, 254],         # dim slice => excluded
            "std_intensity_c0_z0": [5.1, 6.2],      # dim slice => excluded

            # non-numeric and non-scalar
            "type": ["file", "file"],               # string
            "tags": [["qc_ok"], ["qc_fail"]],        # list => excluded
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
            "mean_intensity_s2": [5.0, 6.0],    # -> s has multiple indices => included

            "max_intensity_c0": [255, 254],     # only one c index => should be ignored
            "min_intensity_z0": [0, 1],         # only one z index => should be ignored

            # not matching regex (upper-case dim letter)
            "std_intensity_T0": [1.0, 1.0],
        }
    )

    dims = du.get_all_available_dimensions(df)
    assert dims == {"t": ["0", "1"], "s": ["0", "1", "2"]}


def test_get_dim_aware_column_prefers_best_dim_specific_match():
    cols = [
        "mean_intensity",               # base
        "mean_intensity_t0",            # t slice
        "mean_intensity_s1",            # s slice
        "mean_intensity_t0_s1",         # combined slices
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
    # a toy but realistic “counts over bins” case
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
    A minimal-but-plausible “report table” shape:
    - file-ish metadata
    - grouping column used by report
    - a filterable categorical column
    - intensity metric base + dim-sliced variants
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

            # another dim family for testing “any dimension column non-null”
            "mean_intensity_s0": [1.0, None, None, 2.0, None],
            "mean_intensity_s1": [None, 3.0, None, None, None],
            "mean_intensity_s2": [None, None, 5.0, None, 6.0],
        }
    )


def test_compute_filtered_indices_none_when_no_filters_or_dims():
    df = _base_df_for_global_controls()
    assert gc.compute_filtered_indices(df, {"filter": {}, "dimensions": {}}) is None
    assert gc.compute_filtered_indices(df, None) is None


def test_compute_filtered_indices_with_value_filter_and_dimension_filter():
    df = _base_df_for_global_controls()

    # Filter to only .tif and require the chosen dim slice to exist (non-null)
    cfg = {
        "filter": {"file_extension": [".tif"]},
        "dimensions": {"t": "0"},
    }

    # .tif rows are 100,101,102,104; then t=0 requires any *_t0 non-null => 100,102,104
    assert gc.compute_filtered_indices(df, cfg) == [100, 102, 104]


def test_filter_rows_by_any_dimension_selects_rows_with_any_matching_dim_column():
    df = _base_df_for_global_controls()

    # s=1 should keep rows where any column containing "_s1" is non-null
    out = gc.filter_rows_by_any_dimension(df, {"s": "1"})
    assert set(out["unique_id"].to_list()) == {101}

    # dimension present but no matching columns => empty
    out2 = gc.filter_rows_by_any_dimension(df, {"z": "9"})
    assert out2.height == 0


def test_prepare_widget_data_resolves_dim_aware_metric_and_warns_when_missing():
    df = _base_df_for_global_controls()

    # dims t=0, metric_base mean_intensity => resolve to mean_intensity_t0 and drop nulls
    df_f, group_col, resolved, warn, _order = gc.prepare_widget_data(
        df=df,
        subset_indices=None,
        global_config={"group_col": ["report_group"], "filter": {}, "dimensions": {"t": "0"}},
        metric_base="mean_intensity",
    )

    assert group_col == "report_group"
    assert resolved == "mean_intensity_t0"
    assert warn is None
    assert set(df_f["unique_id"].to_list()) == {100, 102, 104}

    # dims t=9 => no such slice column => empty + warning
    df_f2, _gc2, resolved2, warn2, _order = gc.prepare_widget_data(
        df=df,
        subset_indices=None,
        global_config={"group_col": ["report_group"], "filter": {}, "dimensions": {"t": "9"}},
        metric_base="mean_intensity",
    )
    assert df_f2.height == 0
    assert resolved2 is None
    assert warn2 is not None


def test_apply_global_row_filters_and_grouping_filters_then_resolves_group_col():
    df = _base_df_for_global_controls()

    df2, group_col = gc.apply_global_row_filters_and_grouping(
        df,
        {
            "group_col": ["report_group"],
            "filter": {"file_extension": [".png"]},
            "dimensions": {},
        },
    )

    assert group_col == "report_group"
    assert set(df2["unique_id"].to_list()) == {103}


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
