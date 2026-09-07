import re
import polars as pl

from pixel_patrol_base.utils.df_utils import add_parent_level_columns

_PARENT_LEVEL_RE = re.compile(r"^parent\d+$")


def _parent_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if _PARENT_LEVEL_RE.match(c)]


def test_root_only_no_parent_columns():
    """All files directly in base: no parent columns created."""
    df = pl.DataFrame({"path": ["a.tif", "b.tif"]})
    result = add_parent_level_columns(df)
    assert _parent_cols(result) == []


def test_various_depths_values_and_nulls():
    """parent values correct; shallower files get null for deeper columns."""
    df = pl.DataFrame({"path": [
        "file.tif",
        "a/file.tif",
        "a/b/file.tif",
        "a/b/c/file.tif",
    ]})
    result = add_parent_level_columns(df)
    assert sorted(_parent_cols(result)) == ["parent0", "parent1", "parent2"]

    rows = {r["path"]: r for r in result.to_dicts()}
    assert rows["file.tif"]["parent0"] is None
    assert rows["a/file.tif"]["parent0"] == "a"
    assert rows["a/file.tif"]["parent1"] is None
    assert rows["a/b/file.tif"]["parent0"] == "a"
    assert rows["a/b/file.tif"]["parent1"] == "b"
    assert rows["a/b/file.tif"]["parent2"] is None
    assert rows["a/b/c/file.tif"]["parent0"] == "a"
    assert rows["a/b/c/file.tif"]["parent1"] == "b"
    assert rows["a/b/c/file.tif"]["parent2"] == "c"


def test_deep_nesting():
    """N levels deep: exactly N parent columns, correct values throughout."""
    n = 52
    path = "/".join(f"level{i}" for i in range(n)) + "/file.tif"
    df = pl.DataFrame({"path": [path]})
    result = add_parent_level_columns(df)
    cols = _parent_cols(result)
    assert len(cols) == n
    row = result.to_dicts()[0]
    for i in range(n):
        assert row[f"parent{i}"] == f"level{i}"


def test_no_path_column_unchanged():
    df = pl.DataFrame({"name": ["a", "b"]})
    result = add_parent_level_columns(df)
    assert result.columns == ["name"]


def test_empty_df_unchanged():
    df = pl.DataFrame({"path": pl.Series([], dtype=pl.String)})
    result = add_parent_level_columns(df)
    assert result.columns == ["path"]
