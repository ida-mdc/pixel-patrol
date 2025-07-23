import pytest
import polars as pl
from datetime import datetime
from polars.testing import assert_frame_equal
from pixel_patrol.core.processing import build_paths_df, PATHS_DF_EXPECTED_SCHEMA


def test_schema_unmodified(mock_temp_file_system, patch_tree):
    patch_tree(pl.DataFrame([], schema=PATHS_DF_EXPECTED_SCHEMA))
    df = build_paths_df(mock_temp_file_system)
    assert set(df.columns) == set(PATHS_DF_EXPECTED_SCHEMA.keys())
    for col, dtype in PATHS_DF_EXPECTED_SCHEMA.items():
        assert df[col].dtype == dtype, f"Column {col} has type {df[col].dtype}, expected {dtype}"


def test_basic_paths_df(mock_temp_file_system, patch_tree, mock_paths_df_content, build_expected_paths_df):
    dir1, dir2 = mock_temp_file_system

    df1 = mock_paths_df_content.filter(pl.col("imported_path") == str(dir1))
    df2 = mock_paths_df_content.filter(pl.col("imported_path") == str(dir2))

    mock_fetch = patch_tree(df1)         # now _fetch_single_directory_tree returns df1 first
    mock_fetch.side_effect = [df1, df2]  # then df2 on the second call

    actual = build_paths_df([dir1, dir2])

    exp1 = build_expected_paths_df(df1.to_dicts())
    exp2 = build_expected_paths_df(df2.to_dicts())
    expected = exp1.vstack(exp2).sort("path")

    actual = actual.select(*expected.columns)
    assert_frame_equal(actual, expected)


def test_complex_size_aggregation(mock_temp_file_system_complex, patch_tree, build_expected_paths_df):
    paths = mock_temp_file_system_complex
    root = paths[0]

    sub_a = root / "sub_dir_a"
    sub_a2 = sub_a / "sub_sub_dir_a"
    sub_b = root / "sub_dir_b"
    raw = [
        {"path": str(root),    "name": root.name,  "type": "folder", "parent": None,         "depth": 0, "size_bytes": 0,  "file_extension": None, "modification_date": datetime.now(), "imported_path": str(root)},
        {"path": str(sub_a),    "name": "sub_dir_a",  "type": "folder", "parent": str(root),  "depth": 1, "size_bytes": 0,  "file_extension": None, "modification_date": datetime.now(), "imported_path": str(root)},
        {"path": str(sub_a / "fileA.txt"), "name": "fileA.txt", "type": "file",   "parent": str(sub_a), "depth": 2, "size_bytes": 10, "file_extension": "txt",  "modification_date": datetime.now(), "imported_path": str(root)},
        {"path": str(sub_a2),   "name": "sub_sub_dir_a", "type": "folder", "parent": str(sub_a), "depth": 2, "size_bytes": 0,  "file_extension": None, "modification_date": datetime.now(), "imported_path": str(root)},
        {"path": str(sub_a2 / "fileB.txt"), "name": "fileB.txt", "type": "file",   "parent": str(sub_a2), "depth": 3, "size_bytes": 20, "file_extension": "txt",  "modification_date": datetime.now(), "imported_path": str(root)},
        {"path": str(sub_b),    "name": "sub_dir_b",  "type": "folder", "parent": str(root),  "depth": 1, "size_bytes": 0,  "file_extension": None, "modification_date": datetime.now(), "imported_path": str(root)},
        {"path": str(sub_b / "fileC.txt"), "name": "fileC.txt", "type": "file",   "parent": str(sub_b), "depth": 2, "size_bytes": 30, "file_extension": "txt",  "modification_date": datetime.now(), "imported_path": str(root)},
    ]
    patch_tree(pl.DataFrame(raw))
    df = build_paths_df(paths)
    expected = build_expected_paths_df(raw)
    df = df.select(*expected.columns)
    assert_frame_equal(df, expected)


def test_file_extension_edge_cases(mock_temp_file_system_edge_cases, patch_tree):
    paths = mock_temp_file_system_edge_cases
    base = paths[0]
    raw = []
    for p in base.iterdir():
        file_ext = p.suffix.lstrip('.') if p.is_file() else None
        raw.append({
            "path": str(p), "name": p.name, "type": "file", "parent": str(base),
            "depth": 1, "size_bytes": p.stat().st_size, "file_extension": file_ext, # Updated line
            "modification_date": datetime.now(), "imported_path": str(base)
        })
    raw.append({
        "path": str(base), "name": base.name, "type": "folder", "parent": str(base.parent),
        "depth": 0, "size_bytes": 0, "file_extension": None,
        "modification_date": datetime.now(), "imported_path": str(base)
    })
    patch_tree(pl.DataFrame(raw))
    df = build_paths_df(paths)
    ext_df = df.filter(pl.col("type")=="file").select("name","file_extension").sort("name")
    expected = pl.DataFrame([
        {"name": ".hidden_file",   "file_extension": ""},
        {"name": "archive.tar.gz","file_extension": "gz"},
        {"name": "file_no_ext",    "file_extension": ""},
        {"name": "image.JPEG",     "file_extension": "jpeg"},
        {"name": "regular.png",    "file_extension": "png"},
    ], schema={"name": pl.String, "file_extension": pl.String}).sort("name")
    assert_frame_equal(ext_df, expected)
