# TODO: Do we need to write tests for the helper functions too?

import polars as pl
from pathlib import Path
from typing import List
from datetime import datetime
import logging
import pytest

from pixel_patrol.core.processing import build_paths_df, PATHS_DF_EXPECTED_SCHEMA
from pixel_patrol.utils.utils import format_bytes_to_human_readable

TEST_FIXED_DATETIME = datetime(2023, 1, 15, 10, 30, 0)


def create_mock_fetch_single_directory_tree_side_effect(existing_dir: Path, non_existent_path: Path, mock_data: pl.DataFrame):
    """
    Returns a side effect function for _fetch_single_directory_tree mock.
    """
    def _side_effect(path):
        if path == existing_dir:
            return mock_data
        elif path == non_existent_path:
            raise ValueError(f"Path does not exist or is not a directory: '{path}'")
        else:
            # Fallback for unexpected paths during test; might indicate an issue
            return pl.DataFrame([], schema=PATHS_DF_EXPECTED_SCHEMA)
    return _side_effect


def test_build_paths_df_full_scan(mock_temp_file_system: List[Path], mocker):
    """
    Tests the build_paths_df function, verifying its structure,
    content, including imported_path, and aggregated folder sizes.
    """
    paths_to_scan = mock_temp_file_system

    dir1_path = paths_to_scan[0]
    dir2_path = paths_to_scan[1]
    subdir_a_path = dir1_path / "subdir_a"

    file_a_path = dir1_path / "fileA.jpg"
    file_c_path = subdir_a_path / "fileC.txt"
    file_b_path = dir2_path / "fileB.png"

    file_a_size = file_a_path.stat().st_size
    file_c_size = file_c_path.stat().st_size
    file_b_size = file_b_path.stat().st_size

    dir1_size = file_a_size + file_c_size
    subdir_a_size = file_c_size
    dir2_size = file_b_size

    mock_tree_data_dir1 = pl.DataFrame([
        {"path": str(dir1_path), "name": "test_dir1", "type": "folder", "parent": None, "depth": 0, "size_bytes": 0, "file_extension": None, "modification_date": TEST_FIXED_DATETIME, "imported_path": str(dir1_path)},
        {"path": str(file_a_path), "name": "fileA.jpg", "type": "file", "parent": str(dir1_path), "depth": 1, "size_bytes": file_a_size, "file_extension": "jpg", "modification_date": TEST_FIXED_DATETIME, "imported_path": str(dir1_path)},
        {"path": str(subdir_a_path), "name": "subdir_a", "type": "folder", "parent": str(dir1_path), "depth": 1, "size_bytes": 0, "file_extension": None, "modification_date": TEST_FIXED_DATETIME, "imported_path": str(dir1_path)},
        {"path": str(file_c_path), "name": "fileC.txt", "type": "file", "parent": str(subdir_a_path), "depth": 2, "size_bytes": file_c_size, "file_extension": "txt", "modification_date": TEST_FIXED_DATETIME, "imported_path": str(dir1_path)},
    ])

    mock_tree_data_dir2 = pl.DataFrame([
        {"path": str(dir2_path), "name": "test_dir2", "type": "folder", "parent": None, "depth": 0, "size_bytes": 0, "file_extension": None, "modification_date": TEST_FIXED_DATETIME, "imported_path": str(dir2_path)},
        {"path": str(file_b_path), "name": "fileB.png", "type": "file", "parent": str(dir2_path), "depth": 1, "size_bytes": file_b_size, "file_extension": "png", "modification_date": TEST_FIXED_DATETIME, "imported_path": str(dir2_path)},
    ])

    mocker.patch(
        "pixel_patrol.core.processing._fetch_single_directory_tree",
        side_effect=[mock_tree_data_dir1, mock_tree_data_dir2]
    )

    actual_df = build_paths_df(paths_to_scan)

    assert actual_df is not None
    assert isinstance(actual_df, pl.DataFrame)

    full_expected_schema = PATHS_DF_EXPECTED_SCHEMA.copy()
    full_expected_schema["size_readable"] = pl.String

    assert set(actual_df.columns) == set(full_expected_schema.keys()), \
        f"Columns mismatch.\nActual: {actual_df.columns}\nExpected: {list(full_expected_schema.keys())}"

    for col, expected_dtype in full_expected_schema.items():
        assert col in actual_df.columns, f"Column '{col}' missing from DataFrame."
        assert actual_df[col].dtype == expected_dtype, \
            f"Column '{col}' has wrong dtype. Expected {expected_dtype}, got {actual_df[col].dtype}."

    expected_data = [
        {"path": str(dir1_path), "name": "test_dir1", "type": "folder", "parent": None,
         "depth": 0, "size_bytes": dir1_size, "file_extension": None, "modification_date": TEST_FIXED_DATETIME, "imported_path": str(dir1_path)},
        {"path": str(file_a_path), "name": "fileA.jpg", "type": "file", "parent": str(dir1_path),
         "depth": 1, "size_bytes": file_a_size, "file_extension": "jpg", "modification_date": TEST_FIXED_DATETIME, "imported_path": str(dir1_path)},
        {"path": str(subdir_a_path), "name": "subdir_a", "type": "folder", "parent": str(dir1_path),
         "depth": 1, "size_bytes": subdir_a_size, "file_extension": None, "modification_date": TEST_FIXED_DATETIME, "imported_path": str(dir1_path)},
        {"path": str(file_c_path), "name": "fileC.txt", "type": "file", "parent": str(subdir_a_path),
         "depth": 2, "size_bytes": file_c_size, "file_extension": "txt", "modification_date": TEST_FIXED_DATETIME, "imported_path": str(dir1_path)},
        {"path": str(dir2_path), "name": "test_dir2", "type": "folder", "parent": None,
         "depth": 0, "size_bytes": dir2_size, "file_extension": None, "modification_date": TEST_FIXED_DATETIME, "imported_path": str(dir2_path)},
        {"path": str(file_b_path), "name": "fileB.png", "type": "file", "parent": str(dir2_path),
         "depth": 1, "size_bytes": file_b_size, "file_extension": "png", "modification_date": TEST_FIXED_DATETIME, "imported_path": str(dir2_path)},
    ]

    expected_df = pl.DataFrame(expected_data)
    expected_df = expected_df.with_columns(
        pl.col("size_bytes")
        .map_elements(format_bytes_to_human_readable, return_dtype=pl.String)
        .alias("size_readable")
    )

    for col_name, col_dtype in full_expected_schema.items():
        expected_df = expected_df.with_columns(pl.col(col_name).cast(col_dtype))

    # Define the fixed column order for consistent comparison
    fixed_expected_column_order = [
        "path", "name", "type", "parent", "depth", "size_bytes",
        "file_extension", "modification_date", "imported_path", "size_readable"
    ]

    # Apply the fixed column order to both DataFrames before sorting and comparing
    expected_df_for_comparison = expected_df.select(fixed_expected_column_order).sort("path")
    actual_df_for_comparison = actual_df.select(fixed_expected_column_order).sort("path") # Add .select here

    assert actual_df_for_comparison.equals(expected_df_for_comparison), \
        f"DataFrames are not equal.\nActual:\n{actual_df_for_comparison}\nExpected:\n{expected_df_for_comparison}"


def test_build_paths_df_empty_input(mocker): # Add 'mocker' fixture
    """
    Tests build_paths_df with an empty list of paths.
    Should return an empty DataFrame with the correct schema.
    """
    paths_to_scan = []

    # For empty input, _fetch_single_directory_tree should not be called in the loop.
    # However, if it were called (e.g., if paths_to_scan had non-existent paths),
    # we would mock it. For this test, it's not strictly necessary to patch,
    # but for consistency you could mock it to return an empty DataFrame for any call
    # if you want to ensure it's not trying to hit the real file system.
    mocker.patch(
        "pixel_patrol.core.processing._fetch_single_directory_tree",
        return_value=pl.DataFrame([], schema=PATHS_DF_EXPECTED_SCHEMA) # Provide a default empty schema
    )

    actual_df = build_paths_df(paths_to_scan)

    # 1. Assert it's a DataFrame and is empty
    assert actual_df is not None
    assert isinstance(actual_df, pl.DataFrame)
    assert actual_df.is_empty()

    # 2. Assert schema is correct
    expected_full_schema = PATHS_DF_EXPECTED_SCHEMA.copy()
    actual_schema = actual_df.schema

    for col, expected_dtype in expected_full_schema.items():
        assert col in actual_schema, f"Column '{col}' missing from empty DataFrame."
        assert actual_schema[col] == expected_dtype, \
            f"Column '{col}' has wrong dtype in empty DataFrame. Expected {expected_dtype}, got {actual_schema[col]}."


def test_build_paths_df_non_existent_paths_are_skipped(tmp_path: Path, mocker, caplog):
    """
    Tests that build_paths_df gracefully skips non-existent paths.
    """
    existing_dir = tmp_path / "existing_dir"
    existing_dir.mkdir()
    (existing_dir / "file1.txt").write_bytes(b"content") # 7 bytes

    non_existent_path = tmp_path / "non_existent_dir"

    paths_to_scan = [existing_dir, non_existent_path]

    # Use a fixed datetime for consistent testing results
    test_fixed_datetime = datetime(2023, 1, 1, 10, 0, 0, 123456)

    # Prepare the mock DataFrame with the fixed datetime
    mock_tree_data_existing = pl.DataFrame([
        {"path": str(existing_dir), "name": "existing_dir", "type": "folder", "parent": None,
         "depth": 0, "size_bytes": 0, "file_extension": None, "modification_date": test_fixed_datetime, "imported_path": str(existing_dir)},
        {"path": str(existing_dir / "file1.txt"), "name": "file1.txt", "type": "file", "parent": str(existing_dir),
         "depth": 1, "size_bytes": 7, "file_extension": "txt", "modification_date": test_fixed_datetime, "imported_path": str(existing_dir)},
    ])

    # Patch the function using the helper to create the side effect
    mocker.patch(
        "pixel_patrol.core.processing._fetch_single_directory_tree", # Patch where it's *used*
        side_effect=create_mock_fetch_single_directory_tree_side_effect(
            existing_dir, non_existent_path, mock_tree_data_existing
        )
    )

    # Use caplog to capture and assert on logging messages
    with caplog.at_level(logging.WARNING):
        actual_df = build_paths_df(paths_to_scan)

    # Assert that the expected warning message was logged
    assert "Error processing path" in caplog.text
    assert str(non_existent_path) in caplog.text
    assert "Path does not exist or is not a directory" in caplog.text

    file1_size = (existing_dir / "file1.txt").stat().st_size
    # For a simple directory with one file, the directory's aggregated size will be the file's size
    dir_size = file1_size

    # Prepare the expected DataFrame with the fixed datetime
    expected_data = [
        {"path": str(existing_dir), "name": "existing_dir", "type": "folder", "parent": None,
         "depth": 0, "size_bytes": dir_size, "file_extension": None, "imported_path": str(existing_dir), "modification_date": test_fixed_datetime},
        {"path": str(existing_dir / "file1.txt"), "name": "file1.txt", "type": "file", "parent": str(existing_dir),
         "depth": 1, "size_bytes": file1_size, "file_extension": "txt", "imported_path": str(existing_dir), "modification_date": test_fixed_datetime},
    ]
    expected_df = pl.DataFrame(expected_data)

    # Add 'size_readable' column
    expected_df = expected_df.with_columns(
        pl.col("size_bytes")
        .map_elements(format_bytes_to_human_readable, return_dtype=pl.String)
        .alias("size_readable")
    )

    # Ensure all columns from PATHS_DF_EXPECTED_SCHEMA are present in expected_df
    # Add any missing columns with nulls to match the actual_df's full schema
    for col, dtype in PATHS_DF_EXPECTED_SCHEMA.items():
        if col not in expected_df.columns:
            expected_df = expected_df.with_columns(pl.lit(None, dtype=dtype).alias(col))


    # Define the final column order for consistent comparison
    fixed_expected_column_order = [
        "path", "name", "type", "parent", "depth", "size_bytes",
        "file_extension", "size_readable", "imported_path", "modification_date"
    ]

    # Select and sort columns for both DataFrames before comparison
    expected_df = expected_df.select(fixed_expected_column_order).sort("path")
    actual_df_for_comparison = actual_df.sort("path")
    # Ensure actual_df has the same columns in the same order as expected_df
    actual_df_for_comparison = actual_df_for_comparison.select(expected_df.columns)


    # Final assertion using .equals()
    assert actual_df_for_comparison.equals(expected_df), \
        f"Non-existent paths not skipped correctly.\nActual:\n{actual_df_for_comparison}\nExpected:\n{expected_df}"


def test_build_paths_df_file_extension_edge_cases(mock_temp_file_system_edge_cases: List[Path], mocker):
    """
    Tests correct handling of file extensions for various edge cases.
    """
    paths_to_scan = mock_temp_file_system_edge_cases
    root_path = paths_to_scan[0] # Assuming one root for edge cases

    # Create mock data that _fetch_single_directory_tree would return for this scenario.
    # CRITICAL FIX: Ensure 'file_extension' is an empty string ("") for files without
    # extensions, and the correct lowercase extension string for others.
    # It should only be None for folders.
    mock_tree_data = pl.DataFrame([
        # .hidden_file: No explicit extension, should resolve to ""
        {"path": str(root_path / ".hidden_file"), "name": ".hidden_file", "type": "file",
         "parent": str(root_path), "depth": 1, "size_bytes": 1, "file_extension": "", # FIX: Changed from None to ""
         "modification_date": datetime.now(), "imported_path": str(root_path)},

        # archive.tar.gz: Double extension, `_fetch_single_directory_tree` might give "gz" or "".
        # For the test, we want to ensure build_paths_df handles it correctly.
        # If `_fetch_single_directory_tree` itself already extracts "gz", use "gz" here.
        # If `_fetch_single_directory_tree` gives "" and build_paths_df extracts "gz",
        # then mock it as "". Let's assume _fetch_single_directory_tree gives the last suffix.
        {"path": str(root_path / "archive.tar.gz"), "name": "archive.tar.gz", "type": "file",
         "parent": str(root_path), "depth": 1, "size_bytes": 1, "file_extension": "gz", # FIX: Changed from None to "gz"
         "modification_date": datetime.now(), "imported_path": str(root_path)},

        # file_no_ext: No extension, should resolve to ""
        {"path": str(root_path / "file_no_ext"), "name": "file_no_ext", "type": "file",
         "parent": str(root_path), "depth": 1, "size_bytes": 1, "file_extension": "", # FIX: Changed from None to ""
         "modification_date": datetime.now(), "imported_path": str(root_path)},

        # image.JPEG: Capitalized extension, should resolve to "jpeg"
        {"path": str(root_path / "image.JPEG"), "name": "image.JPEG", "type": "file",
         "parent": str(root_path), "depth": 1, "size_bytes": 1, "file_extension": "jpeg", # FIX: Changed from None to "jpeg"
         "modification_date": datetime.now(), "imported_path": str(root_path)},

        # regular.png: Standard extension, should resolve to "png"
        {"path": str(root_path / "regular.png"), "name": "regular.png", "type": "file",
         "parent": str(root_path), "depth": 1, "size_bytes": 1, "file_extension": "png", # FIX: Changed from None to "png"
         "modification_date": datetime.now(), "imported_path": str(root_path)},

        # Add a folder entry for completeness, file_extension should be None for folders
        {"path": str(root_path), "name": root_path.name, "type": "folder",
         "parent": str(root_path.parent), "depth": 0, "size_bytes": 0, "file_extension": None,
         "modification_date": datetime.now(), "imported_path": str(root_path)},
    ])

    # Patch the _fetch_single_directory_tree function to return our carefully constructed mock data
    mocker.patch(
        "pixel_patrol.core.processing._fetch_single_directory_tree", # Ensure this path is correct
        return_value=mock_tree_data
    )

    # Call the actual build_paths_df function (which is under test)
    actual_df = build_paths_df(paths_to_scan)

    assert actual_df is not None
    assert isinstance(actual_df, pl.DataFrame)

    # Define the expected full schema for verification
    expected_full_schema = PATHS_DF_EXPECTED_SCHEMA.copy()
    expected_full_schema["size_readable"] = pl.String # build_paths_df adds this column

    # Verify schema and columns
    assert set(actual_df.columns) == set(expected_full_schema.keys()), \
        f"Columns mismatch.\nActual: {actual_df.columns}\nExpected: {list(expected_full_schema.keys())}"

    for col, expected_dtype in expected_full_schema.items():
        assert col in actual_df.columns, f"Column '{col}' missing from DataFrame."
        assert actual_df[col].dtype == expected_dtype, \
            f"Column '{col}' has wrong dtype. Expected {expected_dtype}, got {actual_df[col].dtype}."

    # Extract relevant columns and sort for comparison for files
    actual_df_ext = actual_df.filter(pl.col("type") == "file").select(
        pl.col("name"), pl.col("file_extension")
    ).sort("name")

    expected_data = [
        {"name": ".hidden_file", "file_extension": ""},
        {"name": "archive.tar.gz", "file_extension": "gz"},
        {"name": "file_no_ext", "file_extension": ""},
        {"name": "image.JPEG", "file_extension": "jpeg"},
        {"name": "regular.png", "file_extension": "png"},
    ]
    expected_df_ext = pl.DataFrame(expected_data, schema={"name": pl.String, "file_extension": pl.String}).sort("name")

    # Assert that actual and expected DataFrames are equal
    assert actual_df_ext.equals(expected_df_ext), \
        f"File extension handling incorrect.\nActual:\n{actual_df_ext}\nExpected:\n{expected_df_ext}"

    # Additionally, verify that folders have None/Null file_extension
    folder_rows = actual_df.filter(pl.col("type") == "folder")
    # Using is_null() and all() ensures all values are nulls for the column
    assert folder_rows["file_extension"].is_null().all(), "Folders should have null file_extension"


def test_build_paths_df_single_file_and_single_empty_directory(tmp_path: Path, mocker):
    """
    Tests build_paths_df with a single file and a single empty directory.
    """
    # Setup for single file
    single_file_dir = tmp_path / "single_file_test"
    single_file_dir.mkdir()
    test_file_path = single_file_dir / "my_single_file.txt"
    test_file_path.write_bytes(b"test data")  # 9 bytes
    file_size = test_file_path.stat().st_size

    # Capture a specific datetime to use consistently in mocks and expected data
    test_dt = datetime.now()

    # Setup for single empty directory
    empty_dir_path = tmp_path / "empty_dir_test"
    empty_dir_path.mkdir()

    mock_tree_single_file_dir = pl.DataFrame([
        {"path": str(single_file_dir), "name": "single_file_test", "type": "folder", "parent": None,
         "depth": 0, "size_bytes": 0, "file_extension": None, "modification_date": test_dt,
         "imported_path": str(single_file_dir)},
        {"path": str(test_file_path), "name": "my_single_file.txt", "type": "file", "parent": str(single_file_dir),
         "depth": 1, "size_bytes": file_size, "file_extension": "txt", "modification_date": test_dt,
         "imported_path": str(single_file_dir)},
    ], schema=PATHS_DF_EXPECTED_SCHEMA)  # <--- ADD THIS SCHEMA

    mock_tree_empty_dir = pl.DataFrame([
        {"path": str(empty_dir_path), "name": "empty_dir_test", "type": "folder", "parent": None,
         "depth": 0, "size_bytes": 0, "file_extension": None, "modification_date": test_dt,
         "imported_path": str(empty_dir_path)},
    ], schema=PATHS_DF_EXPECTED_SCHEMA)

    # Test with single file
    # We need to configure the mock to return different values on successive calls.
    # mocker.patch can take a list of return_value for sequential calls.
    mocker.patch(
        "pixel_patrol.core.processing._fetch_single_directory_tree",
        side_effect=[mock_tree_single_file_dir, mock_tree_empty_dir]
    )

    # First call to build_paths_df will use mock_tree_single_file_dir
    actual_df_file = build_paths_df([single_file_dir])

    # Expected data for single file
    expected_data_file = [
        # For folders, size_bytes should reflect aggregated size (file_size) after _aggregate_folder_sizes
        {"path": str(single_file_dir), "name": "single_file_test", "type": "folder", "parent": None,
         "depth": 0, "size_bytes": file_size, "file_extension": None, "modification_date": test_dt,
         "imported_path": str(single_file_dir)},
        {"path": str(test_file_path), "name": "my_single_file.txt", "type": "file", "parent": str(single_file_dir),
         "depth": 1, "size_bytes": file_size, "file_extension": "txt", "modification_date": test_dt,
         "imported_path": str(single_file_dir)},
    ]

    # Create the expected DataFrame, adding size_readable and casting/selecting all columns
    expected_df_file = pl.DataFrame(expected_data_file).with_columns(
        pl.col("size_bytes").map_elements(format_bytes_to_human_readable, return_dtype=pl.String).alias("size_readable")
    )

    expected_df_file = expected_df_file.select(
        *[pl.col(col).cast(PATHS_DF_EXPECTED_SCHEMA[col])
          for col in PATHS_DF_EXPECTED_SCHEMA.keys()]  # <-- Removed the redundant part
    ).sort("path")

    actual_df_file_for_comparison = actual_df_file.sort("path")

    # Assert comparison
    assert actual_df_file_for_comparison.equals(expected_df_file), \
        f"Single file processing incorrect.\nActual:\n{actual_df_file_for_comparison}\nExpected:\n{expected_df_file}"

    # Test with single empty directory (this will trigger the second side_effect return value)
    actual_df_empty_dir = build_paths_df([empty_dir_path])

    # Expected data for empty directory
    expected_data_empty_dir = [
        {"path": str(empty_dir_path), "name": "empty_dir_test", "type": "folder", "parent": None,
         "depth": 0, "size_bytes": 0, "file_extension": None, "modification_date": test_dt,
         "imported_path": str(empty_dir_path)},
    ]

    # Create the expected DataFrame, adding size_readable and casting/selecting all columns
    expected_df_empty_dir = pl.DataFrame(expected_data_empty_dir).with_columns(
        pl.col("size_bytes").map_elements(format_bytes_to_human_readable, return_dtype=pl.String).alias("size_readable")
    )

    expected_df_empty_dir = expected_df_empty_dir.select(
        *[pl.col(col).cast(PATHS_DF_EXPECTED_SCHEMA[col])
          for col in PATHS_DF_EXPECTED_SCHEMA.keys()]
    ).sort("path")

    actual_df_empty_dir_for_comparison = actual_df_empty_dir.sort("path")

    # Assert comparison
    assert actual_df_empty_dir_for_comparison.equals(expected_df_empty_dir), \
        f"Single empty directory processing incorrect.\nActual:\n{actual_df_empty_dir_for_comparison}\nExpected:\n{expected_df_empty_dir}"


def test_build_paths_df_complex_size_aggregation(mock_temp_file_system_complex: List[Path], mocker): # Add 'mocker'
    """
    Tests the recursive size aggregation for a more complex nested folder structure.
    """
    paths_to_scan = mock_temp_file_system_complex
    root_dir = paths_to_scan[0]
    sub_dir_a = root_dir / "sub_dir_a"
    file_a = sub_dir_a / "fileA.txt"
    sub_sub_dir_a = sub_dir_a / "sub_sub_dir_a"
    file_b = sub_sub_dir_a / "fileB.txt"
    sub_dir_b = root_dir / "sub_dir_b"
    file_c = sub_dir_b / "fileC.txt"

    # Expected raw file sizes (from your fixture's definition, not aggregated)
    size_file_a = 10
    size_file_b = 20
    size_file_c = 30

    # Mock data that _fetch_single_directory_tree would return.
    # Note: size_bytes here are *raw file sizes*, not aggregated folder sizes,
    # as aggregation is done by _aggregate_folder_sizes in the real build_paths_df.
    mock_tree_data_complex = pl.DataFrame([
        {"path": str(root_dir), "name": "root_dir", "type": "folder", "parent": None, "depth": 0, "size_bytes": 0, "file_extension": None, "modification_date": datetime.now(), "imported_path": str(root_dir)},
        {"path": str(sub_dir_a), "name": "sub_dir_a", "type": "folder", "parent": str(root_dir), "depth": 1, "size_bytes": 0, "file_extension": None, "modification_date": datetime.now(), "imported_path": str(root_dir)},
        {"path": str(file_a), "name": "fileA.txt", "type": "file", "parent": str(sub_dir_a), "depth": 2, "size_bytes": size_file_a, "file_extension": "txt", "modification_date": datetime.now(), "imported_path": str(root_dir)},
        {"path": str(sub_sub_dir_a), "name": "sub_sub_dir_a", "type": "folder", "parent": str(sub_dir_a), "depth": 2, "size_bytes": 0, "file_extension": None, "modification_date": datetime.now(), "imported_path": str(root_dir)},
        {"path": str(file_b), "name": "fileB.txt", "type": "file", "parent": str(sub_sub_dir_a), "depth": 3, "size_bytes": size_file_b, "file_extension": "txt", "modification_date": datetime.now(), "imported_path": str(root_dir)},
        {"path": str(sub_dir_b), "name": "sub_dir_b", "type": "folder", "parent": str(root_dir), "depth": 1, "size_bytes": 0, "file_extension": None, "modification_date": datetime.now(), "imported_path": str(root_dir)},
        {"path": str(file_c), "name": "fileC.txt", "type": "file", "parent": str(sub_dir_b), "depth": 2, "size_bytes": size_file_c, "file_extension": "txt", "modification_date": datetime.now(), "imported_path": str(root_dir)},
    ])

    mocker.patch(
        "pixel_patrol.core.processing._fetch_single_directory_tree",
        return_value=mock_tree_data_complex
    )

    actual_df = build_paths_df(paths_to_scan)

    # These expected sizes are the *aggregated* sizes, which the real _aggregate_folder_sizes should compute
    size_sub_sub_dir_a = size_file_b
    size_sub_dir_a = size_file_a + size_sub_sub_dir_a
    size_sub_dir_b = size_file_c
    size_root_dir = size_sub_dir_a + size_sub_dir_b

    actual_sizes_df = actual_df.select(
        pl.col("path"), pl.col("size_bytes"), pl.col("name"), pl.col("type")
    ).sort("path")

    expected_data = [
        {"path": str(root_dir), "name": "root_dir", "type": "folder", "size_bytes": size_root_dir},
        {"path": str(sub_dir_a), "name": "sub_dir_a", "type": "folder", "size_bytes": size_sub_dir_a},
        {"path": str(file_a), "name": "fileA.txt", "type": "file", "size_bytes": size_file_a},
        {"path": str(sub_sub_dir_a), "name": "sub_sub_dir_a", "type": "folder", "size_bytes": size_sub_sub_dir_a},
        {"path": str(file_b), "name": "fileB.txt", "type": "file", "size_bytes": size_file_b},
        {"path": str(sub_dir_b), "name": "sub_dir_b", "type": "folder", "size_bytes": size_sub_dir_b},
        {"path": str(file_c), "name": "fileC.txt", "type": "file", "size_bytes": size_file_c},
    ]
    expected_df_sizes = pl.DataFrame(expected_data).select(
        pl.col("path"), pl.col("size_bytes"), pl.col("name"), pl.col("type")
    ).sort("path")

    assert actual_sizes_df.equals(expected_df_sizes), \
        f"Size aggregation incorrect.\nActual:\n{actual_sizes_df}\nExpected:\n{expected_df_sizes}"
