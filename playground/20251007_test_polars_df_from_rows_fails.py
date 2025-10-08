import sys
import traceback
from typing import List, Dict, Any, Tuple
import numpy as np
import polars as pl
import datetime as dt
import csv

Case = Tuple[str, List[Dict[str, Any]]]

def make_cases() -> List[Case]:
    return [
        # 1) None with int
        ("int_vs_none", [{"a": 1}, {"a": None}]),

        # 2) list[float] vs list[int] in same column
        ("list_float_vs_list_int", [{"hist": [0.1, 0.2]}, {"hist": [1, 2, 3]}]),

        # 3) numpy uint16 vs numpy uint8
        ("uint16_vs_uint8", [{"mx": np.uint16(65535)}, {"mx": np.uint8(255)}]),

        # 4) ndarray of ndarray[numpy.uint8] (nested ndarrays)
        ("ndarray_of_ndarray_uint8", [{"thumb": np.array([np.array([1,2,3], dtype=np.uint8)], dtype=object)},
                                      {"thumb": np.array([np.array([4,5], dtype=np.uint8)], dtype=object)}]),

        # 5) row key mismatch (different columns per row)
        ("row_key_mismatch", [{"a": 1}, {"b": 2}]),

        # 6) scalar vs list in same column
        ("scalar_vs_list", [{"x": 1}, {"x": [1,2,3]}]), # TODO: FAIL SchemaError('failed to determine supertype of i64 and list[i64]')

        # 7) dict value in a column
        ("dict_value", [{"cfg": {"alpha": 1.0}}, {"cfg": {"beta": 2}}]),

        # 8) datetime mixed with string
        ("datetime_vs_string", [{"t": dt.datetime(2024,1,1)}, {"t": "2024-01-01"}]), # TODO: FAIL

        # 9) bool mixed with string
        ("bool_vs_string", [{"b": True}, {"b": "False"}]),

        # 10) float vs numeric string
        ("float_vs_numeric_string", [{"v": 1.23}, {"v": "2.34"}]),

        # 11) empty list vs non-empty list
        ("empty_list_vs_list", [{"lst": []}, {"lst": [1,2,3]}]),

        # 12) list with mixed inner types
        ("list_mixed_inner_types", [{"lst": [1, 2.0, "3"]}, {"lst": [4]}]), # TODO: FAIL TypeError('unexpected value while building Series of type Int64; found value of type Float64')

        # 13) tuple vs list
        ("tuple_vs_list", [{"t": (1,2,3)}, {"t": [1,2,3]}]),

        # 14) ndarray vs list with same content
        ("ndarray_vs_list", [{"arr": np.array([1,2,3])}, {"arr": [1,2,3]}]), # TODO: FAIL SchemaError('failed to determine supertype of object and list[i64]')

        # 15) bytes vs str
        ("bytes_vs_str", [{"s": b"bytes"}, {"s": "text"}]),

        # 16) mixed ndarray shapes
        ("ndarray_mixed_shapes", [{"arr": np.array([1,2,3])}, {"arr": np.array([[1,2,3]])}]),

        # 17) nested list of lists vs list
        ("list_of_lists_vs_list", [{"x": [[1,2], [3,4]]}, {"x": [5,6]}]), # TODO: FAIL SchemaError('failed to determine supertype of list[list[i64]] and list[i64]')

        # 18) None with list
        ("list_vs_none", [{"x": None}, {"x": [1,2,3]}]),

        # 19) consistent uint8 only (control that should PASS)
        ("control_uint8_only", [{"mx": np.uint8(1)}, {"mx": np.uint8(2)}]),

        # 20) consistent Python ints only (control that should PASS)
        ("control_pyint_only", [{"a": 1}, {"a": 2}]),
    ]

def try_build_df(rows: List[Dict[str, Any]]) -> Tuple[bool, str, str]:
    """
    Attempt to build a Polars DataFrame; return (ok, schema_or_err, note)
    """
    try:
        df = pl.DataFrame(rows)  # pl.from_dicts(rows) equivalent
        schema = str(df.schema)
        return True, schema, "ok"
    except Exception as e:
        # Try to extract concise cause
        err = repr(e)
        # Sometimes additional context is helpful
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        return False, err, tb

def main():
    cases = make_cases()
    results = []
    for name, rows in cases:
        ok, schema_or_err, note = try_build_df(rows)
        results.append({
            "case": name,
            "pass": "PASS" if ok else "FAIL",
            "detail": schema_or_err,
            "note": note
        })
    # Pretty print
    print(f"{'CASE':30} {'RESULT':7}  DETAIL/SCHEMA")
    print("-"*100)
    for r in results:
        print(f"{r['case']:30} {r['pass']:7}  {r['detail']}")
    # Write CSV
    out_csv = "polars_row_dtype_cases_results.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case","pass","detail","note"])
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {out_csv}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
