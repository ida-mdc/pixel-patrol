import time, gc, sys
import numpy as np
import polars as pl

def make_deep_rows(n_rows, bins=256, as_numpy=True):
    rows = []
    for i in range(n_rows):
        arr = np.random.randint(0, 1000, size=bins, dtype=np.int32)
        if as_numpy:
            hist = arr  # np.ndarray
        else:
            hist = arr.tolist()  # python list
        rows.append({
            "id": i,
            "histogram_counts": hist,
            "histogram_min": np.float32(arr.min()),
            "histogram_max": np.float32(arr.max()),
        })
    return rows

def run_once(n_rows=2000, bins=256, repeats=3):
    results = []
    for as_numpy in (True, False):
        deep_rows = make_deep_rows(n_rows, bins=bins, as_numpy=as_numpy)
        # rough python-level size for payload
        py_size = sum(sys.getsizeof(r["histogram_counts"]) for r in deep_rows)
        times = []
        for _ in range(repeats):
            gc.collect()
            t0 = time.perf_counter()
            df = pl.DataFrame(deep_rows,
                              nan_to_null=True,
                              strict=False,
                              infer_schema_length=None)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg_time = sum(times) / len(times)
        # Inspect dtype
        col = df["histogram_counts"]
        print("="*60)
        print("as_numpy:", as_numpy)
        print("n_rows:", n_rows, "bins:", bins)
        print("avg build time (s):", avg_time)
        print("python payload bytes (approx):", py_size)
        print("polars dtype:", col.dtype)
        # show one element and its type/len
        cell = col[0]
        print("sample cell python type:", type(cell), "len:", (len(cell) if cell is not None else None))
        # Now force-cast to the desired target
        try:
            cast_df = df.with_columns(pl.col("histogram_counts").cast(pl.List(pl.Int32)).alias("histogram_counts"))
            print("after cast polars dtype:", cast_df["histogram_counts"].dtype)
        except Exception as e:
            print("cast failed:", e)
        results.append((as_numpy, avg_time, str(col.dtype)))
    return results

if __name__ == "__main__":
    print("Running benchmark... (this may use memory proportional to n_rows*bins)")
    run_once(n_rows=2000, bins=256, repeats=3)
    print("Done.")
