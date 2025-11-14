################ RESULTS ################
######## SUMMARY: CHUNKS are great for processing ############
######## But not sure how to write chunks to disk without knowing schema ########

# === SHORT RUN (MANY COLS) ===
# rows=10000, base_cols=200, extra_pool=200, p_extra=0.05, p_missing=0.02, p_nan=0.05
# df from full list of dicts   1.0225s   df size=(10000, 401)
# write once at end              0.0754s
# -- batch size sweep --
# df from chunks (bs=   500)    0.9760s  shape=(10000, 401)
# df from chunks (bs=  1000)    0.8901s  shape=(10000, 401)
# df from chunks (bs=  2000)    0.9549s  shape=(10000, 401)
# df from chunks (bs=  5000)    0.9757s  shape=(10000, 401)
# df from chunks (bs= 10000)    0.9620s  shape=(10000, 401)
# df from chunks (bs= 20000)    0.9338s  shape=(10000, 401)
# df from chunks (bs= 50000)    0.9408s  shape=(10000, 401)
#
# === LONG RUN (100k rows × 10k cols) ===
# rows=50000, base_cols=1000, extra_pool=0, p_extra=0.0, p_missing=0.02, p_nan=0.02
# df from full list of dicts  19.8582s   df size=(50000, 1001)
# write once at end              1.3602s
# -- batch size sweep --
# df from chunks (bs=   500)   19.1656s  shape=(50000, 1001)
# df from chunks (bs=  1000)   17.8349s  shape=(50000, 1001)
# df from chunks (bs=  2000)   23.3457s  shape=(50000, 1001)
# df from chunks (bs=  5000)   22.1695s  shape=(50000, 1001)
# df from chunks (bs= 10000)   18.6401s  shape=(50000, 1001)
# df from chunks (bs= 20000)   19.6546s  shape=(50000, 1001)
# df from chunks (bs= 50000)   18.9893s  shape=(50000, 1001)



import time, random
from typing import Dict, List
import polars as pl
import os
# Because polars can't append to parquet directly:
import pyarrow as pa
import pyarrow.parquet as pq

import os
import time
import random

def time_call(label, fn, *args, **kwargs):
    random.seed(42)                               # fair, repeatable rows
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"{label:28s} {dt:8.4f}s")
    return res, dt


def build_record_dict(i: int,
                 base_cols: int,
                 extra_pool: int,
                 p_extra: float,
                 p_missing: float,
                 p_nan: float) -> Dict[str, object]:
    rec: Dict[str, object] = {}
    for k in range(base_cols):
        if random.random() >= p_missing:
            v = i * 1000 + k
            if random.random() < p_nan:
                v = float("nan")
            rec[f"c{k}"] = v
    for _ in range(extra_pool):
        if random.random() < p_extra:
            k = random.randrange(extra_pool)
            v = i * 2000 + k
            if random.random() < p_nan:
                v = float("nan")
            rec[f"x{k}"] = v
    rec["path"] = f"/fake/path/{i:08d}.bin"
    return rec

def build_rows_as_list_of_dicts(n: int, **kw) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for i in range(n):
        rows.append(build_record_dict(i, **kw))
    return rows

def build_polars_from_rows(rows: List[Dict[str, object]]) -> pl.DataFrame:
    ## TODO: Strange that so far we don't need all those params in pp!
    ## Here we would crash without it. Maybe we need to better investigate.
    ## See 20251007_test_polars_df_from_rows_fails.py
    return pl.DataFrame(rows, nan_to_null=True, strict=False, infer_schema_length=None)

# def build_df_from_chunks(base_cols: int, extra_pool: int, n: int,
#                                 p_extra: float, p_missing: float, p_nan: float,
#                                 batch_size: int = 10_000) -> pl.DataFrame:
#     df = pl.DataFrame([])  # empty start; we'll concat diagonally
#     buf: List[Dict[str, object]] = []
#
#     for i in range(n):
#         buf.append(build_record_dict(i, base_cols=base_cols, extra_pool=extra_pool,
#                                      p_extra=p_extra, p_missing=p_missing, p_nan=p_nan))
#         if len(buf) == batch_size:
#             chunk = pl.DataFrame(buf, nan_to_null=True, strict=False, infer_schema_length=None)
#             df = pl.concat([df, chunk], how="diagonal_relaxed", rechunk=False)
#             buf.clear()
#
#     if buf:  # flush remaining rows (necessary so the last partial batch isn't dropped)
#         chunk = pl.DataFrame(buf, nan_to_null=True, strict=False, infer_schema_length=None)
#         df = pl.concat([df, chunk], how="diagonal_relaxed", rechunk=False)
#
#     return df

def build_df_from_chunks(base_cols: int, extra_pool: int, n: int,
                         p_extra: float, p_missing: float, p_nan: float,
                         batch_size: int = 10_000,
                         write: bool = False,
                         write_path: str | None = None,
                         compression: str = "zstd") -> pl.DataFrame:
    """
    Incrementally build a DF by concatenating chunks (diagonal_relaxed).
    If write=True, ALSO append each chunk to a SINGLE Parquet file (PyArrow).
    Returns the full in-memory DF regardless.
    """
    import pyarrow.parquet as pq

    df = pl.DataFrame([])
    buf: List[Dict[str, object]] = []
    writer = None  # PyArrow ParquetWriter

    for i in range(n):
        buf.append(build_record_dict(i, base_cols=base_cols, extra_pool=extra_pool,
                                     p_extra=p_extra, p_missing=p_missing, p_nan=p_nan))
        if len(buf) == batch_size:
            chunk = pl.DataFrame(buf, nan_to_null=True, strict=False, infer_schema_length=None)
            df = pl.concat([df, chunk], how="diagonal_relaxed", rechunk=False)

            if write:
                table = chunk.to_arrow()
                if writer is None:
                    if not write_path:
                        raise ValueError("write=True requires write_path (single Parquet file).")
                    writer = pq.ParquetWriter(write_path, table.schema, compression=compression)
                else:
                    table = table.cast(writer.schema)  # enforce first-chunk schema
                writer.write_table(table)

            buf.clear()

    if buf:
        chunk = pl.DataFrame(buf, nan_to_null=True, strict=False, infer_schema_length=None)
        df = pl.concat([df, chunk], how="diagonal_relaxed", rechunk=False)

        if write:
            table = chunk.to_arrow()
            if writer is None:
                if not write_path:
                    raise ValueError("write=True requires write_path (single Parquet file).")
                writer = pq.ParquetWriter(write_path, table.schema, compression=compression)
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)

    if writer is not None:
        writer.close()

    return df


def write_parquet_end(df: pl.DataFrame, path: str, compression: str = "zstd") -> float:
    t0 = time.perf_counter()
    # Should we use compression?
    df.write_parquet(path, compression=compression)
    return time.perf_counter() - t0


def run(name: str,
        n: int,
        base_cols: int,
        extra_pool: int,
        p_extra: float,
        p_missing: float,
        p_nan: float):
    print(f"\n=== {name} ===")
    print(f"rows={n}, base_cols={base_cols}, extra_pool={extra_pool}, p_extra={p_extra}, p_missing={p_missing}, p_nan={p_nan}")

    random.seed(42)
    t = time.perf_counter()
    df = build_df_from_full_list_of_dicts(base_cols, extra_pool, n, p_extra, p_missing, p_nan)
    t = time.perf_counter() - t
    print(f"df from full list of dicts {t:8.4f}s   df size={df.shape}")

    # end-once write timing
    out_end = f"/tmp/pp_end_{name.replace(' ', '_')}.parquet"
    if os.path.exists(out_end): os.remove(out_end)
    _, t_end = time_call("write once at end",
                         df.write_parquet, out_end, compression="zstd")

    # TODO: not running as its complicated to write to parquet without knowing the schema.
    # Maybe run chunks individually?
    # print("-- chunk sweep --")
    # for bs in [1_000, 5_000, 10_000, 20_000, 50_000]:
    #     # incremental build, no writing
    #     df_inc, _ = time_call(f"chunks bs={bs} (no write)",
    #                           build_df_from_chunks,
    #                           base_cols, extra_pool, n,
    #                           p_extra, p_missing, p_nan,
    #                           batch_size=bs, write=False)
    #
    #     # incremental build, append to ONE parquet via PyArrow
    #     out_one = f"/tmp/pp_append_{name.replace(' ', '_')}_bs{bs}.parquet"
    #     if os.path.exists(out_one): os.remove(out_one)
    #     _, _ = time_call(f"chunks bs={bs} (append 1 file)",
    #                      build_df_from_chunks,
    #                      base_cols, extra_pool, n,
    #                      p_extra, p_missing, p_nan,
    #                      batch_size=bs, write=True,
    #                      write_path=out_one, compression="zstd")


    print("-- batch size sweep --")
    bench_chunk_batch_sizes(base_cols, extra_pool, n,
                            p_extra, p_missing, p_nan,
                            sizes=[500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000])


def build_df_from_full_list_of_dicts(base_cols, extra_pool, n, p_extra, p_missing, p_nan):
    rows = build_rows_as_list_of_dicts(n, base_cols=base_cols, extra_pool=extra_pool,
                                       p_extra=p_extra, p_missing=p_missing, p_nan=p_nan)
    df = build_polars_from_rows(rows)
    return df

def bench_chunk_batch_sizes(base_cols: int, extra_pool: int, n: int,
                            p_extra: float, p_missing: float, p_nan: float,
                            sizes: list[int]):
    for bs in sizes:
        random.seed(42)
        t0 = time.perf_counter()
        df = build_df_from_chunks(base_cols, extra_pool, n,
                                  p_extra, p_missing, p_nan,
                                  batch_size=bs)
        dt = time.perf_counter() - t0
        print(f"df from chunks (bs={bs:>6})  {dt:8.4f}s  shape={df.shape}")


if __name__ == "__main__":
    # SHORT RUN (many cols, moderate rows)
    run("SHORT RUN (MANY COLS)",
        n=10_000,
        base_cols=200,
        extra_pool=200,
        p_extra=0.05,
        p_missing=0.02,
        p_nan=0.05)

    # LONG RUN (toy sizes per your example)
    run("LONG RUN (100k rows × 10k cols)",
        n=50_000,
        base_cols=1_000,
        extra_pool=0,
        p_extra=0.0,
        p_missing=0.02,
        p_nan=0.02)
