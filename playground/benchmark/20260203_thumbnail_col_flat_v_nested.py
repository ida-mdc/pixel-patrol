import numpy as np
import polars as pl
import time

N = 500_000
S = 64
FLAT_SIZE = S * S

print(f"Generating raw data for {N} rows...")
raw_3d = np.random.randint(0, 256, (N, S, S), dtype=np.uint8)


def benchmark_flat():
    print("Running: Flat Array (pl.Array(u8, 4096))...")
    # Flatten the 3D data to 2D for the constructor
    raw_flat = raw_3d.reshape(N, FLAT_SIZE)

    t0 = time.perf_counter()
    df = pl.DataFrame({"thub": raw_flat}, schema={"thub": pl.Array(pl.UInt8, FLAT_SIZE)})
    ct = time.perf_counter() - t0

    t1 = time.perf_counter()
    # Direct view and reshape
    _ = df["thub"].to_numpy().reshape(-1, S, S)
    at = time.perf_counter() - t1
    return ct, at, df.estimated_size()


def benchmark_nested():
    print("Running: Nested (pl.List(pl.Array(u8, 64)))...")
    t0 = time.perf_counter()
    ##
    ## New:
    # This simulates your processor creating a series of arrays per cell
    # To be 'fair' and avoid crashes, we use a more direct constructor
    nested_data = [pl.Series(row, dtype=pl.Array(pl.UInt8, S)) for row in raw_3d]
    df = pl.DataFrame({"thub": pl.Series(nested_data)})
    ##
    ct = time.perf_counter() - t0

    t1 = time.perf_counter()
    ##
    ## New:
    # Accessing nested structures requires converting to list then stacking
    # This is usually where the SIGKILL happens if memory isn't managed
    _ = np.stack([x.to_numpy() for x in df["thub"]])
    ##
    at = time.perf_counter() - t1
    return ct, at, df.estimated_size()


# Execution
res_flat = benchmark_flat()
res_nested = benchmark_nested()

print("\n" + "=" * 70)
print(f"{'Method':<20} | {'Create (s)':<10} | {'Access (s)':<10} | {'Size (MB)':<10}")
print("-" * 70)
print(f"{'Flat Array':<20} | {res_flat[0]:<10.4f} | {res_flat[1]:<10.4f} | {res_flat[2] / 1024 ** 2:<10.2f}")
print(
    f"{'Nested List-Array':<20} | {res_nested[0]:<10.4f} | {res_nested[1]:<10.4f} | {res_nested[2] / 1024 ** 2:<10.2f}")