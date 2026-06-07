"""Generates the tiny "Pixel Sky Watch" dataset used by this example.

Each "sky patch" is a 16x16 grayscale snapshot of the sky stored as a
`.parquet` table - one `uint8` column per pixel column (X), one row per pixel
row (Y). A patch's overall brightness depends on the time of day it was
"logged" (bright at midday, dark at night), with a few bright "stars"
sprinkled on top - mostly at night, rarely during the day - so
SkyPatchLoader / StarSpotterProcessor have something intuitive to read and
count. Fake "image metadata" describing the patch (time_of_day, cloud_cover)
is stashed in the parquet schema metadata, the same slot real formats use for
instrument metadata.

Run once to (re)create `data/`:

    uv run python make_dataset.py
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
SPRITE_SIZE = 16
SEED = 42

# time_of_day -> (base brightness, min/max number of stars sprinkled on top).
# It's brightest at midday and darkest at night - and that's exactly when you
# actually see stars, by construction.
TIME_PROFILES = {
    "dawn":  {"brightness": 110, "stars": (0, 2)},
    "day":   {"brightness": 205, "stars": (0, 0)},
    "dusk":  {"brightness": 95,  "stars": (0, 2)},
    "night": {"brightness": 25,  "stars": (8, 14)},
}

# (folder, file stem, time_of_day, cloud_cover)
PATCHES = [
    ("rooftop_log",  "patch_dawn",  "dawn",  "clear"),
    ("rooftop_log",  "patch_day",   "day",   "clear"),
    ("rooftop_log",  "patch_dusk",  "dusk",  "cloudy"),
    ("rooftop_log",  "patch_night", "night", "clear"),
    ("campsite_log", "patch_dawn",  "dawn",  "cloudy"),
    ("campsite_log", "patch_day",   "day",   "cloudy"),
    ("campsite_log", "patch_dusk",  "dusk",  "clear"),
    ("campsite_log", "patch_night", "night", "clear"),
]


def make_patch(rng: np.random.Generator, brightness: int, star_range: tuple[int, int]) -> np.ndarray:
    """Paint a patch of sky: a soft vertical gradient, some noise, and a sprinkle of stars."""
    gradient = np.linspace(-15, 15, SPRITE_SIZE).reshape(-1, 1)
    sky = brightness + gradient + rng.integers(-10, 11, size=(SPRITE_SIZE, SPRITE_SIZE))
    sky = np.clip(sky, 0, 255).astype(np.uint8)

    lo, hi = star_range
    n_stars = int(rng.integers(lo, hi + 1))
    for _ in range(n_stars):
        y, x = rng.integers(0, SPRITE_SIZE, size=2)
        sky[y, x] = rng.integers(225, 256)

    return sky


def write_patch(path: Path, patch: np.ndarray, time_of_day: str, cloud_cover: str) -> None:
    columns = {f"px_{x:02d}": pa.array(patch[:, x], type=pa.uint8()) for x in range(patch.shape[1])}
    table = pa.table(columns).replace_schema_metadata({"time_of_day": time_of_day, "cloud_cover": cloud_cover})
    pq.write_table(table, path)


def main() -> None:
    rng = np.random.default_rng(SEED)
    data_dir = HERE / "data"

    for folder, stem, time_of_day, cloud_cover in PATCHES:
        out_dir = data_dir / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        profile = TIME_PROFILES[time_of_day]
        patch = make_patch(rng, profile["brightness"], profile["stars"])
        out_path = out_dir / f"{stem}.parquet"
        write_patch(out_path, patch, time_of_day, cloud_cover)
        print(f"wrote {out_path.relative_to(HERE)}  [{time_of_day} · {cloud_cover}]")


if __name__ == "__main__":
    main()
