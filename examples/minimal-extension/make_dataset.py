"""Generates the tiny "Pixel HAI Watch" dataset used by this example.

Each "dive patch" is a 16x16 grayscale snapshot from a deep-sea camera,
stored as a `.parquet` table - one `uint8` column per pixel column (X), one
row per pixel row (Y). A patch's overall brightness depends on the ocean
layer it was logged in (`depth_zone`): bright near the surface where sunlight
reaches, almost black in the deep, with a soft top-to-bottom gradient as that
sunlight fades on its way down. A few bright "glows" are sprinkled on top -
rarely near the surface, generously in the deep - standing in for
bioluminescence, the light some sharks make themselves down where the sun
never reaches. SharkCamLoader / GlowSpotterProcessor have something intuitive
to read and count: of course there's more glow the deeper - and darker - it
gets. Fake "image metadata" (`depth_zone`) is stashed in the parquet schema
metadata - the same slot real formats use for instrument metadata - and
read out by SharkCamLoader.

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

# A short, charming blurb - passed as `process_files(..., description=...)`
# in create_and_show_report.py, where the viewer displays it below the
# project title (and stores it in the report's own metadata).
DESCRIPTION = (
    "Some deep-sea sharks make their own light: in 2021 researchers confirmed "
    "the kitefin shark as the largest known glowing vertebrate, producing a "
    "soft blue-green bioluminescence in ocean layers sunlight never reaches. "
    "https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2021.633582/full"
)

# depth_zone -> (base brightness, min/max number of glows sprinkled on top).
# It's brightest near the surface, where the sun still reaches, and darkest
# in the deep - and that's exactly where bioluminescence shows up, by
# construction: the deeper and darker the water, the more glows light up.
DEPTH_PROFILES = {
    "sunlit":   {"brightness": 205, "glows": (0, 0)},
    "twilight": {"brightness": 120, "glows": (0, 2)},
    "midnight": {"brightness": 55,  "glows": (3, 6)},
    "abyss":    {"brightness": 20,  "glows": (6, 10)},
}

# (folder, file stem, depth_zone)
PATCHES = [
    ("azores_log",   "patch_sunlit",   "sunlit"),
    ("azores_log",   "patch_twilight", "twilight"),
    ("azores_log",   "patch_midnight", "midnight"),
    ("azores_log",   "patch_abyss",    "abyss"),
    ("kermadec_log", "patch_sunlit",   "sunlit"),
    ("kermadec_log", "patch_twilight", "twilight"),
    ("kermadec_log", "patch_midnight", "midnight"),
    ("kermadec_log", "patch_abyss",    "abyss"),
]


def stamp_glow(water: np.ndarray, rng: np.random.Generator, peak: float) -> None:
    """Add one soft, round glow - a bright core fading out over a couple of
    pixels, the way bioluminescent light blooms and scatters underwater
    (rather than appearing as a sharp point, like a star against open sky)."""
    size = water.shape[0]
    cy, cx = rng.integers(0, size, size=2)
    yy, xx = np.ogrid[:size, :size]
    falloff = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 2.0)
    water += peak * falloff


def make_patch(rng: np.random.Generator, brightness: int, glow_range: tuple[int, int]) -> np.ndarray:
    """Paint a patch of ocean: a soft top-to-bottom gradient (sunlight fading
    with depth), some noise, and a sprinkle of bioluminescent glows."""
    gradient = np.linspace(15, -15, SPRITE_SIZE).reshape(-1, 1)
    water = brightness + gradient + rng.integers(-10, 11, size=(SPRITE_SIZE, SPRITE_SIZE))
    water = water.astype(np.float32)

    lo, hi = glow_range
    for _ in range(int(rng.integers(lo, hi + 1))):
        stamp_glow(water, rng, peak=float(rng.integers(140, 200)))

    return np.clip(water, 0, 255).astype(np.uint8)


def write_patch(path: Path, patch: np.ndarray, depth_zone: str) -> None:
    columns = {f"px_{x:02d}": pa.array(patch[:, x], type=pa.uint8()) for x in range(patch.shape[1])}
    table = pa.table(columns).replace_schema_metadata({"depth_zone": depth_zone})
    pq.write_table(table, path)


def main() -> None:
    rng = np.random.default_rng(SEED)
    data_dir = HERE / "data"

    for folder, stem, depth_zone in PATCHES:
        out_dir = data_dir / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        profile = DEPTH_PROFILES[depth_zone]
        patch = make_patch(rng, profile["brightness"], profile["glows"])
        out_path = out_dir / f"{stem}.parquet"
        write_patch(out_path, patch, depth_zone)
        print(f"wrote {out_path.relative_to(HERE)}  [{depth_zone}]")


if __name__ == "__main__":
    main()
