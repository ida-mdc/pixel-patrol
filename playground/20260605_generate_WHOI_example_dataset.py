"""
Generate a colorised subset of the WHOI plankton dataset for use as an example report.

Takes 10 randomly selected images from each of the 4 conditions, preserving the
species subdirectory structure. Each image is converted from grayscale to 3-channel
(YXS) by multiplying with a randomly assigned vivid color. The S dimension is what
bioio exposes as color channels.

Source: https://darchive.mblwhoilibrary.org/entities/publication/48f7a9a7-23f5-5584-a7de-6a5a64d01a37
"""

import random
import numpy as np
from pathlib import Path
from PIL import Image

SRC_BASE   = Path(__file__).parent.parent / "plankton_processed"
DST_BASE   = Path(__file__).parent
CONDITIONS = ["condition1_org", "condition2_bl", "condition3_comp", "condition4_nois"]
N_PER_CONDITION = 10
SEED = 42

# Vivid, visually distinct colors assigned randomly per image
PALETTE = [
    (1.0, 0.15, 0.15),  # red
    (0.15, 0.85, 0.15),  # green
    (0.15, 0.15, 1.0),  # blue
    (1.0, 0.75, 0.05),  # amber
    (0.85, 0.15, 0.85),  # magenta
    (0.05, 0.85, 0.85),  # cyan
    (1.0, 0.45, 0.05),  # orange
    (0.5, 0.15, 1.0),   # purple
    (0.15, 1.0, 0.5),   # mint
    (1.0, 0.15, 0.5),   # pink
]

rng = random.Random(SEED)

for condition in CONDITIONS:
    src_cond = SRC_BASE / condition
    all_images = (sorted(src_cond.rglob("*.png")) +
                  sorted(src_cond.rglob("*.jpeg")) +
                  sorted(src_cond.rglob("*.jpg")))
    selected   = rng.sample(all_images, min(N_PER_CONDITION, len(all_images)))

    for src_path in selected:
        rel      = src_path.relative_to(src_cond)
        dst_path = DST_BASE / condition / rel          # keep original extension
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        gray  = np.array(Image.open(src_path).convert('L'), dtype=np.float32)   # YX
        color = rng.choice(PALETTE)
        rgb   = np.stack([gray * color[0], gray * color[1], gray * color[2]], axis=-1)
        rgb   = np.clip(rgb, 0, 255).astype(np.uint8)               # YXS

        suffix = src_path.suffix.lower()
        if suffix in ('.jpg', '.jpeg'):
            Image.fromarray(rgb).save(dst_path, format='JPEG', quality=85)
        else:
            Image.fromarray(rgb).save(dst_path, format='PNG')

    print(f"  {condition}: {len(selected)} images written")

print("Done.")
