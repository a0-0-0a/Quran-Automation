#!/usr/bin/env python3
"""Scale a texture's 1K tiles darker by a factor (keeps the colour ratios).

    python3 tools/scale_texture.py --dir assets/textures/Marble019 --factor 0.10

Then re-merge with tools/merge_texture.py.  The factor multiplies every RGB
channel equally, so hue/saturation stay correct; only the brightness drops.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--factor", required=True, type=float)
    a = ap.parse_args()
    if not 0.0 < a.factor <= 1.0:
        raise SystemExit("factor must be in (0, 1]")

    for p in sorted(a.dir.glob("tile_*.png")):
        im = Image.open(p).convert("RGB")
        arr = np.asarray(im).astype(np.float32) * a.factor
        arr = np.clip(arr + 0.5, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(arr, "RGB").save(p, optimize=True)
        print(f"{p.name}: mean {arr.mean():.1f}")


if __name__ == "__main__":
    main()