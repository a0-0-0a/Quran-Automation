#!/usr/bin/env python3
"""Merge the 1K tiles of one texture back into a 4K albedo zip.

    python3 tools/merge_texture.py --dir assets/textures/Marble016

Writes assets/textures/Marble016.zip containing texture.png (the 4K albedo).
The zip is a build artifact (gitignored); the 16 tiles are the committed source.
"""
import argparse
import json
import zipfile
from pathlib import Path

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    meta = json.loads((a.dir / "meta.json").read_text())
    w, h, tile = meta["width"], meta["height"], meta["tile"]
    cols, rows, count = meta["cols"], meta["rows"], meta["count"]

    im = Image.new("RGB", (w, h))
    for idx in range(count):
        r, c = divmod(idx, cols)
        t = Image.open(a.dir / f"tile_{idx:02d}.png").convert("RGB")
        if t.size != (tile, tile):
            raise SystemExit(f"tile {idx} is {t.size}, expected {(tile, tile)}")
        im.paste(t, (c * tile, r * tile))

    out = a.out or a.dir.with_suffix(".zip")
    tmp = a.dir / "_merged.png"
    im.save(tmp)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tmp, "texture.png")
    tmp.unlink()
    print(f"merged {count} tiles -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()