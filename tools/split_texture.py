#!/usr/bin/env python3
"""Split a 4K albedo texture into 1K tiles that fit GitHub's 25 MB file limit.

Each output folder contains:
    meta.json      width/height/tile/grid for the merge step
    material.txt   floor material (reflection) parameters
    tile_00.png .. one 1024x1024 tile per grid cell

    python3 tools/split_texture.py --zip ../../Assets/texture/Marble016_4K-PNG.zip --name Marble016
    python3 tools/split_texture.py --zip textures.zip --name green_marble

The Color/albedo PNG is the only map the engine uses, so only that one is
tiled.  The merge step (tools/merge_texture.py) rebuilds the 4K image in CI.
"""
import argparse
import io
import json
import zipfile
from pathlib import Path

from PIL import Image

TILE = 1024

# Floor look per texture family.  Polished stone reflects more sharply; wood is
# rougher and weaker.  Tweak these or edit the generated material.txt.
MATERIALS = {
    "green_marble": dict(f0=0.08, reflect_rough=0.05, gloss_rough=0.32,
                         albedo_gain=2.0, sharpen=0.6, lod_bias=-0.35,
                         vein_gloss=0.35, tile_m=1.4),
    "Marble016": dict(f0=0.08, reflect_rough=0.05, gloss_rough=0.32,
                      albedo_gain=2.0, sharpen=0.6, lod_bias=-0.35,
                      vein_gloss=0.35, tile_m=1.4),
    "Marble019": dict(f0=0.08, reflect_rough=0.045, gloss_rough=0.30,
                      albedo_gain=2.0, sharpen=0.6, lod_bias=-0.35,
                      vein_gloss=0.35, tile_m=1.4),
    "Onyx001": dict(f0=0.09, reflect_rough=0.04, gloss_rough=0.28,
                    albedo_gain=1.8, sharpen=0.6, lod_bias=-0.35,
                    vein_gloss=0.30, tile_m=1.4),
    "Wood030": dict(f0=0.04, reflect_rough=0.18, gloss_rough=0.45,
                    albedo_gain=1.0, sharpen=0.35, lod_bias=-0.25,
                    vein_gloss=0.25, tile_m=1.0),
    "Wood095": dict(f0=0.04, reflect_rough=0.18, gloss_rough=0.45,
                    albedo_gain=1.0, sharpen=0.35, lod_bias=-0.25,
                    vein_gloss=0.25, tile_m=1.0),
}
DEFAULT_MAT = MATERIALS["green_marble"]


def albedo_from_zip(z: zipfile.ZipFile) -> Image.Image:
    for i in z.infolist():
        n = i.filename.lower()
        if n.endswith("_color.png") or (n.endswith(".png") and "color" in n):
            return Image.open(io.BytesIO(z.read(i.filename))).convert("RGB")
    # fall back to the first png
    for i in z.infolist():
        if i.filename.lower().endswith(".png"):
            return Image.open(io.BytesIO(z.read(i.filename))).convert("RGB")
    raise SystemExit("no albedo png in zip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="assets/textures", type=Path)
    a = ap.parse_args()

    with zipfile.ZipFile(a.zip) as z:
        im = albedo_from_zip(z)
    im = im.convert("RGB")
    w, h = im.size
    if w % TILE or h % TILE:
        raise SystemExit(f"{w}x{h} is not a multiple of {TILE}")

    out = a.out / a.name
    out.mkdir(parents=True, exist_ok=True)
    cols, rows = w // TILE, h // TILE
    idx = 0
    for r in range(rows):
        for c in range(cols):
            im.crop((c * TILE, r * TILE, (c + 1) * TILE, (r + 1) * TILE)).save(
                out / f"tile_{idx:02d}.png", optimize=True
            )
            idx += 1

    (out / "meta.json").write_text(json.dumps(
        {"name": a.name, "width": w, "height": h, "tile": TILE,
         "cols": cols, "rows": rows, "count": idx}, indent=2))

    m = MATERIALS.get(a.name, DEFAULT_MAT)
    (out / "material.txt").write_text(
        f"# Floor material for {a.name}\n"
        f"texture       = assets/textures/{a.name}.zip\n"
        f"tile_m        = {m['tile_m']}\n"
        f"albedo_gain   = {m['albedo_gain']}\n"
        f"f0            = {m['f0']}\n"
        f"reflect_rough = {m['reflect_rough']}\n"
        f"gloss_rough   = {m['gloss_rough']}\n"
        f"sharpen       = {m['sharpen']}\n"
        f"lod_bias      = {m['lod_bias']}\n"
        f"vein_gloss    = {m['vein_gloss']}\n"
        f"env_gloss     = 0.0\n"
    )

    total = sum(p.stat().st_size for p in out.glob("tile_*.png"))
    print(f"{a.name}: {w}x{h} -> {idx} tiles ({total/1e6:.1f} MB total) in {out}")


if __name__ == "__main__":
    main()