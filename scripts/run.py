#!/usr/bin/env python3
"""
run.py — Entry point.

Usage
─────
    python scripts/run.py                  # random sound + random dark/normal
    python scripts/run.py --dark           # force dark mode
    python scripts/run.py --no-dark        # force normal mode
    python scripts/run.py --sound 4        # specific sound by base number
    python scripts/run.py --sound 4 --dark

Sound selection
───────────────
Sound files named with a trailing 'x' or 'X' (e.g. "4x.wav") default to
dark mode when --dark / --no-dark are not specified.

Info folder matching
────────────────────
For sound "4.wav" or "4x.wav", the info folder is "info/4" or "info/4x".
The loader tries both variants automatically.

CI / headless usage
───────────────────
•  No interactive steps — all inputs are file-based.
•  No GUI or display required.
•  Set PYTHONPATH or run from the project root.
•  All outputs go to output/.
•  After generating, writes output/last_video.txt with the full path
   so the upload step can find the video without globbing.
"""
import argparse
import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image

from settings import (
    SOUNDS_DIR, IMAGES_DIR, INFO_DIR, OUT_DIR,
    N_BG_IMAGES, MOTION_NONE_CHANCE,
)
from pipeline import load_text_svg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run")

AUDIO_EXTS = {".wav", ".mp3", ".aac", ".m4a", ".ogg", ".flac"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# SOUND / PATH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _sounds() -> list[Path]:
    s = [p for p in SOUNDS_DIR.iterdir()
         if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    if not s:
        log.error(f"No audio files found in {SOUNDS_DIR}")
        sys.exit(1)
    return s


def _duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(path)],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr.strip()}")
    for s in json.loads(r.stdout).get("streams", []):
        if "duration" in s:
            return float(s["duration"])
    raise RuntimeError(f"No duration found in {path}")


def _base_num(p: Path) -> str:
    return p.stem.rstrip("xX")


def _is_dark(p: Path) -> bool:
    return p.stem.upper().endswith("X")


def _info_folder(num: str) -> Path | None:
    for candidate in [INFO_DIR / num, INFO_DIR / (num + "x"), INFO_DIR / (num + "X")]:
        if candidate.is_dir():
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ITEM LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_items(info_folder: Path) -> list[dict]:
    """
    Load text items from an info folder.
    Expected layout:
        info/<n>/1.txt        — timing file: lines of  "index: timestamp"
        info/<n>/<index>.svg  — SVG text image (preferred)
        info/<n>/<index>.png  — PNG fallback
    """
    txt = info_folder / "1.txt"
    if not txt.exists():
        raise FileNotFoundError(f"Missing timing file: {txt}")

    all_files = sorted(p.name for p in info_folder.iterdir() if p.is_file())
    log.info(f"Info folder: {info_folder.name}  ({len(all_files)} files)")

    items   = []
    skipped = []

    with open(txt, encoding="utf-8") as f:
        lines = [l for l in f.readlines() if l.strip() and ":" in l]
    log.info(f"  Timing entries: {len(lines)}")

    for raw in lines:
        left, right = raw.strip().split(":", 1)
        idx = "".join(c for c in left.strip() if c.isalnum() or c in "_-")
        if not idx:
            log.warning(f"  ⚠ Malformed index: {raw.strip()!r}")
            skipped.append(raw.strip())
            continue
        try:
            ts = float(right.strip())
        except ValueError:
            log.warning(f"  ⚠ Bad timestamp: {raw.strip()!r}")
            skipped.append(raw.strip())
            continue

        svg_p = info_folder / f"{idx}.svg"
        png_p = info_folder / f"{idx}.png"

        if svg_p.exists():
            try:
                img, w = load_text_svg(svg_p)
            except Exception as e:
                log.error(f"  ✗ SVG render failed [{idx}]: {e}")
                skipped.append(raw.strip())
                continue
        elif png_p.exists():
            raw_img = Image.open(png_p).convert("RGBA")
            bbox    = raw_img.getchannel("A").getbbox()
            img     = raw_img.crop(bbox) if bbox else raw_img
            w       = img.width
        else:
            similar = [f for f in all_files if Path(f).stem == idx]
            hint    = f" (found: {similar})" if similar else ""
            log.warning(f"  ⚠ No file for [{idx}]{hint}")
            skipped.append(raw.strip())
            continue

        bbox = img.getchannel("A").getbbox()
        if bbox is None:
            log.warning(f"  ⚠ Empty alpha [{idx}]")
            skipped.append(raw.strip())
            continue
        img = img.crop(bbox)
        items.append({"t": ts, "img": img, "w": bbox[2] - bbox[0]})

    if skipped:
        log.warning(f"  Skipped {len(skipped)}: {skipped}")
    if not items:
        raise ValueError(
            f"No items loaded from {info_folder}\n"
            f"  Files: {all_files}\n"
            f"  Ensure SVG/PNG filenames match indices in 1.txt.")
    log.info(f"  Loaded: {len(items)} text item(s)  ✓")
    return sorted(items, key=lambda x: x["t"])


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE / MOTION / PRESET SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def _pick_images(n: int = N_BG_IMAGES) -> list[Path]:
    pool = [p for p in IMAGES_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not pool:
        log.error(f"No images in {IMAGES_DIR}")
        sys.exit(1)
    while len(pool) < n:
        pool = pool + pool
    return random.sample(pool, n)


def _load_img(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"), dtype=np.uint8)


MOTION_CHOICES = ["zoom_in", "zoom_out", "drift_left", "drift_right", "none"]


def _is_drift(m: str) -> bool:
    return m in ("drift_left", "drift_right")


def _pick_motions(n: int = N_BG_IMAGES) -> list[str]:
    """
    Pick n motion types with two hard rules:
    1. 'none' appears at most ceil(n * MOTION_NONE_CHANCE) times.
    2. No two consecutive drift effects (drift_left / drift_right).
    """
    max_none  = max(0, round(n * MOTION_NONE_CHANCE + 0.5))
    non_drift = [m for m in MOTION_CHOICES if not _is_drift(m)]
    w_base    = [1.0, 1.0, 1.0, 1.0, max(0.01, MOTION_NONE_CHANCE * 4)]
    wt        = sum(w_base)
    w_base    = [x / wt for x in w_base]

    for _ in range(400):
        result = []
        for i in range(n):
            if i > 0 and _is_drift(result[-1]):
                choice = random.choice(non_drift)
            else:
                choice = random.choices(MOTION_CHOICES, weights=w_base, k=1)[0]
            result.append(choice)

        if result.count("none") <= max_none:
            return result

    for i in range(n):
        if result.count("none") <= max_none:
            break
        if result[i] == "none":
            prev_is_drift = i > 0 and _is_drift(result[i - 1])
            next_is_drift = i < n - 1 and _is_drift(result[i + 1])
            pool = [m for m in MOTION_CHOICES
                    if m != "none"
                    and not (prev_is_drift and _is_drift(m))
                    and not (next_is_drift and _is_drift(m))]
            result[i] = random.choice(pool or non_drift)
    return result


def _pick_presets(n: int) -> list[int]:
    from bokeh import FOCUS_PRESETS
    return [random.randint(0, len(FOCUS_PRESETS) - 1) for _ in range(n)]


def _dark_mode(sound_path: Path, force: bool | None) -> bool:
    if force is not None:
        return force
    if _is_dark(sound_path):
        log.info("  Dark mode: forced by 'x' suffix")
        return True
    return random.random() < 0.50


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP CHECKS
# ─────────────────────────────────────────────────────────────────────────────
def _startup() -> None:
    errs = []
    for d, label in [
        (SOUNDS_DIR, "sounds"),
        (IMAGES_DIR, "images"),
        (INFO_DIR,   "info"),
    ]:
        if not d.is_dir():
            errs.append(f"Missing directory: {d}  ({label})")
    for cmd in ("ffmpeg", "ffprobe"):
        if subprocess.run([cmd, "-version"], capture_output=True).returncode != 0:
            errs.append(f"'{cmd}' not found on PATH")
    if errs:
        for e in errs:
            log.error(f"✗ {e}")
        sys.exit(1)
    log.info("Startup checks ✓")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated video creation pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dark",    action="store_true",  help="Force dark mode")
    parser.add_argument("--no-dark", action="store_true",  help="Force normal mode")
    parser.add_argument("--sound",   type=str, default=None,
                        help="Sound base number (e.g. 4 for '4.wav' or '4x.wav')")
    args = parser.parse_args()
    t0   = time.time()

    _startup()

    sounds = _sounds()

    # ── Sound selection ────────────────────────────────────────────────────
    if args.sound:
        matches = [s for s in sounds if _base_num(s) == args.sound.rstrip("xX")]
        if not matches:
            log.error(f"Sound '{args.sound}' not found in {SOUNDS_DIR}")
            sys.exit(1)
        sound_path = matches[0]
    else:
        sound_path = random.choice(sounds)

    force_dark = True if args.dark else (False if args.no_dark else None)
    dark_mode  = _dark_mode(sound_path, force_dark)
    log.info(f"Sound: {sound_path.name}  ·  {'🌑 Dark' if dark_mode else '☀️  Normal'}")

    # ── Info folder search with retry ─────────────────────────────────────
    base     = _base_num(sound_path)
    info_dir = _info_folder(base)
    tried    = {sound_path}
    MAX_RETRY = 15

    for attempt in range(MAX_RETRY):
        if info_dir is not None:
            break
        log.warning(f"  Info folder '{base}' not found — retry {attempt + 1}/{MAX_RETRY}")
        remaining = [s for s in sounds if s not in tried]
        if not remaining:
            log.error("All sounds exhausted without a matching info folder")
            sys.exit(1)
        sound_path = random.choice(remaining)
        tried.add(sound_path)
        base       = _base_num(sound_path)
        dark_mode  = _dark_mode(sound_path, force_dark)
        info_dir   = _info_folder(base)
        log.info(f"  Trying: {sound_path.name}")
    else:
        log.error(f"No matching info folder after {MAX_RETRY} attempts")
        sys.exit(1)

    # ── Load all assets ───────────────────────────────────────────────────
    items      = load_items(info_dir)
    duration   = _duration(sound_path)
    log.info(f"Duration: {duration:.1f}s")

    img_paths  = _pick_images()
    images_arr = [_load_img(p) for p in img_paths]
    motions    = _pick_motions()
    presets    = _pick_presets(N_BG_IMAGES)

    log.info(f"Images:  {[p.name for p in img_paths]}")
    log.info(f"Motions: {motions}")
    log.info(f"Presets: {presets}")

    # ── Output path ───────────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "dark" if dark_mode else "normal"
    out = OUT_DIR / f"{sound_path.stem}_{tag}_{ts}.mp4"

    # ── Render ────────────────────────────────────────────────────────────
    from pipeline import run_render
    run_render(
        items=items,
        images_arr=images_arr,
        motion_types=motions,
        preset_idxs=presets,
        sound_path=sound_path,
        sound_duration=duration,
        dark_mode=dark_mode,
        out_path=out,
    )

    # ── Write output path for upload step ─────────────────────────────────
    # upload_youtube.py reads this file to find the generated video.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    marker = OUT_DIR / "last_video.txt"
    marker.write_text(str(out), encoding="utf-8")
    log.info(f"Output path written to {marker.relative_to(OUT_DIR.parent)}")

    log.info(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
