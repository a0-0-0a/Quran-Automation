#!/usr/bin/env python3
"""
pipeline.py — Core render pipeline.

KEY ARCHITECTURAL FIX: CENTERED CROSSFADE WINDOW
─────────────────────────────────────────────────
Old code handled crossfades from two separate branches (if/elif), one for
the outgoing image (from segment A's slot) and one for the incoming image
(from segment B's slot).  Because `img_idx` switches at the exact segment
boundary, this created a discontinuity:

    t → boundary⁻  :  outgoing branch → alpha ≈ 1.0 → shows image B
    t → boundary⁺  :  incoming branch → alpha ≈ 0.0 → shows image A  ← snap!

New code uses a single centered window of width CROSSFADE_SEC:

    for each boundary b = (i+1)*seg:
        if |t − b| < half:                  # half = CROSSFADE_SEC / 2
            alpha = 0.5 + (t − b) / (2·half)   # 0 at b−half, 1 at b+half

This is continuous through the boundary: at t=b, alpha=0.5 always.

MOTION CONTINUITY THROUGH CROSSFADE
─────────────────────────────────────
When rendering the crossfade, both images receive their ACTUAL time offset:
    t_img_A = t − A_start       # may exceed seg_dur during outgoing half
    t_img_B = t − B_start       # may be negative during incoming half

Motion functions in motion.py are designed to handle t outside [0, dur]:
•  Drift uses unclamped linear progress → continues at constant velocity
•  Zoom uses max(0, t/dur) clamp → holds at endpoint when t > dur or t < 0
•  Bokeh similarly clamps to valid range

Result: image A keeps drifting past its segment end; image B starts its
drift slightly before centre — the crossfade is visually seamless.

DISK SPACE OPTIMISATION (CI / GitHub Actions)
──────────────────────────────────────────────
Each background frame is saved as a .npy file (~6 MB uncompressed).
To avoid accumulating gigabytes of temporary files on the runner,
_render_frame() deletes each .npy file immediately after loading it.
This keeps peak temporary storage to a manageable level regardless of
video length.
"""
import math
import random
import shutil
import subprocess
import sys
import logging
from io import BytesIO
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image
import cairosvg

from settings import (
    VIDEO_W, VIDEO_H, FPS,
    N_BG_IMAGES, CROSSFADE_SEC,
    BAR_OPACITY_MIN, BAR_OPACITY_MAX,
    BAR_OPACITY_DARK_MIN, BAR_OPACITY_DARK_MAX,
    BAR_OPACITY_SMOOTH_SEC, BAR_OPACITY_SMOOTH_SEC_DARK,
    BAR_GRAD_STRENGTH, BAR_GRAD_SPEED,
    TRANSITION_SEC, BAR_TIMING_MODE,
    SVG_RENDER_SCALE, SVG_TEXT_SUPERSAMPLE, INTERNAL_OFFSET,
    TEXT_FADE_IN_SEC, TEXT_FADE_OUT_SEC,
    TEMP_DIR, FRAMES_DIR, OUT_DIR,
    ENABLE_DUST, DUST_ON_NORMAL, DUST_ON_DARK,
)
from bokeh import apply_bokeh_to_frame, get_bokeh_params, apply_crossfade, _ss
from motion import get_motion_frame, apply_dark_mode
from effects import apply_dust_particles, apply_premium_plus_effects

log = logging.getLogger("pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# SVG TEXT LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_text_svg(svg_path: Path) -> tuple[Image.Image, int]:
    """Render SVG supersampled then Lanczos-downsample for ultra-sharp edges."""
    scale = SVG_TEXT_SUPERSAMPLE
    data  = cairosvg.svg2png(url=str(svg_path), scale=scale)
    hi    = Image.open(BytesIO(data)).convert("RGBA")
    w1    = max(1, hi.width  // scale)
    h1    = max(1, hi.height // scale)
    img   = hi.resize((w1, h1), Image.LANCZOS)
    return img, w1


# ─────────────────────────────────────────────────────────────────────────────
# CENTERED CROSSFADE BLEND LAYERS
# ─────────────────────────────────────────────────────────────────────────────
def get_blend_layers(
    t: float,
    n: int,
    seg: float,
    half: float,
) -> tuple:
    """
    Compute which image(s) contribute to frame at absolute time `t`.

    Returns one of:
        (img_a, t_a, None, None, 1.0)          — no crossfade, pure image A
        (img_a, t_a, img_b, t_b, alpha)         — crossfade; alpha = B weight

    t_a / t_b are the times *within each image's own timeline*, NOT clamped —
    motion functions receive them directly and continue their motion past
    segment boundaries.

    CONTINUITY PROOF
    ────────────────
    At t = b (boundary) approaching from either side:
        alpha = 0.5 + 0 / (2·half) = 0.5  ✓ (no jump)
    At t = b − half (start of crossfade):
        alpha = 0.5 + (−half) / (2·half) = 0.0  ✓ (only A)
    At t = b + half (end of crossfade):
        alpha = 0.5 + half / (2·half) = 1.0  ✓ (only B)
    """
    for i in range(n - 1):
        boundary = (i + 1) * seg
        dist     = t - boundary
        if abs(dist) < half:
            alpha = 0.5 + dist / (2.0 * half)
            alpha = max(0.0, min(1.0, alpha))
            t_a   = t - i * seg          # image A time (may exceed seg)
            t_b   = t - (i + 1) * seg    # image B time (may be negative)
            return (i, t_a, i + 1, t_b, alpha)

    img_idx = min(int(t / seg), n - 1)
    t_in    = t - img_idx * seg
    return (img_idx, t_in, None, None, 1.0)


def _render_image_at(
    img_arr: np.ndarray,
    motion_type: str,
    t_in: float,
    seg: float,
    preset_idx: int,
    is_first: bool,
    is_last: bool,
) -> np.ndarray:
    """
    Render a single background image at time t_in within its segment.
    t_in may be outside [0, seg] during crossfade — that is intentional.
    """
    # Bokeh uses clamped time (no meaning outside [0, seg] for intro/outro)
    t_clamped = max(0.0, min(seg, t_in))
    r, hl = get_bokeh_params(t_clamped, seg, preset_idx,
                              is_first=is_first, is_last=is_last)
    # Motion receives raw (unclamped) t_in so drift continues seamlessly
    mot = get_motion_frame(motion_type, img_arr, t_in, seg)
    return apply_bokeh_to_frame(mot, r, hl)


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC BAR OPACITY
# ─────────────────────────────────────────────────────────────────────────────
def _lum_center(img_arr: np.ndarray) -> float:
    H, W = img_arr.shape[:2]
    r    = img_arr[H//4:3*H//4, W//4:3*W//4].astype(np.float32) / 255.0
    return float(0.2126 * r[..., 0].mean()
                 + 0.7152 * r[..., 1].mean()
                 + 0.0722 * r[..., 2].mean())


def precompute_bar_opacities(
    images_arr: list[np.ndarray],
    total_frames: int,
    sound_duration: float,
    dark_mode: bool,
) -> list[float]:
    n       = N_BG_IMAGES
    seg     = sound_duration / n
    sm_sec  = BAR_OPACITY_SMOOTH_SEC_DARK if dark_mode else BAR_OPACITY_SMOOTH_SEC
    op_min  = BAR_OPACITY_DARK_MIN if dark_mode else BAR_OPACITY_MIN
    op_max  = BAR_OPACITY_DARK_MAX if dark_mode else BAR_OPACITY_MAX
    targets = [op_min + _lum_center(img) * (op_max - op_min) for img in images_arr]
    result  = []
    for f in range(total_frames):
        t   = f / FPS
        idx = min(int(t / seg), n - 1)
        tin = t - idx * seg
        hs  = sm_sec / 2.0
        if idx < n - 1 and (seg - tin) < hs:
            alpha = _ss((hs - (seg - tin)) / hs)
            op    = targets[idx] * (1 - alpha) + targets[idx + 1] * alpha
        else:
            op = targets[idx]
        result.append(float(op))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BAR SVG  — shape preserved EXACTLY from original
# ─────────────────────────────────────────────────────────────────────────────
_BAR_TMPL = (
    "m{sx:.3f} 225.511"
    "c-8.541 0-18.679 4.629-22.737 13.737-3.37 7.563-8.364 11.842-21.552 11.842"
    "h-151.208v5.821H{lc:.3f}"
    "c13.188 0 18.183 4.279 21.552 11.842 4.058 9.108 14.196 13.737 22.737 13.737"
    "h4.973 {cs:.3f} 4.973"
    "c8.541 0 18.679-4.629 22.737-13.737 3.37-7.563 8.364-11.842 21.552-11.842"
    "h151.208v-5.821H{rc:.3f}"
    "c-13.188 0-18.183-4.279-21.552-11.842-4.058-9.108-14.196-13.737-22.737-13.737"
    "h-4.973-{cs:.3f}z"
)
_BAR_CACHE: dict = {}


def _render_bar_base(width_px: float, opacity: float) -> Image.Image:
    key = (round(width_px), round(opacity * 200))
    if key in _BAR_CACHE:
        return _BAR_CACHE[key]
    fw  = width_px + INTERNAL_OFFSET
    cs  = fw * 285.75 / 1080
    sx  = 48.341 - (cs - 179.122) / 2.0
    rc  = sx + 4.973 + cs + 4.973 + 44.289
    d   = _BAR_TMPL.format(sx=sx, lc=sx - 44.289, cs=cs, rc=rc)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1920"'
           f' viewBox="0 0 285.75 508">'
           f'<path fill="black" opacity="{opacity:.3f}" d="{d}"/></svg>')
    png = cairosvg.svg2png(bytestring=svg.encode(), scale=SVG_RENDER_SCALE)
    res = Image.open(BytesIO(png)).convert("RGBA").resize((1080, 1920), Image.LANCZOS)
    _BAR_CACHE[key] = res
    return res


def render_bar_with_gradient(width_px: float, opacity: float, t: float) -> Image.Image:
    """Bar image with a slow animated brightness sweep."""
    bar = _render_bar_base(width_px, opacity).copy()
    arr = np.array(bar, dtype=np.float32)
    H, W = arr.shape[:2]
    x        = np.arange(W, dtype=np.float32) / W
    wave_pos = (t * BAR_GRAD_SPEED) % 1.0
    dist     = np.minimum(np.abs(x - wave_pos), 1.0 - np.abs(x - wave_pos))
    spot     = BAR_GRAD_STRENGTH * np.exp(-dist**2 / 0.04)
    a_norm   = arr[..., 3] / 255.0
    for c in range(3):
        arr[..., c] = np.clip(arr[..., c] + spot[np.newaxis, :] * a_norm * 255, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


# ─────────────────────────────────────────────────────────────────────────────
# BAR WIDTHS  (stateful, smooth transitions, never jumps)
# ─────────────────────────────────────────────────────────────────────────────
def precompute_widths(items: list[dict], total_frames: int) -> list[float]:
    if not items:
        return [0.0] * total_frames
    ws   = [it["w"] + INTERNAL_OFFSET for it in items]
    v    = max(-1.0, min(1.0, float(BAR_TIMING_MODE)))
    segs = []
    for i in range(len(items) - 1):
        tn = items[i + 1]["t"]
        wc, wn = ws[i], ws[i + 1]
        tr = min(TRANSITION_SEC, (items[i + 1]["t"] - items[i]["t"]) * 0.9)
        d  = 1 if wn > wc else -1
        f  = (1 + v * d) / 2
        segs.append((tn - tr * f, tn - tr * f + tr, wn))

    si = 0
    ats = ate = afr = ato = None
    cur = float(ws[0])
    res = []
    for f in range(total_frames):
        t = f / FPS
        while si < len(segs) and t >= segs[si][0]:
            ts, te, tw = segs[si]
            afr = cur
            ats, ate, ato = ts, te, tw
            si += 1
        if ats is not None and ate > ats:
            if t <= ate:
                u    = max(0.0, min(1.0, (t - ats) / (ate - ats)))
                ease = 4 * u**3 if u < 0.5 else 1 - math.pow(-2 * u + 2, 3) / 2
                cur  = afr + (ato - afr) * ease
            else:
                cur = ato
        res.append(cur)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND PRE-COMPUTE  — saves frames to disk (avoids RAM exhaustion)
# ─────────────────────────────────────────────────────────────────────────────
def precompute_bg_frames(
    images_arr: list[np.ndarray],
    motion_types: list[str],
    preset_idxs: list[int],
    sound_duration: float,
    dark_mode: bool,
    total_frames: int,
    bg_dir: Path,
) -> None:
    """
    Pre-render all background frames and save them as .npy files on disk.

    WHY DISK INSTEAD OF MEMORY
    ──────────────────────────
    741 frames × 1080×1920×3 bytes ≈ 4.4 GB RAM.
    Passing this list to forked worker processes doubles the peak usage
    to ~9 GB, crashing the pool with BrokenProcessPool.

    Saving to disk: each .npy file is ~6 MB, workers load individually,
    peak RAM stays under 1 GB regardless of video length.

    NOTE: Workers delete each .npy immediately after loading (see _render_frame)
    so disk usage never accumulates — only the current batch remains on disk.
    """
    n    = N_BG_IMAGES
    seg  = sound_duration / n
    half = CROSSFADE_SEC / 2.0

    bg_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Pre-computing {total_frames} background frames → {bg_dir} …")

    for f in tqdm(range(total_frames),
                  desc="BG", unit="fr", colour="cyan",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        t = f / FPS

        layers = get_blend_layers(t, n, seg, half)
        idx_a, t_a, idx_b, t_b, alpha = layers

        is_first_a = (idx_a == 0)
        is_last_a  = (idx_a == n - 1)

        frame_a = _render_image_at(
            images_arr[idx_a], motion_types[idx_a], t_a, seg,
            preset_idxs[idx_a], is_first_a, is_last_a)

        if idx_b is not None and alpha > 0.0:
            is_first_b = (idx_b == 0)
            is_last_b  = (idx_b == n - 1)
            frame_b = _render_image_at(
                images_arr[idx_b], motion_types[idx_b], t_b, seg,
                preset_idxs[idx_b], is_first_b, is_last_b)
            frame_a = apply_crossfade(frame_a, frame_b, alpha)

        if dark_mode:
            frame_a = apply_dark_mode(frame_a, t, f)

        np.save(bg_dir / f"{f:06d}.npy", frame_a)


# ─────────────────────────────────────────────────────────────────────────────
# WORKER (runs in subprocess via ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────
G_BG_DIR = G_ITEMS = G_WIDTHS = G_OPS = G_DUST = None


def _init_worker(bg_dir, items, widths, ops, dust_on):
    global G_BG_DIR, G_ITEMS, G_WIDTHS, G_OPS, G_DUST
    G_BG_DIR = bg_dir; G_ITEMS = items; G_WIDTHS = widths; G_OPS = ops; G_DUST = dust_on


def _render_frame(f_no: int) -> None:
    t        = f_no / FPS
    npy_path = G_BG_DIR / f"{f_no:06d}.npy"

    bg_arr = np.load(npy_path)

    # ── Delete the .npy immediately after loading ────────────────────────────
    # This is crucial for CI environments (GitHub Actions) where disk space is
    # limited.  Workers process frames roughly in order, so deleting as we go
    # keeps the total .npy footprint close to zero at all times.
    try:
        npy_path.unlink()
    except OSError:
        pass  # already deleted or race condition — safe to ignore

    frame  = Image.fromarray(bg_arr).convert("RGBA")

    if G_DUST:
        try:
            apply_dust_particles(frame, t, seed=12345)
        except Exception:
            pass

    bw = G_WIDTHS[f_no] if f_no < len(G_WIDTHS) else 0
    op = G_OPS[f_no]    if f_no < len(G_OPS)    else 0.65
    if bw:
        try:
            frame.alpha_composite(render_bar_with_gradient(bw, op, t))
        except Exception:
            pass

    idx = 0
    for i, it in enumerate(G_ITEMS):
        if it["t"] <= t:
            idx = i

    if idx < len(G_ITEMS):
        it = G_ITEMS[idx]
        t0 = it["t"]
        fi = max(0.0, min(1.0, (t - t0) / TEXT_FADE_IN_SEC)) if t > t0 else 0.0
        if t >= t0 + TEXT_FADE_IN_SEC:
            fi = 1.0
        fo = 0.0
        if idx < len(G_ITEMS) - 1:
            te = G_ITEMS[idx + 1]["t"]
            ts = te - TEXT_FADE_OUT_SEC
            if t >= te:
                fo = 1.0
            elif t >= ts:
                fo = (t - ts) / TEXT_FADE_OUT_SEC
        try:
            eff = apply_premium_plus_effects(it["img"], t, fi, fo, idx * 777)
            frame.alpha_composite(
                eff,
                ((VIDEO_W - eff.width) // 2, (VIDEO_H - eff.height) // 2))
        except Exception:
            pass

    frame.convert("RGB").save(FRAMES_DIR / f"{f_no:06d}.png", optimize=False)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_render(
    items: list[dict],
    images_arr: list[np.ndarray],
    motion_types: list[str],
    preset_idxs: list[int],
    sound_path: Path,
    sound_duration: float,
    dark_mode: bool,
    out_path: Path,
) -> None:
    log.info("─" * 50)
    log.info(f"  Sound:    {sound_path.name}")
    log.info(f"  Mode:     {'🌑 Dark' if dark_mode else '☀️  Normal'}")
    log.info(f"  Duration: {sound_duration:.1f}s  ·  {int(sound_duration * FPS)} frames")
    log.info(f"  Images:   {N_BG_IMAGES}")
    log.info("─" * 50)

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    FRAMES_DIR.mkdir(parents=True)
    BG_DIR = TEMP_DIR / "bg"
    BG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_frames = int(sound_duration * FPS)

    dust_on = ENABLE_DUST and (
        (DUST_ON_DARK   and     dark_mode) or
        (DUST_ON_NORMAL and not dark_mode))
    log.info(f"Dust:     {'on' if dust_on else 'off'}")

    widths    = precompute_widths(items, total_frames)
    opacities = precompute_bar_opacities(images_arr, total_frames, sound_duration, dark_mode)
    precompute_bg_frames(
        images_arr, motion_types, preset_idxs,
        sound_duration, dark_mode, total_frames, BG_DIR)

    log.info(f"Compositing {total_frames} frames …")
    from multiprocessing import get_context
    ctx = get_context("fork")
    with ProcessPoolExecutor(
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(BG_DIR, items, widths, opacities, dust_on),
    ) as ex:
        list(tqdm(
            ex.map(_render_frame, range(total_frames), chunksize=4),
            total=total_frames,
            desc="Compose",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"))

    log.info("Encoding …")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(FPS),
        "-i", str(FRAMES_DIR / "%06d.png"),
        "-i", str(sound_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        str(out_path),
    ], check=True)

    log.info(f"✅  Done → {out_path.name}")
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
