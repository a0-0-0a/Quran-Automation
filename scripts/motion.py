#!/usr/bin/env python3
"""
motion.py — Sub-pixel accurate background motion engine + dark-mode.

KEY ARCHITECTURAL FIXES vs. old bg_motion.py
─────────────────────────────────────────────
1.  DRIFT BUG FIX — "drift stops before the next image appears"

    Root cause: the old travel formula was:
        travel = (slack_px / W) * 0.90
    But the valid cx range is [src_w/(2W), 1 - src_w/(2W)], so the
    maximum offset from the centre is (W - src_w) / (2·W).
    The old formula used twice that limit, so cx hit the affine clamp
    (max(0, min(W-src_w, x0))) at roughly p=0.45 and then STAYED
    clamped for the rest of the segment — visually frozen.

    Fix:
        max_offset = (W - src_w) / (2.0 * W)   # true maximum
        travel     = max_offset * 0.88           # 88 % of range (safety margin)

2.  CROSSFADE CONTINUITY — motion functions accept t outside [0, dur]

    During the crossfade window the pipeline renders the outgoing image
    at t slightly beyond its nominal segment end, and the incoming image
    at t slightly before its nominal segment start (negative).

    All motion functions handle this gracefully:
    •  _p() is NOT clamped here — the pipeline uses a centered crossfade
       window so values outside [0,1] only occur during the short ±half
       crossfade overlap, and the affine clamp provides the final safety net.
    •  Drift uses raw p (unclamped) so it continues drifting through the
       crossfade at constant velocity — no dead frames, no pauses.

Sub-pixel accuracy
──────────────────
PIL Image.transform(AFFINE, BICUBIC) operates at sub-pixel precision,
so every frame differs from the previous one by a fraction of a pixel,
giving perfectly smooth motion at any FPS.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image

from settings import (
    VIDEO_W, VIDEO_H, FPS,
    MOTION_ZOOM_AMOUNT, MOTION_DRIFT_AMOUNT,
    DARK_VIGNETTE_STRENGTH, DARK_SATURATION, DARK_BRIGHTNESS,
    DARK_FLICKER_BASE_SPEED, DARK_FLICKER_BASE_AMP,
    DARK_FLICKER_MID_SPEED,  DARK_FLICKER_MID_AMP,
    DARK_FLICKER_FRAME_AMP,
    DARK_FLICKER_DIP_CHANCE, DARK_FLICKER_DIP_AMP, DARK_FLICKER_DIP_DECAY,
)

# ─────────────────────────────────────────────────────────────────────────────
# VIGNETTE CACHE
# ─────────────────────────────────────────────────────────────────────────────
_VIG_CACHE: dict = {}

def _make_vignette(w: int, h: int) -> np.ndarray:
    if (w, h) in _VIG_CACHE:
        return _VIG_CACHE[(w, h)]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / w - 0.5) * 2
    ny = (yy / h - 0.5) * 2
    r  = np.sqrt(nx**2 * 0.78 + ny**2 * 1.20)
    m  = np.power(np.clip(1.0 - np.clip(r, 0, 1)**2 * (3 - 2 * np.clip(r, 0, 1)), 0, 1), 0.80)
    _VIG_CACHE[(w, h)] = m.astype(np.float32)
    return _VIG_CACHE[(w, h)]


# ─────────────────────────────────────────────────────────────────────────────
# CORE: SUB-PIXEL AFFINE WARP
# ─────────────────────────────────────────────────────────────────────────────
def _affine_frame(
    img_arr: np.ndarray,
    cx: float,
    cy: float,
    zoom: float,
    ow: int = VIDEO_W,
    oh: int = VIDEO_H,
) -> np.ndarray:
    """
    Extract a region centred at (cx, cy) [normalised 0–1] with the given zoom
    factor and resize to (ow × oh) using PIL AFFINE + BICUBIC — fully sub-pixel.

    PIL AFFINE data = (a, b, c, d, e, f) maps DST pixel (u,v) → SRC pixel (x,y):
        x = a*u + b*v + c
        y = d*u + e*v + f
    For pure scale+translate (no rotation): a=e=sc, b=d=0, c=x0, f=y0.
    """
    H, W = img_arr.shape[:2]

    base        = max(ow / W, oh / H)   # base scale to fill the frame
    src_per_dst = base * zoom            # total source pixels per output pixel

    src_w = ow / src_per_dst            # source region width
    src_h = oh / src_per_dst            # source region height

    # Centre position clamped so the region stays inside the source image
    x0 = cx * W - src_w / 2
    y0 = cy * H - src_h / 2
    x0 = max(0.0, min(float(W) - src_w, x0))
    y0 = max(0.0, min(float(H) - src_h, y0))

    sc = 1.0 / src_per_dst             # output pixel → source pixel scale

    pil_img = Image.fromarray(img_arr, "RGB")
    out     = pil_img.transform(
        (ow, oh),
        Image.AFFINE,
        (sc, 0, x0, 0, sc, y0),
        resample=Image.BICUBIC,
    )
    return np.array(out, dtype=np.uint8)


def _no_motion(img_arr: np.ndarray) -> np.ndarray:
    """Full-frame: uses the entire source image."""
    pil = Image.fromarray(img_arr, "RGB")
    return np.array(pil.resize((VIDEO_W, VIDEO_H), Image.LANCZOS), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# MOTION TYPES
# ─────────────────────────────────────────────────────────────────────────────

def _zoom_needed_for_drift(W: int, base: float) -> float:
    """
    Compute minimum zoom so that MOTION_DRIFT_AMOUNT of travel is physically
    possible without cx hitting the affine clamp.

    Constraint:  max_offset = (W - src_w) / (2·W)  ≥  MOTION_DRIFT_AMOUNT / 0.88
    → src_w ≤ W · (1 − 2 · MOTION_DRIFT_AMOUNT / 0.88)
    → zoom ≥ VIDEO_W / (base · W · (1 − 2 · MOTION_DRIFT_AMOUNT / 0.88))
    """
    min_slack_fraction = MOTION_DRIFT_AMOUNT * 2.0 / 0.88   # ≈ 0.091
    denominator = max(1.0 - min_slack_fraction, 0.05)
    return max(VIDEO_W / (base * W * denominator), 1.0)


def make_zoom_in(img_arr: np.ndarray, t: float, dur: float) -> np.ndarray:
    """Slow zoom-in over the segment duration. t may exceed dur slightly."""
    p = t / max(dur, 1e-9)
    return _affine_frame(img_arr, 0.5, 0.5, 1.0 + MOTION_ZOOM_AMOUNT * max(0.0, p))


def make_zoom_out(img_arr: np.ndarray, t: float, dur: float) -> np.ndarray:
    """Slow zoom-out over the segment duration. t may be negative or exceed dur."""
    p = t / max(dur, 1e-9)
    return _affine_frame(img_arr, 0.5, 0.5,
                         (1.0 + MOTION_ZOOM_AMOUNT) - MOTION_ZOOM_AMOUNT * max(0.0, p))


def _ease_drift(p: float) -> float:
    """
    Gentle ease-in-out for drift.  65% linear + 35% sine-ease.

    Why this blend:
    • Pure sine-ease (smoothstep) has velocity=0 at both ends — too heavy,
      the image appears to hang at the start and stop.
    • Pure linear snaps velocity — jarring when drift-right meets drift-left.
    • 65/35 blend feels mostly constant-speed but with a soft ramp so
      consecutive drifts in opposite directions cross-fade smoothly.

    Derivative (velocity) at the boundaries:
        pure linear      →  1.00  (no ease at all)
        pure sine-ease   →  0.00  (full stop)
        this blend       →  0.65  (gently slowed, still clearly moving)

    Outside [0, 1] (during crossfade overlap) the function extrapolates
    at constant slope 0.65 so the image keeps drifting through the blend
    without freezing or reversing.
    """
    if p <= 0.0:
        return p * 0.65          # linear extrapolation at boundary slope
    if p >= 1.0:
        return 1.0 + (p - 1.0) * 0.65
    sine_ease = (1.0 - math.cos(p * math.pi)) / 2.0
    return 0.65 * p + 0.35 * sine_ease


def make_drift(img_arr: np.ndarray, t: float, dur: float, direction: int) -> np.ndarray:
    """
    Horizontal drift with gentle ease-in-out.

    The drift feels mostly constant-speed (no heavy arc) but starts and
    ends with a soft velocity ramp.  When drift-right is followed by
    drift-left both images slow gently toward the crossfade boundary,
    making the direction change look natural instead of jarring.

    t is not clamped — _ease_drift() extrapolates linearly outside [0,1]
    so motion continues smoothly through the crossfade window.
    """
    H, W = img_arr.shape[:2]
    base = max(VIDEO_W / W, VIDEO_H / H)

    zoom = _zoom_needed_for_drift(W, base)

    src_w      = VIDEO_W / (base * zoom)
    max_offset = max(0.0, W - src_w) / (2.0 * W)
    travel     = max_offset * 0.88

    p  = t / max(dur, 1e-9)
    cx = 0.5 + direction * travel * _ease_drift(p)

    return _affine_frame(img_arr, cx, 0.5, zoom)


def get_motion_frame(
    motion_type: str,
    img_arr: np.ndarray,
    t: float,
    duration: float,
) -> np.ndarray:
    """
    Dispatch to the appropriate motion function.
    t can be outside [0, duration] during crossfades — all functions handle it.
    """
    try:
        if motion_type == "zoom_in":    return make_zoom_in(img_arr, t, duration)
        if motion_type == "zoom_out":   return make_zoom_out(img_arr, t, duration)
        if motion_type == "drift_left":  return make_drift(img_arr, t, duration, -1)
        if motion_type == "drift_right": return make_drift(img_arr, t, duration, +1)
        return _no_motion(img_arr)
    except Exception:
        return _no_motion(img_arr)


# ─────────────────────────────────────────────────────────────────────────────
# PRO FILM FLICKER
# ─────────────────────────────────────────────────────────────────────────────
def pro_film_flicker(t: float, frame_seed: int = 0) -> float:
    """
    Multi-frequency brightness oscillation simulating analogue film projector
    flicker.  Returns a multiplier; 1.0 = normal brightness.
    """
    slow = DARK_FLICKER_BASE_AMP * math.sin(t * DARK_FLICKER_BASE_SPEED * 2 * math.pi)
    mid  = (DARK_FLICKER_MID_AMP
            * math.sin(t * DARK_FLICKER_MID_SPEED * 2 * math.pi + 1.731)
            * math.cos(t * DARK_FLICKER_MID_SPEED * 0.43 * 2 * math.pi + 0.42))

    frng   = random.Random(int(t * FPS) * 104723 + frame_seed)
    jitter = frng.gauss(0.0, DARK_FLICKER_FRAME_AMP)

    dip    = 0.0
    dip_iv = 1.0 / max(DARK_FLICKER_DIP_CHANCE, 0.01)
    for k in range(4):
        ei = int(t / dip_iv) - k
        if ei < 0:
            continue
        erng = random.Random(ei * 77777 + frame_seed * 3 + 11)
        if erng.random() > 0.65:
            continue
        t_ev = ei * dip_iv + erng.random() * dip_iv
        dt   = t - t_ev
        if 0.0 <= dt < 0.6:
            dip += (-erng.uniform(0.5, 1.0) * DARK_FLICKER_DIP_AMP
                    * math.exp(-dt * DARK_FLICKER_DIP_DECAY))

    return max(0.62, min(1.30, 1.0 + slow + mid + jitter + dip))


# ─────────────────────────────────────────────────────────────────────────────
# DARK MODE
# ─────────────────────────────────────────────────────────────────────────────
def apply_dark_mode(frame_uint8: np.ndarray, t: float, frame_seed: int = 0) -> np.ndarray:
    """
    Apply cinematic dark-mode to a frame:
    desaturate → dim → film flicker → radial vignette.
    """
    arr  = frame_uint8.astype(np.float32) / 255.0
    gray = (0.2126 * arr[..., 0]
            + 0.7152 * arr[..., 1]
            + 0.0722 * arr[..., 2])[..., None]
    arr  = arr * DARK_SATURATION + gray * (1.0 - DARK_SATURATION)
    arr *= DARK_BRIGHTNESS

    flicker = pro_film_flicker(t, frame_seed)
    arr    *= flicker

    vig = _make_vignette(arr.shape[1], arr.shape[0])[..., None]
    rs  = max(0.0, min(1.0, DARK_VIGNETTE_STRENGTH * (1.0 + 0.08 * (1.0 - flicker))))
    arr = arr * (rs + (1.0 - rs) * vig)

    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
