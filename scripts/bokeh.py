#!/usr/bin/env python3
"""
bokeh.py — Fast cinematic bokeh / depth-of-field renderer.

Unchanged from the original bokeh_bg.py — this system is working perfectly.
Renamed for architectural consistency.

Speed architecture
──────────────────
• Working resolution 360×640 (¼ of full pixel count → ~4× faster FFT)
• PIL GaussianBlur for the base layer (separable, fast)
• FFT convolution only for bokeh CIRCLE layers (ring shape needs FFT)
• Kernels are cached across calls (same radius → reused)
• Color preserved: no gamma conversion

Timing
──────
• First image: skip intro → starts sharp immediately
• Last  image: skip outro → holds sharp until segment ends
• All others: intro blur → cinematic focus-pull → sharp hold → outro blur

Crossfade
─────────
• Symmetric exposure-bloom cross-dissolve
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image, ImageFilter

from settings import (
    VIDEO_W, VIDEO_H,
    BOKEH_INTRO_SEC, BOKEH_FOCUS_SEC, BOKEH_OUTRO_SEC,
    MAX_BOKEH_RADIUS, BOKEH_BLEND_RADIUS,
    BOKEH_W, BOKEH_H,
    BOKEH_LOW_THR, BOKEH_HIGH_THR,
)

# ─────────────────────────────────────────────────────────────────────────────
# 10 CINEMATIC FOCUS PRESETS
# t_norm 0→1 = BOKEH_FOCUS_SEC seconds
# radius 1.0 = MAX_BOKEH_RADIUS, 0.0 = sharp
# ─────────────────────────────────────────────────────────────────────────────
FOCUS_PRESETS = [
    # 0 ─ Smooth Pro
    {"r": {0.00:1.00, 0.22:0.76, 0.48:0.34, 0.74:0.05, 1.00:0.00},
     "h": {0.00:0.88, 0.32:0.62, 0.64:0.18, 1.00:0.00}},
    # 1 ─ Quick Snap + micro bounce
    {"r": {0.00:1.00, 0.22:0.58, 0.38:0.03, 0.50:0.11, 0.60:0.00, 1.00:0.00},
     "h": {0.00:1.00, 0.24:0.70, 0.44:0.16, 0.62:0.00, 1.00:0.00}},
    # 2 ─ AF Hunt (controlled overshoot)
    {"r": {0.00:1.00, 0.20:0.52, 0.36:0.04, 0.50:0.22, 0.64:0.05, 0.78:0.00, 1.00:0.00},
     "h": {0.00:0.95, 0.34:0.56, 0.52:0.38, 0.76:0.08, 1.00:0.00}},
    # 3 ─ Lazy then Rush
    {"r": {0.00:1.00, 0.36:0.90, 0.56:0.48, 0.78:0.05, 1.00:0.00},
     "h": {0.00:0.86, 0.40:0.78, 0.70:0.32, 1.00:0.02}},
    # 4 ─ Already Near
    {"r": {0.00:0.50, 0.24:0.36, 0.52:0.10, 0.80:0.00, 1.00:0.00},
     "h": {0.00:0.48, 0.38:0.28, 0.72:0.06, 1.00:0.00}},
    # 5 ─ Documentary (mid-breath pause)
    {"r": {0.00:1.00, 0.14:0.84, 0.36:0.60, 0.52:0.58, 0.70:0.18, 0.90:0.01, 1.00:0.00},
     "h": {0.00:0.84, 0.34:0.62, 0.58:0.46, 0.76:0.16, 1.00:0.00}},
    # 6 ─ Deep Cinema
    {"r": {0.00:1.00, 0.08:1.00, 0.26:0.86, 0.54:0.46, 0.82:0.03, 1.00:0.00},
     "h": {0.00:1.00, 0.18:0.90, 0.46:0.56, 0.80:0.10, 1.00:0.00}},
    # 7 ─ Double Hunt (misses once then locks)
    {"r": {0.00:1.00, 0.18:0.62, 0.32:0.06, 0.46:0.32, 0.60:0.02, 0.74:0.00, 1.00:0.00},
     "h": {0.00:0.94, 0.30:0.58, 0.52:0.28, 0.74:0.02, 1.00:0.00}},
    # 8 ─ Breathing Organic
    {"r": {0.00:1.00, 0.12:0.82, 0.24:0.70, 0.34:0.74, 0.48:0.50,
           0.58:0.54, 0.70:0.28, 0.84:0.03, 1.00:0.00},
     "h": {0.00:0.90, 0.30:0.66, 0.58:0.28, 1.00:0.00}},
    # 9 ─ Ultrafast Lock
    {"r": {0.00:1.00, 0.12:0.82, 0.26:0.36, 0.36:0.00, 1.00:0.00},
     "h": {0.00:1.00, 0.18:0.74, 0.30:0.16, 0.38:0.00, 1.00:0.00}},
]


# ─────────────────────────────────────────────────────────────────────────────
# CATMULL-ROM SPLINE  (smooth keyframe interpolation, zero jitter)
# ─────────────────────────────────────────────────────────────────────────────
def _cr(t_norm: float, kf: dict) -> float:
    keys = sorted(kf)
    n    = len(keys)
    if n == 0:              return 0.0
    if t_norm <= keys[0]:   return float(kf[keys[0]])
    if t_norm >= keys[-1]:  return float(kf[keys[-1]])
    seg = next(i for i in range(n - 1) if keys[i] <= t_norm <= keys[i + 1])
    k0, k1 = keys[seg], keys[seg + 1]
    span = k1 - k0
    if span < 1e-9:
        return float(kf[k1])
    u  = (t_norm - k0) / span
    p0 = float(kf[keys[max(seg - 1, 0)]])
    p1 = float(kf[k0])
    p2 = float(kf[k1])
    p3 = float(kf[keys[min(seg + 2, n - 1)]])
    v  = 0.5 * (u**3 * (-p0 + 3*p1 - 3*p2 + p3)
                + u**2 * (2*p0 - 5*p1 + 4*p2 - p3)
                + u * (-p0 + p2)
                + 2 * p1)
    return max(0.0, float(v))


def _ss(x: float) -> float:
    """Smoothstep 0→1."""
    t = max(0.0, min(1.0, x))
    return t * t * (3 - 2 * t)


def _lum(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


# ─────────────────────────────────────────────────────────────────────────────
# RING BOKEH KERNEL  (cached by radius)
# ─────────────────────────────────────────────────────────────────────────────
_KERN_CACHE: dict = {}


def _ring_kernel(radius: float, ring_w: float = 0.22, ring_b: float = 2.1,
                 feather: float = 0.12) -> np.ndarray:
    key = (round(radius * 4) / 4,)
    if key in _KERN_CACHE:
        return _KERN_CACHE[key]
    r  = max(1, int(round(radius)))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    dist   = np.sqrt(xx**2 + yy**2) / max(r, 0.5)
    k      = np.zeros_like(dist, dtype=np.float32)
    inner  = 1.0 - ring_w
    mi     = dist < inner
    if np.any(mi):
        k[mi] = 0.28 + 0.52 * (dist[mi] / max(inner, 1e-6))
    k[(dist >= inner) & (dist <= 1.0)] = ring_b
    me = (dist > 1.0) & (dist <= 1.0 + feather)
    if np.any(me):
        k[me] = ring_b * np.cos((dist[me] - 1.0) / feather * math.pi / 2) ** 1.8
    s = k.sum()
    k = k / s if s > 0 else k
    _KERN_CACHE[key] = k
    return k


# ─────────────────────────────────────────────────────────────────────────────
# FFT CONVOLUTION
# ─────────────────────────────────────────────────────────────────────────────
def _fft_conv3(rgb: np.ndarray, kern: np.ndarray) -> np.ndarray:
    """Convolve 3-channel image; kernel FFT computed once for all channels."""
    kh, kw = kern.shape
    py, px = kh // 2, kw // 2
    H, W   = rgb.shape[:2]
    ph, pw = H + 2*py, W + 2*px
    shape  = (ph + kh - 1, pw + kw - 1)
    f_kern = np.fft.rfftn(kern, s=shape, axes=(0, 1))
    out    = np.empty_like(rgb, dtype=np.float32)
    for c in range(3):
        pad  = np.pad(rgb[..., c], ((py, py), (px, px)), mode="reflect")
        full = np.fft.irfftn(np.fft.rfftn(pad, s=shape, axes=(0, 1)) * f_kern,
                              s=shape, axes=(0, 1))
        sy, sx = (kh - 1) // 2, (kw - 1) // 2
        same   = full[sy:sy+ph, sx:sx+pw]
        out[..., c] = same[py:py+H, px:px+W]
    return out


def _fft_conv3_ca(rgb: np.ndarray, kr: np.ndarray, kg: np.ndarray,
                  kb: np.ndarray) -> np.ndarray:
    """3-channel convolution with per-channel kernels (chromatic aberration)."""
    out = np.empty_like(rgb, dtype=np.float32)
    for c, kern in enumerate((kr, kg, kb)):
        kh, kw = kern.shape
        py, px = kh // 2, kw // 2
        H, W   = rgb.shape[:2]
        ph, pw = H + 2*py, W + 2*px
        shape  = (ph + kh - 1, pw + kw - 1)
        pad  = np.pad(rgb[..., c], ((py, py), (px, px)), mode="reflect")
        full = np.fft.irfftn(
            np.fft.rfftn(pad, s=shape, axes=(0, 1))
            * np.fft.rfftn(kern, s=shape, axes=(0, 1)),
            s=shape, axes=(0, 1))
        sy, sx = (kh - 1) // 2, (kw - 1) // 2
        same   = full[sy:sy+ph, sx:sx+pw]
        out[..., c] = same[py:py+H, px:px+W]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BOKEH RENDER  (at working resolution)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_bokeh(small_f32: np.ndarray, radius: float, hl: float) -> np.ndarray:
    """
    small_f32: float32 (BOKEH_H, BOKEH_W, 3) in [0, 1].
    Returns float32 [0, 1] blended result.
    Base blur: PIL GaussianBlur (separable, fast).
    Bokeh circles: ring-kernel FFT (vivid bokeh highlights).
    No gamma conversion → no colour shift.
    """
    pil_blur = Image.fromarray((small_f32 * 255).astype(np.uint8), "RGB")
    base = np.array(
        pil_blur.filter(ImageFilter.GaussianBlur(radius=radius * 0.75)),
        dtype=np.float32) / 255.0

    if hl < 0.02:
        return np.clip(base, 0, 1)

    lum    = _lum(small_f32)
    lo, hi = BOKEH_LOW_THR, BOKEH_HIGH_THR
    m_soft = np.clip((lum - lo) / max(hi - lo, 1e-6), 0, 1) ** 3.2
    m_hard = np.clip((lum - (lo + 0.12)) / max(hi - lo - 0.12, 1e-6), 0, 1) ** 9.0

    bloom = np.array(
        Image.fromarray(
            (np.clip(small_f32 * m_soft[..., None], 0, 1) * 255).astype(np.uint8), "RGB")
        .filter(ImageFilter.GaussianBlur(radius=radius * 1.2)),
        dtype=np.float32) / 255.0

    def circ(mask: np.ndarray, rs: float) -> np.ndarray:
        rw = 0.22 + 0.04 * (rs / max(MAX_BOKEH_RADIUS, 1))
        rb = 1.80 + 0.55 * hl
        kr = _ring_kernel(rs * 1.04, rw, rb, 0.07)
        kg = _ring_kernel(rs * 1.00, rw, rb, 0.07)
        kb = _ring_kernel(rs * 0.96, rw, rb, 0.07)
        src = np.clip(small_f32 * mask[..., None], 0, None)
        return np.clip(_fft_conv3_ca(src, kr, kg, kb) * 1.08, 0, 2.0)

    hb = 0.38 + 1.50 * hl
    cb = 0.72 + 1.10 * hl

    result  = base.copy()
    result += bloom                               * (0.13 * hb)
    result += circ(m_hard, max(1.0, radius*1.55)) * (0.11 * hb)
    result += circ(m_hard, max(1.0, radius*1.08)) * (0.21 * hb)
    result += circ(m_hard, max(1.0, radius*0.72)) * (0.28 * cb)
    return np.clip(result, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: apply bokeh to a full-res uint8 frame
# ─────────────────────────────────────────────────────────────────────────────
def apply_bokeh_to_frame(
    motion_frame_uint8: np.ndarray,
    radius: float,
    hl: float,
) -> np.ndarray:
    """
    Returns (H, W, 3) uint8 at full video resolution.
    blend=0 (radius near 0) → exact copy.  Colour preserved.
    """
    blend = _ss(radius / max(BOKEH_BLEND_RADIUS, 1e-3))
    if blend < 0.005:
        return motion_frame_uint8.copy()

    H, W  = motion_frame_uint8.shape[:2]
    bw, bh = BOKEH_W, BOKEH_H
    small  = np.array(
        Image.fromarray(motion_frame_uint8).resize((bw, bh), Image.LANCZOS),
        dtype=np.float32) / 255.0

    bk_small = _compute_bokeh(small, radius, hl)
    bk_up    = np.array(
        Image.fromarray((np.clip(bk_small, 0, 1) * 255).astype(np.uint8), "RGB")
        .resize((W, H), Image.LANCZOS),
        dtype=np.float32)

    orig   = motion_frame_uint8.astype(np.float32)
    result = orig * (1 - blend) + bk_up * blend
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# BOKEH TIMING
# ─────────────────────────────────────────────────────────────────────────────
def get_bokeh_params(
    t_in_seg: float,
    seg_dur: float,
    preset_idx: int,
    is_first: bool = False,
    is_last: bool = False,
) -> tuple[float, float]:
    """
    Returns (radius, highlight) for the current time within a segment.
    is_first → skip intro + focus pull; start sharp immediately.
    is_last  → skip outro; hold sharp until end.
    """
    if is_first:
        hold_end = seg_dur - (0.0 if is_last else BOKEH_OUTRO_SEC)
        if t_in_seg <= hold_end:
            return 0.0, 0.0
        to   = (t_in_seg - hold_end) / max(BOKEH_OUTRO_SEC, 1e-6)
        ease = _ss(to)
        return float(ease * MAX_BOKEH_RADIUS), float(ease * 0.86)

    intro    = BOKEH_INTRO_SEC
    focus    = BOKEH_FOCUS_SEC
    outro    = 0.0 if is_last else BOKEH_OUTRO_SEC
    bookend  = intro + focus + outro

    if seg_dur < bookend:
        s = seg_dur / bookend
        intro *= s; focus *= s; outro *= s

    focus_end = intro + focus
    hold_end  = seg_dur - outro

    if t_in_seg <= intro:
        return float(MAX_BOKEH_RADIUS), 1.0

    if t_in_seg <= focus_end:
        tf    = (t_in_seg - intro) / focus
        pset  = FOCUS_PRESETS[preset_idx % len(FOCUS_PRESETS)]
        r_raw = _cr(tf, pset["r"])
        r_sm  = _ss(r_raw)
        r_blend = r_sm * 0.70 + r_raw * 0.30
        r    = max(0.0, r_blend) * MAX_BOKEH_RADIUS
        hl   = max(0.0, _cr(tf, pset["h"]))
        return float(r), float(hl)

    if t_in_seg <= hold_end:
        return 0.0, 0.0

    if is_last:
        return 0.0, 0.0

    to = (t_in_seg - hold_end) / max(outro, 1e-6)
    return float(_ss(to) * MAX_BOKEH_RADIUS), float(_ss(to) * 0.86)


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-FADE
# ─────────────────────────────────────────────────────────────────────────────
def apply_crossfade(arr_a: np.ndarray, arr_b: np.ndarray, alpha: float) -> np.ndarray:
    """Symmetric exposure-bloom cross-dissolve."""
    ease  = _ss(alpha)
    bloom = 1.0 + 0.13 * math.sin(ease * math.pi)
    a, b  = arr_a.astype(np.float32), arr_b.astype(np.float32)
    return np.clip((a * (1 - ease) + b * ease) * bloom, 0, 255).astype(np.uint8)
