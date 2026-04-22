#!/usr/bin/env python3
"""
settings.py — The ONLY file you need to edit.

All tuneable parameters live here. Sub-systems import from this module;
they never hard-code values themselves.
"""
from pathlib import Path

# ── PATHS ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
SOUNDS_DIR  = ROOT / "sounds"
IMAGES_DIR  = ROOT / "images"
INFO_DIR    = ROOT / "info"
TEMP_DIR    = ROOT / "_temp_render"
FRAMES_DIR  = TEMP_DIR / "frames"
OUT_DIR     = ROOT / "output"

# ── VIDEO CORE ───────────────────────────────────────────────────────────────
VIDEO_W     = 1080
VIDEO_H     = 1920
FPS         = 60
N_BG_IMAGES = 4

# ── SVG TEXT RENDERING ───────────────────────────────────────────────────────
# Render SVG at N× then Lanczos-downsample → ultra-sharp edges
SVG_TEXT_SUPERSAMPLE = 5

# ── BAR (dynamic opacity + animated gradient) ────────────────────────────────
# Opacity range (0–1). Bar auto-adjusts to background luminance.
BAR_OPACITY_MIN              = 0.50
BAR_OPACITY_MAX              = 0.80
BAR_OPACITY_DARK_MIN         = 0.53
BAR_OPACITY_DARK_MAX         = 0.66
BAR_OPACITY_SMOOTH_SEC       = 3.5   # normal mode: crossfade speed
BAR_OPACITY_SMOOTH_SEC_DARK  = 9.0   # dark mode: slower, less jumpy
# Animated gradient sweep across bar pill
BAR_GRAD_STRENGTH            = 0.10  # brightness lift at peak (0=off)
BAR_GRAD_SPEED               = 0.15  # cycles/second (slow sweep)

TRANSITION_SEC   = 1.5
BAR_TIMING_MODE  = 0.5
SVG_RENDER_SCALE = 5.0
INTERNAL_OFFSET  = -8

# ── WARP ─────────────────────────────────────────────────────────────────────
WARP_ENABLED = True
WARP_SPEED   = 0.05
WARP_AMOUNT  = 0.3
WARP_FREQ    = 0.002
WIND_DRIFT_X = -2

# ── GLOW (5 feathered zones over text) ───────────────────────────────────────
GLOW_OPACITY_MIN  = 250
GLOW_OPACITY_MAX  = 400
GLOW_FLOOR        = 0.5
GLOW_RADIUS_MIN   = 25
GLOW_RADIUS_MAX   = 40
GLOW_SPEED_MIN    = 2.0
GLOW_SPEED_MAX    = 3.0
GLOW_FEATHER_MIN  = 0.85
GLOW_FEATHER_MAX  = 1.35
GLOW_PAD          = 50

# ── TEXT TRANSITIONS ─────────────────────────────────────────────────────────
TEXT_FADE_IN_SEC   = 0.30
TEXT_FADE_OUT_SEC  = 0.22
TEXT_SCALE_IN_FROM = 0.96
TEXT_DRIFT_OUT_PX  = 5

# ── GRADIENT / SHIMMER ───────────────────────────────────────────────────────
GRADIENT_SPEED   = 2.0
BRIGHTNESS_BOOST = 1.3
SHIMMER_SPEED    = 100
SHIMMER_WIDTH    = 100
SHIMMER_OPACITY  = 160

# ── DUST PARTICLES ───────────────────────────────────────────────────────────
# DUST_DIRECTION_DEG:
#   0   = moves purely right (horizontal)
#   25  = gentle diagonal (right + slightly down)  ← recommended
#   45  = true diagonal (equal right and down)
#   90  = moves purely down
#
# DUST_BASE_SPEED:
#   Master speed scalar in source-pixels per second for the fastest layer.
#   Layers apply their own speed_mult on top.
#
# DUST_GLOBAL_OPACITY:
#   Master opacity multiplier 0.0–1.0 applied after per-layer opacity.
#
# DUST_LAYER_OPACITIES:
#   Per-layer opacity multiplier [farthest → closest].
#
# DUST_DEPTHS entries: (min_px, max_px, speed_mult, min_base_op, max_base_op)
#   Layer 0 = farthest (tiny specks, dim, slow)
#   Layer 4 = closest  (large circles, bright, fast)

ENABLE_DUST             = True
DUST_ON_NORMAL          = False
DUST_ON_DARK            = True

DUST_DIRECTION_DEG      = 25       # degrees: 0=right, 90=down, 25=diagonal ↘
DUST_SPEED_JITTER       = 0.28     # ±fraction of per-particle speed variation
DUST_BASE_SPEED         = 14       # px/s for fastest layer at speed_mult=1

DUST_GLOBAL_OPACITY     = 1.0
DUST_LAYER_OPACITIES    = [0.22, 0.40, 0.60, 0.80, 1.00]
DUST_TOTAL              = 100

DUST_DEPTHS = [
    # (min_px, max_px, speed_mult, base_min_op, base_max_op)
    (1,  1,  0.12,   8,  18),
    (1,  2,  0.35,  16,  32),
    (2,  3,  0.72,  26,  48),
    (3,  5,  1.60,  40,  72),
    (5,  9,  3.20,  60, 105),
]

DUST_ULTRA_CLOSE_ENABLED = True
DUST_ULTRA_CLOSE_BLUR    = 8      # blur radius (depth-of-field simulation)
DUST_ULTRA_CLOSE_COUNT   = 2      # max per frame (keep small)
DUST_ULTRA_CLOSE_SIZE    = (18, 55)
DUST_ULTRA_CLOSE_OPACITY = (35, 90)

# ── EXTRA TEXT EFFECTS ───────────────────────────────────────────────────────
ENABLE_CHROMATIC = True
ENABLE_SHADOWS   = True

# ── BOKEH ────────────────────────────────────────────────────────────────────
# BOKEH_INTRO_SEC   : seconds at max blur at each segment start
#                     (first image skips → starts sharp immediately)
# BOKEH_FOCUS_SEC   : cinematic focus-pull duration (seconds)
# BOKEH_OUTRO_SEC   : seconds ramping back to blur at segment end
#                     (last image skips → holds sharp until end)
# BOKEH_BLEND_RADIUS: below this radius, smoothly blend back to sharp
# BOKEH_W/H         : working resolution for bokeh computation
#                     270×480 is ~4× faster than 540×960
# MAX_BOKEH_RADIUS  : peak blur circle radius at working resolution
BOKEH_INTRO_SEC    = 0.65
BOKEH_FOCUS_SEC    = 1.75
BOKEH_OUTRO_SEC    = 0.55
MAX_BOKEH_RADIUS   = 14
BOKEH_BLEND_RADIUS = 2.5
BOKEH_W            = 360
BOKEH_H            = 640
BOKEH_LOW_THR      = 0.60
BOKEH_HIGH_THR     = 0.97
CROSSFADE_SEC      = 0.50    # full cross-dissolve window (centered at boundary)

# ── BACKGROUND MOTION (Ken Burns) ────────────────────────────────────────────
# MOTION_ZOOM_AMOUNT  : total zoom change over segment (0.035 = 3.5% size change)
# MOTION_DRIFT_AMOUNT : maximum drift offset as fraction of image half-width
#                       (0.040 = 4% of image half-width = safe range)
# MOTION_NONE_CHANCE  : probability of "no motion" for a segment (≤ 20%)
MOTION_ZOOM_AMOUNT  = 0.035
MOTION_DRIFT_AMOUNT = 0.040
MOTION_NONE_CHANCE  = 0.20

# ── DARK MODE ────────────────────────────────────────────────────────────────
DARK_VIGNETTE_STRENGTH    = 0.78
DARK_SATURATION           = 0.05
DARK_BRIGHTNESS           = 0.55
# Film flicker: multi-frequency oscillation + per-frame Gaussian jitter + dip events
DARK_FLICKER_BASE_SPEED   = 1.1    # Hz
DARK_FLICKER_BASE_AMP     = 0.030
DARK_FLICKER_MID_SPEED    = 4.3    # Hz
DARK_FLICKER_MID_AMP      = 0.016
DARK_FLICKER_FRAME_AMP    = 0.012  # per-frame Gaussian noise
DARK_FLICKER_DIP_CHANCE   = 1.8    # dip events per second
DARK_FLICKER_DIP_AMP      = 0.10   # brightness drop amount
DARK_FLICKER_DIP_DECAY    = 14.0   # recovery speed (higher = faster recovery)
