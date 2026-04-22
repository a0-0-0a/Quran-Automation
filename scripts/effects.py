#!/usr/bin/env python3
"""
effects.py — Text-layer visual effects + depth-aware dust particles.

KEY CHANGE vs. old effects.py
──────────────────────────────
DUST DIRECTION FIX
    Old code computed vy from a tiny random per-particle angle (~0.04–0.14 rad)
    relative to vx, giving vy ≈ 4–14% of vx — barely diagonal, mostly rightward.

    New code uses a single global DUST_DIRECTION_DEG setting:
        angle_rad = radians(DUST_DIRECTION_DEG)
        vx = speed * cos(angle_rad)   # rightward component
        vy = speed * sin(angle_rad)   # downward component

    With DUST_DIRECTION_DEG = 25:
        vx ≈ 0.906 * speed  (mostly right)
        vy ≈ 0.423 * speed  (noticeably downward — true diagonal ↘)

    With DUST_DIRECTION_DEG = 45:
        vx = vy = 0.707 * speed  (perfect 45° diagonal)

    Per-particle jitter (DUST_SPEED_JITTER) adds natural variation in
    SPEED while the direction stays consistent with the global angle.
    An optional per-particle angle jitter (±DUST_ANGLE_JITTER_DEG) makes
    particles spread in a cone around the main direction.

Wrapping
    x: modular cycle that includes an off-screen "rest" period so particles
       re-enter at staggered times (no mechanical synchronised bursts).
    y: simple modulo H — particle wraps from bottom back to top when vy > 0.
       Works correctly for any angle 0–90°.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageChops, ImageDraw

from settings import (
    VIDEO_W, VIDEO_H,
    WARP_ENABLED, WARP_SPEED, WARP_AMOUNT, WARP_FREQ, WIND_DRIFT_X,
    ENABLE_CHROMATIC,
    GRADIENT_SPEED, BRIGHTNESS_BOOST,
    SHIMMER_SPEED, SHIMMER_WIDTH, SHIMMER_OPACITY,
    ENABLE_SHADOWS,
    GLOW_OPACITY_MIN, GLOW_OPACITY_MAX, GLOW_FLOOR,
    GLOW_RADIUS_MIN, GLOW_RADIUS_MAX,
    GLOW_SPEED_MIN, GLOW_SPEED_MAX,
    GLOW_FEATHER_MIN, GLOW_FEATHER_MAX, GLOW_PAD,
    TEXT_SCALE_IN_FROM, TEXT_DRIFT_OUT_PX,
    TEXT_FADE_IN_SEC, TEXT_FADE_OUT_SEC,
    ENABLE_DUST, DUST_ON_NORMAL, DUST_ON_DARK,
    DUST_DIRECTION_DEG,
    DUST_SPEED_JITTER, DUST_BASE_SPEED,
    DUST_GLOBAL_OPACITY, DUST_LAYER_OPACITIES,
    DUST_TOTAL, DUST_DEPTHS,
    DUST_ULTRA_CLOSE_ENABLED, DUST_ULTRA_CLOSE_BLUR,
    DUST_ULTRA_CLOSE_COUNT, DUST_ULTRA_CLOSE_SIZE, DUST_ULTRA_CLOSE_OPACITY,
)

# Per-particle angle spread around the global direction (degrees).
# 0 = all particles move at exactly DUST_DIRECTION_DEG.
# 12 = particles spread in a ±12° cone — adds organic variation.
DUST_ANGLE_JITTER_DEG = 12.0


# ─────────────────────────────────────────────────────────────────────────────
# TEXT WARP
# ─────────────────────────────────────────────────────────────────────────────
def apply_gentle_warp(img: Image.Image, t: float, seed: float) -> Image.Image:
    if not WARP_ENABLED:
        return img
    w, h = img.size
    pad  = 60
    pw, ph = w + pad, h + pad
    arr  = np.array(img)
    out  = np.zeros((ph, pw, 4), dtype=np.uint8)
    dy   = np.arange(ph, dtype=np.float64)
    sx   = (np.sin((dy * WARP_FREQ * 10) + (t * WARP_SPEED * 15) + seed)
            * (WARP_AMOUNT * 10) + WIND_DRIFT_X)
    sy   = np.cos(dy * 0.01 + t * 1.5) * 1.5
    bx   = np.arange(pw, dtype=np.float64) - (pad // 2)
    for j in range(ph):
        sj = int(j - (pad // 2) - sy[j])
        if not (0 <= sj < h):
            continue
        sxj = (bx - sx[j]).astype(np.int32)
        ok  = (sxj >= 0) & (sxj < w)
        out[j, ok] = arr[sj, sxj[ok]]
    return Image.fromarray(out, "RGBA")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CHROMATIC ABERRATION
# ─────────────────────────────────────────────────────────────────────────────
def apply_chromatic_aberration(img: Image.Image, amount: int = 1) -> Image.Image:
    if not ENABLE_CHROMATIC:
        return img
    r, g, b, a = img.split()
    return Image.merge("RGBA", (
        ImageChops.offset(r,  amount, 0),
        g,
        ImageChops.offset(b, -amount, 0),
        a,
    ))


# ─────────────────────────────────────────────────────────────────────────────
# ANIMATED GRADIENT
# ─────────────────────────────────────────────────────────────────────────────
def apply_animated_gradient(img: Image.Image, t: float, seed: float) -> Image.Image:
    w, h = img.size
    lum  = np.clip(
        212 + 43 * np.sin((np.arange(w, dtype=np.float32) + t * 60 * GRADIENT_SPEED) * 0.04),
        0, 255).astype(np.uint8)
    grad = Image.fromarray(np.broadcast_to(lum[np.newaxis, :], (h, w)).copy(), "L")
    res  = Image.merge("RGBA", (grad, grad, grad, img.getchannel("A")))
    return ImageEnhance.Brightness(res).enhance(BRIGHTNESS_BOOST)


# ─────────────────────────────────────────────────────────────────────────────
# DUST PARTICLES — diagonal, parametrised direction
# ─────────────────────────────────────────────────────────────────────────────
def apply_dust_particles(canvas: Image.Image, t: float, seed: int) -> None:
    """
    Draw depth-layered dust particles onto `canvas` (RGBA, in-place).

    Direction is controlled by DUST_DIRECTION_DEG:
        vx = speed * cos(direction_rad)  — rightward component
        vy = speed * sin(direction_rad)  — downward component

    Per-particle jitter varies SPEED (±DUST_SPEED_JITTER) and DIRECTION
    (±DUST_ANGLE_JITTER_DEG) for organic variety.

    Wrapping:
        x: modular cycle including an off-screen delay so particles
           re-enter at staggered times rather than in synchronised waves.
        y: simple modulo H — wraps from bottom back to top.
    """
    if not ENABLE_DUST:
        return

    draw    = ImageDraw.Draw(canvas)
    W, H    = canvas.size
    n_l     = len(DUST_DEPTHS)
    per_l   = max(1, DUST_TOTAL // n_l)

    base_angle_rad = math.radians(DUST_DIRECTION_DEG)

    for d_idx, (mn_sz, mx_sz, spd, mn_op, mx_op) in enumerate(DUST_DEPTHS):
        lrng     = random.Random(seed + d_idx * 104729)
        lop_mult = DUST_LAYER_OPACITIES[d_idx] * DUST_GLOBAL_OPACITY

        for _p in range(per_l):
            # ── Stable per-particle identity ──────────────────────────────
            bx       = lrng.random()                               # x birth phase [0,1]
            by       = lrng.random()                               # y birth position [0,1]
            sz       = lrng.randint(mn_sz, mx_sz)
            b_op     = lrng.randint(mn_op, mx_op)
            speed_j  = lrng.uniform(1 - DUST_SPEED_JITTER, 1 + DUST_SPEED_JITTER)
            angle_j  = math.radians(lrng.uniform(-DUST_ANGLE_JITTER_DEG, DUST_ANGLE_JITTER_DEG))
            re_delay = lrng.uniform(0.8, 3.5)                     # off-screen rest time (s)

            # ── Velocity from direction angle ──────────────────────────────
            final_angle = base_angle_rad + angle_j
            speed       = DUST_BASE_SPEED * spd * speed_j
            vx          = speed * math.cos(final_angle)
            vy          = speed * math.sin(final_angle)

            # ── X: slides across, exits right, waits re_delay, repeats ───
            margin_x = sz + 40
            cycle_x  = W + margin_x + vx * re_delay
            raw_x    = (bx * cycle_x + t * vx) % cycle_x
            x        = raw_x - vx * re_delay * 0.5

            if x < -sz or x > W + sz:
                continue   # particle in off-screen delay zone

            # ── Y: constant downward drift, wraps modulo H ────────────────
            y = (by * H + t * vy) % H

            shim = 0.70 + 0.30 * math.sin(t * spd * 2.4 + bx * 13.7 + by * 9.3)
            op   = max(0, min(255, int(b_op * shim * lop_mult)))
            draw.ellipse([x, y, x + sz, y + sz], fill=(255, 255, 255, op))

    # ── Ultra-close layer: large defocused blobs ──────────────────────────
    if not (DUST_ULTRA_CLOSE_ENABLED and DUST_ULTRA_CLOSE_COUNT > 0):
        return

    ucl_rng = random.Random(seed + 999983)
    for _p in range(DUST_ULTRA_CLOSE_COUNT):
        bx    = ucl_rng.random()
        b_op  = ucl_rng.randint(*DUST_ULTRA_CLOSE_OPACITY)
        sz    = ucl_rng.randint(*DUST_ULTRA_CLOSE_SIZE)
        re_t  = ucl_rng.uniform(1.5, 5.0)
        by_seed = ucl_rng.random()

        # Ultra-close particles move faster and at the global angle
        uc_speed = DUST_BASE_SPEED * 5.8 * ucl_rng.uniform(0.75, 1.25)
        uc_vx    = uc_speed * math.cos(base_angle_rad)
        uc_vy    = uc_speed * math.sin(base_angle_rad)

        margin = sz + DUST_ULTRA_CLOSE_BLUR * 4 + 60
        cycle  = W + margin + uc_vx * re_t
        x      = (bx * W + t * uc_vx) % cycle
        x     -= uc_vx * re_t * 0.4

        if x < -margin or x > W + margin:
            continue

        cycle_num = int((bx * W + t * uc_vx) / cycle)
        cy_rng    = random.Random(seed + _p * 7 + cycle_num * 31337)
        y         = cy_rng.random() * H
        # Apply vertical drift within the cycle too
        y         = (y + t * uc_vy) % H

        op = max(0, min(255, int(b_op * DUST_GLOBAL_OPACITY)))

        blob_sz = sz + DUST_ULTRA_CLOSE_BLUR * 4
        blob    = Image.new("RGBA", (blob_sz * 2, blob_sz * 2), (0, 0, 0, 0))
        bd      = ImageDraw.Draw(blob)
        off     = blob_sz // 2
        bd.ellipse([off, off, off + sz, off + sz], fill=(255, 255, 255, op))
        blob = blob.filter(ImageFilter.GaussianBlur(radius=DUST_ULTRA_CLOSE_BLUR))
        try:
            canvas.alpha_composite(blob, (int(x) - off, int(y) - off))
        except (ValueError, OverflowError):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 5-ZONE GLOW
# ─────────────────────────────────────────────────────────────────────────────
def apply_5_blocks_glow(
    text_img: Image.Image,
    t: float,
    opacity: int,
    item_seed: int,
) -> Image.Image:
    """Animated 5-zone horizontal glow over the text alpha channel."""
    w, h = text_img.size
    bw   = w / 5.0
    P    = GLOW_PAD
    pw, ph = w + 2*P, h + 2*P

    af = Image.new("L", (pw, ph), 0)
    af.paste(text_img.getchannel("A"), (P, P))
    aa = np.array(af, dtype=np.float32) / 255.0
    xa = np.arange(pw, dtype=np.float32)
    acc = np.zeros((ph, pw), dtype=np.float32)

    for j in range(5):
        rng   = random.Random(item_seed * 7919 + j * 104729 + 31337)
        spd   = rng.uniform(GLOW_SPEED_MIN, GLOW_SPEED_MAX)
        ph_   = rng.uniform(0, 2 * math.pi)
        stay  = rng.uniform(0.4, 1.6)
        rad   = rng.uniform(GLOW_RADIUS_MIN, GLOW_RADIUS_MAX)
        peak  = min(255.0, rng.uniform(GLOW_OPACITY_MIN, GLOW_OPACITY_MAX))
        feat  = rng.uniform(GLOW_FEATHER_MIN, GLOW_FEATHER_MAX)

        wave  = math.sin(t * spd + ph_) * stay + math.cos(t * spd * 0.53 + ph_ * 1.3) * 0.6
        norm  = max(0.0, min(1.0, (wave + 1.6) / 3.2))
        inten = GLOW_FLOOR * peak + norm * (1 - GLOW_FLOOR) * peak
        if inten < 1.0:
            continue

        cx    = P + (j + 0.5) * bw
        gauss = np.exp(-0.5 * ((xa - cx) / (bw * feat)) ** 2)
        layer = np.clip(gauss[np.newaxis, :] * aa * inten, 0, 255).astype(np.uint8)
        acc  += np.array(
            Image.fromarray(layer, "L").filter(ImageFilter.GaussianBlur(radius=rad)),
            dtype=np.float32)

    acc = np.clip(acc, 0, 255)
    if opacity < 255:
        acc *= opacity / 255.0
    res = Image.new("RGBA", (pw, ph), (255, 255, 255, 0))
    res.putalpha(Image.fromarray(acc.astype(np.uint8), "L"))
    return res


# ─────────────────────────────────────────────────────────────────────────────
# FULL EFFECT STACK
# ─────────────────────────────────────────────────────────────────────────────
def apply_premium_plus_effects(
    text_img: Image.Image,
    t: float,
    fade_in: float,
    fade_out: float,
    item_seed: int,
) -> Image.Image:
    """
    Full text effect stack:
    gradient → warp → chromatic aberration → scale → glow → shimmer → shadow → composite.
    """
    ei = fade_in  * fade_in  * (3 - 2 * fade_in)
    eo = fade_out * fade_out * (3 - 2 * fade_out)
    op = max(0, min(255, int(255 * ei * (1 - eo))))

    scale   = TEXT_SCALE_IN_FROM + (1 - TEXT_SCALE_IN_FROM) * ei
    drift_y = -int(TEXT_DRIFT_OUT_PX * eo)

    colored = apply_animated_gradient(text_img, t, item_seed)
    warped  = apply_gentle_warp(colored, t, item_seed)
    warped  = apply_chromatic_aberration(warped, amount=1)
    w, h    = warped.size

    if abs(scale - 1.0) > 0.001:
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        buf    = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        buf.paste(warped.resize((sw, sh), Image.LANCZOS), ((w - sw) // 2, (h - sh) // 2))
        warped = buf

    mask   = warped.getchannel("A")
    glow   = apply_5_blocks_glow(warped, t, op, item_seed)
    P      = GLOW_PAD

    shimmer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(shimmer)
    pos     = (t * SHIMMER_SPEED) % (w + 400)
    cx_s    = w - pos + 200
    draw.polygon(
        [(cx_s, 0), (cx_s + SHIMMER_WIDTH, 0),
         (cx_s + SHIMMER_WIDTH - 60, h), (cx_s - 60, h)],
        fill=(255, 255, 255, SHIMMER_OPACITY))
    shimmer  = shimmer.filter(ImageFilter.GaussianBlur(radius=15))
    sh_out   = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    sh_out.putalpha(ImageChops.multiply(shimmer.getchannel("A"), mask))

    margin = 50 + P
    canvas = Image.new("RGBA",
                        (w + margin * 2, h + margin * 2 + max(0, -drift_y)),
                        (0, 0, 0, 0))
    off = margin

    if ENABLE_SHADOWS:
        shadow = Image.new("RGBA", (w, h), (0, 0, 0, 100))
        shadow.putalpha(mask)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=15))
        canvas.alpha_composite(shadow, (off + 3, off + 6 + drift_y))

    canvas.alpha_composite(glow,  (off - P, off - P + drift_y))

    t_text = warped.copy()
    if op < 255:
        t_text.putalpha(Image.eval(t_text.getchannel("A"),
                                    lambda x: int(x * op / 255)))
    canvas.alpha_composite(t_text, (off, off + drift_y))
    canvas.alpha_composite(sh_out, (off, off + drift_y))
    return canvas
