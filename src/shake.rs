//! The handheld "human hand shake" of local1 — re-implemented as TIME keyframes.
//!
//! local1 pre-computed the shake once per FRAME:
//!
//!     shake_x += amp * np.sin(2*np.pi*freq*np.arange(ntotal)/fps + phx)
//!
//! so the motion was welded to the frame rate: at 24 fps you got a different
//! hand movement than at 60 fps, and a re-encode at another frame rate changed
//! the motion.
//!
//! Here the shake is a table of keyframes laid out on a fixed TIME grid
//! (`KEY_DT` seconds apart) and interpolated with Catmull-Rom.  The clip is
//! sampled by absolute time, so the exact same hand movement is played back at
//! any frame rate — 24, 30, 60 fps all trace the identical curve.
//!
//! The curve itself is local1's: three sine partials with the exact amplitudes,
//! frequencies and the exact random phases that local1 draws from
//! `np.random.default_rng(cfg.seed + 123)` with the default seed 7.
//!
//!   shake_x(t) = SUM  amp_i * sin(2*pi*f_i*t + phx_i)
//!   shake_y(t) = SUM (0.6*amp_i) * sin(2*pi*(0.9*f_i)*t + phy_i)
//!
//! local1 adds these to the camera TARGET, i.e. they translate the whole rig
//! (camera + look direction) sideways and up/down: a pure image translation.

/// local1's shake partials: (frequency Hz, amplitude em, phase x, phase y).
const SHAKE_PARTS: [(f32, f32, f32, f32); 3] = [
    (0.35, 0.30, 2.848_044_7, 0.490_301_07),
    (0.80, 0.18, 1.922_287_2, 2.557_619_7),
    (1.50, 0.09, 4.733_278_5, 2.867_725_8),
];

/// Keyframe spacing in seconds of animation time (120 Hz key grid).
pub const KEY_DT: f32 = 1.0 / 120.0;

pub struct Shake {
    xs: Vec<f32>,
    ys: Vec<f32>,
    dt: f32,
}

#[inline]
fn catmull(p0: f32, p1: f32, p2: f32, p3: f32, f: f32) -> f32 {
    let f2 = f * f;
    let f3 = f2 * f;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * f
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * f2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * f3)
}

/// A per-seed phase offset in radians.  Seed 0 keeps local1's exact phases
/// (used by the tests); any other seed gives every render a different hand.
fn phase(seed: u32, k: u32) -> f32 {
    if seed == 0 {
        return 0.0;
    }
    let mut x = seed ^ k.wrapping_mul(0x9E37_79B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    (x as f32 / u32::MAX as f32) * std::f32::consts::TAU
}

impl Shake {
    /// Build the keyframe table for a clip of `duration` seconds.
    /// `scale` multiplies the whole shake (1.0 = local1's exact motion).
    /// `seed` randomises the phases (0 = local1's exact motion).
    pub fn new(duration: f32, scale: f32, seed: u32) -> Shake {
        let dt = KEY_DT;
        // a few guard keys past the end so the tail of the clip still has
        // neighbours to interpolate against
        let n = ((duration / dt).ceil() as usize).saturating_add(4).max(4);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let tau = std::f32::consts::TAU;
        let parts: Vec<(f32, f32, f32, f32)> = SHAKE_PARTS
            .iter()
            .enumerate()
            .map(|(i, &(f, a, px, py))| {
                (f, a, px + phase(seed, i as u32), py + phase(seed, (i + 3) as u32))
            })
            .collect();
        for i in 0..n {
            let t = i as f32 * dt;
            let (mut x, mut y) = (0.0f32, 0.0f32);
            for &(f, a, px, py) in parts.iter() {
                x += a * (tau * f * t + px).sin();
                y += 0.6 * a * (tau * f * 0.9 * t + py).sin();
            }
            xs.push(x * scale);
            ys.push(y * scale);
        }
        Shake { xs, ys, dt }
    }

    /// Sample the shake at an absolute time (seconds).  Frame-rate independent.
    pub fn sample(&self, t: f32) -> (f32, f32) {
        let n = self.xs.len();
        if n < 2 {
            return (0.0, 0.0);
        }
        let u = (t / self.dt).clamp(0.0, (n - 1) as f32 - 1.0);
        let i = u.floor() as usize;
        let f = u - i as f32;
        let g = |k: isize| -> usize { (i as isize + k).clamp(0, n as isize - 1) as usize };
        let x = catmull(
            self.xs[g(-1)],
            self.xs[g(0)],
            self.xs[g(1)],
            self.xs[g(2)],
            f,
        );
        let y = catmull(
            self.ys[g(-1)],
            self.ys[g(0)],
            self.ys[g(1)],
            self.ys[g(2)],
            f,
        );
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// local1's continuous shake, sampled at 30 fps (numpy, seed 7+123).
    #[test]
    fn matches_local1_continuous() {
        let sh = Shake::new(2.0, 1.0, 0);
        let cases = [
            (0.0, 0.16582, 0.15891),
            (1.0 / 30.0, 0.136838, 0.139899),
            (2.0 / 30.0, 0.111447, 0.119484),
            (1.0 / 60.0, 0.150862, 0.149674),
        ];
        for (t, ex, ey) in cases {
            let (x, y) = sh.sample(t);
            assert!((x - ex).abs() < 2e-3, "x t={t}: {x} vs {ex}");
            assert!((y - ey).abs() < 2e-3, "y t={t}: {y} vs {ey}");
        }
    }

    /// The clip is a time curve: sampling at 30 fps or 60 fps gives the same
    /// values at the same instants.
    #[test]
    fn same_curve_at_any_fps() {
        let sh = Shake::new(2.0, 1.0, 0);
        for i in 0..60 {
            let t30 = i as f32 / 30.0;
            let t60 = (2 * i) as f32 / 60.0;
            assert_eq!(sh.sample(t30), sh.sample(t60));
        }
    }
}
