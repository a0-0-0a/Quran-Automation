//! local1's word-to-word camera slide, ported verbatim.
//!
//! local1 builds a per-frame `cam_xs` array by critically damping the camera
//! towards the centre of whichever word is being recited (`smooth_damp`), then
//! looks the array up `lead` seconds ahead.  The camera keeps this exact motion;
//! only the *shake* is re-expressed as time keyframes (see `shake.rs`).

#[derive(Clone, Copy, Debug)]
pub struct Segment {
    /// Word centre, in the same world units as `Mesh::recitation_x`.
    pub center: f32,
    /// Segment start / end, seconds.
    pub start: f32,
    pub end: f32,
}

/// local1 `target_center_at`: the centre of the word being recited at `t`
/// (seconds), or the next/previous word when `t` falls in a gap.
pub fn target_center_at(t: f32, segs: &[Segment]) -> f32 {
    if segs.is_empty() {
        return 0.0;
    }
    for e in segs {
        if e.start <= t && t <= e.end {
            return e.center;
        }
    }
    for e in segs {
        if t < e.start {
            return e.center;
        }
    }
    segs[segs.len() - 1].center
}

/// local1 `smooth_damp` (Game Programming Gems 4, critically damped spring).
pub fn smooth_damp(current: f32, target: f32, vel: f32, smooth_time: f32, dt: f32) -> (f32, f32) {
    let smooth_time = smooth_time.max(0.0001);
    let omega = 2.0 / smooth_time;
    let x = omega * dt;
    let exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x);
    let change = (current - target).clamp(-1.0e9, 1.0e9);
    let target_adj = current - change;
    let temp = (vel + omega * change) * dt;
    let new_vel = (vel - omega * temp) * exp;
    let new_current = target_adj + (change + temp) * exp;
    (new_current, new_vel)
}

/// The sampled slide, exactly like local1's `cam_xs`.
pub struct Slide {
    xs: Vec<f32>,
    fps: u32,
    lead: f32,
}

impl Slide {
    pub fn build(
        segs: &[Segment],
        duration: f32,
        fps: u32,
        smooth_time: f32,
        lead: f32,
    ) -> Slide {
        let nframes = (duration * fps as f32).round() as usize;
        let extra = ((lead + 0.5) * fps as f32).round() as usize + 5;
        let ntotal = nframes + extra;
        let mut xs = vec![0.0f32; ntotal];
        if segs.is_empty() || ntotal == 0 {
            return Slide { xs, fps, lead };
        }
        let mut cur = segs[0].center;
        let mut vel = 0.0f32;
        let dt = 1.0 / fps as f32;
        let last = (duration - 0.01).max(0.0);
        for i in 0..ntotal {
            let t = i as f32 / fps as f32;
            let tgt = target_center_at(t.min(last), segs);
            let (c, v) = smooth_damp(cur, tgt, vel, smooth_time, dt);
            cur = c;
            vel = v;
            xs[i] = cur;
        }
        Slide { xs, fps, lead }
    }

    /// Camera target X at render time `t` (seconds).
    pub fn at(&self, t: f32) -> f32 {
        if self.xs.is_empty() {
            return 0.0;
        }
        let idx = ((t + self.lead) * self.fps as f32).round();
        let idx = idx.clamp(0.0, (self.xs.len() - 1) as f32) as usize;
        self.xs[idx]
    }

    pub fn len(&self) -> usize {
        self.xs.len()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Values produced by local1's Python `render_baqarah.py` for the same
    /// segments (center 0/10/20, 1 s each, smooth_time 1.5, lead 0.9, 10 fps).
    #[test]
    fn matches_local1() {
        let segs = [
            Segment { center: 0.0, start: 0.0, end: 1.0 },
            Segment { center: 10.0, start: 1.0, end: 2.0 },
            Segment { center: 20.0, start: 2.0, end: 3.0 },
        ];
        let sl = Slide::build(&segs, 3.0, 10, 1.5, 0.9);
        let expect = [
            (0.0, 0.0),
            (0.5, 0.998233),
            (1.0, 3.362908),
            (1.5, 6.554005),
            (2.0, 10.547977),
            (2.5, 13.836295),
            (3.0, 16.161204),
        ];
        for (t, x) in expect {
            assert!((sl.at(t) - x).abs() < 1e-4, "t={t}: {} vs {x}", sl.at(t));
        }
    }

    #[test]
    fn starts_on_the_first_word() {
        // local1 initialises `cur` to the first word's centre, so even with a
        // lead the camera starts there instead of at the old `start_x`.
        let segs = [Segment { center: 5.0, start: 2.0, end: 3.0 }];
        let sl = Slide::build(&segs, 3.0, 30, 1.5, 0.9);
        assert!((sl.at(0.0) - 5.0).abs() < 1e-4, "{}", sl.at(0.0));
    }
}
