//! Pre-recorded lamp motion (assets/lamp_keys.bin, made by `gen_lamp_sims`).
//! Keyframes are stored at a fixed time step, so sampling is by absolute time: any fps gives the same motion.

use crate::vec::*;

pub struct LampBank {
    pub clips: usize,
    pub keys: usize,
    pub hz: f32,
    data: Vec<f32>, // per key: displacement(3) + axis(3)
}

fn rd(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

impl LampBank {
    pub fn load(path: &str) -> Result<LampBank, String> {
        let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if b.len() < 20 || &b[0..4] != b"LAMP" {
            return Err("not a lamp key file".into());
        }
        let clips = rd(&b, 8) as usize;
        let keys = rd(&b, 12) as usize;
        let hz = f32::from_bits(rd(&b, 16));
        let n = clips * keys * 6;
        if b.len() < 20 + n * 4 {
            return Err("lamp key file truncated".into());
        }
        let data = (0..n).map(|i| f32::from_bits(rd(&b, 20 + i * 4))).collect();
        Ok(LampBank { clips, keys, hz, data })
    }

    pub fn duration(&self) -> f32 {
        (self.keys - 1) as f32 / self.hz
    }

    fn key(&self, clip: usize, k: i64) -> [f32; 6] {
        let k = k.clamp(0, self.keys as i64 - 1) as usize;
        let o = (clip * self.keys + k) * 6;
        let mut a = [0f32; 6];
        a.copy_from_slice(&self.data[o..o + 6]);
        a
    }

    /// (displacement from rest, unit axis) at absolute time t (seconds). Catmull-Rom between keys.
    /// Past the end of a clip the recording is played back and forth (ping-pong).
    pub fn sample(&self, clip: usize, t: f32) -> (V, V) {
        let d = self.duration();
        let mut tt = t.rem_euclid(2.0 * d);
        if tt > d {
            tt = 2.0 * d - tt;
        }
        let f = tt * self.hz;
        let k = f.floor() as i64;
        let u = f - k as f32;
        let (p0, p1, p2, p3) = (self.key(clip, k - 1), self.key(clip, k), self.key(clip, k + 1), self.key(clip, k + 2));
        let mut o = [0f32; 6];
        for i in 0..6 {
            o[i] = 0.5 * ((2.0 * p1[i]) + (-p0[i] + p2[i]) * u + (2.0 * p0[i] - 5.0 * p1[i] + 4.0 * p2[i] - p3[i]) * u * u + (-p0[i] + 3.0 * p1[i] - 3.0 * p2[i] + p3[i]) * u * u * u);
        }
        (v(o[0], o[1], o[2]), v(o[3], o[4], o[5]).norm())
    }
}
