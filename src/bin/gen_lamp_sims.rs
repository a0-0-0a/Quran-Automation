//! Offline physics recorder for the hanging spot lamps.
//!
//! Model: ceiling anchor -> inextensible wire (chain of 10 light particles) -> lamp (heavy body on a short rigid rod).
//! Solver: XPBD-style position-based dynamics with small sub-steps (Macklin et al. "Small Steps in Physics Simulation"),
//! distance constraints, air drag, gravity. A "ship" sways the ceiling anchor a little, and random "touches"
//! (impulses on the wire or on the lamp) excite wiggle / swing / swirl. 100 different 20 s recordings are written as
//! 60 Hz keyframes (time-based, so any render frame rate samples the same motion).
//!
//! Run:  cargo run --release --bin gen_lamp_sims        -> assets/lamp_keys.bin (+ assets/lamp_keys_index.txt)

use std::io::Write;

const CLIPS: usize = 100;
const DUR: f32 = 20.0;
const HZ: f32 = 60.0;
const WIRE_LEN: f32 = 1.0;
const WIRE_N: usize = 10;
const ROD: f32 = 0.14;
const SUB_HZ: f32 = 480.0;

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Rng {
        Rng(s.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03)
    }
    fn u(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / 16777216.0
    }
    fn r(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.u()
    }
}

type V3 = [f32; 3];
fn sub(a: V3, b: V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn len(a: V3) -> f32 { (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt() }

#[derive(Clone)]
struct Event {
    t: f32,
    particle: usize, // 1..=WIRE_N wire, WIRE_N = attach point, WIRE_N+1 = lamp
    dv: V3,
}
#[derive(Clone)]
struct Plan {
    kind: usize,
    events: Vec<Event>,
    sway: [f32; 6], // ax, ay amplitude; periods; phases
}

const KINDS: [&str; 7] = ["lamp push", "wire pluck", "lamp taps (wiggle)", "push + swirl", "double bump", "ship sway only", "mixed"];

fn make_plan(i: usize, rng: &mut Rng) -> Plan {
    let kind = i % 7;
    let mut ev = Vec::new();
    let lamp = WIRE_N + 1;
    let dir = |rng: &mut Rng| {
        let a = rng.r(0.0, 6.2831853);
        [a.cos(), a.sin()]
    };
    match kind {
        0 => {
            for _ in 0..rng.r(1.0, 2.99) as usize {
                let d = dir(rng);
                let m = rng.r(0.15, 0.32);
                ev.push(Event { t: rng.r(1.0, 16.0), particle: lamp, dv: [d[0] * m, d[1] * m, 0.0] });
            }
        }
        1 => {
            for _ in 0..rng.r(2.0, 4.99) as usize {
                let d = dir(rng);
                let m = rng.r(0.25, 0.6);
                ev.push(Event { t: rng.r(1.0, 16.0), particle: rng.r(3.0, 8.99) as usize, dv: [d[0] * m, d[1] * m, 0.0] });
            }
        }
        2 => {
            let t0 = rng.r(1.0, 15.0);
            for k in 0..3 {
                let d = dir(rng);
                let m = rng.r(0.06, 0.15);
                ev.push(Event { t: t0 + k as f32 * rng.r(0.4, 0.9), particle: lamp, dv: [d[0] * m, d[1] * m, 0.0] });
            }
        }
        3 => {
            let d = dir(rng);
            let m = rng.r(0.15, 0.28);
            let t0 = rng.r(1.0, 15.0);
            ev.push(Event { t: t0, particle: lamp, dv: [d[0] * m, d[1] * m, 0.0] });
            // quarter period later (pendulum T ~ 2.1 s) a push 90 degrees off -> elliptical / circular swirl
            ev.push(Event { t: t0 + 0.55, particle: lamp, dv: [-d[1] * m * 0.9, d[0] * m * 0.9, 0.0] });
        }
        4 => {
            let d = dir(rng);
            let m = rng.r(0.12, 0.25);
            let t0 = rng.r(1.0, 15.0);
            ev.push(Event { t: t0, particle: lamp, dv: [d[0] * m, d[1] * m, 0.0] });
            ev.push(Event { t: t0 + rng.r(1.2, 2.2), particle: lamp, dv: [-d[0] * m * 0.8, -d[1] * m * 0.8, 0.0] });
        }
        5 => {}
        _ => {
            for _ in 0..rng.r(2.0, 3.99) as usize {
                let d = dir(rng);
                let m = rng.r(0.08, 0.28);
                let p = if rng.u() < 0.5 { lamp } else { rng.r(3.0, 8.99) as usize };
                ev.push(Event { t: rng.r(1.0, 16.0), particle: p, dv: [d[0] * m, d[1] * m, 0.0] });
            }
        }
    }
    ev.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    let sway_amp = if kind == 5 { rng.r(0.012, 0.03) } else { rng.r(0.003, 0.012) };
    Plan {
        kind,
        events: ev,
        sway: [sway_amp, sway_amp * rng.r(0.4, 1.0), rng.r(4.0, 8.0), rng.r(4.5, 9.0), rng.r(0.0, 6.28), rng.r(0.0, 6.28)],
    }
}

struct Out {
    keys: Vec<[f32; 6]>, // lamp displacement from rest (3) + unit axis (3)
    max_off: f32,
    rms: f32,
    max_tilt: f32,
    stretch: f32,
}

fn simulate(plan: &Plan, scale: f32) -> Out {
    let n = WIRE_N + 2; // 0 anchor, 1..=WIRE_N wire (last = attach), WIRE_N+1 lamp
    let seg = WIRE_LEN / WIRE_N as f32;
    let mut x: Vec<V3> = (0..n)
        .map(|i| if i <= WIRE_N { [0.0, 0.0, -(i as f32) * seg] } else { [0.0, 0.0, -(WIRE_LEN + ROD)] })
        .collect();
    let mut v = vec![[0.0f32; 3]; n];
    let rest_len: Vec<f32> = (0..n - 1).map(|i| if i < WIRE_N { seg } else { ROD }).collect();
    let mass: Vec<f32> = (0..n)
        .map(|i| if i == 0 { 0.0 } else if i == n - 1 { 0.6 } else if i == WIRE_N { 0.03 } else { 0.01 })
        .collect();
    let w: Vec<f32> = mass.iter().map(|&m| if m > 0.0 { 1.0 / m } else { 0.0 }).collect();
    // calm, realistic drag (1/s): thin wire loses energy fast, heavy lamp keeps swinging for a long time
    let drag: Vec<f32> = (0..n).map(|i| if i == n - 1 { 0.10 } else { 0.9 }).collect();
    let dt = 1.0 / SUB_HZ;
    let steps = (DUR * SUB_HZ) as usize;
    let rest = [0.0, 0.0, -(WIRE_LEN + ROD)];
    let anchor_at = |t: f32| -> V3 {
        let s = &plan.sway;
        [scale * s[0] * (6.2831853 * t / s[2] + s[4]).sin(), scale * s[1] * (6.2831853 * t / s[3] + s[5]).sin(), 0.0]
    };
    let mut ev_i = 0;
    let key_every = (SUB_HZ / HZ) as usize;
    let mut keys = Vec::new();
    let (mut max_off, mut sum2, mut cnt, mut max_tilt, mut stretch) = (0.0f32, 0.0f32, 0usize, 0.0f32, 0.0f32);
    let record = |x: &Vec<V3>, keys: &mut Vec<[f32; 6]>| -> (f32, f32) {
        let b = x[n - 1];
        let a = x[n - 2];
        let ax = sub(b, a);
        let l = len(ax);
        let u = [ax[0] / l, ax[1] / l, ax[2] / l];
        let d = sub(b, rest);
        keys.push([d[0], d[1], d[2], u[0], u[1], u[2]]);
        ((d[0] * d[0] + d[1] * d[1]).sqrt(), (u[0] * u[0] + u[1] * u[1]).sqrt().atan2(-u[2]))
    };
    record(&x, &mut keys);
    for s in 0..steps {
        let t = s as f32 * dt;
        while ev_i < plan.events.len() && plan.events[ev_i].t <= t {
            let e = &plan.events[ev_i];
            for k in 0..3 {
                v[e.particle][k] += e.dv[k] * scale;
            }
            ev_i += 1;
        }
        let a0 = anchor_at(t + dt);
        let mut p = x.clone();
        for i in 1..n {
            v[i][2] -= 9.81 * dt;
            let f = (-drag[i] * dt).exp();
            for k in 0..3 {
                v[i][k] *= f;
                p[i][k] = x[i][k] + v[i][k] * dt;
            }
        }
        p[0] = a0;
        for it in 0..8 {
            let order: Vec<usize> = if it % 2 == 0 { (0..n - 1).collect() } else { (0..n - 1).rev().collect() };
            for &i in &order {
                let (a, b) = (i, i + 1);
                let d = sub(p[b], p[a]);
                let l = len(d).max(1e-9);
                let ws = w[a] + w[b];
                let c = (l - rest_len[i]) / ws;
                for k in 0..3 {
                    let dir = d[k] / l;
                    p[a][k] += w[a] * c * dir;
                    p[b][k] -= w[b] * c * dir;
                }
            }
        }
        for i in 1..n {
            for k in 0..3 {
                v[i][k] = (p[i][k] - x[i][k]) / dt;
            }
        }
        x = p;
        if (s + 1) % key_every == 0 {
            let (off, tilt) = record(&x, &mut keys);
            if t > 4.0 {
                max_off = max_off.max(off);
                sum2 += off * off;
                cnt += 1;
                max_tilt = max_tilt.max(tilt);
            }
            let total: f32 = (0..n - 1).map(|i| len(sub(x[i + 1], x[i]))).sum();
            stretch = stretch.max((total - (WIRE_LEN + ROD)).abs() / (WIRE_LEN + ROD));
        }
    }
    Out { keys, max_off, rms: (sum2 / cnt.max(1) as f32).sqrt(), max_tilt, stretch }
}

fn main() {
    let mut all: Vec<Vec<[f32; 6]>> = Vec::new();
    let mut index = String::from("clip kind scale max_offset_m rms_m max_tilt_deg stretch\n");
    let mut kind_count = [0usize; 7];
    let mut worst_stretch = 0.0f32;
    for i in 0..CLIPS {
        let mut rng = Rng::new(1000 + i as u64);
        let plan = make_plan(i, &mut rng);
        // keep it chill: tune the interaction strength so the lamp sways a few cm up to ~11 cm, never crazy
        let mut scale = 1.0f32;
        let mut out = simulate(&plan, scale);
        for _ in 0..5 {
            if out.max_off > 0.115 || out.max_off < 0.045 {
                scale *= (0.085 / out.max_off.max(1e-4)).clamp(0.4, 2.5);
                out = simulate(&plan, scale);
            } else {
                break;
            }
        }
        worst_stretch = worst_stretch.max(out.stretch);
        kind_count[plan.kind] += 1;
        index += &format!(
            "{:3} {:<20} {:5.2} {:8.4} {:8.4} {:6.2} {:7.4}\n",
            i, KINDS[plan.kind], scale, out.max_off, out.rms, out.max_tilt.to_degrees(), out.stretch
        );
        all.push(out.keys);
    }
    let nk = all[0].len();
    let mut f = std::io::BufWriter::new(std::fs::File::create("assets/lamp_keys.bin").unwrap());
    f.write_all(b"LAMP").unwrap();
    for u in [1u32, CLIPS as u32, nk as u32] {
        f.write_all(&u.to_le_bytes()).unwrap();
    }
    f.write_all(&HZ.to_le_bytes()).unwrap();
    for c in &all {
        assert_eq!(c.len(), nk);
        for k in c {
            for x in k {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }
    }
    f.flush().unwrap();
    std::fs::write("assets/lamp_keys_index.txt", &index).unwrap();
    println!("{} clips x {} keys @ {} Hz ({} s each), kinds {:?}, worst wire stretch {:.3}%", CLIPS, nk, HZ, DUR, kind_count, worst_stretch * 100.0);
    let offs: Vec<f32> = index.lines().skip(1).map(|l| l.split_whitespace().rev().nth(3).unwrap().parse().unwrap()).collect();
    println!("max lamp offset over clips: min {:.3} m, max {:.3} m", offs.iter().cloned().fold(9.0, f32::min), offs.iter().cloned().fold(0.0, f32::max));
}
