//! Per-frame work is deliberately tiny:
//!   primary ray -> heightfield hit -> look up BAKED irradiance -> albedo * E
//!   + one short glossy reflection ray on the floor (the only view-dependent light transport)
//!   + analytic GGX highlights from the (few) spot lights
//! then edge-only supersampling and a cheap "macro lens" post (variable blur, bloom, vignette, filmic).

use crate::bake::Maps;
use crate::scene::*;
use crate::tex::Tex;
use crate::vec::*;
use rayon::prelude::*;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub struct Camera {
    pub pos: V,
    pub fwd: V,
    pub right: V,
    pub up: V,
}
impl Camera {
    pub fn look_at(pos: V, target: V) -> Camera {
        let fwd = (target - pos).norm();
        let right = fwd.cross(UP).norm();
        let up = right.cross(fwd);
        Camera { pos, fwd, right, up }
    }
}

pub struct Params {
    pub f0_floor: f32,
    pub alpha_floor: f32,
    pub alpha_hl: f32, // roughness used for the lamp highlight on the floor
    pub lod_bias: f32,
    pub vein_gloss: f32,
    pub env_gloss: f32,
    pub env_box: (f32, f32), // centre (x,y) of the faint ceiling light reflected in the floor
    pub grade: f32,        // colour grade strength (0 = off)
    pub text_albedo: V,
    pub f0_text: f32,
    pub alpha_text: f32,
    pub tex_scale: f32,
    pub ambient: f32,
    pub floor_gain: f32,
    pub env: f32,
    pub exposure: f32,
    pub coc_k: f32, // blur-circle diameter (px) per unit |z-zf|/z : f^2 / (N (zf-f)) / pixel_pitch
    pub focus: f32, // focus distance (m)
    pub vignette: f32,
    pub bloom: f32,
    /// Small constant floor irradiance: the baked map only covers the text
    /// area, so without this the top/bottom of the frame are pure black.
    /// It reveals the floor texture everywhere and fades into the lit middle.
    pub floor_fill: f32,
}

#[derive(Default, Clone, Copy)]
pub struct Timings {
    pub trace: Duration,
    pub aa: Duration,
    pub post: Duration,
}

pub struct Renderer {
    pub w: usize,
    pub h: usize,
    pub focal: f32,
    pub scene: Scene,
    pub maps: Maps,
    pub tex: Tex,
    pub p: Params,
    hdr: Vec<f32>,
    cls: Vec<u8>,
    pub timings: Timings,
    pub lights: Vec<Light>,
    zbuf: Vec<f32>,
    pyr: Vec<Vec<f32>>,
    pyr_dst: Vec<f32>,
    pyr_tmp: Vec<f32>,
    lut: Vec<f32>,
}

#[inline(always)]
fn schlick(f0: f32, c: f32) -> f32 {
    f0 + (1.0 - f0) * (1.0 - c).max(0.0).powi(5)
}

// ---------- separable box blur on interleaved RGB (running sums) ----------
fn box_h(src: &[f32], dst: &mut [f32], w: usize, r: usize) {
    let inv = 1.0 / (2 * r + 1) as f32;
    dst.par_chunks_mut(w * 3).zip(src.par_chunks(w * 3)).for_each(|(d, s)| {
        for c in 0..3 {
            let mut acc = 0.0f32;
            for i in -(r as i32)..=(r as i32) {
                acc += s[(i.clamp(0, w as i32 - 1) as usize) * 3 + c];
            }
            for x in 0..w {
                d[x * 3 + c] = acc * inv;
                let add = (x + r + 1).min(w - 1);
                let sub = (x as i32 - r as i32).max(0) as usize;
                acc += s[add * 3 + c] - s[sub * 3 + c];
            }
        }
    });
}
fn box_v(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize) {
    let inv = 1.0 / (2 * r + 1) as f32;
    let row = w * 3;
    let mut acc = vec![0f32; row];
    for i in -(r as i32)..=(r as i32) {
        let y = i.clamp(0, h as i32 - 1) as usize;
        for k in 0..row {
            acc[k] += src[y * row + k];
        }
    }
    for y in 0..h {
        for k in 0..row {
            dst[y * row + k] = acc[k] * inv;
        }
        let add = (y + r + 1).min(h - 1);
        let sub = (y as i32 - r as i32).max(0) as usize;
        for k in 0..row {
            acc[k] += src[add * row + k] - src[sub * row + k];
        }
    }
}
fn blur2(src: &[f32], dst: &mut [f32], tmp: &mut [f32], w: usize, h: usize, r: usize) {
    box_h(src, tmp, w, r);
    box_v(tmp, dst, w, h, r);
    box_h(dst, tmp, w, r);
    box_v(tmp, dst, w, h, r);
}

#[inline]
fn aces(x: f32) -> f32 {
    ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
}
#[inline]
fn to_srgb(x: f32) -> f32 {
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

impl Renderer {
    pub fn new(w: usize, h: usize, focal: f32, scene: Scene, maps: Maps, tex: Tex, p: Params) -> Self {
        let n = w * h * 3;
        Renderer {
            w,
            h,
            focal,
            scene,
            maps,
            tex,
            p,
            hdr: vec![0.0; n],
            cls: vec![0; w * h],
            timings: Timings::default(),
            lights: Vec::new(),
            zbuf: vec![0.0; w * h],
            pyr: {
                let (mut lw, mut lh) = (w, h);
                (0..3).map(|_| { lw = (lw + 1) / 2; lh = (lh + 1) / 2; vec![0.0f32; lw * lh * 3] }).collect()
            },
            pyr_dst: vec![0.0; ((w + 1) / 2) * ((h + 1) / 2) * 3],
            pyr_tmp: vec![0.0; ((w + 1) / 2) * ((h + 1) / 2) * 3],
            lut: (0..=4096).map(|i| { let t = i as f32 / 4095.0; to_srgb(aces(16.0 * t * t)) }).collect(),
        }
    }

    // ---------------- shading ----------------
    /// Dynamic spot-light loop (runs every frame, so shadows follow the swinging lamps).
    /// Returns (diffuse irradiance, specular radiance, state bits: 1 = some light shadowed, 2 = some light reaches).
    /// `jit` picks a random point on each light's emitting disc (used by the AA samples -> real penumbra);
    /// otherwise the disc centre is used, which gives a razor-sharp shadow edge.
    fn lights(&self, p: V, n: V, vd: V, alpha: f32, f0: f32, shadow: bool, rng: &mut Rng, jit: bool) -> (V, V, u8) {
        let (mut e, mut sp, mut st) = (0.0f32, 0.0f32, 0u8);
        let o = p + n * 3e-4;
        for l in &self.lights {
            let dx = p.x - l.cx;
            let dy = p.y - l.cy;
            if dx * dx + dy * dy > l.reach2 {
                continue;
            }
            let q = if jit {
                let a = rng.f() * 6.2831853;
                let rr = l.rad * rng.f().sqrt();
                l.pos + l.t1 * (rr * a.cos()) + l.t2 * (rr * a.sin())
            } else {
                l.pos
            };
            let w = q - p;
            let r2 = w.dot(w);
            let inv = 1.0 / r2.sqrt();
            let r = r2 * inv;
            let wn = w * inv;
            let cosp = n.dot(wn);
            if cosp <= 0.0 {
                continue;
            }
            let fall = smoothstep(l.cos_out, l.cos_in, -wn.dot(l.axis));
            if fall <= 0.0 {
                continue;
            }
            if shadow {
                if self.scene.occluded(o, wn, r) {
                    st |= 1;
                    continue;
                }
                st |= 2;
            }
            e += fall * cosp / r2;
            let a = (alpha + l.rad / (3.0 * r)).min(1.0);
            sp += fall * ggx_nl(n, vd, wn, a, f0) / r2;
        }
        let c = self.scene.lcol * self.scene.i0;
        (c * e, c * sp, st)
    }

    fn surf(&self, h: &Hit, vd: V, rng: &mut Rng, jit: bool) -> (V, u8) {
        let amb = self.p.env;
        match h.kind {
            TOP => {
                let (e, s, st) = self.lights(h.p, UP, vd, self.p.alpha_text, self.p.f0_text, false, rng, jit);
                let fr = schlick(self.p.f0_text, vd.z.max(0.0));
                (self.p.text_albedo * (e + self.maps.top(h.p.x, h.p.y)) * ((1.0 - fr) / PI) + s + v(amb, amb, amb), st)
            }
            WALL => {
                let (e, s, st) = self.lights(h.p, h.n, vd, self.p.alpha_text, self.p.f0_text, true, rng, jit);
                let fr = schlick(self.p.f0_text, h.n.dot(vd).max(0.0));
                (self.p.text_albedo * (e + self.maps.wall(h.p.x, h.p.y)) * ((1.0 - fr) / PI) + s + v(amb, amb, amb), st)
            }
            _ => (v(amb, amb, amb), 0),
        }
    }

    /// What the floor reflects when the ray escapes: darkness plus (optionally) a faint soft ceiling light,
    /// which is what makes a polished floor read as glossy. Fixed in the world, so it slides as the camera moves.
    #[inline]
    fn sky(&self, p: V, r: V) -> V {
        let e = self.p.env;
        if self.p.env_gloss <= 0.0 || r.z < 0.05 {
            return v(e, e, e);
        }
        let t = (2.2 - p.z) / r.z;
        let (x, y) = (p.x + r.x * t - self.p.env_box.0, p.y + r.y * t - self.p.env_box.1);
        let m = (1.0 - smoothstep(0.55, 0.95, x.abs())) * (1.0 - smoothstep(0.20, 0.50, y.abs()));
        let b = self.p.env_gloss * m;
        v(e + b, e + b * 0.96, e + b * 0.9)
    }

    fn reflect_color(&self, p: V, r: V, rng: &mut Rng) -> (V, u8) {
        let hit = self.scene.intersect(v(p.x, p.y, 1e-4), r);
        let k = match hit.kind {
            TOP => 1,
            WALL => 2,
            _ => 0,
        };
        if hit.kind == SKY {
            return (self.sky(p, r), k);
        }
        (self.surf(&hit, -r, rng, false).0, k)
    }

    fn shade(&self, o: V, d: V, rng: &mut Rng, jit: bool) -> (V, u8, f32) {
        let hit = self.scene.intersect(o, d);
        match hit.kind {
            FLOOR => {
                let p = hit.p;
                let nv = (-d.z).max(1e-3);
                let fr = schlick(self.p.f0_floor, nv);
                // The floor is one continuous surface: sample the texture across
                // the WHOLE frame (not just the lamp pools), so there is no
                // hard line where the pool ends.  The texture is seamless, so
                // the tiles fade into each other on their own.
                let fp = hit.t / self.focal / nv.max(0.08);
                let lod = (fp / (self.p.tex_scale / self.tex.size0)).max(1.0).log2() + self.p.lod_bias;
                let a = self.tex.sample(p.x / self.p.tex_scale, p.y / self.p.tex_scale, lod);
                let ratio = ((a.x + a.y + a.z) / (self.tex.avg.x + self.tex.avg.y + self.tex.avg.z).max(1e-6)).min(2.0);
                let alb = a;
                let rmul = 1.0 + self.p.vein_gloss * 0.5 * (ratio - 1.0); // brighter mineral areas a little duller
                let (ed, sd_spec, st) = self.lights(p, UP, -d, self.p.alpha_hl * rmul, self.p.f0_floor, true, rng, jit);
                // One small fill everywhere (not just outside the bake): the
                // unlit centre and the top/bottom are the same "out of light"
                // level, so they match exactly.  In the lit pool the direct
                // light dominates, so the fill changes the lit centre by <1%.
                let fill = self.p.floor_fill;
                let e = ed + self.maps.floor_clamped(p.x, p.y) + v(fill, fill, fill);
                let mut c = alb * self.p.floor_gain * e * ((1.0 - fr) / PI) + sd_spec;
                // glossy reflection of the slab (the only view-dependent transport): skip when out of reach
                let r = v(d.x, d.y, -d.z);
                let sdist = self.scene.sdf_at(p.x, p.y);
                let reach = self.scene.h * (r.x * r.x + r.y * r.y).sqrt() / r.z.max(1e-3);
                let mut rc = self.sky(p, r);
                let mut k = 0u8;
                if sdist < reach * 1.15 {
                    let (c0, k0) = self.reflect_color(p, r, rng);
                    rc = c0;
                    k = k0;
                    let m = sample_ggx(UP, self.p.alpha_floor * rmul, rng);
                    let mut rj = d - m * (2.0 * d.dot(m));
                    if rj.z <= 0.0 {
                        rj = r;
                    }
                    rc = (rc + self.reflect_color(p, rj, rng).0) * 0.5;
                }
                c += rc * fr;
                (c, FLOOR | (k << 3) | (st << 5), hit.t)
            }
            SKY => (v(self.p.env, self.p.env, self.p.env), SKY, 1.0e4),
            k => {
                let (c, st) = self.surf(&hit, -d, rng, jit);
                (c, k | (st << 5), hit.t)
            }
        }
    }

    #[inline]
    fn ray(&self, cam: &Camera, px: f32, py: f32) -> V {
        let sx = (px - self.w as f32 * 0.5) / self.focal;
        let sy = (self.h as f32 * 0.5 - py) / self.focal;
        (cam.fwd + cam.right * sx + cam.up * sy).norm()
    }

    /// Render one frame, write 8-bit RGB (w*h*3) into `out`.
    pub fn render(&mut self, cam: &Camera, frame: u32, out: &mut [u8]) {
        let (w, h) = (self.w, self.h);
        let mut hdr = std::mem::take(&mut self.hdr);
        let mut cls = std::mem::take(&mut self.cls);
        let mut zbuf = std::mem::take(&mut self.zbuf);
        let t0 = Instant::now();
        {
            let me = &*self;
            // pass 1: one sample per pixel
            hdr.par_chunks_mut(w * 3).zip(cls.par_chunks_mut(w)).zip(zbuf.par_chunks_mut(w)).enumerate().for_each(|(y, ((row, crow), zrow))| {
                let mut rng = Rng::new(y as u32 * 2654435 + frame * 97 + 5);
                for x in 0..w {
                    let d = me.ray(cam, x as f32 + 0.5, y as f32 + 0.5);
                    let (c, k, tz) = me.shade(cam.pos, d, &mut rng, false);
                    zrow[x] = (tz * d.dot(cam.fwd)).min(1.0e4);
                    row[x * 3] = c.x.min(60.0);
                    row[x * 3 + 1] = c.y.min(60.0);
                    row[x * 3 + 2] = c.z.min(60.0);
                    crow[x] = k;
                }
            });
        }
        self.timings.trace += t0.elapsed();
        let t1 = Instant::now();
        {
            let me = &*self;
            let cls_r = &cls;
            // pass 2: supersample only where the surface class changes (silhouettes, reflection edges)
            hdr.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
                if y == 0 || y + 1 >= h {
                    return;
                }
                let mut rng = Rng::new(y as u32 * 40503 + frame * 131 + 11);
                for x in 1..w - 1 {
                    let k = cls_r[y * w + x];
                    if cls_r[y * w + x - 1] == k && cls_r[y * w + x + 1] == k && cls_r[(y - 1) * w + x] == k && cls_r[(y + 1) * w + x] == k {
                        continue;
                    }
                    let mut acc = v(row[x * 3], row[x * 3 + 1], row[x * 3 + 2]);
                    const N: usize = 6;
                    for s in 0..N {
                        // R2 low-discrepancy jitter with per-pixel rotation
                        let a = ((s as f32 + 1.0) * 0.7548777 + rng.f()).fract();
                        let b = ((s as f32 + 1.0) * 0.5698403 + rng.f()).fract();
                        let d = me.ray(cam, x as f32 + a, y as f32 + b);
                        let c = me.shade(cam.pos, d, &mut rng, true).0;
                        acc += v(c.x.min(60.0), c.y.min(60.0), c.z.min(60.0));
                    }
                    acc = acc / (N as f32 + 1.0);
                    row[x * 3] = acc.x;
                    row[x * 3 + 1] = acc.y;
                    row[x * 3 + 2] = acc.z;
                }
            });
        }
        self.timings.aa += t1.elapsed();
        let t2 = Instant::now();
        self.post(&hdr, &zbuf, frame, out);
        self.timings.post += t2.elapsed();
        self.hdr = hdr;
        self.cls = cls;
        self.zbuf = zbuf;
    }

    fn post(&mut self, hdr: &[f32], zbuf: &[f32], frame: u32, out: &mut [u8]) {
        let (w, h) = (self.w, self.h);
        // low-res blur pyramid (L1=1/2, L2=1/4, L3=1/8): the lens blur is smooth, so it is nearly free
        let mut dims = vec![(w, h)];
        let mut lv = std::mem::take(&mut self.pyr);
        let mut dst_all = std::mem::take(&mut self.pyr_dst);
        let mut tmp_all = std::mem::take(&mut self.pyr_tmp);
        for l in 1..=3 {
            let (pw, ph) = dims[l - 1];
            let (nw, nh) = ((pw + 1) / 2, (ph + 1) / 2);
            let n = nw * nh * 3;
            {
                let dst = &mut dst_all[..n];
                let src: &[f32] = if l == 1 { hdr } else { &lv[l - 2] };
                dst.par_chunks_mut(nw * 3).enumerate().for_each(|(y, row)| {
                    let y0 = (2 * y).min(ph - 1);
                    let y1 = (2 * y + 1).min(ph - 1);
                    for x in 0..nw {
                        let x0 = (2 * x).min(pw - 1);
                        let x1 = (2 * x + 1).min(pw - 1);
                        for c in 0..3 {
                            row[x * 3 + c] = 0.25
                                * (src[(y0 * pw + x0) * 3 + c] + src[(y0 * pw + x1) * 3 + c] + src[(y1 * pw + x0) * 3 + c] + src[(y1 * pw + x1) * 3 + c]);
                        }
                    }
                });
            }
            let (head, tail) = lv.split_at_mut(l - 1);
            let _ = head;
            blur2(&dst_all[..n], &mut tail[0], &mut tmp_all[..n], nw, nh, if l == 3 { 2 } else { 1 });
            dims.push((nw, nh));
        }
        struct Lvl<'a> {
            w: usize,
            h: usize,
            sc: f32,
            d: &'a [f32],
        }
        let lvls: Vec<Lvl> = (1..=3).map(|l| Lvl { w: dims[l].0, h: dims[l].1, sc: 1.0 / (1u32 << l) as f32, d: &lv[l - 1] }).collect();
        let samp3 = |l: usize, x: usize, y: usize| -> [f32; 3] {
            // bilinear lookup in pyramid level l (1..=3) for full-res pixel (x,y), all 3 channels at once
            let lv = &lvls[l - 1];
            let fx = ((x as f32 + 0.5) * lv.sc - 0.5).clamp(0.0, (lv.w - 1) as f32);
            let fy = ((y as f32 + 0.5) * lv.sc - 0.5).clamp(0.0, (lv.h - 1) as f32);
            let (ix, iy) = (fx as usize, fy as usize);
            let (tx, ty) = (fx - ix as f32, fy - iy as f32);
            let ix1 = (ix + 1).min(lv.w - 1);
            let iy1 = (iy + 1).min(lv.h - 1);
            let (i00, i10, i01, i11) = ((iy * lv.w + ix) * 3, (iy * lv.w + ix1) * 3, (iy1 * lv.w + ix) * 3, (iy1 * lv.w + ix1) * 3);
            let (w00, w10, w01, w11) = ((1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty);
            let mut o = [0f32; 3];
            for c in 0..3 {
                // indices are clamped above, so these are always in range
                unsafe {
                    o[c] = *lv.d.get_unchecked(i00 + c) * w00 + *lv.d.get_unchecked(i10 + c) * w10 + *lv.d.get_unchecked(i01 + c) * w01 + *lv.d.get_unchecked(i11 + c) * w11;
                }
            }
            o
        };
        // sRGB LUT
        let lut = std::mem::take(&mut self.lut);
        let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
        let p = &self.p;
        let (coc_k, focus, vig, bloom, expo) = (p.coc_k, p.focus, p.vignette, p.bloom, p.exposure);
        let grade = p.grade;
        out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let dx = (x as f32 + 0.5 - cx) / cx;
                let dy = (y as f32 + 0.5 - cy) / cy;
                let rho = (dx * dx + (0.55 * dy) * (0.55 * dy)).sqrt(); // only used for the vignette
                // real thin-lens depth of field: blur-circle radius (px) from the pixel's scene depth
                let z = zbuf[y * w + x];
                let r = 0.5 * coc_k * (z - focus).abs() / z.max(1e-3);
                // pyramid levels ~ radius 0, 4, 9, 30 px
                let a = if r <= 4.0 { r * 0.25 } else if r <= 9.0 { 1.0 + (r - 4.0) * 0.2 } else { (2.0 + (r - 9.0) / 21.0).min(3.0) };
                let a = if r < 0.35 { 0.0 } else { a };
                let i = a.floor().min(2.0) as usize;
                let f = (a - i as f32).min(1.0);
                let o = (y * w + x) * 3;
                let vg = 1.0 - vig * rho * rho;
                let h1 = hash((y * w + x) as u32 ^ frame.wrapping_mul(0x9E3779B1));
                let h2 = hash(h1 ^ 0x68bc21eb);
                let bl = samp3(3, x, y);
                let mut base = [hdr[o], hdr[o + 1], hdr[o + 2]];
                if a > 1e-3 {
                    let l0 = if i == 0 { base } else { samp3(i, x, y) };
                    let l1 = samp3(i + 1, x, y);
                    for c in 0..3 {
                        base[c] = l0[c] * (1.0 - f) + l1[c] * f;
                    }
                }
                let mut sr = [0f32; 3];
                for c in 0..3 {
                    let col = base[c] + bloom * bl[c];
                    let t = ((col * vg * expo) * (1.0 / 16.0)).max(0.0).sqrt().min(1.0) * 4095.0;
                    let ti = (t as usize).min(4095);
                    let tf = t - ti as f32;
                    sr[c] = lut[ti] * (1.0 - tf) + lut[ti + 1] * tf;
                }
                if grade > 0.0 {
                    // gentle cinematic grade: cool shadows, warm highlights, soft S-curve
                    let luma = 0.2126 * sr[0] + 0.7152 * sr[1] + 0.0722 * sr[2];
                    let (sh, hi) = ((1.0 - luma) * (1.0 - luma), luma * luma);
                    sr[0] += grade * (-0.004 * sh + 0.010 * hi);
                    sr[1] += grade * (0.004 * sh + 0.003 * hi);
                    sr[2] += grade * (0.010 * sh - 0.010 * hi);
                    for c in 0..3 {
                        let q = sr[c].clamp(0.0, 1.0);
                        sr[c] = q + grade * 0.18 * (q * q * (3.0 - 2.0 * q) - q);
                    }
                }
                for c in 0..3 {
                    let n1 = ((h1 >> (10 * c)) & 1023) as f32 / 1023.0;
                    let n2 = ((h2 >> (10 * c)) & 1023) as f32 / 1023.0;
                    let dither = (n1 + n2 - 1.0) / 255.0; // triangular, +-1 LSB
                    row[x * 3 + c] = ((sr[c] + dither) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        });
        self.lut = lut;
        self.pyr = lv;
        self.pyr_dst = dst_all;
        self.pyr_tmp = tmp_all;
    }
}
