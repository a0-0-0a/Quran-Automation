//! BAKE ONCE. Path-traces the whole static scene into irradiance maps:
//!   floor (G), text top (T), text walls (W)
//! Pass 0: direct light from the area spot lights (stratified disc samples, soft shadows).
//! Pass 1..B: multi-bounce path tracing: cosine-sampled rays gather the *previous* bounce's
//! baked radiance (radiance cache), so each extra bounce costs one short ray, not a full path.
//! Nothing here runs per frame.

use crate::scene::*;
use crate::vec::*;
use rayon::prelude::*;

pub struct Maps {
    pub x0: f32,
    pub y0: f32,
    pub t: f32,
    pub w: usize,
    pub h: usize,
    pub g: Vec<V>,
    pub tp: Vec<V>,
    pub wl: Vec<V>,
}

#[inline]
fn bil(a: &[V], w: usize, h: usize, x0: f32, y0: f32, t: f32, x: f32, y: f32) -> V {
    let fx = (x - x0) / t - 0.5;
    let fy = (y - y0) / t - 0.5;
    if fx < 0.0 || fy < 0.0 || fx >= (w - 1) as f32 || fy >= (h - 1) as f32 {
        return V0;
    }
    let ix = fx as usize;
    let iy = fy as usize;
    let tx = fx - ix as f32;
    let ty = fy - iy as f32;
    let i = iy * w + ix;
    let r0 = a[i] * (1.0 - tx) + a[i + 1] * tx;
    let r1 = a[i + w] * (1.0 - tx) + a[i + w + 1] * tx;
    r0 * (1.0 - ty) + r1 * ty
}

/// Bilinear lookup with edge clamping: outside the baked grid the border value
/// is extended, so the floor's ambient continues seamlessly to the frame edges.
#[inline]
fn bil_clamp(a: &[V], w: usize, h: usize, x0: f32, y0: f32, t: f32, x: f32, y: f32) -> V {
    // Keep ix <= w-2 and iy <= h-2 so i+w+1 is always a valid index.
    if w < 2 || h < 2 || a.len() < w * h || t <= 0.0 {
        return V0;
    }
    let fx = ((x - x0) / t - 0.5).clamp(0.0, (w - 2) as f32);
    let fy = ((y - y0) / t - 0.5).clamp(0.0, (h - 2) as f32);
    let ix = fx as usize;
    let iy = fy as usize;
    let tx = fx - ix as f32;
    let ty = fy - iy as f32;
    let i = iy * w + ix;
    let r0 = a[i] * (1.0 - tx) + a[i + 1] * tx;
    let r1 = a[i + w] * (1.0 - tx) + a[i + w + 1] * tx;
    r0 * (1.0 - ty) + r1 * ty
}

impl Maps {
    #[inline]
    pub fn floor(&self, x: f32, y: f32) -> V {
        bil(&self.g, self.w, self.h, self.x0, self.y0, self.t, x, y)
    }
    /// Floor irradiance with the border extended (no black outside the bake).
    #[inline]
    pub fn floor_clamped(&self, x: f32, y: f32) -> V {
        bil_clamp(&self.g, self.w, self.h, self.x0, self.y0, self.t, x, y)
    }
    #[inline]
    pub fn top(&self, x: f32, y: f32) -> V {
        bil(&self.tp, self.w, self.h, self.x0, self.y0, self.t, x, y)
    }
    #[inline]
    pub fn wall(&self, x: f32, y: f32) -> V {
        bil(&self.wl, self.w, self.h, self.x0, self.y0, self.t, x, y)
    }
}

pub struct BakeCfg {
    pub texel: f32,
    pub margin: f32,
    pub bounces: usize,
    pub spp: usize,
    pub light_samples: usize,
    pub floor_albedo: V,
    pub text_albedo: V,
    pub ambient: f32,
}

fn dilate(a: &mut [V], valid: &mut [bool], w: usize, h: usize, iters: usize) {
    for _ in 0..iters {
        let src = a.to_vec();
        let vs = valid.to_vec();
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let i = y * w + x;
                if vs[i] {
                    continue;
                }
                let mut s = V0;
                let mut c = 0.0;
                for (dx, dy) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)] {
                    let j = (y as i32 + dy) as usize * w + (x as i32 + dx) as usize;
                    if vs[j] {
                        s += src[j];
                        c += 1.0;
                    }
                }
                if c > 0.0 {
                    a[i] = s / c;
                    valid[i] = true;
                }
            }
        }
    }
}

fn smooth_masked(a: &mut [V], mask: &[bool], w: usize, h: usize, r: i32) {
    let src = a.to_vec();
    let out: Vec<V> = (0..w * h)
        .into_par_iter()
        .map(|i| {
            if !mask[i] {
                return src[i];
            }
            let (x, y) = ((i % w) as i32, (i / w) as i32);
            let mut s = V0;
            let mut c = 0.0f32;
            for dy in -r..=r {
                for dx in -r..=r {
                    let (xx, yy) = (x + dx, y + dy);
                    if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                        continue;
                    }
                    let j = yy as usize * w + xx as usize;
                    if mask[j] {
                        s += src[j];
                        c += 1.0;
                    }
                }
            }
            s / c.max(1.0)
        })
        .collect();
    a.copy_from_slice(&out);
}

/// `bb` = text bounding box (x0, y0, x1, y1). The returned maps hold ONLY ambient + indirect bounce light
/// (smooth, weak, static); direct light and its shadows are evaluated per frame because the spots move.
pub fn bake(sc: &Scene, bb: (f32, f32, f32, f32), cfg: &BakeCfg) -> Maps {
    let t = cfg.texel;
    let x0 = bb.0 - cfg.margin;
    let y0 = bb.1 - cfg.margin;
    let w = ((bb.2 - bb.0 + 2.0 * cfg.margin) / t).ceil() as usize;
    let h = ((bb.3 - bb.1 + 2.0 * cfg.margin) / t).ceil() as usize;
    let hh = sc.h;
    let band = 3.0 * t;
    let amb = V { x: cfg.ambient, y: cfg.ambient, z: cfg.ambient };

    let mut g_dir = vec![V0; w * h];
    let mut tp = vec![V0; w * h];
    let mut wl_dir = vec![V0; w * h];
    let mut vg = vec![false; w * h];
    let mut vt = vec![false; w * h];

    // ---- pass 0: direct light (area spot lights, soft shadows) ----
    let ns = cfg.light_samples;
    {
        let rows: Vec<(Vec<V>, Vec<V>, Vec<V>, Vec<bool>, Vec<bool>)> = (0..h)
            .into_par_iter()
            .map(|j| {
                let mut rng = Rng::new(j as u32 * 7919 + 1);
                let mut rg = vec![V0; w];
                let mut rt = vec![V0; w];
                let mut rw = vec![V0; w];
                let mut fg = vec![false; w];
                let mut ft = vec![false; w];
                let y = y0 + (j as f32 + 0.5) * t;
                for i in 0..w {
                    let x = x0 + (i as f32 + 0.5) * t;
                    let s = sc.sdf_at(x, y);
                    if s > 0.0004 {
                        rg[i] = sc.direct(v(x, y, 1e-4), UP, ns, &mut rng, true);
                        fg[i] = true;
                    } else if s < -0.0004 {
                        rt[i] = sc.direct(v(x, y, hh + 1e-4), UP, ns, &mut rng, false);
                        ft[i] = true;
                    }
                    if s.abs() < band {
                        let n = sc.grad(x, y);
                        let px = x - n.x * s;
                        let py = y - n.y * s;
                        let p = v(px + n.x * 3e-4, py + n.y * 3e-4, hh * 0.5);
                        rw[i] = sc.direct(p, n, ns, &mut rng, true);
                    }
                }
                (rg, rt, rw, fg, ft)
            })
            .collect();
        for (j, (rg, rt, rw, fg, ft)) in rows.into_iter().enumerate() {
            g_dir[j * w..(j + 1) * w].copy_from_slice(&rg);
            tp[j * w..(j + 1) * w].copy_from_slice(&rt);
            wl_dir[j * w..(j + 1) * w].copy_from_slice(&rw);
            vg[j * w..(j + 1) * w].copy_from_slice(&fg);
            vt[j * w..(j + 1) * w].copy_from_slice(&ft);
        }
    }
    // fill texels under/outside the slab so bilinear taps near walls never leak black
    dilate(&mut g_dir, &mut vg, w, h, 3);
    dilate(&mut tp, &mut vt, w, h, 3);
    for p in g_dir.iter_mut().chain(tp.iter_mut()) {
        *p += amb;
    }
    // wall texels live in a thin band around the outline; denoise within it
    let wmask: Vec<bool> = (0..w * h)
        .map(|i| {
            let x = x0 + ((i % w) as f32 + 0.5) * t;
            let y = y0 + ((i / w) as f32 + 0.5) * t;
            sc.sdf_at(x, y).abs() < band
        })
        .collect();
    smooth_masked(&mut wl_dir, &wmask, w, h, 2);
    for p in wl_dir.iter_mut() {
        *p += amb;
    }

    let mut maps = Maps { x0, y0, t, w, h, g: g_dir.clone(), tp: tp.clone(), wl: wl_dir.clone() };
    let mut g_ind = vec![V0; w * h];
    let mut w_ind = vec![V0; w * h];

    // ---- passes 1..B: path-traced bounces against the radiance cache ----
    let k = 4usize; // floor indirect is smooth: solve at 4x coarser grid
    let cw = (w + k - 1) / k + 1;
    let ch = (h + k - 1) / k + 1;
    let ct = t * k as f32;
    for _ in 0..cfg.bounces {
        let mp = &maps;
        let gather = |p: V, n: V, rng: &mut Rng| -> V {
            let mut acc = V0;
            let o = p + n * 2e-4;
            for _ in 0..cfg.spp {
                let d = cos_hemisphere(n, rng);
                let hit = sc.intersect(o, d);
                match hit.kind {
                    FLOOR => acc += cfg.floor_albedo * mp.floor(hit.p.x, hit.p.y),
                    TOP => acc += cfg.text_albedo * mp.top(hit.p.x, hit.p.y),
                    WALL => acc += cfg.text_albedo * mp.wall(hit.p.x, hit.p.y),
                    _ => {}
                }
            }
            acc / cfg.spp as f32
        };
        // floor (coarse, only close to the slab where bounce light matters)
        let crow: Vec<Vec<V>> = (0..ch)
            .into_par_iter()
            .map(|j| {
                let mut rng = Rng::new(j as u32 * 104729 + 17);
                let y = y0 + (j as f32 + 0.5) * ct;
                (0..cw)
                    .map(|i| {
                        let x = x0 + (i as f32 + 0.5) * ct;
                        let s = sc.sdf_at(x, y);
                        if s > 0.0004 && s < 0.08 {
                            gather(v(x, y, 1e-4), UP, &mut rng)
                        } else {
                            V0
                        }
                    })
                    .collect()
            })
            .collect();
        let cflat: Vec<V> = crow.into_iter().flatten().collect();
        // walls (fine)
        let wrows: Vec<Vec<V>> = (0..h)
            .into_par_iter()
            .map(|j| {
                let mut rng = Rng::new(j as u32 * 15485863 + 3);
                let y = y0 + (j as f32 + 0.5) * t;
                (0..w)
                    .map(|i| {
                        let x = x0 + (i as f32 + 0.5) * t;
                        let s = sc.sdf_at(x, y);
                        if s.abs() < band {
                            let n = sc.grad(x, y);
                            gather(v(x - n.x * s, y - n.y * s, hh * 0.5), n, &mut rng)
                        } else {
                            V0
                        }
                    })
                    .collect()
            })
            .collect();
        let mut wrows_flat: Vec<V> = wrows.into_iter().flatten().collect();
        smooth_masked(&mut wrows_flat, &wmask, w, h, 3);
        let mut g_new = g_dir.clone();
        let mut w_new = wl_dir.clone();
        for j in 0..h {
            for i in 0..w {
                let x = x0 + (i as f32 + 0.5) * t;
                let y = y0 + (j as f32 + 0.5) * t;
                g_new[j * w + i] += bil(&cflat, cw, ch, x0, y0, ct, x, y);
                w_new[j * w + i] += wrows_flat[j * w + i];
            }
        }
        for j in 0..h {
            for i in 0..w {
                g_ind[j * w + i] = g_new[j * w + i] - g_dir[j * w + i];
                w_ind[j * w + i] = w_new[j * w + i] - wl_dir[j * w + i];
            }
        }
        maps.g = g_new;
        maps.wl = w_new;
    }
    // final maps: ambient + indirect only
    let fin = |ind: &Vec<V>| -> Vec<V> { ind.iter().map(|&p| p + amb).collect() };
    Maps { x0, y0, t, w, h, g: fin(&g_ind), tp: vec![amb; w * h], wl: fin(&w_ind) }
}
