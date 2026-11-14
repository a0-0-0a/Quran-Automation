//! The scene is a heightfield: flat floor (z=0) + extruded text slab (z=0..h).
//! The slab footprint is stored as a signed-distance grid (sub-texel exact via coverage),
//! so every ray query (camera, shadow, bounce) is a short 2D sphere-trace. No triangles.

use crate::text::Mesh;
use crate::vec::*;

pub const SKY: u8 = 0;
pub const FLOOR: u8 = 1;
pub const TOP: u8 = 2;
pub const WALL: u8 = 3;

#[derive(Clone, Copy)]
pub struct Hit {
    pub kind: u8,
    pub t: f32,
    pub p: V,
    pub n: V,
}

/// A hanging spot light: position + (tilting) axis, small emitting disc, smooth cone.
#[derive(Clone, Copy)]
pub struct Light {
    pub pos: V,
    pub axis: V, // unit, direction the light points (down the wire)
    pub t1: V,   // disc tangent basis
    pub t2: V,
    pub rad: f32,
    pub cos_in: f32,
    pub cos_out: f32,
    pub cx: f32, // where the axis meets the floor
    pub cy: f32,
    pub reach2: f32, // squared xy radius beyond which the cone cannot reach any surface
}
impl Light {
    /// `r_in`/`r_out`: pool radii (inner full-intensity / outer zero) measured at `rest_h` below the lamp.
    pub fn new(pos: V, axis: V, rad: f32, r_in: f32, r_out: f32, rest_h: f32) -> Light {
        let axis = axis.norm();
        let (t1, t2) = onb(axis);
        let ang = |r: f32| (r / rest_h).atan().cos();
        let (ci, co) = (ang(r_in), ang(r_out));
        let tan_o = (1.0 - co * co).sqrt() / co;
        let dist = pos.z / (-axis.z).max(1e-3);
        let c = pos + axis * dist;
        let reach = 1.3 * dist * tan_o + rad + 0.02;
        Light { pos, axis, t1, t2, rad, cos_in: ci, cos_out: co, cx: c.x, cy: c.y, reach2: reach * reach }
    }
}

pub struct Scene {
    pub h: f32,
    pub lights: Vec<Light>,
    pub i0: f32,
    pub lcol: V,
    pub gx0: f32,
    pub gy0: f32,
    pub gx1: f32,
    pub gy1: f32,
    pub ds: f32,
    inv_ds: f32,
    nx: usize,
    ny: usize,
    sdf: Vec<f32>,
    nrm: Vec<[f32; 2]>, // smoothed outward wall normal per texel
    margin: f32,
    eps: f32,
    minstep: f32,
}

// ---------- coverage rasteriser (exact area, font-rs style accumulation) ----------
fn draw_line(a: &mut [f32], w: usize, h: usize, p0: (f32, f32), p1: (f32, f32)) {
    if p0.1 == p1.1 {
        return;
    }
    let (dir, p0, p1) = if p0.1 < p1.1 { (1.0, p0, p1) } else { (-1.0, p1, p0) };
    let dxdy = (p1.0 - p0.0) / (p1.1 - p0.1);
    let mut x = p0.0;
    let y0 = p0.1 as usize;
    if p0.1 < 0.0 {
        x -= p0.1 * dxdy;
    }
    for y in y0..h.min(p1.1.ceil() as usize) {
        let ls = y * w;
        let dy = ((y + 1) as f32).min(p1.1) - (y as f32).max(p0.1);
        let xn = x + dxdy * dy;
        let d = dy * dir;
        let (x0, x1) = if x < xn { (x, xn) } else { (xn, x) };
        let x0f = x0.floor();
        let x0i = x0f as i32;
        let x1c = x1.ceil();
        let x1i = x1c as i32;
        if x1i <= x0i + 1 {
            let xmf = 0.5 * (x + xn) - x0f;
            a[ls + x0i as usize] += d - d * xmf;
            a[ls + (x0i + 1) as usize] += d * xmf;
        } else {
            let s = 1.0 / (x1 - x0);
            let x0r = x0 - x0f;
            let a0 = 0.5 * s * (1.0 - x0r) * (1.0 - x0r);
            let x1r = x1 - x1c + 1.0;
            let am = 0.5 * s * x1r * x1r;
            a[ls + x0i as usize] += d * a0;
            if x1i == x0i + 2 {
                a[ls + (x0i + 1) as usize] += d * (1.0 - a0 - am);
            } else {
                let a1 = s * (1.5 - x0r);
                a[ls + (x0i + 1) as usize] += d * (a1 - a0);
                for xi in x0i + 2..x1i - 1 {
                    a[ls + xi as usize] += d * s;
                }
                let a2 = a1 + (x1i - x0i - 3) as f32 * s;
                a[ls + (x1i - 1) as usize] += d * (1.0 - a2 - am);
            }
            a[ls + x1i as usize] += d * am;
        }
        x = xn;
    }
}

// ---------- Felzenszwalb exact euclidean distance transform (squared) ----------
fn dt1d(f: &[f32], d: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    let mut k = 0usize;
    v[0] = 0;
    z[0] = -1e20;
    z[1] = 1e20;
    for q in 1..n {
        let mut s;
        loop {
            let p = v[k];
            s = ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32)) / (2.0 * (q as f32 - p as f32));
            if s <= z[k] && k > 0 {
                k -= 1;
            } else {
                break;
            }
        }
        if s <= z[k] {
            v[0] = q;
            z[0] = -1e20;
            z[1] = 1e20;
            k = 0;
        } else {
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = 1e20;
        }
    }
    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let p = v[k];
        let dq = q as f32 - p as f32;
        d[q] = dq * dq + f[p];
    }
}

fn edt2(sq: &mut [f32], w: usize, h: usize) {
    let n = w.max(h);
    let mut f = vec![0f32; n];
    let mut d = vec![0f32; n];
    let mut v = vec![0usize; n];
    let mut z = vec![0f32; n + 2];
    for x in 0..w {
        for y in 0..h {
            f[y] = sq[y * w + x];
        }
        dt1d(&f[..h], &mut d[..h], &mut v, &mut z);
        for y in 0..h {
            sq[y * w + x] = d[y];
        }
    }
    for y in 0..h {
        f[..w].copy_from_slice(&sq[y * w..(y + 1) * w]);
        dt1d(&f[..w], &mut d[..w], &mut v, &mut z);
        sq[y * w..(y + 1) * w].copy_from_slice(&d[..w]);
    }
}

impl Scene {
    pub fn build(mesh: &Mesh, h: f32, ds: f32, margin: f32) -> Scene {
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for i in 0..mesh.vx.len() {
            x0 = x0.min(mesh.vx[i]);
            x1 = x1.max(mesh.vx[i]);
            y0 = y0.min(mesh.vy[i]);
            y1 = y1.max(mesh.vy[i]);
        }
        let gx0 = x0 - margin;
        let gy0 = y0 - margin;
        let nx = ((x1 - x0 + 2.0 * margin) / ds).ceil() as usize + 2;
        let ny = ((y1 - y0 + 2.0 * margin) / ds).ceil() as usize + 2;
        let inv = 1.0 / ds;

        // 1) exact coverage
        let mut acc = vec![0f32; nx * ny + nx + 8];
        for i in 0..mesh.vx.len() {
            let j = mesh.next[i] as usize;
            let p0 = ((mesh.vx[i] - gx0) * inv, (mesh.vy[i] - gy0) * inv);
            let p1 = ((mesh.vx[j] - gx0) * inv, (mesh.vy[j] - gy0) * inv);
            draw_line(&mut acc, nx, ny, p0, p1);
        }
        let mut cov = vec![0f32; nx * ny];
        let mut run = 0.0f32;
        for i in 0..nx * ny {
            run += acc[i];
            cov[i] = run.abs().min(1.0);
        }
        drop(acc);

        // 2) coarse signed distance from the binary mask (texels)
        let big = 1e12f32;
        let mut d_out: Vec<f32> = cov.iter().map(|&c| if c >= 0.5 { 0.0 } else { big }).collect();
        let mut d_in: Vec<f32> = cov.iter().map(|&c| if c >= 0.5 { big } else { 0.0 }).collect();
        edt2(&mut d_out, nx, ny);
        edt2(&mut d_in, nx, ny);
        let mut sdf = vec![0f32; nx * ny];
        for i in 0..nx * ny {
            let inside = cov[i] >= 0.5;
            let s = if inside { -(d_in[i].sqrt() - 0.5) } else { d_out[i].sqrt() - 0.5 };
            sdf[i] = s;
        }
        // 3) sub-texel refinement inside the anti-aliased band
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let i = y * nx + x;
                let c = cov[i];
                if c > 0.0 && c < 1.0 {
                    let gx = cov[i + 1] - cov[i - 1];
                    let gy = cov[i + nx] - cov[i - nx];
                    let g = (gx * gx + gy * gy).sqrt();
                    if g > 1e-4 {
                        let nmax = gx.abs().max(gy.abs()) / g;
                        sdf[i] = (0.5 - c) * nmax;
                    }
                }
            }
        }
        for s in sdf.iter_mut() {
            *s *= ds;
        }
        // smooth wall normals: gradient of a lightly blurred distance field (2 passes of [1 2 1], ~1 mm)
        let mut sm = sdf.clone();
        let mut tmp = vec![0f32; nx * ny];
        for _ in 0..2 {
            for y in 0..ny {
                for x in 0..nx {
                    let l = sm[y * nx + x.saturating_sub(1)];
                    let r = sm[y * nx + (x + 1).min(nx - 1)];
                    tmp[y * nx + x] = 0.25 * l + 0.5 * sm[y * nx + x] + 0.25 * r;
                }
            }
            for y in 0..ny {
                let (u, d) = (y.saturating_sub(1), (y + 1).min(ny - 1));
                for x in 0..nx {
                    sm[y * nx + x] = 0.25 * tmp[u * nx + x] + 0.5 * tmp[y * nx + x] + 0.25 * tmp[d * nx + x];
                }
            }
        }
        let mut nrm = vec![[0f32; 2]; nx * ny];
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let gx = sm[y * nx + x + 1] - sm[y * nx + x - 1];
                let gy = sm[(y + 1) * nx + x] - sm[(y - 1) * nx + x];
                let l = (gx * gx + gy * gy).sqrt();
                if l > 1e-12 {
                    nrm[y * nx + x] = [gx / l, gy / l];
                }
            }
        }
        // True outline normals: per-vertex normals (averaged across smooth joins, kept split at real corners),
        // interpolated along every edge and splatted into the grid with a small gaussian -> smooth, facet-free walls.
        {
            let ne = mesh.vx.len();
            let mut prev = vec![0usize; ne];
            for i in 0..ne {
                prev[mesh.next[i] as usize] = i;
            }
            let cos_corner = 0.80f32; // joins sharper than ~37 degrees stay corners
            let end_normal = |edge: usize, vtx: usize| -> (f32, f32) {
                let (ax, ay) = (mesh.nx[prev[vtx]], mesh.ny[prev[vtx]]);
                let (bx, by) = (mesh.nx[vtx], mesh.ny[vtx]);
                if ax * bx + ay * by > cos_corner {
                    let (sx, sy) = (ax + bx, ay + by);
                    let l = (sx * sx + sy * sy).sqrt().max(1e-9);
                    (sx / l, sy / l)
                } else {
                    (mesh.nx[edge], mesh.ny[edge])
                }
            };
            let mut acc = vec![[0f32; 3]; nx * ny];
            let sigma2 = 1.2f32 * 1.2;
            for i in 0..ne {
                let j = mesh.next[i] as usize;
                let (x0, y0) = ((mesh.vx[i] - gx0) * inv, (mesh.vy[i] - gy0) * inv);
                let (x1, y1) = ((mesh.vx[j] - gx0) * inv, (mesh.vy[j] - gy0) * inv);
                let (n0, n1) = (end_normal(i, i), end_normal(i, j));
                let l = ((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)).sqrt();
                let steps = (l * 2.0).ceil().max(1.0) as usize;
                for st in 0..=steps {
                    let t = st as f32 / steps as f32;
                    let (px, py) = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
                    let (mut qx, mut qy) = (n0.0 * (1.0 - t) + n1.0 * t, n0.1 * (1.0 - t) + n1.1 * t);
                    let ql = (qx * qx + qy * qy).sqrt().max(1e-9);
                    qx /= ql;
                    qy /= ql;
                    let (cx, cy) = (px.floor() as i32, py.floor() as i32);
                    for yy in cy - 3..=cy + 3 {
                        for xx in cx - 3..=cx + 3 {
                            if xx < 0 || yy < 0 || xx >= nx as i32 || yy >= ny as i32 {
                                continue;
                            }
                            let (dx, dy) = (xx as f32 + 0.5 - px, yy as f32 + 0.5 - py);
                            let w = (-(dx * dx + dy * dy) / (2.0 * sigma2)).exp();
                            let a = &mut acc[yy as usize * nx + xx as usize];
                            a[0] += qx * w;
                            a[1] += qy * w;
                            a[2] += w;
                        }
                    }
                }
            }
            for k in 0..nx * ny {
                if acc[k][2] > 0.02 {
                    let l = (acc[k][0] * acc[k][0] + acc[k][1] * acc[k][1]).sqrt();
                    if l > 1e-6 {
                        nrm[k] = [acc[k][0] / l, acc[k][1] / l];
                    }
                }
            }
        }
        Scene {
            h,
            lights: vec![],
            i0: 1.0,
            lcol: v(1.0, 1.0, 1.0),
            gx0,
            gy0,
            gx1: gx0 + (nx - 1) as f32 * ds,
            gy1: gy0 + (ny - 1) as f32 * ds,
            ds,
            inv_ds: inv,
            nx,
            ny,
            sdf,
            nrm,
            margin,
            eps: ds * 0.02,
            minstep: ds * 0.25,
        }
    }

    #[inline(always)]
    pub fn sdf_at(&self, x: f32, y: f32) -> f32 {
        let fx = (x - self.gx0) * self.inv_ds - 0.5;
        let fy = (y - self.gy0) * self.inv_ds - 0.5;
        if fx < 0.0 || fy < 0.0 || fx >= (self.nx - 2) as f32 || fy >= (self.ny - 2) as f32 {
            let dx = (self.gx0 - x).max(x - self.gx1).max(0.0);
            let dy = (self.gy0 - y).max(y - self.gy1).max(0.0);
            return (dx * dx + dy * dy).sqrt() + self.margin * 0.9;
        }
        let ix = fx as usize;
        let iy = fy as usize;
        let tx = fx - ix as f32;
        let ty = fy - iy as f32;
        let i = iy * self.nx + ix;
        let a = self.sdf[i] + (self.sdf[i + 1] - self.sdf[i]) * tx;
        let b = self.sdf[i + self.nx] + (self.sdf[i + self.nx + 1] - self.sdf[i + self.nx]) * tx;
        a + (b - a) * ty
    }

    /// Outward horizontal normal of the slab wall at (x,y) (bilinear lookup in the smoothed normal map).
    pub fn grad(&self, x: f32, y: f32) -> V {
        let fx = ((x - self.gx0) * self.inv_ds - 0.5).clamp(0.0, (self.nx - 2) as f32 - 1e-3);
        let fy = ((y - self.gy0) * self.inv_ds - 0.5).clamp(0.0, (self.ny - 2) as f32 - 1e-3);
        let (ix, iy) = (fx as usize, fy as usize);
        let (tx, ty) = (fx - ix as f32, fy - iy as f32);
        let i = iy * self.nx + ix;
        let (a, b, c, d) = (self.nrm[i], self.nrm[i + 1], self.nrm[i + self.nx], self.nrm[i + self.nx + 1]);
        let gx = (a[0] * (1.0 - tx) + b[0] * tx) * (1.0 - ty) + (c[0] * (1.0 - tx) + d[0] * tx) * ty;
        let gy = (a[1] * (1.0 - tx) + b[1] * tx) * (1.0 - ty) + (c[1] * (1.0 - tx) + d[1] * tx) * ty;
        let l = (gx * gx + gy * gy).sqrt().max(1e-12);
        v(gx / l, gy / l, 0.0)
    }

    fn clip(&self, x0: f32, y0: f32, dx: f32, dy: f32) -> Option<(f32, f32)> {
        let (mut t0, mut t1) = (0.0f32, 1.0f32);
        for (p, dp, lo, hi) in [(x0, dx, self.gx0, self.gx1), (y0, dy, self.gy0, self.gy1)] {
            if dp.abs() < 1e-12 {
                if p < lo || p > hi {
                    return None;
                }
            } else {
                let a = (lo - p) / dp;
                let b = (hi - p) / dp;
                let (a, b) = if a < b { (a, b) } else { (b, a) };
                t0 = t0.max(a);
                t1 = t1.min(b);
                if t0 > t1 {
                    return None;
                }
            }
        }
        Some((t0, t1))
    }

    /// First point (as fraction of the segment) where the 2D segment enters the slab footprint.
    pub fn march2d(&self, x0: f32, y0: f32, x1: f32, y1: f32) -> Option<f32> {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            return if self.sdf_at(x0, y0) <= self.eps { Some(0.0) } else { None };
        }
        let (t0, t1) = self.clip(x0, y0, dx, dy)?;
        let (ux, uy) = (dx / len, dy / len);
        let mut u = t0 * len;
        let end = t1 * len;
        for _ in 0..80 {
            let s = self.sdf_at(x0 + ux * u, y0 + uy * u);
            if s <= self.eps {
                return Some(u / len);
            }
            u += s.max(self.minstep);
            if u > end {
                return if self.sdf_at(x0 + ux * end, y0 + uy * end) <= self.eps { Some(end / len) } else { None };
            }
        }
        None
    }

    fn wall_hit(&self, o: V, d: V, t: f32) -> Hit {
        let p = o + d * t;
        Hit { kind: WALL, t, p, n: self.grad(p.x, p.y) }
    }

    pub fn intersect(&self, o: V, d: V) -> Hit {
        let h = self.h;
        let sky = Hit { kind: SKY, t: 1e9, p: o, n: UP };
        if o.z >= h {
            if d.z >= -1e-7 {
                return sky;
            }
            let ta = (h - o.z) / d.z;
            let a = o + d * ta;
            let sa = self.sdf_at(a.x, a.y);
            if sa <= 0.0 {
                return Hit { kind: TOP, t: ta, p: a, n: UP };
            }
            let tb = -o.z / d.z;
            let seg = (tb - ta) * (d.x * d.x + d.y * d.y).sqrt();
            if sa < seg {
                if let Some(u) = self.march2d(a.x, a.y, o.x + d.x * tb, o.y + d.y * tb) {
                    return self.wall_hit(o, d, ta + (tb - ta) * u);
                }
            }
            let p = o + d * tb;
            return Hit { kind: FLOOR, t: tb, p: v(p.x, p.y, 0.0), n: UP };
        }
        if d.z < -1e-7 {
            let tb = -o.z / d.z;
            let b = o + d * tb;
            if let Some(u) = self.march2d(o.x, o.y, b.x, b.y) {
                return self.wall_hit(o, d, tb * u);
            }
            return Hit { kind: FLOOR, t: tb, p: v(b.x, b.y, 0.0), n: UP };
        }
        if d.z > 1e-7 {
            let ta = (h - o.z) / d.z;
            let a = o + d * ta;
            if let Some(u) = self.march2d(o.x, o.y, a.x, a.y) {
                return self.wall_hit(o, d, ta * u);
            }
            return sky;
        }
        let e = o + d * 2.0;
        if let Some(u) = self.march2d(o.x, o.y, e.x, e.y) {
            return self.wall_hit(o, d, 2.0 * u);
        }
        sky
    }

    pub fn occluded(&self, o: V, d: V, maxt: f32) -> bool {
        if o.z >= self.h {
            return false;
        }
        if d.z <= 1e-6 {
            let e = o + d * maxt;
            return self.march2d(o.x, o.y, e.x, e.y).is_some();
        }
        let ta = ((self.h - o.z) / d.z).min(maxt);
        // nearest solid farther than the whole horizontal segment -> cannot be shadowed (cheap exit)
        if self.sdf_at(o.x, o.y) > ta * (d.x * d.x + d.y * d.y).sqrt() + self.ds {
            return false;
        }
        let a = o + d * ta;
        self.march2d(o.x, o.y, a.x, a.y).is_some()
    }

    /// Direct irradiance (RGB) at p with normal n from all spot lights; area lights, soft shadows.
    pub fn direct(&self, p: V, n: V, ns: usize, rng: &mut Rng, shadow: bool) -> V {
        let mut e = 0.0f32;
        for l in &self.lights {
            let w0 = l.pos - p;
            let r0 = w0.len();
            if -w0.dot(l.axis) / r0 <= l.cos_out - 0.03 {
                continue;
            }
            let rot = rng.f() * 6.2831853;
            let mut acc = 0.0f32;
            for k in 0..ns {
                let rr = l.rad * ((k as f32 + 0.5) / ns as f32).sqrt();
                let th = k as f32 * 2.3999632 + rot;
                let q = l.pos + l.t1 * (rr * th.cos()) + l.t2 * (rr * th.sin());
                let w = q - p;
                let r2 = w.dot(w);
                let r = r2.sqrt();
                let wn = w / r;
                let cosp = n.dot(wn);
                if cosp <= 0.0 {
                    continue;
                }
                let fall = smoothstep(l.cos_out, l.cos_in, -wn.dot(l.axis));
                if fall <= 0.0 {
                    continue;
                }
                if shadow && self.occluded(p, wn, r) {
                    continue;
                }
                acc += fall * cosp / r2;
            }
            e += acc / ns as f32;
        }
        self.lcol * (e * self.i0)
    }
}
