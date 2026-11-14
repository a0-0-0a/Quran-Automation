//! Tiny math helpers (no external deps).
use std::ops::*;

pub const PI: f32 = std::f32::consts::PI;

#[derive(Clone, Copy, Debug, Default)]
pub struct V {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
pub const V0: V = V { x: 0.0, y: 0.0, z: 0.0 };
pub const UP: V = V { x: 0.0, y: 0.0, z: 1.0 };
#[inline(always)]
pub const fn v(x: f32, y: f32, z: f32) -> V {
    V { x, y, z }
}
impl Add for V {
    type Output = V;
    #[inline(always)]
    fn add(self, o: V) -> V {
        v(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for V {
    type Output = V;
    #[inline(always)]
    fn sub(self, o: V) -> V {
        v(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f32> for V {
    type Output = V;
    #[inline(always)]
    fn mul(self, s: f32) -> V {
        v(self.x * s, self.y * s, self.z * s)
    }
}
impl Mul<V> for V {
    type Output = V;
    #[inline(always)]
    fn mul(self, o: V) -> V {
        v(self.x * o.x, self.y * o.y, self.z * o.z)
    }
}
impl Div<f32> for V {
    type Output = V;
    #[inline(always)]
    fn div(self, s: f32) -> V {
        let i = 1.0 / s;
        v(self.x * i, self.y * i, self.z * i)
    }
}
impl Neg for V {
    type Output = V;
    #[inline(always)]
    fn neg(self) -> V {
        v(-self.x, -self.y, -self.z)
    }
}
impl AddAssign for V {
    #[inline(always)]
    fn add_assign(&mut self, o: V) {
        *self = *self + o;
    }
}
impl V {
    #[inline(always)]
    pub fn dot(self, o: V) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    #[inline(always)]
    pub fn cross(self, o: V) -> V {
        v(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }
    #[inline(always)]
    pub fn len(self) -> f32 {
        self.dot(self).sqrt()
    }
    #[inline(always)]
    pub fn norm(self) -> V {
        self / self.len().max(1e-20)
    }
    #[inline(always)]
    pub fn max_c(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }
}

#[inline(always)]
pub fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Orthonormal basis around n (Duff et al.).
#[inline]
pub fn onb(n: V) -> (V, V) {
    let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    (v(1.0 + s * n.x * n.x * a, s * b, -s * n.x), v(b, s + n.y * n.y * a, -n.y))
}

pub struct Rng(u32);
#[inline(always)]
pub fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}
impl Rng {
    pub fn new(seed: u32) -> Rng {
        Rng(hash(seed ^ 0x9e3779b9) | 1)
    }
    #[inline(always)]
    pub fn f(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        (x >> 8) as f32 * (1.0 / 16777216.0)
    }
}

#[inline]
pub fn cos_hemisphere(n: V, r: &mut Rng) -> V {
    let (r1, r2) = (r.f(), r.f());
    let phi = 2.0 * PI * r1;
    let rr = r2.sqrt();
    let (t, b) = onb(n);
    (t * (rr * phi.cos()) + b * (rr * phi.sin()) + n * (1.0 - r2).max(0.0).sqrt()).norm()
}

/// GGX microfacet normal sample around n.
#[inline]
pub fn sample_ggx(n: V, alpha: f32, r: &mut Rng) -> V {
    let (u1, u2) = (r.f(), r.f());
    let t2 = alpha * alpha * u1 / (1.0 - u1).max(1e-6);
    let c = 1.0 / (1.0 + t2).sqrt();
    let s = (1.0 - c * c).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    let (t, b) = onb(n);
    (t * (s * phi.cos()) + b * (s * phi.sin()) + n * c).norm()
}

/// GGX specular BRDF * (n.l). View vector `vd` and light vector `l` point away from the surface.
#[inline]
pub fn ggx_nl(n: V, vd: V, l: V, alpha: f32, f0: f32) -> f32 {
    let nl = n.dot(l);
    if nl <= 0.0 {
        return 0.0;
    }
    let nv = n.dot(vd).max(1e-3);
    let hh = l + vd;
    let h = hh * (1.0 / hh.dot(hh).sqrt().max(1e-20));
    let nh = n.dot(h).max(0.0);
    let vh = vd.dot(h).max(0.0);
    let a2 = alpha * alpha;
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * dd * dd);
    let k = alpha * 0.5;
    let g = (nl / (nl * (1.0 - k) + k)) * (nv / (nv * (1.0 - k) + k));
    let f = f0 + (1.0 - f0) * (1.0 - vh).powi(5);
    d * g * f / (4.0 * nv)
}

#[inline]
fn lat(ix: i32, iy: i32) -> f32 {
    (hash((ix as u32).wrapping_mul(0x9E3779B1) ^ (iy as u32).wrapping_mul(0x85EBCA6B)) >> 8) as f32 * (1.0 / 16777216.0)
}
/// Smooth 2D value noise in 0..1 (lattice spacing = 1).
#[inline]
pub fn vnoise(x: f32, y: f32) -> f32 {
    let (fx, fy) = (x.floor(), y.floor());
    let (tx, ty) = (x - fx, y - fy);
    let (ix, iy) = (fx as i32, fy as i32);
    let (sx, sy) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let a = lat(ix, iy) * (1.0 - sx) + lat(ix + 1, iy) * sx;
    let b = lat(ix, iy + 1) * (1.0 - sx) + lat(ix + 1, iy + 1) * sx;
    a * (1.0 - sy) + b * sy
}
