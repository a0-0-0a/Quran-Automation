//! Camera rig — a port of local1's `Camera` class (local1/engin/quran_3d.py).
//!
//! local1 does not use a look-at camera: it orbits a TARGET that sits on the
//! text plane at a fixed elevation / azimuth and then rolls the image.  That is
//! what gives the "macro photo of a name plate" look, and it is what c2
//! reproduces, formula for formula:
//!
//!     to_cam = (sin(az)*cos(el), sin(el), cos(az)*cos(el))
//!     C      = target + to_cam * dist
//!     fwd    = norm(target - C)
//!     right  = norm(cross(fwd, up_world))
//!     up     = cross(right, fwd)
//!     (right, up) are then rolled by cam_roll about the view axis.
//!
//! Frame mapping (local1 -> c2).  local1 puts the text on the y = 0 plane and
//! extrudes it along +y (up); c2 puts the text on the z = 0 plane and extrudes
//! it along +z (up).  The two frames are related by a proper rotation
//! (x, y_up, z_depth) -> (x, -z_depth, y_up), so the rendered image is NOT
//! mirrored and the perspective is identical.
//!
//!   local1 x       -> c2 X   reading axis (+X = right of frame)
//!   local1 z_depth -> c2 -Y  depth (+Y = away from the camera = up-screen)
//!   local1 y_up    -> c2 Z   up (the extrusion axis; ground is Z = 0)

#[derive(Clone, Copy)]
pub struct Camera {
    pub pos: [f32; 3],
    pub fwd: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
}

#[inline]
fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    [v[0] / l, v[1] / l, v[2] / l]
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The orbit rig: everything that does not change from frame to frame.
pub struct Rig {
    /// distance from the target to the camera, in world units (em)
    pub dist: f32,
    /// unit vector from the target towards the camera
    pub to_cam: [f32; 3],
    /// roll about the view axis, radians
    pub roll: f32,
}

impl Rig {
    pub fn new(focal_px: f32, dist: f32, elev_deg: f32, azim_deg: f32, roll_deg: f32) -> Rig {
        let _ = focal_px; // kept in the signature: the rig and the lens belong together
        let el = elev_deg.to_radians();
        let az = azim_deg.to_radians();
        // local1: (sin az cos el, sin el, cos az cos el) in (x, y_up, z_depth)
        // -> c2:    (sin az cos el, -cos az cos el, sin el) in (X, Y, Z)
        let to_cam = [
            az.sin() * el.cos(),
            -az.cos() * el.cos(),
            el.sin(),
        ];
        Rig {
            dist,
            to_cam,
            roll: roll_deg.to_radians(),
        }
    }

    /// Build the camera for a target that sits on the text plane.
    pub fn aim(&self, target: [f32; 3]) -> Camera {
        let c = [
            target[0] + self.to_cam[0] * self.dist,
            target[1] + self.to_cam[1] * self.dist,
            target[2] + self.to_cam[2] * self.dist,
        ];
        let fwd = norm([
            target[0] - c[0],
            target[1] - c[1],
            target[2] - c[2],
        ]);
        // "up_world" for the right vector is the vertical axis (+Z in c2).
        let right = norm(cross(fwd, [0.0, 0.0, 1.0]));
        let up = cross(right, fwd);
        // roll the image about the view axis (local1's cam_roll)
        let (s, co) = (self.roll.sin(), self.roll.cos());
        let r2 = [
            right[0] * co + up[0] * s,
            right[1] * co + up[1] * s,
            right[2] * co + up[2] * s,
        ];
        let u2 = [
            -right[0] * s + up[0] * co,
            -right[1] * s + up[1] * co,
            -right[2] * s + up[2] * co,
        ];
        let (right, up) = (r2, u2);
        Camera {
            pos: c,
            fwd,
            right,
            up,
        }
    }
}
