//! Floor texture: read straight from `textures.zip` (first .png inside), mip-mapped, sRGB->linear.

use crate::vec::*;
use std::io::Read;

pub struct Tex {
    levels: Vec<(usize, usize, Vec<u8>)>, // w, h, RGB8
    lut: [f32; 256],
    pub avg: V,
    pub size0: f32,
}

fn srgb_to_lin(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Unsharp mask on an RGB8 image: out = img + amount * (img - blur(img)), blur = two 5x5 box passes.
fn unsharp(img: &mut [u8], w: usize, h: usize, amount: f32) {
    use rayon::prelude::*;
    let r = 2usize;
    let win = (2 * r + 1) as u32;
    // horizontal pass (parallel over rows) into u16 sums, then vertical running sums
    let mut hsum = vec![0u32; w * h * 3];
    hsum.par_chunks_mut(w * 3).zip(img.par_chunks(w * 3)).for_each(|(d, s)| {
        for c in 0..3 {
            // window centred at x=0 is [-r, r], clamped at the border
            let mut acc: u32 = (0..=2 * r).map(|i| s[((i as i32 - r as i32).clamp(0, w as i32 - 1) as usize) * 3 + c] as u32).sum();
            for x in 0..w {
                d[x * 3 + c] = acc;
                let add = (x + r + 1).min(w - 1);
                let sub = (x as i32 - r as i32).max(0) as usize;
                acc = acc + s[add * 3 + c] as u32 - s[sub * 3 + c] as u32;
            }
        }
    });
    let row = w * 3;
    let mut col = vec![0u32; row];
    for i in -(r as i32)..=(r as i32) {
        let y = i.clamp(0, h as i32 - 1) as usize;
        for k in 0..row {
            col[k] += hsum[y * row + k];
        }
    }
    let norm = 1.0 / (win * win) as f32;
    for y in 0..h {
        for k in 0..row {
            let blur = col[k] as f32 * norm;
            let o = img[y * row + k] as f32;
            img[y * row + k] = (o + amount * (o - blur)).clamp(0.0, 255.0) as u8;
        }
        let add = (y + r + 1).min(h - 1);
        let sub = (y as i32 - r as i32).max(0) as usize;
        for k in 0..row {
            col[k] = col[k] + hsum[add * row + k] - hsum[sub * row + k];
        }
    }
}

impl Tex {
    pub fn from_zip(path: &str, sharpen: f32) -> Result<Tex, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut z = zip::ZipArchive::new(f).map_err(|e| e.to_string())?;
        let mut data = Vec::new();
        let mut found = false;
        for i in 0..z.len() {
            let mut e = z.by_index(i).map_err(|e| e.to_string())?;
            let n = e.name().to_lowercase();
            if n.ends_with(".png") && !n.contains("__macosx") {
                e.read_to_end(&mut data).map_err(|e| e.to_string())?;
                found = true;
                break;
            }
        }
        if !found {
            return Err("no .png inside textures.zip".into());
        }
        let mut dec = png::Decoder::new(&data[..]);
        dec.set_transformations(png::Transformations::EXPAND);
        let mut rd = dec.read_info().map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; rd.output_buffer_size()];
        let info = rd.next_frame(&mut buf).map_err(|e| e.to_string())?;
        let (w, h) = (info.width as usize, info.height as usize);
        if !w.is_power_of_two() || !h.is_power_of_two() {
            return Err("texture size must be a power of two (e.g. 2048/4096)".into());
        }
        let ch = match info.color_type {
            png::ColorType::Rgb => 3,
            png::ColorType::Rgba => 4,
            png::ColorType::Grayscale => 1,
            png::ColorType::GrayscaleAlpha => 2,
            _ => return Err("unsupported png colour type".into()),
        };
        if info.bit_depth != png::BitDepth::Eight {
            return Err("only 8-bit png supported".into());
        }
        let mut rgb = vec![0u8; w * h * 3];
        for i in 0..w * h {
            for c in 0..3 {
                rgb[i * 3 + c] = if ch >= 3 { buf[i * ch + c] } else { buf[i * ch] };
            }
        }
        drop(buf);
        let mut rgb = rgb;
        if sharpen > 0.0 {
            unsharp(&mut rgb, w, h, sharpen);
        }
        let mut levels = vec![(w, h, rgb)];
        loop {
            let (lw, lh, ref src) = levels[levels.len() - 1];
            if lw <= 1 && lh <= 1 {
                break;
            }
            let nw = (lw / 2).max(1);
            let nh = (lh / 2).max(1);
            let mut dst = vec![0u8; nw * nh * 3];
            for y in 0..nh {
                for x in 0..nw {
                    let x1 = (x * 2 + 1).min(lw - 1);
                    let y1 = (y * 2 + 1).min(lh - 1);
                    for c in 0..3 {
                        let s = src[((y * 2) * lw + x * 2) * 3 + c] as u32
                            + src[((y * 2) * lw + x1) * 3 + c] as u32
                            + src[(y1 * lw + x * 2) * 3 + c] as u32
                            + src[(y1 * lw + x1) * 3 + c] as u32;
                        dst[(y * nw + x) * 3 + c] = ((s + 2) / 4) as u8;
                    }
                }
            }
            levels.push((nw, nh, dst));
        }
        let mut lut = [0f32; 256];
        for i in 0..256 {
            lut[i] = srgb_to_lin(i as f32 / 255.0);
        }
        let last = &levels[levels.len() - 1].2;
        let avg = v(lut[last[0] as usize], lut[last[1] as usize], lut[last[2] as usize]);
        Ok(Tex { levels, lut, avg, size0: w as f32 })
    }

    #[inline]
    fn bilin(&self, l: usize, u: f32, vv: f32) -> V {
        let (w, h, ref d) = self.levels[l];
        let fx = u * w as f32 - 0.5;
        let fy = vv * h as f32 - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let (xi, yi) = (x0 as i32, y0 as i32);
        let (mx, my) = ((w - 1) as i32, (h - 1) as i32); // textures are power-of-two
        let xa = (xi & mx) as usize;
        let xb = ((xi + 1) & mx) as usize;
        let ya = (yi & my) as usize;
        let yb = ((yi + 1) & my) as usize;
        let mut out = [0f32; 3];
        for c in 0..3 {
            let p00 = self.lut[d[(ya * w + xa) * 3 + c] as usize];
            let p10 = self.lut[d[(ya * w + xb) * 3 + c] as usize];
            let p01 = self.lut[d[(yb * w + xa) * 3 + c] as usize];
            let p11 = self.lut[d[(yb * w + xb) * 3 + c] as usize];
            out[c] = (p00 * (1.0 - tx) + p10 * tx) * (1.0 - ty) + (p01 * (1.0 - tx) + p11 * tx) * ty;
        }
        v(out[0], out[1], out[2])
    }

    /// Trilinear, wrapped. u,v in tile units.
    pub fn sample(&self, u: f32, vv: f32, lod: f32) -> V {
        let u = u - u.floor();
        let vv = vv - vv.floor();
        let max = (self.levels.len() - 1) as f32;
        let lod = lod.clamp(0.0, max);
        let l0 = lod.floor();
        let f = lod - l0;
        let a = self.bilin(l0 as usize, u, vv);
        if l0 + 1.0 > max || f < 0.25 {
            return a;
        }
        if f > 0.75 {
            return self.bilin(l0 as usize + 1, u, vv);
        }
        let b = self.bilin(l0 as usize + 1, u, vv);
        a * (1.0 - f) + b * f
    }
}
