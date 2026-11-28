//! Text -> extruded 3D mesh (outline contours + outward wall normals).
//! Arabic shaping via rustybuzz (HarfBuzz port); outlines via ttf-parser.
//! The mesh is built ONCE; nothing here runs per frame.
//!
//! Layout follows local1: the line is laid out RIGHT-TO-LEFT (the logically
//! first word sits at the largest X), every word is shaped on its own and
//! packed into the next cell, and the whole line is centred on the origin
//! afterwards.
//!
//! World frame (see camera.rs):
//!   X : reading axis   (+X = right of frame; RTL text starts at +X)
//!   Y : depth          (+Y = away from the camera = up-screen)
//!   Z : up             (extrusion axis; the ground plane is Z = 0)
//!
//! One world unit == one em (the font's em square), exactly like local1, so
//! the camera distances and the shake amplitudes are directly comparable.

use rustybuzz::ttf_parser::OutlineBuilder;
use rustybuzz::{ttf_parser, Face, UnicodeBuffer};

/// One word of the line.
///
/// `recitation` is `Some(index)` for a word that is actually spoken (and has a
/// timing entry) and `None` for a non-spoken word such as the ayah-end marker.
/// The indices are in reading order and are what the camera timeline targets.
#[derive(Clone, Copy)]
pub struct TextWord<'a> {
    pub text: &'a str,
    pub recitation: Option<usize>,
}

/// A closed contour: edge i runs from vertex i to `next[i]`.
pub struct Mesh {
    pub vx: Vec<f32>,
    pub vy: Vec<f32>,
    pub next: Vec<u32>,
    /// Outward-pointing unit normal (horizontal) of edge i.
    pub nx: Vec<f32>,
    pub ny: Vec<f32>,
    /// (first edge, edge count) of every closed contour.
    pub contours: Vec<(u32, u32)>,
    /// extrusion height (world units = em)
    pub height: f32,
    /// ink width of the whole line
    pub width: f32,
    /// ink height of the whole line
    pub text_height: f32,
    /// world Y of the glyph baseline (the line is centred on the origin, so this
    /// is generally not 0 — the camera aims at the baseline, like local1).
    pub baseline_y: f32,
    /// centre of the logically first word (the right-most one), world X
    pub first_word_x: f32,
    /// centre X of every word, in logical (reading) order, world em
    pub word_x: Vec<f32>,
    /// centre X of every *spoken* word, indexed by its recitation index.
    /// Empty when the mesh was built from plain text.
    pub recitation_x: Vec<f32>,
    /// mean word advance, in em
    pub avg_advance: f32,
    /// number of words that carry ink
    pub word_count: usize,
}

struct Builder {
    contours: Vec<Vec<(f32, f32)>>,
    cur: Vec<(f32, f32)>,
    last: (f32, f32),
    ox: f32,
    oy: f32,
    tol: f32, // flatten tolerance, in font units
}

impl Builder {
    fn push(&mut self, p: (f32, f32)) {
        self.last = p;
        self.cur.push((p.0 + self.ox, p.1 + self.oy));
    }
}

impl ttf_parser::OutlineBuilder for Builder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.close();
        self.push((x, y));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.push((x, y));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p0 = self.last;
        let ddx = p0.0 - 2.0 * x1 + x;
        let ddy = p0.1 - 2.0 * y1 + y;
        let dd = (ddx * ddx + ddy * ddy).sqrt();
        let n = ((dd / (4.0 * self.tol)).sqrt().ceil() as usize).clamp(1, 64);
        for i in 1..=n {
            let t = i as f32 / n as f32;
            let u = 1.0 - t;
            let px = u * u * p0.0 + 2.0 * u * t * x1 + t * t * x;
            let py = u * u * p0.1 + 2.0 * u * t * y1 + t * t * y;
            self.push((px, py));
        }
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let p0 = self.last;
        let a = ((p0.0 - 2.0 * x1 + x2).powi(2) + (p0.1 - 2.0 * y1 + y2).powi(2)).sqrt();
        let b = ((x1 - 2.0 * x2 + x).powi(2) + (y1 - 2.0 * y2 + y).powi(2)).sqrt();
        let m = a.max(b);
        let n = ((0.75 * m / self.tol).sqrt().ceil() as usize).clamp(1, 64);
        for i in 1..=n {
            let t = i as f32 / n as f32;
            let u = 1.0 - t;
            let px = u * u * u * p0.0 + 3.0 * u * u * t * x1 + 3.0 * u * t * t * x2 + t * t * t * x;
            let py = u * u * u * p0.1 + 3.0 * u * u * t * y1 + 3.0 * u * t * t * y2 + t * t * t * y;
            self.push((px, py));
        }
    }
    fn close(&mut self) {
        if self.cur.len() >= 3 {
            let c = std::mem::take(&mut self.cur);
            self.contours.push(c);
        } else {
            self.cur.clear();
        }
    }
}

/// Shape one word and return its glyphs (id, pen x, pen y in font units) and
/// its advance.  RTL is detected per buffer, exactly like local1 shapes every
/// word on its own.
fn shape_word(face: &Face, word: &str) -> (Vec<(u16, f32, f32)>, f32) {
    let mut buf = UnicodeBuffer::new();
    buf.push_str(word);
    buf.guess_segment_properties();
    let out = rustybuzz::shape(face, &[], buf);
    let infos = out.glyph_infos();
    let pos = out.glyph_positions();
    let mut pen = 0.0f32;
    let mut glyphs = Vec::with_capacity(infos.len());
    for (i, p) in infos.iter().zip(pos.iter()) {
        glyphs.push((i.glyph_id as u16, pen + p.x_offset as f32, p.y_offset as f32));
        pen += p.x_advance as f32;
    }
    (glyphs, pen)
}

/// Build the mesh for a (possibly multi-word) line of text.
///
/// * `depth_em` — extrusion height, in em
/// * `gap_em`   — gap between words, in em
/// * `tol_em`   — curve flattening tolerance, in em
pub fn build_mesh(
    font_data: &[u8],
    text: &str,
    depth_em: f32,
    gap_em: f32,
    tol_em: f32,
) -> Result<Mesh, String> {
    let words: Vec<TextWord> = text
        .split_whitespace()
        .map(|w| TextWord { text: w, recitation: None })
        .collect();
    build_mesh_words(font_data, &words, depth_em, gap_em, tol_em)
}

/// Average shaped advance of a line, in em.  This is the reference used for the
/// camera zoom: it must NOT be recomputed from the selected text, otherwise a
/// narrower font/text would be scaled up and the zoom would change.
pub fn average_advance(font_data: &[u8], text: &str) -> Result<f32, String> {
    let face = Face::from_slice(font_data, 0).ok_or("cannot parse font")?;
    let scale = 1.0 / face.units_per_em() as f32;
    let mut sum = 0.0f32;
    let mut n = 0usize;
    for w in text.split_whitespace() {
        let (_, adv) = shape_word(&face, w);
        if adv > 0.0 {
            sum += adv * scale;
            n += 1;
        }
    }
    if n == 0 {
        return Err("no words".into());
    }
    Ok(sum / n as f32)
}

/// Build the mesh for an explicit list of words (used by the Quran renderer,
/// where the words and their recitation indices come from the Quran text data).
pub fn build_mesh_words(
    font_data: &[u8],
    words_in: &[TextWord<'_>],
    depth_em: f32,
    gap_em: f32,
    tol_em: f32,
) -> Result<Mesh, String> {
    let face = Face::from_slice(font_data, 0).ok_or("cannot parse font")?;
    let upem = face.units_per_em() as f32;
    let scale = 1.0 / upem; // font units -> em
    let tol_fu = (tol_em / scale).max(0.05);

    // ---- 1. shape every word, keep its contours in em (pen space) ---------
    let mut words: Vec<(Vec<Vec<(f32, f32)>>, f32, Option<usize>)> = Vec::new(); // (contours, advance em, recitation index)
    for w in words_in {
        let (glyphs, adv_fu) = shape_word(&face, w.text);
        if glyphs.is_empty() {
            continue;
        }
        let mut b = Builder {
            contours: vec![],
            cur: vec![],
            last: (0.0, 0.0),
            ox: 0.0,
            oy: 0.0,
            tol: tol_fu,
        };
        for (gid, px, py) in glyphs {
            b.ox = px;
            b.oy = py;
            if face.outline_glyph(ttf_parser::GlyphId(gid), &mut b).is_some() {
                b.close();
            }
        }
        if b.contours.is_empty() {
            continue;
        }
        let contours: Vec<Vec<(f32, f32)>> = b
            .contours
            .iter()
            .map(|c| c.iter().map(|&(x, y)| (x * scale, y * scale)).collect())
            .collect();
        words.push((contours, adv_fu * scale, w.recitation));
    }
    if words.is_empty() {
        return Err("no outlines produced (font lacks glyphs? If this is DigitalKhattV2.otf, use DigitalKhattV2.ttf)".into());
    }

    // ---- 2. right-to-left placement --------------------------------------
    // Cell edges march to the left; every word's INK is flush with its cell's
    // right edge, so the visual gap between two words is exactly `gap_em`.
    let mut x_right = 0.0f32;
    let mut placed: Vec<(Vec<Vec<(f32, f32)>>, f32, Option<usize>)> = Vec::new(); // (contours, ink centre, recitation index)
    let mut adv_sum = 0.0f32;
    let mut spoken_adv_sum = 0.0f32;
    let mut spoken_count = 0usize;
    for (contours, adv, recitation) in words {
        let (mut x0, mut x1) = (f32::MAX, f32::MIN);
        for c in &contours {
            for &(x, _y) in c {
                x0 = x0.min(x);
                x1 = x1.max(x);
            }
        }
        if x1 <= x0 {
            continue;
        }
        let ink_w = x1 - x0;
        let dx = x_right - x1;
        let shifted: Vec<Vec<(f32, f32)>> = contours
            .iter()
            .map(|c| c.iter().map(|&(x, y)| (x + dx, y)).collect())
            .collect();
        placed.push((shifted, x_right - ink_w * 0.5, recitation));
        adv_sum += adv;
        if recitation.is_some() {
            spoken_adv_sum += adv;
            spoken_count += 1;
        }
        x_right -= ink_w + gap_em;
    }
    let word_count = placed.len();
    if word_count == 0 {
        return Err("no outlines produced (font lacks glyphs? If this is DigitalKhattV2.otf, use DigitalKhattV2.ttf)".into());
    }
    // The world scale is set from the SPOKEN words only.  The ayah-end markers
    // are not recited and are much narrower, so including them would shrink the
    // average advance and silently zoom the camera in.
    let avg_advance = if spoken_count > 0 {
        spoken_adv_sum / spoken_count as f32
    } else {
        adv_sum / word_count as f32
    };

    // ---- 3. centre the whole line on the origin ---------------------------
    let (mut bx0, mut by0, mut bx1, mut by1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (contours, _, _) in &placed {
        for c in contours {
            for &(x, y) in c {
                bx0 = bx0.min(x);
                bx1 = bx1.max(x);
                by0 = by0.min(y);
                by1 = by1.max(y);
            }
        }
    }
    let (cx, cy) = ((bx0 + bx1) * 0.5, (by0 + by1) * 0.5);

    // ---- 4. outer-contour orientation -------------------------------------
    // The font winds outer contours one way and holes the other; the sign of
    // the total signed area tells us which way the OUTER contours run, so the
    // wall normals can point outwards.
    let mut total = 0.0f64;
    for (contours, _, _) in &placed {
        for c in contours {
            let mut a = 0.0f64;
            for i in 0..c.len() {
                let (p, q) = (c[i], c[(i + 1) % c.len()]);
                a += (p.0 as f64) * (q.1 as f64) - (q.0 as f64) * (p.1 as f64);
            }
            total += a;
        }
    }
    // material on the left of travel if total>0 (CCW outer) -> outward normal = right = (dy,-dx)
    let sgn = if total >= 0.0 { 1.0f32 } else { -1.0f32 };

    // ---- 5. emit the mesh --------------------------------------------------
    let n_rec = placed
        .iter()
        .filter_map(|p| p.2)
        .max()
        .map(|i| i + 1)
        .unwrap_or(0);
    let mut recitation_x = vec![0.0f32; n_rec];
    for (_, centre, rec) in &placed {
        if let Some(i) = rec {
            recitation_x[*i] = centre - cx;
        }
    }
    let mut m = Mesh {
        vx: vec![],
        vy: vec![],
        next: vec![],
        nx: vec![],
        ny: vec![],
        contours: vec![],
        height: depth_em,
        width: bx1 - bx0,
        text_height: by1 - by0,
        baseline_y: -cy,
        first_word_x: placed[0].1 - cx,
        word_x: placed.iter().map(|p| p.1 - cx).collect(),
        recitation_x,
        avg_advance,
        word_count,
    };
    for (word_contours, _, _) in &placed {
        for c in word_contours {
            // drop near-duplicate points
            let mut pts: Vec<(f32, f32)> = Vec::with_capacity(c.len());
            for &(x, y) in c.iter() {
                let p = (x - cx, y - cy);
                if let Some(&l) = pts.last() {
                    if (p.0 - l.0).abs() + (p.1 - l.1).abs() < tol_em * 0.05 {
                        continue;
                    }
                }
                pts.push(p);
            }
            if pts.len() < 3 {
                continue;
            }
            let base = m.vx.len() as u32;
            let n = pts.len() as u32;
            for (i, &(x, y)) in pts.iter().enumerate() {
                let (qx, qy) = pts[(i + 1) % pts.len()];
                let (dx, dy) = (qx - x, qy - y);
                let len = (dx * dx + dy * dy).sqrt().max(1e-12);
                m.vx.push(x);
                m.vy.push(y);
                m.next.push(base + (i as u32 + 1) % n);
                m.nx.push(sgn * dy / len);
                m.ny.push(sgn * -dx / len);
            }
            m.contours.push((base, n));
        }
    }
    Ok(m)
}
