mod audio;
mod bake;
mod data;
mod lamps;
mod material;
mod quran;
mod render;
mod rig;
mod scene;
mod shake;
mod slide;
mod tex;
mod text;
mod vec;

use render::{Camera, Params, Renderer};
use scene::{Light, Scene};
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::Instant;
use vec::*;

struct Args {
    text: String,
    font: String,
    material: String,
    tex: Option<String>,
    grade: f32,
    out: String,
    still: Option<String>,
    at: f32,
    w: usize,
    h: usize,
    fps: u32,
    secs: f32,
    threads: usize,
    crf: u32,
    preset: String,
    bench: bool,
    // camera rig (local1 numbers)
    focal_mm: f32,
    sensor_mm: f32,
    elev: f32,
    azim: f32,
    roll: f32,
    frame_words: f32,
    pan_wps: f32,
    shake: f32,
    // scene
    depth: f32,
    gap: f32,
    // lights
    pitch: f32,
    light_h: f32,
    lamp_file: String,
    lamp_seed: usize,
    segments: Option<String>,
    fstop: f32,
    spot: f32,
    intensity: f32,
    disc: f32,
    // look
    blur: f32, // lens-blur gain
    exposure: f32,
    spp: usize,
    bounces: usize,
    // quran range mode
    range: Option<String>,
    audio_dir: String,
    quran_json: String,
    timings: String,
    work_dir: String,
    smooth_time: f32,
    lead: f32,
    no_audio: bool,
    seed: Option<u32>,
    sfx: Option<String>,
    sfx_dir: String,
    no_sfx: bool,
    sfx_gain: f32,
    reciter_gain: f32,
}

fn parse() -> Args {
    let mut a = Args {
        text: "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ".into(),
        font: "assets/DigitalKhattV2.ttf".into(),
        material: "assets/materials/green_marble.txt".into(),
        tex: None,
        grade: 1.0,
        out: "out.mp4".into(),
        still: None,
        at: 0.0,
        w: 1080,
        h: 1920,
        fps: 60,
        secs: 1.0,
        threads: 0,
        crf: 12,
        preset: "medium".into(),
        bench: false,
        focal_mm: 70.0,
        sensor_mm: 36.0,
        elev: 63.0,
        azim: 21.0,
        roll: -9.0,
        frame_words: 3.8,
        pan_wps: 1.5,
        shake: 1.0,
        depth: 0.15,
        gap: 0.10,
        pitch: 0.22,
        light_h: 0.26,
        lamp_file: "assets/lamp_keys.bin".into(),
        lamp_seed: 0,
        segments: None,
        fstop: 0.7,
        spot: 3.6,
        intensity: 2.0,
        disc: 0.02,
        blur: 3.0, // original lens-blur gain
        exposure: 1.5,
        spp: 48,
        bounces: 2,
        range: None,
        audio_dir: "../../Assets/reciters/yasser-al-dosarie/quran".into(),
        quran_json: "assets/digital-khatt-v2.json.zip".into(),
        timings: "assets/timings_db/Yasser_Ad-Dussary.json".into(),
        work_dir: "target/quran_work".into(),
        smooth_time: 1.5,
        lead: 0.9,
        no_audio: false,
        seed: None,
        sfx: None,
        sfx_dir: "assets/sfx".into(),
        no_sfx: false,
        sfx_gain: 0.1,
        reciter_gain: 1.0,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        if v[i] == "--bench" {
            a.bench = true;
            i += 1;
            continue;
        }
        let val = v.get(i + 1).cloned().unwrap_or_default();
        let f = |s: &str| -> f32 { s.parse().expect("number expected") };
        match v[i].as_str() {
            "--text" => a.text = val,
            "--font" => a.font = val,
            "--material" => a.material = val,
            "--tex" => a.tex = Some(val),
            "--grade" => a.grade = f(&val),
            "--out" => a.out = val,
            "--still" => a.still = Some(val),
            "--at" => a.at = f(&val),
            "--w" => a.w = val.parse().unwrap(),
            "--h" => a.h = val.parse().unwrap(),
            "--fps" => a.fps = val.parse().unwrap(),
            "--secs" => a.secs = f(&val),
            "--threads" => a.threads = val.parse().unwrap(),
            "--crf" => a.crf = val.parse().unwrap(),
            "--preset" => a.preset = val,
            "--focal" => a.focal_mm = f(&val),
            "--sensor" => a.sensor_mm = f(&val),
            "--elev" => a.elev = f(&val),
            "--azim" => a.azim = f(&val),
            "--roll" => a.roll = f(&val),
            "--frame-words" => a.frame_words = f(&val),
            "--pan" => a.pan_wps = f(&val),
            "--shake" => a.shake = f(&val),
            "--depth" => a.depth = f(&val),
            "--gap" => a.gap = f(&val),
            "--pitch" => a.pitch = f(&val),
            "--light-h" => a.light_h = f(&val),
            "--lamps" => a.lamp_file = val,
            "--lamp-seed" => a.lamp_seed = val.parse().unwrap(),
            "--segments" => a.segments = Some(val),
            "--fstop" => a.fstop = f(&val),
            "--spot" => a.spot = f(&val),
            "--intensity" => a.intensity = f(&val),
            "--disc" => a.disc = f(&val),
            "--blur" => a.blur = f(&val),
            "--exposure" => a.exposure = f(&val),
            "--spp" => a.spp = val.parse().unwrap(),
            "--bounces" => a.bounces = val.parse().unwrap(),
            "--range" => a.range = Some(val),
            "--audio-dir" => a.audio_dir = val,
            "--quran-json" => a.quran_json = val,
            "--timings" => a.timings = val,
            "--work-dir" => a.work_dir = val,
            "--smooth-time" => a.smooth_time = f(&val),
            "--lead" => a.lead = f(&val),
            "--no-audio" => {
                a.no_audio = true;
                i += 1;
                continue;
            }
            "--seed" => a.seed = Some(val.parse().unwrap()),
            "--sfx" => a.sfx = Some(val),
            "--sfx-dir" => a.sfx_dir = val,
            "--no-sfx" => {
                a.no_sfx = true;
                i += 1;
                continue;
            }
            "--sfx-gain" => a.sfx_gain = f(&val),
            "--reciter-gain" => a.reciter_gain = f(&val),
            x => {
                eprintln!("unknown arg {x}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    a
}

struct QuranJob {
    range: quran::Range,
    words: Vec<quran::QuranWord>,
    timeline: Vec<quran::TimelineWord>,
    track: Option<audio::AudioTrack>,
    duration: f32,
}

/// Pick a background sound: a random file from `--sfx-dir` (or from `--sfx`
/// when that is a folder).  The render seed chooses it, so the whole render is
/// reproducible with `--seed`.  `--no-sfx` disables it.
fn pick_sfx(a: &Args, seed: u32) -> Option<std::path::PathBuf> {
    if a.no_sfx {
        return None;
    }
    let dir = match &a.sfx {
        Some(s) => {
            let p = std::path::Path::new(s);
            if p.is_dir() {
                p.to_path_buf()
            } else {
                return Some(p.to_path_buf());
            }
        }
        None => std::path::PathBuf::from(&a.sfx_dir),
    };
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && matches!(
                    p.extension().and_then(|x| x.to_str()),
                    Some("mp3" | "ogg" | "wav" | "opus" | "m4a" | "aac")
                )
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return None;
    }
    Some(files[(seed as usize) % files.len()].clone())
}

/// Load everything a `--range` render needs: text, word timings, and audio.
fn load_quran_job(a: &Args, seed: u32) -> QuranJob {
    let range = quran::Range::parse(a.range.as_ref().unwrap())
        .and_then(|r| r.validate_safe().map(|_| r))
        .unwrap_or_else(|e| panic!("{e}"));
    let text = quran::QuranText::load_zip(&a.quran_json).expect("quran json");
    let timings = quran::Timings::load(&a.timings).expect("timings");
    let words = text.range_words(range).expect("quran words");

    let (track, offsets, duration) = if a.no_audio {
        // Testing mode: derive the timeline from the word timings alone.
        let mut offs = Vec::new();
        let mut t = 0.0f32;
        for ay in range.from..=range.to {
            offs.push(t);
            let w = timings
                .ayah(range.surah, ay)
                .unwrap_or_else(|| panic!("no timings for {}.{}", range.surah, ay));
            t += w.last().map(|x| x.end).unwrap_or(0.0);
        }
        (None, offs, t)
    } else {
        let track = audio::build(
            std::path::Path::new(&a.audio_dir),
            range,
            std::path::Path::new(&a.work_dir),
        )
        .unwrap_or_else(|e| panic!("audio: {e}"));
        let mut track = track;
        if let Some(sfx) = pick_sfx(a, seed) {
            let mixed = track.path.with_file_name("soundtrack.wav");
            audio::mix_sfx(
                &track.path,
                &sfx,
                &mixed,
                track.duration,
                a.sfx_gain,
                a.reciter_gain,
            )
            .unwrap_or_else(|e| panic!("sfx: {e}"));
            eprintln!(
                "sfx: {} at {:.0}% of the reciter (reciter gain {:.2})",
                sfx.display(),
                a.sfx_gain * 100.0,
                a.reciter_gain
            );
            track.path = mixed;
        }
        let offs = track.offsets.clone();
        let d = track.duration;
        (Some(track), offs, d)
    };

    let timeline = quran::build_timeline(range, &words, &timings, &offsets, duration)
        .unwrap_or_else(|e| panic!("timeline: {e}"));

    eprintln!(
        "quran: {} ({}) {} spoken words, {} ayahs, {:.2}s audio",
        range,
        data::surah_name(range.surah),
        timeline.len(),
        range.to - range.from + 1,
        duration
    );

    QuranJob { range, words, timeline, track, duration }
}

fn main() {
    let a = parse();
    if a.threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(a.threads).build_global().unwrap();
    }
    eprintln!("threads: {}", rayon::current_num_threads());
    let t_all = Instant::now();

    // ---------- build (once) ----------
    let font = std::fs::read(&a.font).expect("font file");
    // The zoom is pinned to the DEFAULT line (the basmala), so every text and
    // every ayah range is shown at exactly the same scale.  Deriving it from
    // the selected text would scale narrow words up (the "zoomed in" bug).
    let ref_adv = text::average_advance(&font, &a.text).unwrap_or(1.0);
    let focal = a.focal_mm / a.sensor_mm * a.w.max(a.h) as f32; // px (local1 lens)
    let seed = a.seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u32
    });
    let job = a.range.as_ref().map(|_| load_quran_job(&a, seed));
    let duration = job.as_ref().map(|j| j.duration).unwrap_or(a.secs);
    let t = Instant::now();

    // Two passes: a coarse mesh only to learn the average advance (which sets
    // the world scale and the flattening tolerance), then the final mesh.
    let mut mesh = match &job {
        Some(j) => {
            let tw: Vec<text::TextWord> = j
                .words
                .iter()
                .map(|w| text::TextWord { text: &w.text, recitation: w.recitation })
                .collect();
            text::build_mesh_words(&font, &tw, a.depth, a.gap, 0.002).expect("mesh")
        }
        None => text::build_mesh(&font, &a.text, a.depth, a.gap, 0.002).expect("mesh"),
    };
    let frame_em = a.frame_words * ref_adv; // em visible across the frame (fixed reference)
    let tol_em = 0.08 * frame_em / a.w as f32; // curve flattening tolerance: 0.08 px (was 0.2)
    mesh = match &job {
        Some(j) => {
            let tw: Vec<text::TextWord> = j
                .words
                .iter()
                .map(|w| text::TextWord { text: &w.text, recitation: w.recitation })
                .collect();
            text::build_mesh_words(&font, &tw, a.depth, a.gap, tol_em).expect("mesh")
        }
        None => text::build_mesh(&font, &a.text, a.depth, a.gap, tol_em).expect("mesh"),
    };
    // world unit: metres. The frame (at the target) is 1 m wide -> 1 em = k metres.
    let k = 1.0 / frame_em;
    for x in mesh.vx.iter_mut().chain(mesh.vy.iter_mut()) {
        *x *= k;
    }
    mesh.height *= k;
    mesh.width *= k;
    mesh.text_height *= k;
    mesh.baseline_y *= k;
    mesh.first_word_x *= k;
    for x in mesh.word_x.iter_mut() {
        *x *= k;
    }
    for x in mesh.recitation_x.iter_mut() {
        *x *= k;
    }
    mesh.avg_advance *= k;
    eprintln!(
        "scale: avg_advance {:.4} m, frame_em {:.4}, k {:.4}, baseline_y {:.4} m, text_h {:.4} m",
        mesh.avg_advance, frame_em, k, mesh.baseline_y, mesh.text_height
    );
    if let Some(j) = &job {
        assert_eq!(
            mesh.recitation_x.len(),
            j.timeline.len(),
            "mesh produced {} spoken words but the timeline has {}",
            mesh.recitation_x.len(),
            j.timeline.len()
        );
    }
    let mut mat = material::Material::load(&a.material).unwrap_or_else(|e| {
        eprintln!("material: {e} -> using built-in green marble");
        material::Material::green_marble()
    });
    if let Some(t) = &a.tex {
        mat.texture = t.clone();
    }
    let tex = tex::Tex::from_zip(&mat.texture, mat.sharpen).expect("texture zip");
    eprintln!("material: {} (f0 {:.2}, reflect_rough {:.2}, tile {:.2} m)", mat.name, mat.f0, mat.reflect_rough, mat.tile_m);
    eprintln!(
        "load: {} edges, {} words, line {:.2} x {:.2} m, extrude {:.3} m, texture {}px  [{:?}]",
        mesh.vx.len(), mesh.word_count, mesh.width, mesh.text_height, mesh.height, tex.size0, t.elapsed()
    );

    let t = Instant::now();
    let mut sc = Scene::build(&mesh, mesh.height, 0.0007, 0.05);
    let (mut bx0, mut by0, mut bx1, mut by1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for i in 0..mesh.vx.len() {
        bx0 = bx0.min(mesh.vx[i]);
        bx1 = bx1.max(mesh.vx[i]);
        by0 = by0.min(mesh.vy[i]);
        by1 = by1.max(mesh.vy[i]);
    }
    let yc = 0.5 * (by0 + by1);
    // row of hanging spots, evenly spaced along the line, pools almost touching
    let n = ((bx1 - bx0) / a.pitch).round().max(1.0) as usize;
    let s = (bx1 - bx0) / n as f32;
    let r_out = 0.5 * s * a.spot;
    let r_in = 0.12 * r_out;
    let anchors: Vec<V> = (0..n).map(|i| v(bx0 + (i as f32 + 0.5) * s, yc, a.light_h)).collect(); // one row, rest positions
    sc.i0 = a.intensity * a.light_h * a.light_h;
    sc.lcol = v(1.0, 0.86, 0.68); // warm halogen-ish
    // rest pose (used only by the bake): lights hang straight down
    for an in &anchors {
        sc.lights.push(Light::new(*an, v(0.0, 0.0, -1.0), a.disc, r_in, r_out, a.light_h));
    }
    eprintln!("scene: sdf grid + {} swinging spots (pitch {:.2} m)  [{:?}]", n, s, t.elapsed());

    let text_albedo = v(0.80, 0.74, 0.62);
    let t = Instant::now();
    let cfg = bake::BakeCfg {
        texel: 0.002,
        margin: 0.30,
        bounces: a.bounces,
        spp: a.spp,
        light_samples: 16,
        floor_albedo: tex.avg * mat.albedo_gain,
        text_albedo,
        ambient: 0.05,
    };
    let maps = bake::bake(&sc, (bx0, by0, bx1, by1), &cfg);
    let bake_t = t.elapsed();
    eprintln!("BAKE (path-traced ambient + bounce, once): {}x{} texels, {} bounces x {} spp  [{:?}]", maps.w, maps.h, a.bounces, a.spp, bake_t);

    // thin-lens blur circle (px) = f^2 / (N (zf - f)) * |z - zf| / z / pixel pitch
    let dist = focal * 1.0 / a.w as f32; // metres from target to camera (frame = 1 m wide)
    let f_m = a.focal_mm / 1000.0;
    let pitch_m = a.sensor_mm / 1000.0 / a.w.max(a.h) as f32;
    let coc_k = a.blur * f_m * f_m / (a.fstop * (dist - f_m)) / pitch_m;
    let params = Params {
        f0_floor: mat.f0,
        alpha_floor: mat.reflect_rough,
        alpha_hl: mat.gloss_rough,
        lod_bias: mat.lod_bias,
        vein_gloss: mat.vein_gloss,
        env_gloss: mat.env_gloss,
        env_box: (0.5 * (bx0 + bx1), yc + 0.35),
        grade: a.grade,
        text_albedo,
        f0_text: 0.05,
        alpha_text: 0.22,
        tex_scale: mat.tile_m,
        ambient: 0.05,
        floor_gain: mat.albedo_gain,
        env: 0.0015,
        exposure: a.exposure,
        coc_k: coc_k,
        focus: dist,
        vignette: 0.40,
        bloom: 0.05,
        floor_fill: 0.0,
    };
    let (rest_h, disc) = (a.light_h, a.disc);
    let mut r = Renderer::new(a.w, a.h, focal, sc, maps, tex, params);
    // ---------- camera (local1 rig: orbit + roll + pan + hand shake) ----------
    let rig = rig::Rig::new(focal, dist, a.elev, a.azim, a.roll);
    eprintln!("shake seed: {seed}");
    let shk = shake::Shake::new(duration + 1.0, a.shake, seed);
    let pan_speed = a.pan_wps * mesh.avg_advance;
    let start_x = mesh.first_word_x - 0.30;
    let baseline_y = mesh.baseline_y;
    // Word slide (local1): a critically damped follow of the word being recited.
    // In `--range` mode the timeline comes from the recitation timings; otherwise
    // from a `--segments` file (one "start end" line per word).  Without either,
    // the original constant pan is kept.
    let segs: Vec<slide::Segment> = if let Some(j) = &job {
        j.timeline
            .iter()
            .map(|tw| slide::Segment {
                center: mesh.recitation_x[tw.rec_index],
                start: tw.start,
                end: tw.end,
            })
            .collect()
    } else if let Some(path) = &a.segments {
        let txt = std::fs::read_to_string(path).expect("segments file");
        let mut segs = Vec::new();
        for (i, line) in txt
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
            .enumerate()
        {
            let nums: Vec<f32> = line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if nums.len() >= 2 && i < mesh.word_x.len() {
                segs.push(slide::Segment {
                    center: mesh.word_x[i],
                    start: nums[0],
                    end: nums[1],
                });
            }
        }
        segs
    } else {
        Vec::new()
    };
    let sliding = !segs.is_empty();
    let sl = slide::Slide::build(&segs, duration, a.fps, a.smooth_time, a.lead);
    let cam_x = |t: f32| -> f32 {
        if sliding {
            sl.at(t)
        } else {
            start_x - pan_speed * t
        }
    };
    let cam_at = |t: f32| -> Camera {
        let (sx, sy) = shk.sample(t);
        let c = rig.aim([cam_x(t) + sx * k, baseline_y, sy * k]);
        Camera { pos: v(c.pos[0], c.pos[1], c.pos[2]), fwd: v(c.fwd[0], c.fwd[1], c.fwd[2]), right: v(c.right[0], c.right[1], c.right[2]), up: v(c.up[0], c.up[1], c.up[2]) }
    };
    // Depth (z-buffer) of the text's ink centre at time t.  The camera aims at
    // the BASELINE, which is below the visual centre, so a constant focus made
    // the bottom sharp and the top soft.  Tracking the ink centre keeps the
    // middle of the text sharp even while the hand shake moves the rig.
    let focus_at = |t: f32, cam: &Camera| -> f32 {
        let (sx, _) = shk.sample(t);
        (v(cam_x(t) + sx * k, 0.0, 0.0) - cam.pos).dot(cam.fwd)
    };

    // ---------- lamps: recorded physics keyframes (time based), one clip per lamp ----------
    let bank = lamps::LampBank::load(&a.lamp_file).expect("lamp keyframes (run: cargo run --release --bin gen_lamp_sims)");
    eprintln!("lamp keyframes: {} clips x {:.0} s @ {} Hz", bank.clips, bank.duration(), bank.hz);
    let lamps_at = |t: f32| -> Vec<Light> {
        anchors
            .iter()
            .enumerate()
            .map(|(i, an)| {
                let clip = (a.lamp_seed * 13 + i * 29 + 7) % bank.clips;
                let off = 4.0 + (i as f32 * 5.3 + a.lamp_seed as f32 * 1.1) % 10.0; // start mid-motion, 4..14 s into the 20 s clip
                let (d, axis) = bank.sample(clip, t + off);
                Light::new(*an + d, axis, disc, r_in, r_out, rest_h)
            })
            .collect()
    };

    let fsz = a.w * a.h * 3;
    if let Some(path) = &a.still {
        let mut buf = vec![0u8; fsz];
        r.lights = lamps_at(a.at);
        let cam = cam_at(a.at);
        // Focus on the text's ink centre, not the baseline: the baseline sits
        // below the visual centre, which made the bottom sharp and the top soft.
        r.p.focus = focus_at(a.at, &cam);
        let t = Instant::now();
        r.render(&cam, (a.at * 1000.0) as u32, &mut buf);
        eprintln!("still frame: {:?}", t.elapsed());
        let f = std::fs::File::create(path).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(f), a.w as u32, a.h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&buf).unwrap();
        return;
    }

    let tf = |f: u32| f as f32 / a.fps as f32;
    if a.bench {
        let mut buf = vec![0u8; fsz];
        r.lights = lamps_at(0.0);
        let cam0 = cam_at(0.0);
        r.p.focus = focus_at(0.0, &cam0);
        r.render(&cam0, 0, &mut buf);
        r.timings = Default::default();
        let n = 5u32;
        let t = Instant::now();
        for f in 0..n {
            r.lights = lamps_at(tf(f));
            let cam = cam_at(tf(f));
            r.p.focus = focus_at(tf(f), &cam);
            r.render(&cam, f, &mut buf);
        }
        let el = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
        eprintln!(
            "RENDER ONLY {}x{}: {:.1} ms/frame | trace {:.1} | edge-AA {:.1} | post {:.1}",
            a.w, a.h, el,
            r.timings.trace.as_secs_f64() * 1000.0 / n as f64,
            r.timings.aa.as_secs_f64() * 1000.0 / n as f64,
            r.timings.post.as_secs_f64() * 1000.0 / n as f64
        );
        return;
    }

    // ---------- video ----------
    let total = (duration * a.fps as f32).round() as u32;
    let out_path = std::path::PathBuf::from(&a.out);
    let video_path = out_path.with_file_name(format!(
        "{}.video.mp4",
        out_path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "out.mp4".into())
    ));
    let mut ff = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"])
        .args(["-s", &format!("{}x{}", a.w, a.h), "-r", &a.fps.to_string(), "-i", "-"])
        .args(["-vf", "scale=out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int"])
        .args(["-c:v", "libx264", "-preset", &a.preset, "-crf", &a.crf.to_string()])
        .args(["-x264-params", "aq-mode=3:aq-strength=1.0:deblock=-1,-1:psy-rd=1.0,0.0", "-pix_fmt", "yuv420p"])
        .args(["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"])
        .args(["-movflags", "+faststart", &video_path.to_string_lossy()])
        .stdin(Stdio::piped())
        .spawn()
        .expect("ffmpeg not found");
    let mut stdin = ff.stdin.take().unwrap();
    let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(2);
    let (rtx, rrx) = mpsc::channel::<Vec<u8>>();
    for _ in 0..3 {
        rtx.send(vec![0u8; fsz]).unwrap();
    }
    let writer = std::thread::spawn(move || {
        for b in rx {
            if stdin.write_all(&b).is_err() {
                break;
            }
            let _ = rtx.send(b);
        }
    });
    let t0 = Instant::now();
    let mut per = Vec::new();
    for f in 0..total {
        let mut buf = rrx.recv().unwrap();
        let t = Instant::now();
        r.lights = lamps_at(tf(f));
        let cam = cam_at(tf(f));
        // Per-frame focus on the text's ink centre: the shake moves the rig, so
        // a constant focus distance would drift onto the bottom half.
        r.p.focus = focus_at(tf(f), &cam);
        r.render(&cam, f, &mut buf);
        per.push(t.elapsed().as_secs_f64() * 1000.0);
        tx.send(buf).unwrap();
    }
    drop(tx);
    writer.join().unwrap();
    if !ff.wait().unwrap().success() {
        panic!("ffmpeg failed to encode the video");
    }
    // Mux the recitation without re-encoding the video.
    let mut audio_muxed = false;
    if !a.no_audio {
        if let Some(j) = &job {
            if let Some(track) = &j.track {
                let st = Command::new("ffmpeg")
                    .args(["-y", "-hide_banner", "-loglevel", "error"])
                    .arg("-i")
                    .arg(&video_path)
                    .arg("-i")
                    .arg(&track.path)
                    .args(["-map", "0:v", "-map", "1:a"])
                    .args(["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"])
                    .args(["-shortest", "-movflags", "+faststart"])
                    .arg(&out_path)
                    .status()
                    .expect("ffmpeg mux");
                if !st.success() {
                    panic!("ffmpeg failed to mux the recitation");
                }
                let _ = std::fs::remove_file(&video_path);
                audio_muxed = true;
            }
        }
    }
    if !audio_muxed {
        std::fs::rename(&video_path, &out_path).expect("rename video");
    }
    let el = t0.elapsed().as_secs_f64();
    let avg = per.iter().sum::<f64>() / per.len() as f64;
    let mx = per.iter().cloned().fold(0.0, f64::max);
    let mut sorted = per.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if sorted.len() % 2 == 0 { 0.5 * (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) } else { sorted[sorted.len() / 2] };
    let mn = sorted[0];
    let nn = total as f64;
    eprintln!(
        "FRAME TIME: median {:.1} ms | avg {:.1} | min {:.1} | max {:.1} | trace {:.1} | edge-AA {:.1} | post {:.1}  [{}x{}]",
        med, avg, mn, mx,
        r.timings.trace.as_secs_f64() * 1000.0 / nn,
        r.timings.aa.as_secs_f64() * 1000.0 / nn,
        r.timings.post.as_secs_f64() * 1000.0 / nn,
        a.w, a.h
    );
    eprintln!(
        "VIDEO: {} frames in {:.2}s wall (incl. encode) | one-off bake {:.2}s | total {:.2}s -> {}{}",
        total, el, bake_t.as_secs_f64(), t_all.elapsed().as_secs_f64(), out_path.display(),
        if audio_muxed { " (with recitation)" } else { "" }
    );
}
