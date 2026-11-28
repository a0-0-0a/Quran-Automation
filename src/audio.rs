//! Join the per-ayah recitation files into one track with short crossfades and
//! measure the result.
//!
//! GitHub Actions will not have `Assets/reciters` in the repo; a fetch step
//! downloads the same per-ayah files (everyayah.com or the B2 mirror) into a
//! cache directory and points `--audio-dir` at it.  Everything below works on
//! plain files, so local and CI take exactly the same path.

use crate::quran::Range;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct AudioTrack {
    /// A single PCM wav holding the whole recitation.
    pub path: PathBuf,
    /// Exact duration of `path` in seconds.
    pub duration: f32,
    /// Absolute start time (seconds) of each ayah in `path`.
    pub offsets: Vec<f32>,
    /// Decoded duration of each ayah's own file, in seconds.  Word timings are
    /// clamped to these, not to the shorter crossfaded slots, so a word that
    /// runs into the fade is not cut short in the timeline.
    pub durations: Vec<f32>,
}

pub fn probe_duration(p: &Path) -> Result<f32, String> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
        ])
        .arg(p)
        .output()
        .map_err(|e| format!("ffprobe {}: {e}", p.display()))?;
    if !out.status.success() {
        return Err(format!(
            "ffprobe {}: {}",
            p.display(),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .map_err(|_| format!("ffprobe {}: bad duration", p.display()))
}

/// Length of the crossfade at every ayah join, in seconds.  Short enough that
/// it only softens the seam (and never eats a whole short ayah), long enough to
/// remove the hard cut.  The timeline is shifted by exactly this amount at each
/// join (see `build`), so word timings stay locked to the audio.
const CROSSFADE_SECS: f32 = 0.12;

/// Decode one ayah to the exact PCM format used by the crossfade step and
/// return its path plus real duration.  Container durations include MP3
/// encoder padding, so they drift by a few milliseconds per ayah; the decoded
/// PCM does not.
fn decode_ayah(file: &Path, work_dir: &Path, tag: &str) -> Result<(PathBuf, f32), String> {
    let wav = work_dir.join(format!("ayah_{tag}.wav"));
    let st = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .arg("-i")
        .arg(file)
        .args(["-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le"])
        .arg(&wav)
        .status()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    if !st.success() {
        return Err(format!("ffmpeg failed to decode {}", file.display()));
    }
    let d = probe_duration(&wav)?;
    Ok((wav, d))
}

/// Build `recitation.wav` from the ayah files of `range`.
///
/// Consecutive ayahs are joined with a short equal-power crossfade instead of a
/// hard concat.  Each join shortens the track by the fade length, so the start
/// of ayah `i+1` is `offset[i] + duration[i] - fade`; `offsets` below is
/// computed with the same arithmetic.  The output duration is checked against
/// that prediction, which guarantees the word timeline and the audio cannot
/// drift apart.
pub fn build(audio_dir: &Path, range: Range, work_dir: &Path) -> Result<AudioTrack, String> {
    std::fs::create_dir_all(work_dir).map_err(|e| format!("{}: {e}", work_dir.display()))?;

    let mut files = Vec::new();
    for a in range.from..=range.to {
        let p = crate::quran::ayah_audio(audio_dir, range.surah, a).ok_or_else(|| {
            format!(
                "missing recitation for {}.{} in {}",
                range.surah,
                a,
                audio_dir.display()
            )
        })?;
        files.push(p);
    }

    let mut decoded = Vec::with_capacity(files.len());
    for (i, f) in files.iter().enumerate() {
        decoded.push(decode_ayah(f, work_dir, &format!("{i:04}"))?);
    }
    let durations: Vec<f32> = decoded.iter().map(|(_, d)| *d).collect();

    // Per-join fade, clamped so two fades can never meet inside a short ayah.
    let mut fades = Vec::with_capacity(durations.len().saturating_sub(1));
    for w in durations.windows(2) {
        let f = CROSSFADE_SECS.min(w[0] * 0.4).min(w[1] * 0.4).max(0.005);
        fades.push(f);
    }

    let mut offsets = Vec::with_capacity(durations.len());
    let mut t = 0.0f32;
    for (i, d) in durations.iter().enumerate() {
        offsets.push(t);
        t += d - fades.get(i).copied().unwrap_or(0.0);
    }
    let expected_duration = t;

    let out = work_dir.join("recitation.wav");
    if decoded.len() == 1 {
        std::fs::rename(&decoded[0].0, &out)
            .map_err(|e| format!("{}: {e}", decoded[0].0.display()))?;
    } else {
        let mut filter = String::new();
        for (i, fade) in fades.iter().enumerate() {
            let left = if i == 0 {
                "[0:a]".to_string()
            } else {
                format!("[xf{i}]")
            };
            let right = format!("[{}:a]", i + 1);
            let dest = if i + 1 == fades.len() {
                "[out]".to_string()
            } else {
                format!("[xf{}]", i + 1)
            };
            filter.push_str(&format!(
                "{left}{right}acrossfade=d={fade:.6}:c1=qsin:c2=qsin{dest};"
            ));
        }
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-y", "-hide_banner", "-loglevel", "error"]);
        for (wav, _) in &decoded {
            cmd.arg("-i").arg(wav);
        }
        let st = cmd
            .args(["-filter_complex", &filter, "-map", "[out]"])
            .args(["-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le"])
            .arg(&out)
            .status()
            .map_err(|e| format!("ffmpeg: {e}"))?;
        if !st.success() {
            return Err("ffmpeg failed to crossfade the recitation".into());
        }
    }

    let duration = probe_duration(&out)?;
    let drift = duration - expected_duration;
    if drift.abs() > 0.005 {
        return Err(format!(
            "audio: crossfade timeline drifted by {drift:+.3}s \
             (expected {expected_duration:.3}s, got {duration:.3}s)"
        ));
    }

    for (wav, _) in &decoded {
        let _ = std::fs::remove_file(wav);
    }

    eprintln!(
        "audio: {} ayahs, up to {:.0} ms crossfades, {:.3}s (timeline expected {:.3}s), \
         last ayah starts at {:.3}s",
        files.len(),
        fades.first().copied().unwrap_or(0.0) * 1000.0,
        duration,
        expected_duration,
        offsets.last().copied().unwrap_or(0.0)
    );
    Ok(AudioTrack {
        path: out,
        duration,
        offsets,
        durations,
    })
}
/// Peak level of a file, in dBFS (from ffmpeg volumedetect).
fn probe_mean_volume(p: &Path) -> Option<f32> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-i"])
        .arg(p)
        .args(["-af", "volumedetect", "-f", "null", "-"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stderr);
    for line in s.lines() {
        if let Some(rest) = line.split("mean_volume:").nth(1) {
            if let Some(v) = rest.trim().split_whitespace().next().and_then(|x| x.parse::<f32>().ok()) {
                if v.is_finite() {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Build a seamless loop of `sfx`: the last `xfade` seconds are crossfaded with
/// the first `xfade` seconds, so the loop point fades instead of cutting.
fn loop_with_crossfade(sfx: &Path, out: &Path, xfade: f32) -> Result<(), String> {
    let d = probe_duration(sfx)?;
    let x = xfade.min(d / 3.0).max(0.02);
    let a_end = (d - x).max(0.0);
    let filter = format!(
        "[0:a]asplit=2[a][b];\
         [a]atrim=0:{a_end:.3},asetpts=PTS-STARTPTS[a1];\
         [b]atrim={a_end:.3}:{d:.3},asetpts=PTS-STARTPTS[b1];\
         [b1][a1]acrossfade=d={x:.3}:c1=tri:c2=tri[out]"
    );
    let st = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .arg("-i")
        .arg(sfx)
        .args(["-filter_complex", &filter, "-map", "[out]"])
        .args(["-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le"])
        .arg(out)
        .status()
        .map_err(|e| format!("ffmpeg sfx loop: {e}"))?;
    if !st.success() {
        return Err("ffmpeg failed to build the sfx loop".into());
    }
    Ok(())
}

/// Deterministic 0..1 value from the render seed.
fn rand01(seed: u32) -> f32 {
    let mut x = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    x ^= x >> 13;
    x = x.wrapping_mul(1_274_126_177);
    x ^= x >> 16;
    (x & 0x00ff_ffff) as f32 / 16_777_216.0
}

/// Prepare at least `duration` seconds of SFX.
///
/// Long files are cut from a random offset, so a 13-minute ambience does not
/// always start with the same first minute.  Short files keep the old seamless
/// loop behaviour.  Returns `true` when the result must be looped.
fn prepare_sfx(sfx: &Path, out: &Path, duration: f32, seed: u32) -> Result<bool, String> {
    let d = probe_duration(sfx)?;
    if d > duration + 1.0 {
        let max_offset = (d - duration - 0.5).max(0.0);
        let offset = rand01(seed) * max_offset;
        eprintln!("sfx: start offset {offset:.1}s of {d:.1}s");
        let st = Command::new("ffmpeg")
            .args(["-y", "-hide_banner", "-loglevel", "error"])
            .args(["-ss", &format!("{offset:.3}")])
            .arg("-i")
            .arg(sfx)
            .args(["-t", &format!("{duration:.3}")])
            .args(["-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le"])
            .arg(out)
            .status()
            .map_err(|e| format!("ffmpeg sfx cut: {e}"))?;
        if !st.success() {
            return Err("ffmpeg failed to cut the background sound".into());
        }
        Ok(false)
    } else {
        loop_with_crossfade(sfx, out, 0.5)?;
        Ok(true)
    }
}

/// Mix a background sound under the recitation.
///
/// The sound is first normalised to a safe -6 dBFS peak (so every file sits at
/// its average level is scaled to `sfx_gain` (default 0.10 = 10%) of the
/// recitation average and by `reciter_gain`.  It is smoothly ducked under the
/// recitation (side-chain compressor, attack/release, NEVER cut to silence),
/// recitation is safe at `reciter_gain`, and a final limiter prevents clipping.
pub fn mix_sfx(
    rec: &Path,
    sfx: &Path,
    out: &Path,
    duration: f32,
    sfx_gain: f32,
    reciter_gain: f32,
    seed: u32,
) -> Result<(), String> {
    let work = out.parent().unwrap_or_else(|| Path::new("."));
    let prepared = work.join("sfx_prepared.wav");
    let needs_loop = prepare_sfx(sfx, &prepared, duration, seed)?;
    let fade_out = (duration - 1.0).max(0.0);

    // ---- loudness-aware, safe mixing --------------------------------------
    // 1) measure the recitation's average loudness (dB, once per short);
    // 2) scale SFX so its average = `sfx_gain` (default 0.10 = 10%) of the
    //    recitation average; recitation stays safe at `reciter_gain` (1.0);
    // 3) smooth `sidechaincompress` ducking on the recitation sidechain: SFX
    //    smoothly fades down a little when the reciter is loud, and returns to
    //    its baseline when his voice lowers/stops — SFX is NEVER cut to
    //    silence, never interrupts the reciter (safe spots stay safe);
    // 4) output safely limited (`alimiter=0.98`); fades in/out; deterministic
    //    random SFX + start offset; looped short files unchanged.
    let rec_mean = probe_mean_volume(rec).unwrap_or(-20.0);
    let sfx_mean = probe_mean_volume(&prepared).unwrap_or(-20.0);
    let rec_level_db = rec_mean + 20.0 * reciter_gain.max(1e-4).log10();
    let target_ratio = sfx_gain.clamp(0.0, 1.0).max(1e-4); // 0.10 = 10%
    let sfx_vol_db = rec_level_db - sfx_mean + 20.0 * target_ratio.log10();

    let filter = format!(
        "[1:a]atrim=0:{duration:.3},volume={sfx_vol_db:.3}dB,\
         afade=t=in:d=1,afade=t=out:st={fade_out:.3}:d=1[sfx0];\
         [0:a]volume={reciter_gain:.4},asplit=2[rec][sc];\
         [sfx0][sc]sidechaincompress=threshold=0.05:ratio=4:attack=20:release=300[sfx];\
         [rec][sfx]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.98[out]"
    );
    eprintln!(
        "sfx level: recitation {rec_mean:.1} dB avg -> sfx {sfx_mean:.1} dB avg, sfx vol {sfx_vol_db:+.1} dB (target {:.0}%)",
        target_ratio * 100.0
    );
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-hide_banner", "-loglevel", "error"])
        .arg("-i")
        .arg(rec);
    if needs_loop {
        cmd.args(["-stream_loop", "-1"]);
    }
    let st = cmd
        .arg("-i")
        .arg(&prepared)
        .args(["-filter_complex", &filter, "-map", "[out]"])
        .args(["-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le"])
        .arg(out)
        .status()
        .map_err(|e| format!("ffmpeg sfx mix: {e}"))?;
    if !st.success() {
        return Err("ffmpeg failed to mix the background sound".into());
    }
    Ok(())
}
