//! Concatenate the per-ayah recitation files into one track and measure it.
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
}

fn probe_duration(p: &Path) -> Result<f32, String> {
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

/// Decode one ayah to PCM exactly the way the concat step will, and return its
/// real sample count.  Container durations include MP3 encoder padding, so they
/// drift by a few milliseconds per ayah; the decoded PCM does not.
fn decoded_duration(file: &Path, work_dir: &Path, tag: &str) -> Result<f32, String> {
    let wav = work_dir.join(format!("probe_{tag}.wav"));
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
    let d = probe_duration(&wav);
    let _ = std::fs::remove_file(&wav);
    d
}

/// Build `recitation.wav` from the ayah files of `range`.
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

    let mut offsets = Vec::with_capacity(files.len());
    let mut t = 0.0f32;
    for (i, f) in files.iter().enumerate() {
        offsets.push(t);
        t += decoded_duration(f, work_dir, &format!("{i:04}"))?;
    }

    let list = work_dir.join("audio_concat.txt");
    let mut s = String::new();
    for f in &files {
        let abs = f.canonicalize().unwrap_or_else(|_| f.to_path_buf());
        s.push_str(&format!("file '{}'\n", abs.display()));
    }
    std::fs::write(&list, s).map_err(|e| format!("{}: {e}", list.display()))?;

    let out = work_dir.join("recitation.wav");
    let st = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .args(["-f", "concat", "-safe", "0", "-i"])
        .arg(&list)
        .args(["-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le"])
        .arg(&out)
        .status()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    if !st.success() {
        return Err("ffmpeg failed to concatenate the recitation".into());
    }

    let duration = probe_duration(&out)?;
    eprintln!(
        "audio: {} ayahs, decoded {:.3}s (last ayah starts at {:.3}s)",
        files.len(),
        duration,
        offsets.last().copied().unwrap_or(0.0)
    );
    Ok(AudioTrack {
        path: out,
        duration,
        offsets,
    })
}
/// Peak level of a file, in dBFS (from ffmpeg volumedetect).
fn probe_max_volume(p: &Path) -> Result<f32, String> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-i"])
        .arg(p)
        .args(["-af", "volumedetect", "-f", "null", "-"])
        .output()
        .map_err(|e| format!("ffmpeg {}: {e}", p.display()))?;
    let s = String::from_utf8_lossy(&out.stderr);
    for line in s.lines() {
        if let Some(rest) = line.split("max_volume:").nth(1) {
            if let Some(v) = rest.trim().split_whitespace().next().and_then(|x| x.parse::<f32>().ok()) {
                return Ok(v);
            }
        }
    }
    Err(format!("volumedetect {}: no max_volume", p.display()))
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

/// Mix a background sound under the recitation.
///
/// The sound is first normalised to a safe -6 dBFS peak (so every file sits at
/// the same level regardless of how loud it was), then scaled by `sfx_gain`
/// (default 0.1) and by `reciter_gain`.  It is also ducked under the recitation
/// with a side-chain compressor (threshold -26 dB, ratio 8:1), so the reciter is
/// always clearly on top.  A limiter is the final safety net so nothing clips.
pub fn mix_sfx(
    rec: &Path,
    sfx: &Path,
    out: &Path,
    duration: f32,
    sfx_gain: f32,
    reciter_gain: f32,
) -> Result<(), String> {
    let work = out.parent().unwrap_or_else(|| Path::new("."));
    let looped = work.join("sfx_loop.wav");
    loop_with_crossfade(sfx, &looped, 0.5)?;
    let max = probe_max_volume(&looped)?;
    let norm = -6.0 - max; // bring the loudest sample to -6 dBFS
    let fade_out = (duration - 1.0).max(0.0);
    let filter = format!(
        "[1:a]volume={norm:.3}dB,atrim=0:{duration:.3},volume={sfx_gain:.4},volume={reciter_gain:.4},\
         afade=t=in:d=1,afade=t=out:st={fade_out:.3}:d=1[sfx];\
         [0:a]volume={reciter_gain:.4},asplit=2[rec][sc];\
         [sfx][sc]sidechaincompress=threshold=0.05:ratio=8:attack=20:release=300[duck];\
         [rec][duck]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.98[out]"
    );
    let st = Command::new("ffmpeg")
        .args(["-y", "-hide_banner", "-loglevel", "error"])
        .arg("-i")
        .arg(rec)
        .args(["-stream_loop", "-1", "-i"])
        .arg(&looped)
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
