//! Quran text and recitation timing data.
//!
//! Text: `digital-khatt-v2.json` (a zip) maps every `surah:ayah:word` to the
//! exact Uthmani text that the DigitalKhattV2 font knows how to draw.  The last
//! entry of every ayah is the ayah-end marker (`۝١`); it is rendered but it is
//! not spoken, so it gets no camera key.
//!
//! Timings: the per-reciter DB (`assets/timings_db/<reciter>.json`) maps `surah:ayah` to the spoken words with
//! `start_ms`/`end_ms` relative to that ayah's own audio file.  Concatenating
//! the ayah files shifts each ayah by the sum of the preceding durations.

use crate::data;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// One word as it appears in the mesh.
#[derive(Clone, Debug)]
pub struct QuranWord {
    pub text: String,
    /// Global recitation index (reading order, spoken words only), or `None`
    /// for a non-spoken word such as the ayah-end marker.
    pub recitation: Option<usize>,
}

#[derive(Clone, Debug)]
struct StoredWord {
    text: String,
    spoken: bool,
}

/// The complete Quran text, grouped by ayah.
pub struct QuranText {
    ayahs: BTreeMap<(u16, u16), Vec<StoredWord>>,
}

impl QuranText {
    /// Load from a `.zip` containing `digital-khatt-v2.json`.
    pub fn load_zip(path: &str) -> Result<QuranText, String> {
        let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("{path}: {e}"))?;
        let idx = (0..zip.len())
            .find(|&i| {
                zip.by_index(i)
                    .map(|f| f.name().ends_with(".json"))
                    .unwrap_or(false)
            })
            .ok_or_else(|| format!("{path}: no .json entry"))?;
        let mut f = zip.by_index(idx).map_err(|e| format!("{path}: {e}"))?;
        let mut s = String::new();
        f.read_to_string(&mut s).map_err(|e| format!("{path}: {e}"))?;
        QuranText::from_json(&s)
    }

    /// Load from an already-decoded JSON string.
    pub fn from_json(s: &str) -> Result<QuranText, String> {
        let v: Value = serde_json::from_str(s).map_err(|e| format!("quran json: {e}"))?;
        let obj = v.as_object().ok_or("quran json: expected an object")?;
        let mut ayahs: BTreeMap<(u16, u16), Vec<(u16, StoredWord)>> = BTreeMap::new();
        for (key, val) in obj {
            let mut it = key.split(':');
            let (Some(s), Some(a), Some(w), None) = (it.next(), it.next(), it.next(), it.next())
            else {
                return Err(format!("quran json: bad key {key}"));
            };
            let (s, a, w): (u16, u16, u16) = (
                s.parse().map_err(|_| format!("quran json: bad key {key}"))?,
                a.parse().map_err(|_| format!("quran json: bad key {key}"))?,
                w.parse().map_err(|_| format!("quran json: bad key {key}"))?,
            );
            let text = val
                .get("text")
                .and_then(|x| x.as_str())
                .ok_or_else(|| format!("quran json: {key} has no text"))?;
            if text.contains('\u{06dd}') {
                ayahs.entry((s, a)).or_default().push((
                    w,
                    StoredWord { text: text.to_string(), spoken: false },
                ));
            } else {
                for part in text.split_whitespace() {
                    ayahs.entry((s, a)).or_default().push((
                        w,
                        StoredWord { text: part.to_string(), spoken: true },
                    ));
                }
            }
        }
        // reading order inside an ayah
        for words in ayahs.values_mut() {
            words.sort_by_key(|(w, _)| *w);
        }
        let ayahs = ayahs
            .into_iter()
            .map(|(k, ws)| (k, ws.into_iter().map(|(_, w)| w).collect()))
            .collect();
        Ok(QuranText { ayahs })
    }

    pub fn ayah(&self, surah: u16, ayah: u16) -> Option<&[StoredWord]> {
        self.ayahs.get(&(surah, ayah)).map(|v| v.as_slice())
    }

    /// Flat word list for a range, in reading order.  `recitation` is renumbered
    /// globally so it lines up with the mesh's `recitation_x` and the timeline.
    pub fn range_words(&self, range: Range) -> Result<Vec<QuranWord>, String> {
        let mut out = Vec::new();
        let mut rec = 0usize;
        for a in range.from..=range.to {
            let words = self
                .ayah(range.surah, a)
                .ok_or_else(|| format!("no text for {}.{}", range.surah, a))?;
            for w in words {
                let r = if w.spoken {
                    let i = rec;
                    rec += 1;
                    Some(i)
                } else {
                    None
                };
                out.push(QuranWord { text: w.text.clone(), recitation: r });
            }
        }
        Ok(out)
    }
}

/// One spoken word's timing, in seconds, relative to its ayah's audio file.
#[derive(Clone, Copy, Debug)]
pub struct WordTime {
    pub start: f32,
    pub end: f32,
}

/// Per-ayah word timings.
pub struct Timings {
    map: BTreeMap<(u16, u16), Vec<WordTime>>,
}

impl Timings {
    pub fn load(path: &str) -> Result<Timings, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let v: Value = serde_json::from_str(&s).map_err(|e| format!("{path}: {e}"))?;
        let obj = v.as_object().ok_or_else(|| format!("{path}: expected an object"))?;
        let mut map = BTreeMap::new();
        for (key, val) in obj {
            let mut it = key.split(':');
            let (Some(s), Some(a), None) = (it.next(), it.next(), it.next()) else {
                return Err(format!("{path}: bad key {key}"));
            };
            let (s, a): (u16, u16) = (
                s.parse().map_err(|_| format!("{path}: bad key {key}"))?,
                a.parse().map_err(|_| format!("{path}: bad key {key}"))?,
            );
            let arr = val
                .as_array()
                .ok_or_else(|| format!("{path}: {key} is not an array"))?;
            let mut words = Vec::with_capacity(arr.len());
            for w in arr {
                let start = w
                    .get("start_ms")
                    .and_then(|x| x.as_f64())
                    .ok_or_else(|| format!("{path}: {key} bad start_ms"))?
                    as f32
                    / 1000.0;
                let end = w
                    .get("end_ms")
                    .and_then(|x| x.as_f64())
                    .ok_or_else(|| format!("{path}: {key} bad end_ms"))?
                    as f32
                    / 1000.0;
                words.push(WordTime { start, end });
            }
            words.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
            map.insert((s, a), words);
        }
        Ok(Timings { map })
    }

    pub fn ayah(&self, surah: u16, ayah: u16) -> Option<&[WordTime]> {
        self.map.get(&(surah, ayah)).map(|v| v.as_slice())
    }
}

/// A surah + ayah span, e.g. `2:1-5` (or `2:25` for a single ayah).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    pub surah: u16,
    pub from: u16,
    pub to: u16,
}

impl Range {
    pub fn parse(s: &str) -> Result<Range, String> {
        let s = s.trim();
        let (surah_s, rest) = s.split_once(':').ok_or_else(|| format!("bad range {s}"))?;
        let surah: u16 = surah_s
            .trim()
            .parse()
            .map_err(|_| format!("bad surah in {s}"))?;
        let (from, to) = if let Some((a, b)) = rest.split_once('-') {
            (
                a.trim().parse().map_err(|_| format!("bad range {s}"))?,
                b.trim().parse().map_err(|_| format!("bad range {s}"))?,
            )
        } else if let Some((a, b)) = rest.split_once(':') {
            (
                a.trim().parse().map_err(|_| format!("bad range {s}"))?,
                b.trim().parse().map_err(|_| format!("bad range {s}"))?,
            )
        } else {
            let a: u16 = rest
                .trim()
                .parse()
                .map_err(|_| format!("bad range {s}"))?;
            (a, a)
        };
        if from == 0 || to < from {
            return Err(format!("bad range {s}"));
        }
        Ok(Range { surah, from, to })
    }

    /// Reject anything that is not one of the meaning-safe spans.
    pub fn validate_safe(&self) -> Result<(), String> {
        if data::is_safe_range(self.surah, self.from, self.to) {
            Ok(())
        } else {
            Err(format!(
                "{} ({}):{} is not a meaning-safe range; the reciter may not read it",
                data::surah_name(self.surah),
                self.surah,
                self
            ))
        }
    }

    pub fn ayahs(&self) -> impl Iterator<Item = u16> + '_ {
        self.from..=self.to
    }
}

impl std::fmt::Display for Range {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.from == self.to {
            write!(f, "{}:{}", self.surah, self.from)
        } else {
            write!(f, "{}:{}-{}", self.surah, self.from, self.to)
        }
    }
}

/// One camera key: word `rec_index` is centred from `start` to `end` seconds.
#[derive(Clone, Copy, Debug)]
pub struct TimelineWord {
    pub rec_index: usize,
    pub start: f32,
    pub end: f32,
}

/// Build the absolute camera timeline for a range.
///
/// `offsets[a]` is the absolute start time (seconds) of ayah `from + a` in the
/// concatenated recitation.  The number of spoken words must match the text.
pub fn build_timeline(
    range: Range,
    text_words: &[QuranWord],
    timings: &Timings,
    offsets: &[f32],
    total_duration: f32,
) -> Result<Vec<TimelineWord>, String> {
    if offsets.len() != (range.to - range.from + 1) as usize {
        return Err(format!(
            "timeline: {} ayah offsets for range {}",
            offsets.len(),
            range
        ));
    }
    let mut out = Vec::new();
    let mut rec = 0usize;
    for (i, a) in (range.from..=range.to).enumerate() {
        let words = timings
            .ayah(range.surah, a)
            .ok_or_else(|| format!("no timings for {}.{}", range.surah, a))?;
        // PRIMARY timing = the real downloaded audio file.  The per-ayah length
        // is the difference between the measured offsets (ffprobe), so a word
        // can never run past the end of its own audio file.
        let ayah_dur = if i + 1 < offsets.len() {
            offsets[i + 1] - offsets[i]
        } else {
            total_duration - offsets[i]
        }
        .max(0.001);
        // A single ayah-long segment (no per-word data) is stretched to the
        // real audio length so the camera covers the whole ayah.
        let ayah_only = words.len() == 1 && words[0].start <= 0.001;
        for w in words {
            let st = w.start.clamp(0.0, ayah_dur);
            let mut en = w.end.clamp(0.0, ayah_dur);
            if ayah_only {
                en = ayah_dur;
            }
            out.push(TimelineWord {
                rec_index: rec,
                start: offsets[i] + st,
                end: offsets[i] + en.max(st),
            });
            rec += 1;
        }
    }
    let spoken = text_words.iter().filter(|w| w.recitation.is_some()).count();
    if spoken != out.len() {
        return Err(format!(
            "timeline: {} spoken words in text but {} timing entries for {}",
            spoken,
            out.len(),
            range
        ));
    }
    Ok(out)
}

/// Convenience: the audio file path for one ayah (mp3 / opus / ogg / m4a).
pub fn ayah_audio(dir: &Path, surah: u16, ayah: u16) -> Option<std::path::PathBuf> {
    for ext in ["mp3", "opus", "ogg", "m4a", "aac"] {
        let p = dir.join(format!("{:03}{:03}.{}", surah, ayah, ext));
        if p.is_file() {
            return Some(p);
        }
    }
    None
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_ranges_are_exact() {
        assert!(Range::parse("114:1-6").unwrap().validate_safe().is_ok());
        // Starting/stopping inside the single safe span is not allowed.
        assert!(Range::parse("114:1-5").unwrap().validate_safe().is_err());
        assert!(Range::parse("114:2-6").unwrap().validate_safe().is_err());
        assert!(Range::parse("2:1-7").unwrap().validate_safe().is_err());
    }

    #[test]
    fn loads_text_and_markers() {
        let t = QuranText::load_zip("assets/digital-khatt-v2.json.zip").unwrap();
        let w = t
            .range_words(Range { surah: 1, from: 1, to: 1 })
            .unwrap();
        assert_eq!(w.iter().filter(|x| x.recitation.is_some()).count(), 4);
        assert_eq!(w.last().unwrap().recitation, None); // ayah-end marker
        assert!(w.last().unwrap().text.contains('\u{06dd}'));
    }

    #[test]
    fn timings_match_spoken_words() {
        let t = QuranText::load_zip("assets/digital-khatt-v2.json.zip").unwrap();
        let timings = Timings::load("assets/timings_db/Yasser_Ad-Dussary.json").unwrap();
        let range = Range { surah: 2, from: 6, to: 7 };
        let words = t.range_words(range).unwrap();
        let spoken = words.iter().filter(|x| x.recitation.is_some()).count();
        let timed: usize = (range.from..=range.to)
            .map(|a| timings.ayah(range.surah, a).unwrap().len())
            .sum();
        assert_eq!(spoken, timed);
    }
}
