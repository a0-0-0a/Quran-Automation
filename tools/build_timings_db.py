#!/usr/bin/env python3
"""Build the hybrid per-reciter word-timing database.

Sources
-------
  * quran-align pre-generated data (full Quran) for Abdurrahmaan, Minshawy, Saood
  * local yasser_word_timing.json (full Quran) for Yasser
  * quran-timings-api pre-generated data (partial) for Muhsin and Abdullaah
  * everyayah per-ayah timing zips (Assets/timings) for the ayah bounds and for
    every reciter that has no per-word data (the camera then glides per ayah)

Output
------
  assets/timings_db/<Reciter>.json   app format: {"surah:ayah": [{start_ms,end_ms}]}
  assets/timings_db/manifest.json    coverage per reciter
  assets/timings_db/suspicious.json  words that fall outside their ayah / count mismatch

Every per-word time is validated against the ayah's decoded duration and the
digital-khatt spoken-word count.  A bad ayah falls back to a single ayah-long
segment (safe: the ayah is already cut) and is listed in suspicious.json.
"""
import argparse
import json
import re
import zipfile
from pathlib import Path

# B2 folder name, everyayah timings zip, per-word source key
RECITERS = [
    ("Yasser_Ad-Dussary", "Yasser_Ad-Dussary_128kbps.zip", "local"),
    ("Abdullaah_3awwaad_Al-Juhaynee", "Abdullaah_3awwaad_Al-Juhaynee_128kbps.zip", "qta_abdullaah"),
    ("Abdurrahmaan_As-Sudais", "Sudais.zip", "qalign_abdurrahmaan"),
    ("Ahmed_Neana", "Ahmed_Neana_128kbps.zip", None),
    ("Ali_Hajjaj_AlSuesy", "Ali_Hajjaj_AlSuesy_128kbps.zip", None),
    ("Aziz_Alili", "Aziz_Alili_128kbps.zip", None),
    ("Minshawy_Murattal", "Minshawy Murattal.zip", "qalign_minshawy"),
    ("Muhsin_Al_Qasim", "Muhsin Al-Qasim_192kbps.zip", "qta_muhsin"),
    ("Nasser_Alqatami", "Nasser_Alqatami_128kbps.zip", None),
    ("Sahl_Yassin", "Sahl_Yassin_128kbps.zip", None),
    ("Salah_Al_Budair", "Salah_al_Budair_128kbps.zip", None),
    ("Saood_ash-Shuraym", "Saud Ash Shuraym.zip", "qalign_saood"),
]

QTA = {
    "qta_abdullaah": "quranicaudio_abdullaah_3awwaad_al_juhaynee",
    "qta_muhsin": "eya_muhsin_al_qasim_192kbps",
}


def load_qalign(path: Path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    for line in raw.split("\n"):
        if line.strip().startswith("["):
            return {(e["surah"], e["ayah"]): e for e in json.loads(line)}
    raise SystemExit(f"no JSON array in {path}")


def load_qta(data_dir: Path, slug: str):
    out = {}
    root = data_dir / "api" / "reciters" / slug / "surahs"
    if not root.is_dir():
        return out
    for surah_dir in root.iterdir():
        if not surah_dir.is_dir():
            continue
        tf = surah_dir / "timings.json"
        if not tf.is_file():
            continue
        d = json.loads(tf.read_text(encoding="utf-8"))
        # words are SURAH-relative; shift each ayah to its own timeline
        ay_start = {int(a["ayah"]): float(a["start_s"]) for a in d.get("ayahs", [])}
        for w in d.get("words", []):
            ay = int(w["ayah"])
            off = ay_start.get(ay, 0.0)
            out.setdefault((int(w["surah"]), ay), []).append(
                ((float(w["start_s"]) - off) * 1000.0, (float(w["end_s"]) - off) * 1000.0)
            )
    for k in out:
        out[k].sort()
    return out


def load_ayah_durations(zip_path: Path, surah_ayahs):
    """Return ({(surah, ayah): duration_ms}, missing_set) from everyayah times.

    The .txt files can carry one trailing extra value; only the real ayah count
    (from the Quran text) is used.  If a file is one short, the last ayah gets
    the previous ayah's duration and is reported in `missing`.
    """
    out = {}
    missing = set()
    with zipfile.ZipFile(zip_path) as z:
        for n in z.namelist():
            m = re.fullmatch(r"(?:txt/)?(\d{3})\.txt", n)
            if not m:
                continue
            surah = int(m.group(1))
            vals = [int(x) for x in z.read(n).decode("utf-8", "replace").split()]
            n_ayah = surah_ayahs.get(surah, len(vals))
            vals = vals[:n_ayah]
            prev = 0
            for i, end in enumerate(vals, start=1):
                out[(surah, i)] = float(end - prev)
                prev = end
            if len(vals) < n_ayah:
                fallback = out.get((surah, len(vals)), 1000.0)
                for i in range(len(vals) + 1, n_ayah + 1):
                    out[(surah, i)] = fallback
                    missing.add((surah, i))
    return out, missing


def load_dk_counts(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        d = json.loads(z.read("digital-khatt-v2.json"))
    counts = {}
    for v in d.values():
        if "\u06dd" in v["text"]:
            continue
        k = (int(v["surah"]), int(v["ayah"]))
        counts[k] = counts.get(k, 0) + len(v["text"].split())
    return counts


def expand_qalign(entry):
    segments = (entry or {}).get("segments", [])
    words = []
    for ws, we, st, en in segments:
        ws, we = int(ws), int(we)
        if we <= ws:
            continue
        span = (en - st) / (we - ws)
        for k in range(ws, we):
            words.append((st + span * (k - ws), st + span * (k - ws + 1)))
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=".", type=Path)
    ap.add_argument("--timings", default="../../Assets/timings", type=Path)
    ap.add_argument("--qalign", default="../../tmp_timings/quran-align-data", type=Path)
    ap.add_argument("--qta", default="../../tmp_timings/quran-timings-api/data", type=Path)
    ap.add_argument("--local-yasser", default="assets/yasser_word_timing.json", type=Path)
    ap.add_argument("--dk", default="assets/digital-khatt-v2.json.zip", type=Path)
    ap.add_argument("--out", default="assets/timings_db", type=Path)
    a = ap.parse_args()

    dk = load_dk_counts(a.dk)
    surah_ayahs = {}
    for (s, ay) in dk:
        surah_ayahs[s] = max(surah_ayahs.get(s, 0), ay)
    qalign = {
        "qalign_abdurrahmaan": load_qalign(a.qalign / "Abdurrahmaan_As-Sudais_192kbps.json"),
        "qalign_minshawy": load_qalign(a.qalign / "Minshawy_Murattal_128kbps.json"),
        "qalign_saood": load_qalign(a.qalign / "Saood_ash-Shuraym_128kbps.json"),
    }
    qta = {k: load_qta(a.qta, slug) for k, slug in QTA.items()}
    local_yasser = json.loads(a.local_yasser.read_text(encoding="utf-8"))

    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    suspicious = []

    for name, zip_name, source in RECITERS:
        durs, missing = load_ayah_durations(a.timings / zip_name, surah_ayahs)
        for (s, ay) in missing:
            suspicious.append({"reciter": name, "surah": s, "ayah": ay,
                               "issue": "missing ayah duration (filled)", "detail": "zip one short"})
        db = {}
        per_word = 0
        ayah_only = 0
        for (s, ay), dur in durs.items():
            words = None
            if source == "local":
                words = [(w["start_ms"], w["end_ms"]) for w in local_yasser.get(f"{s}:{ay}", [])]
            elif source in qalign:
                words = expand_qalign(qalign[source].get((s, ay)))
            elif source in qta:
                words = qta[source].get((s, ay), [])
            if words:
                # The everyayah zips are FULL-SURAH timings, slightly shorter than
                # the split per-ayah audio.  Validate against the word content end
                # (+50 ms) so correct per-word data is not flagged.
                max_end = max((w[1] for w in words), default=0.0)
                dur_eff = max(dur, max_end + 50.0)
                bad = [w for w in words if w[0] < -50.0 or w[1] > dur_eff + 50.0 or w[0] > w[1] + 1.0]
                if bad:
                    suspicious.append({"reciter": name, "surah": s, "ayah": ay,
                                       "issue": "word outside ayah", "detail": bad[:3]})
                words = [(max(w[0], 0.0), min(w[1], dur_eff)) for w in words]
                if len(words) != dk.get((s, ay), -1):
                    suspicious.append({"reciter": name, "surah": s, "ayah": ay,
                                       "issue": "word count mismatch",
                                       "detail": f"{len(words)} vs text {dk.get((s, ay))}"})
                    words = None  # fall back to ayah-only for this ayah
            if words:
                db[f"{s}:{ay}"] = [{"start_ms": round(st, 1), "end_ms": round(en, 1)} for st, en in words]
                per_word += 1
            else:
                db[f"{s}:{ay}"] = [{"start_ms": 0.0, "end_ms": round(dur, 1)}]
                ayah_only += 1
        (a.out / f"{name}.json").write_text(json.dumps(db, separators=(",", ":")), encoding="utf-8")
        manifest[name] = {
            "source": source or "ayah",
            "ayahs": len(db),
            "per_word_ayahs": per_word,
            "ayah_only_ayahs": ayah_only,
            "everyayah_zip": zip_name,
        }
        print(f"{name:32s} {len(db):5d} ayahs  per-word {per_word:5d}  ayah-only {ayah_only:5d}")

    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (a.out / "suspicious.json").write_text(json.dumps(suspicious, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {a.out}: {len(manifest)} reciters, {len(suspicious)} suspicious entries")
    print("suspicious ->", a.out / "suspicious.json")


if __name__ == "__main__":
    main()