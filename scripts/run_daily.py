#!/usr/bin/env python3
"""Daily orchestrator: pick the next range, fetch, render, upload, advance state.

Order: the B2 reciter list starting at Yasser; within a reciter, the 711
meaning-safe ranges in order.  One successful post advances one range; when a
reciter finishes all ranges, the next reciter starts.  After all 12 reciters,
it wraps to Yasser.

Fallbacks:
  * a reciter with no audio at all      -> skip to the next reciter
  * a range whose ayahs cannot be fetched -> skip to the next range
  * a render failure                    -> skip to the next range
  * an upload failure                   -> do NOT advance (retry next run)
"""
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import fetch_reciter  # noqa: E402
import state  # noqa: E402
import titling  # noqa: E402

MAX_ATTEMPTS = 8


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ranges = load_json(ROOT / "data" / "ranges.json")
    reciters = load_json(ROOT / "data" / "reciters.json")
    surahs = {s["number"]: s["name"] for s in load_json(ROOT / "data" / "surahs.json")}
    st = state.load()

    for attempt in range(MAX_ATTEMPTS):
        rec = reciters[int(st["reciter_index"]) % len(reciters)]
        rg = ranges[int(st["range_index"]) % len(ranges)]
        rng = f"{rg['surah']}:{rg['from']}-{rg['to']}" if rg["from"] != rg["to"] else f"{rg['surah']}:{rg['from']}"
        print(f"[attempt {attempt + 1}] {rec['name_en']}  {rng}")

        try:
            audio_dir = fetch_reciter.fetch(rec, rg, ROOT / "audio_cache")
        except fetch_reciter.ReciterUnavailable as e:
            print("  reciter unavailable -> next reciter:", e)
            state.save(state.advance_reciter(st, len(ranges), len(reciters)))
            continue
        except fetch_reciter.RangeUnavailable as e:
            print("  range unavailable -> next range:", e)
            state.save(state.advance(st, len(ranges), len(reciters)))
            continue

        tex = random.choice([d.name for d in (ROOT / "assets" / "textures").iterdir() if d.is_dir()])
        print("  texture:", tex)
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "merge_texture.py"), "--dir", f"assets/textures/{tex}"],
            cwd=ROOT,
            check=True,
        )

        out = ROOT / "short.mp4"
        cmd = [
            str(ROOT / "target" / "release" / "lumen3d"),
            "--range", rng,
            "--audio-dir", str(audio_dir),
            "--work-dir", str(ROOT / "work"),
            "--timings", str(ROOT / rec["timings"]),
            "--material", f"assets/textures/{tex}/material.txt",
            "--fps", "60", "--w", "1080", "--h", "1920",
            "--crf", "12", "--threads", "0",
            "--seed", str(random.randint(1, 2**31 - 1)),
            "--out", str(out),
        ]
        print("  render:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print("  render failed -> next range")
            state.save(state.advance(st, len(ranges), len(reciters)))
            continue

        count = rg["to"] - rg["from"] + 1
        ayah = f"{rg['from']}" if count == 1 else f"{rg['from']}-{rg['to']}"
        title = titling.pick("title", rec["name_ar"], surahs[rg["surah"]], ayah, count)
        desc = titling.pick("desc", rec["name_ar"], surahs[rg["surah"]], ayah, count)
        print("  title:", title)

        up = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "upload_youtube.py"),
             "--video", str(out), "--title", title, "--description", desc],
            cwd=ROOT,
        )
        if up.returncode != 0:
            print("  upload failed -> state not advanced (retry next run)")
            return 1

        st = state.advance(st, len(ranges), len(reciters), posted={"reciter": rec["b2"], "range": rng, "title": title})
        state.save(st)
        print("posted:", title)
        return 0

    print("all attempts failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())