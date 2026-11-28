#!/usr/bin/env python3
"""Daily orchestrator: render the next reviewed `recit/<n>` folder and post it.

Each folder is self-contained (created by the review app):

    recit/<n>/short.json   Arabic title/description + word keyframes
    recit/<n>/audio.mp3    192 kbps crop, keyframe times relative to its start

Order is a numeric round-robin over the folders that exist at run time:

  * start after the last folder recorded in `state/progress.json`
  * newly added higher folders are rendered in order
  * after the highest folder, wrap to the lowest one

A folder that is missing, invalid, or fails to render is logged, skipped, and
`last_folder` advances past it.  An upload failure does **not** advance, so the
same folder is retried next run.

The renderer still supports the old `--range` mode; this job only uses
`--short-json`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import b2  # noqa: E402
import state  # noqa: E402

MAX_ATTEMPTS = 12
OUT = ROOT / "short.mp4"
WORK = ROOT / "work"


def log(message: str) -> None:
    print(message, flush=True)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_folders(root: Path) -> list[int]:
    if not root.is_dir():
        return []
    return sorted(
        int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()
    )


def candidates_in_order(folders: list[int], last: int) -> list[int]:
    """Folders after `last` first, then wrap to the lowest ones."""
    after = [n for n in folders if n > last]
    before = [n for n in folders if n <= last]
    return after + before


def choose_texture(rng: random.Random) -> str:
    textures = [
        p
        for p in (ROOT / "assets" / "textures").iterdir()
        if p.is_dir()
        and (p / "material.txt").is_file()
        and (p / "meta.json").is_file()
        and (p / "tile_00.png").is_file()
    ]
    if not textures:
        raise SystemExit("no texture folders under assets/textures")
    tex = rng.choice(textures)
    log(f"texture: {tex.name}")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "merge_texture.py"),
            "--dir",
            str(tex),
        ],
        cwd=ROOT,
        check=True,
    )
    return tex.name


def fetch_folder(number: int, source_dir: Path | None, auth: dict | None) -> Path:
    dest = WORK / "recit" / str(number)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source_dir is not None:
        src = source_dir / str(number)
        if not src.is_dir():
            raise FileNotFoundError(f"{src} is not a folder")
        shutil.copytree(src, dest)
    else:
        assert auth is not None
        b2.download_folder(auth, number, dest)
    return dest


def probe_duration(path: Path) -> float:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip())
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not read audio duration: {exc}") from exc


def read_short(folder: Path, rng: "random.Random | None" = None) -> tuple[dict, Path, str, str]:
    path = folder / "short.json"
    try:
        data = load_json(path)
    except Exception as exc:
        raise ValueError(f"short.json: {exc}") from exc
    if data.get("reviewed") is False:
        raise ValueError("short.json is not marked reviewed")

    try:
        surah = int((data.get("surah") or {}).get("number"))
        first = int((data.get("range") or {}).get("from"))
        last = int((data.get("range") or {}).get("to"))
    except (TypeError, ValueError) as exc:
        raise ValueError("short.json: missing surah/range") from exc
    if not (1 <= surah <= 114 and 1 <= first <= last):
        raise ValueError(f"short.json: invalid surah/range {surah}:{first}-{last}")

    audio = data.get("audio") or {}
    audio_name = str(audio.get("file") or "audio.mp3")
    audio_path = folder / audio_name
    if not audio_path.is_file():
        raise ValueError(f"missing {audio_name}")
    if audio_path.stat().st_size < 1024:
        raise ValueError(f"{audio_name} is empty")
    duration = probe_duration(audio_path)
    if duration < 0.5:
        raise ValueError(f"{audio_name} is only {duration:.2f}s")

    words = data.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError("short.json has no words")
    last_start = -1.0
    for i, word in enumerate(words):
        try:
            start = float(word["start"])
            end = float(word["end"])
            text_index = int(word["text_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"words[{i}] is malformed") from exc
        if text_index < 0:
            raise ValueError(f"words[{i}] has a negative text_index")
        if start < last_start - 1e-6:
            raise ValueError(f"words[{i}] is out of order")
        if end < start - 1e-6:
            raise ValueError(f"words[{i}] ends before it starts")
        if start < -0.001 or end > duration + 0.25:
            raise ValueError(
                f"words[{i}] is outside the {duration:.2f}s audio"
            )
        last_start = start

    title = str(data.get("title") or "").strip()
    description = str(data.get("description") or "").strip()
    if not title:
        raise ValueError("title is empty")
    # choose any random variation when the folder ships multiple (10 each)
    title_pool = [str(t).strip() for t in (data.get("titles") or []) if str(t).strip()]
    desc_pool = [str(d).strip() for d in (data.get("descriptions") or []) if str(d).strip()]
    title_pool = title_pool or [title]
    desc_pool = desc_pool or [description or title]
    chooser = rng if rng is not None else random
    title = chooser.choice(title_pool)
    description = chooser.choice(desc_pool)
    if not description:
        description = title
    return data, audio_path, title, description


def render(
    folder: Path,
    texture: str,
    seed: int,
    no_sfx: bool,
    width: int = 1080,
    height: int = 1920,
    fps: int = 60,
    crf: int = 12,
) -> None:
    cmd = [
        str(ROOT / "target" / "release" / "lumen3d"),
        "--short-json",
        str(folder),
        "--material",
        f"assets/textures/{texture}/material.txt",
        "--fps",
        str(fps),
        "--w",
        str(width),
        "--h",
        str(height),
        "--crf",
        str(crf),
        "--threads",
        "0",
        "--seed",
        str(seed),
        "--out",
        str(OUT),
    ]
    if no_sfx:
        cmd.append("--no-sfx")
    else:
        cmd += ["--sfx-gain", "0.10"]
    log("render: " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"renderer exited with {result.returncode}")


def upload(video: Path, title: str, description: str) -> None:
    missing = [
        name
        for name in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError("missing YouTube secrets: " + ", ".join(missing))
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "upload_youtube.py"),
        "--video",
        str(video),
        "--title",
        title[:100],
        "--description",
        description,
    ]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"YouTube upload exited with {result.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Use local <n>/ folders instead of Backblaze B2 (for testing).",
    )
    ap.add_argument(
        "--folder",
        type=int,
        default=None,
        help="Render this folder only, ignoring the round-robin.",
    )
    ap.add_argument(
        "--skip-upload",
        action="store_true",
        help="Stop after a successful render; do not upload or change state.",
    )
    ap.add_argument("--no-sfx", action="store_true", help="Render without background sound.")
    ap.add_argument("--attempts", type=int, default=MAX_ATTEMPTS)
    ap.add_argument("--w", type=int, default=1080, help="Render width (default 1080).")
    ap.add_argument("--h", type=int, default=1920, help="Render height (default 1920).")
    ap.add_argument("--fps", type=int, default=60, help="Render fps (default 60).")
    ap.add_argument("--crf", type=int, default=12, help="x264 CRF (default 12).")
    args = ap.parse_args()

    st = state.load()
    auth = None
    source_dir = args.source_dir.resolve() if args.source_dir else None
    if source_dir is not None:
        folders = numeric_folders(source_dir)
    else:
        auth = b2.authorize()
        folders = b2.list_folders(auth)
    if not folders:
        log("no recit/<n> folders found")
        return 1

    if args.folder is not None:
        if args.folder not in folders:
            log(f"recit/{args.folder} does not exist")
            return 1
        candidates = [args.folder]
    else:
        candidates = candidates_in_order(folders, int(st.get("last_folder", 0)))

    log(f"folders: {folders[0]}..{folders[-1]} ({len(folders)} total)")
    log(f"last posted/skipped: {st.get('last_folder', 0)}")
    log("queue: " + ", ".join(str(n) for n in candidates[: args.attempts]))

    rng = random.Random()
    texture = choose_texture(rng)
    OUT.unlink(missing_ok=True)
    attempted = 0
    for number in candidates:
        if attempted >= args.attempts:
            break
        attempted += 1
        log(f"[attempt {attempted}] recit/{number}")
        try:
            folder = fetch_folder(number, source_dir, auth)
        except Exception as exc:  # noqa: BLE001
            log(f"  download failed -> skip: {exc}")
            state.save(state.mark_skipped(st, number))
            continue

        try:
            data, audio, title, description = read_short(folder, rng)
        except Exception as exc:  # noqa: BLE001
            log(f"  invalid folder -> skip: {exc}")
            state.save(state.mark_skipped(st, number))
            continue

        log(
            f"  {data.get('surah', {}).get('name_ar', '')} "
            f"{data.get('range', {}).get('label', '')} · {title}"
        )
        try:
            render(
                folder,
                texture,
                rng.randint(1, 2**31 - 1),
                args.no_sfx,
                args.w,
                args.h,
                args.fps,
                args.crf,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"  render failed -> skip: {exc}")
            OUT.unlink(missing_ok=True)
            state.save(state.mark_skipped(st, number))
            continue

        if args.skip_upload:
            log(f"  rendered {OUT} (upload skipped, folder not advanced)")
            return 0

        try:
            upload(OUT, title, description)
        except Exception as exc:  # noqa: BLE001
            log(f"  upload failed -> state not advanced: {exc}")
            return 1

        posted = {
            "folder": number,
            "title": title,
            "surah": data.get("surah", {}).get("number"),
            "range": data.get("range", {}).get("label"),
        }
        state.save(state.mark_posted(st, number, posted))
        log(f"posted recit/{number}: {title}")
        return 0

    log("no folder could be rendered")
    return 1


if __name__ == "__main__":
    sys.exit(main())
