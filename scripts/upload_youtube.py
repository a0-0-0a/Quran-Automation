#!/usr/bin/env python3
"""
upload_youtube.py — Upload the generated video to YouTube.

  1. Reads info/<SOUND_BASE>/titles.txt → picks one random title.
  2. Detects language: Arabic (if title contains Arabic Unicode chars) or English.
  3. Reads SEO/descriptions/descriptions_{ar|en}.txt → picks one random line.
  4. Uploads the latest .mp4 from output/ using YouTube Data API v3.

Requires env vars (set as GitHub Actions secrets):
  YT_CLIENT_ID      — Google OAuth2 client ID
  YT_CLIENT_SECRET  — Google OAuth2 client secret
  YT_REFRESH_TOKEN  — Long-lived refresh token (obtained once via browser)
  SOUND_BASE        — e.g. "3"  (set by sound_tracker.py)

Optional:
  YT_PRIVACY        — "public" | "unlisted" | "private"  (default: public)
  YT_CATEGORY_ID    — YouTube category ID (default: 22 = People & Blogs)
"""

import os
import random
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

ROOT     = Path(__file__).parent.parent
INFO_DIR = ROOT / "info"
OUT_DIR  = ROOT / "output"
SEO_DIR  = ROOT / "SEO" / "descriptions"

# ─────────────────────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────────────────────
YT_CLIENT_ID     = os.environ.get("YT_CLIENT_ID",     "")
YT_CLIENT_SECRET = os.environ.get("YT_CLIENT_SECRET", "")
YT_REFRESH_TOKEN = os.environ.get("YT_REFRESH_TOKEN", "")
SOUND_BASE       = os.environ.get("SOUND_BASE",       "")
PRIVACY_STATUS   = os.environ.get("YT_PRIVACY",       "public")
CATEGORY_ID      = os.environ.get("YT_CATEGORY_ID",   "22")


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _is_arabic(text: str) -> bool:
    """Return True if the text contains at least one Arabic Unicode character."""
    for ch in text:
        cp = ord(ch)
        # Arabic block: 0x0600–0x06FF
        # Arabic Supplement: 0x0750–0x077F
        # Arabic Extended-A: 0x08A0–0x08FF
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0x08A0 <= cp <= 0x08FF:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# TITLE & DESCRIPTION
# ─────────────────────────────────────────────────────────────────────────────

def _find_info_folder(base: str) -> Path:
    """Try info/<base>/, info/<base>x/, info/<base>X/ — return first found."""
    for candidate in [
        INFO_DIR / base,
        INFO_DIR / (base + "x"),
        INFO_DIR / (base + "X"),
    ]:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Info folder for base '{base}' not found in {INFO_DIR}\n"
        f"Tried: {base}, {base}x, {base}X"
    )


def _read_nonempty_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise ValueError(f"File is empty: {path}")
    return lines


def pick_title(info_folder: Path) -> tuple[str, str]:
    """
    Returns (title, lang) where lang is 'ar' or 'en'.
    Picks a random line from info/<folder>/titles.txt.
    """
    lines = _read_nonempty_lines(info_folder / "titles.txt")
    title = random.choice(lines)
    lang  = "ar" if _is_arabic(title) else "en"
    return title, lang


def pick_description(lang: str) -> str:
    """Pick a random line from SEO/descriptions/descriptions_{lang}.txt."""
    desc_file = SEO_DIR / f"descriptions_{lang}.txt"
    lines     = _read_nonempty_lines(desc_file)
    return random.choice(lines)


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO FILE FINDER
# ─────────────────────────────────────────────────────────────────────────────

def _find_output_video() -> Path:
    """
    First checks output/last_video.txt (written by run.py).
    Falls back to the most recently modified .mp4 in output/.
    """
    marker = OUT_DIR / "last_video.txt"
    if marker.exists():
        candidate = Path(marker.read_text(encoding="utf-8").strip())
        if candidate.exists():
            return candidate

    videos = sorted(
        OUT_DIR.glob("*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not videos:
        raise FileNotFoundError(f"No .mp4 found in {OUT_DIR}")
    return videos[0]


# ─────────────────────────────────────────────────────────────────────────────
# YOUTUBE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

def _build_youtube():
    if not YT_CLIENT_ID or not YT_CLIENT_SECRET or not YT_REFRESH_TOKEN:
        raise EnvironmentError(
            "YT_CLIENT_ID, YT_CLIENT_SECRET, and YT_REFRESH_TOKEN must all be set."
        )
    creds = Credentials(
        token=None,
        refresh_token=YT_REFRESH_TOKEN,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=YT_CLIENT_ID,
        client_secret=YT_CLIENT_SECRET,
    )
    creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def upload_video(video_path: Path, title: str, description: str) -> str:
    """Upload video and return the YouTube video ID."""
    youtube = _build_youtube()
    media   = MediaFileUpload(
        str(video_path),
        mimetype="video/mp4",
        resumable=True,
        chunksize=10 * 1024 * 1024,   # 10 MB chunks
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title":       title,
                "description": description,
                "categoryId":  CATEGORY_ID,
            },
            "status": {
                "privacyStatus":              PRIVACY_STATUS,
                "selfDeclaredMadeForKids":    False,
            },
        },
        media_body=media,
    )

    print(f"⬆️  Uploading: {video_path.name}  ({video_path.stat().st_size / 1e6:.1f} MB)")

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"   Progress: {pct}%", end="\r", flush=True)

    print()  # newline after progress
    return response["id"]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not SOUND_BASE:
        print("❌ SOUND_BASE env var not set.  Run sound_tracker.py select first.")
        sys.exit(1)

    print("─── YouTube Upload ───")

    # 1. Find the generated video
    video_path = _find_output_video()
    print(f"📹 Video: {video_path.name}")

    # 2. Find info folder and pick title
    info_folder = _find_info_folder(SOUND_BASE)
    title, lang = pick_title(info_folder)
    print(f"📝 Title ({lang.upper()}): {title}")

    # 3. Pick matching description
    description = pick_description(lang)
    print(f"📄 Description: {description[:80]}{'…' if len(description) > 80 else ''}")

    # 4. Upload
    video_id = upload_video(video_path, title, description)

    print(f"\n✅ Upload complete!")
    print(f"   Video ID : {video_id}")
    print(f"   URL      : https://www.youtube.com/watch?v={video_id}")
    print(f"   Privacy  : {PRIVACY_STATUS}")


if __name__ == "__main__":
    main()
