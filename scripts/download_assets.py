#!/usr/bin/env python3
"""
download_assets.py — Download all assets needed for one video run from B2.

Reads from environment (set by sound_tracker.py):
  SELECTED_SOUND       e.g. "3.wav"
  SELECTED_SOUND_PATH  e.g. "sounds/3.wav"
  SOUND_BASE           e.g. "3"

Downloads:
  1. sounds/<SELECTED_SOUND>           → local sounds/ dir
  2. info/<SOUND_BASE>/ (or /x or /X)  → local info/ dir
  3. N_BG_IMAGES random images          → local images/ dir

Usage:
    python scripts/download_assets.py
"""

import base64
import os
import random
import sys
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from settings import N_BG_IMAGES
from image_list import ALL_IMAGES

# ─────────────────────────────────────────────────────────────────────────────
B2_AUTH_ID   = os.environ.get("B2_AUTH_ID",         "")
B2_APP_KEY   = os.environ.get("B2_APPLICATION_KEY",  "")
B2_BUCKET    = os.environ.get("B2_BUCKET_NAME",      "main-one")

SELECTED_SOUND      = os.environ.get("SELECTED_SOUND",      "")  # "3.wav"
SELECTED_SOUND_PATH = os.environ.get("SELECTED_SOUND_PATH", "")  # "sounds/3.wav"
SOUND_BASE          = os.environ.get("SOUND_BASE",          "")  # "3"

session = requests.Session()


# ─────────────────────────────────────────────────────────────────────────────
# B2 AUTH
# ─────────────────────────────────────────────────────────────────────────────

def _b2_authorize():
    basic = base64.b64encode(
        f"{B2_AUTH_ID}:{B2_APP_KEY}".encode("utf-8")
    ).decode("ascii")

    resp = session.get(
        "https://api.backblazeb2.com/b2api/v4/b2_authorize_account",
        headers={"Authorization": f"Basic {basic}"},
        timeout=60,
    )
    resp.raise_for_status()
    data    = resp.json()
    storage = data["apiInfo"]["storageApi"]
    tok     = data["authorizationToken"]
    api_url = storage["apiUrl"]
    dl_url  = storage["downloadUrl"]

    bucket_id = None
    for b in storage.get("allowed", {}).get("buckets", []):
        if b.get("name") == B2_BUCKET:
            bucket_id = b.get("id")

    if not bucket_id:
        r2 = session.post(
            f"{api_url}/b2api/v4/b2_list_buckets",
            headers={"Authorization": tok},
            json={"accountId": data.get("accountId", "")},
            timeout=60,
        )
        r2.raise_for_status()
        for b in r2.json().get("buckets", []):
            if b.get("bucketName") == B2_BUCKET:
                bucket_id = b.get("bucketId")
                break

    if not bucket_id:
        raise RuntimeError(f"Bucket '{B2_BUCKET}' not found.")

    return tok, api_url, dl_url, bucket_id


# ─────────────────────────────────────────────────────────────────────────────
# LIST & DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def _list_prefix(api_url: str, tok: str, bucket_id: str, prefix: str) -> list[str]:
    files = []
    start = None
    while True:
        params: dict = {
            "bucketId":     bucket_id,
            "prefix":       prefix,
            "maxFileCount": 1000,
        }
        if start:
            params["startFileName"] = start
        resp = session.get(
            f"{api_url}/b2api/v4/b2_list_file_names",
            headers={"Authorization": tok},
            params=params,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("files", []):
            if item.get("action") == "upload":
                files.append(item["fileName"])
        nxt = data.get("nextFileName")
        if not nxt:
            break
        start = nxt
    return files


def _download(dl_url: str, tok: str, b2_path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    safe_path = quote(b2_path, safe="/")
    url       = f"{dl_url}/file/{B2_BUCKET}/{safe_path}"
    resp      = session.get(
        url,
        headers={"Authorization": tok},
        stream=True,
        timeout=180,
    )
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


# ─────────────────────────────────────────────────────────────────────────────
# FIND INFO FOLDER IN B2
# ─────────────────────────────────────────────────────────────────────────────

def _find_info_prefix(api_url: str, tok: str, bucket_id: str, base: str) -> str:
    """
    Try info/<base>/, info/<base>x/, info/<base>X/ and return the
    first prefix that has files in B2.  Raises if none found.
    """
    for candidate in [
        f"info/{base}/",
        f"info/{base}x/",
        f"info/{base}X/",
    ]:
        files = _list_prefix(api_url, tok, bucket_id, candidate)
        real  = [f for f in files if not f.endswith(".bzEmpty")]
        if real:
            return candidate
    raise RuntimeError(
        f"Info folder for base '{base}' not found in B2. "
        f"Tried: info/{base}/, info/{base}x/, info/{base}X/"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not SELECTED_SOUND or not SOUND_BASE:
        print("❌ SELECTED_SOUND or SOUND_BASE env vars not set.")
        print("   Run sound_tracker.py select first.")
        sys.exit(1)

    print("─── Download Assets ───")
    print(f"  Sound : {SELECTED_SOUND}")
    print(f"  Base  : {SOUND_BASE}")
    print(f"  Images: {N_BG_IMAGES} random from fixed list")

    print("\n🔗 Authorizing with B2 …")
    tok, api_url, dl_url, bucket_id = _b2_authorize()

    # ── 1. Download sound ────────────────────────────────────────────────────
    sounds_dir = ROOT / "sounds"
    sounds_dir.mkdir(parents=True, exist_ok=True)
    sound_dest = sounds_dir / SELECTED_SOUND

    print(f"\n⬇️  Sound: {SELECTED_SOUND_PATH}")
    _download(dl_url, tok, SELECTED_SOUND_PATH, sound_dest)
    print(f"   → {sound_dest.relative_to(ROOT)}")

    # ── 2. Download info folder ──────────────────────────────────────────────
    print(f"\n⬇️  Info folder for base '{SOUND_BASE}' …")
    info_prefix = _find_info_prefix(api_url, tok, bucket_id, SOUND_BASE)
    info_files  = _list_prefix(api_url, tok, bucket_id, info_prefix)
    info_files  = [f for f in info_files if not f.endswith(".bzEmpty")]

    print(f"   Found {len(info_files)} file(s) under {info_prefix}")
    for b2_path in info_files:
        # Keep the same relative structure under ROOT
        dest = ROOT / b2_path
        _download(dl_url, tok, b2_path, dest)
        print(f"   → {dest.relative_to(ROOT)}")

    # ── 3. Download random images ────────────────────────────────────────────
    images_dir = ROOT / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Pick N_BG_IMAGES unique random images from the fixed list
    pool = ALL_IMAGES.copy()
    if len(pool) < N_BG_IMAGES:
        pool = pool * (N_BG_IMAGES // len(pool) + 1)
    picks = random.sample(pool, N_BG_IMAGES)

    print(f"\n⬇️  Images ({N_BG_IMAGES} random):")
    for b2_path in picks:
        dest = ROOT / b2_path          # e.g. ROOT/images/3a.webp
        _download(dl_url, tok, b2_path, dest)
        print(f"   → {dest.relative_to(ROOT)}")

    print("\n✅ All assets downloaded successfully")


if __name__ == "__main__":
    main()
