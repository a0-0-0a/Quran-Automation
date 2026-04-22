#!/usr/bin/env python3
"""
sound_tracker.py — Sequential sound queue manager.

How it works:
  1. Connects to B2 and gets the current list of sounds.
  2. Loads sound_queue.json from the repo root.
  3. Syncs the two lists:
       - Sounds in B2 but not in queue → added as unused.
       - Sounds in queue but removed from B2 → removed from queue.
       - Existing sounds keep their "used" status.
  4. Picks the first unused sound (sorted by numeric filename order).
  5. If all sounds are used → resets all to unused → picks the first.
  6. Marks the selected sound as used and saves sound_queue.json locally.
  7. Exports SELECTED_SOUND, SELECTED_SOUND_PATH, and SOUND_BASE to
     $GITHUB_ENV so subsequent workflow steps can read them.

The updated sound_queue.json is committed back to the repo by the
workflow ONLY after a successful upload — so a failed run leaves the
queue unchanged and re-tries the same sound next time.

Usage:
    python scripts/sound_tracker.py select
"""

import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
QUEUE_FILE = ROOT / "sound_queue.json"

B2_AUTH_ID  = os.environ.get("B2_AUTH_ID",         "")
B2_APP_KEY  = os.environ.get("B2_APPLICATION_KEY",  "")
B2_BUCKET   = os.environ.get("B2_BUCKET_NAME",      "main-one")

SOUNDS_PREFIX = "sounds/"
AUDIO_EXTS    = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}

session = requests.Session()


# ─────────────────────────────────────────────────────────────────────────────
# BACKBLAZE HELPERS
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
    data     = resp.json()
    storage  = data["apiInfo"]["storageApi"]
    tok      = data["authorizationToken"]
    api_url  = storage["apiUrl"]

    bucket_id = None
    for b in storage.get("allowed", {}).get("buckets", []):
        if b.get("name") == B2_BUCKET:
            bucket_id = b.get("id")

    if not bucket_id:
        # Fallback: list all buckets
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
        raise RuntimeError(f"Bucket '{B2_BUCKET}' not found or no permission.")

    return tok, api_url, bucket_id


def _list_sounds(api_url: str, tok: str, bucket_id: str) -> list[str]:
    """Returns list of full B2 paths like 'sounds/1.wav', sorted."""
    files = []
    start = None

    while True:
        params: dict = {
            "bucketId":     bucket_id,
            "prefix":       SOUNDS_PREFIX,
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
            if item.get("action") != "upload":
                continue
            name     = item["fileName"]                # "sounds/1.wav"
            relative = name[len(SOUNDS_PREFIX):]       # "1.wav"
            if "/" in relative:
                continue                               # skip sub-folders
            if relative == ".bzEmpty":
                continue
            if Path(relative).suffix.lower() in AUDIO_EXTS:
                files.append(name)

        nxt = data.get("nextFileName")
        if not nxt:
            break
        start = nxt

    return files


# ─────────────────────────────────────────────────────────────────────────────
# SORT KEY  (numeric order: "1" < "2" < "10" < "11x" < "20")
# ─────────────────────────────────────────────────────────────────────────────

def _sort_key(full_path: str) -> tuple:
    stem = Path(full_path).stem.rstrip("xX")   # "10x" → "10"
    try:
        return (0, int(stem), stem)
    except ValueError:
        return (1, 0, stem)


# ─────────────────────────────────────────────────────────────────────────────
# QUEUE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_queue() -> dict:
    if QUEUE_FILE.exists():
        with open(QUEUE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"sounds": [], "last_updated": None}


def _save_queue(queue: dict) -> None:
    queue["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)
    print(f"💾 sound_queue.json saved ({len(queue['sounds'])} sounds)")


def _sync_queue(queue: dict, b2_sounds: list[str]) -> dict:
    """
    Merge current B2 list with stored queue:
      - Add new sounds (used=False)
      - Remove sounds no longer in B2
      - Preserve 'used' status for existing entries
    Result is sorted in numeric order.
    """
    existing_map = {s["name"]: s["used"] for s in queue["sounds"]}
    b2_set       = set(b2_sounds)

    removed = set(existing_map) - b2_set
    added   = b2_set - set(existing_map)

    if removed:
        print(f"🗑  Removed from queue (no longer in B2): {sorted(removed)}")
    if added:
        print(f"➕ Added to queue (new in B2): {sorted(added, key=_sort_key)}")

    merged = []
    for name in sorted(b2_sounds, key=_sort_key):
        merged.append({"name": name, "used": existing_map.get(name, False)})

    queue["sounds"] = merged
    return queue


def _select_next(queue: dict) -> str:
    """
    Returns the full B2 path of the next unused sound.
    If all sounds are used, resets the cycle and returns the first one.
    """
    # Find the first unused sound
    for s in queue["sounds"]:
        if not s["used"]:
            s["used"] = True
            return s["name"]

    # All sounds used — reset cycle and start over
    print("🔄 All sounds have been used — resetting cycle")
    for s in queue["sounds"]:
        s["used"] = False
    queue["sounds"][0]["used"] = True
    return queue["sounds"][0]["name"]


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB ENV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def _export(key: str, value: str) -> None:
    """Write key=value to $GITHUB_ENV (or just print locally)."""
    gh_env = os.environ.get("GITHUB_ENV")
    if gh_env:
        with open(gh_env, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")
    print(f"  → {key}={value}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "select":
        print("Usage: python scripts/sound_tracker.py select")
        sys.exit(1)

    if not B2_AUTH_ID or not B2_APP_KEY:
        print("❌ B2_AUTH_ID or B2_APPLICATION_KEY not set.")
        sys.exit(1)

    print("─── Sound Tracker ───")
    print("🔗 Connecting to Backblaze B2 …")
    tok, api_url, bucket_id = _b2_authorize()

    b2_sounds = _list_sounds(api_url, tok, bucket_id)
    if not b2_sounds:
        print("❌ No audio files found under 'sounds/' in B2.")
        sys.exit(1)

    print(f"📋 Found {len(b2_sounds)} sounds in B2")

    queue    = _load_queue()
    queue    = _sync_queue(queue, b2_sounds)
    selected = _select_next(queue)   # e.g. "sounds/3.wav"
    _save_queue(queue)

    sound_filename = Path(selected).name            # "3.wav"
    sound_stem     = Path(sound_filename).stem      # "3" or "3x"
    sound_base     = sound_stem.rstrip("xX")        # "3"

    print(f"\n✅ Selected: {selected}")
    print("Exporting to GitHub Actions environment:")
    _export("SELECTED_SOUND",      sound_filename)   # "3.wav"
    _export("SELECTED_SOUND_PATH", selected)         # "sounds/3.wav"
    _export("SOUND_STEM",          sound_stem)       # "3" or "3x"
    _export("SOUND_BASE",          sound_base)       # "3"


if __name__ == "__main__":
    main()
