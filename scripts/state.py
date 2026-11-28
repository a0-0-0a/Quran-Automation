"""Progress state: the last reviewed `recit/<n>` folder we finished.

The daily job walks the numeric folders in order and wraps to the lowest one.
`last_folder` is advanced for posted shorts and for folders that are invalid or
fail to render (so the loop does not get stuck on a broken upload).  An upload
failure deliberately does **not** advance it, so the next run retries the same
folder.
"""
from __future__ import annotations

import json
from pathlib import Path

STATE = Path(__file__).resolve().parent.parent / "state" / "progress.json"
DEFAULT = {"last_folder": 0, "last_posted": None}


def load() -> dict:
    if STATE.is_file():
        try:
            data = json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        try:
            last = int(data.get("last_folder", 0) or 0)
        except (TypeError, ValueError):
            last = 0
        return {"last_folder": max(0, last), "last_posted": data.get("last_posted")}
    return dict(DEFAULT)


def save(data: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_folder": int(data.get("last_folder", 0) or 0),
        "last_posted": data.get("last_posted"),
    }
    STATE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def mark_skipped(data: dict, folder: int) -> dict:
    """Advance past an invalid / render-failed folder."""
    data["last_folder"] = int(folder)
    return data


def mark_posted(data: dict, folder: int, posted: dict) -> dict:
    data["last_folder"] = int(folder)
    data["last_posted"] = posted
    return data
