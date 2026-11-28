"""Minimal Backblaze B2 client for the reviewed `recit/<n>/` folders.

The app uploads (manually) two files per folder:

    recit/12/short.json
    recit/12/audio.mp3

This module only needs list + download.  Credentials use the same secret names
as the old repo, with AWS-style names accepted as a fallback.
"""
from __future__ import annotations

import base64
import os
import urllib.parse
import urllib.request
from pathlib import Path

PREFIX = "recit/"


class B2Error(RuntimeError):
    pass


def _credentials():
    key = os.environ.get("B2_AUTH_ID") or os.environ.get("AWS_ACCESS_KEY_ID", "")
    sec = (
        os.environ.get("B2_APPLICATION_KEY")
        or os.environ.get("B2_APP_KEY")
        or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    )
    bucket = os.environ.get("B2_BUCKET_NAME") or os.environ.get("BUCKET_NAME", "main-one")
    if not key or not sec:
        raise B2Error("B2 credentials are missing (B2_AUTH_ID / B2_APPLICATION_KEY).")
    return key, sec, bucket


def authorize() -> dict:
    import requests

    key, sec, bucket_name = _credentials()
    basic = base64.b64encode(f"{key}:{sec}".encode()).decode()
    r = requests.get(
        "https://api.backblazeb2.com/b2api/v4/b2_authorize_account",
        headers={"Authorization": f"Basic {basic}"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    storage = data["apiInfo"]["storageApi"]
    allowed = storage.get("allowed", {})
    bucket_id = allowed.get("bucketId")
    api_url = storage["apiUrl"]
    token = data["authorizationToken"]
    account_id = data["accountId"]

    if not bucket_id:
        rr = requests.post(
            f"{api_url}/b2api/v4/b2_list_buckets",
            headers={"Authorization": token},
            json={"accountId": account_id},
            timeout=60,
        )
        rr.raise_for_status()
        for b in rr.json().get("buckets", []):
            if b.get("bucketName") == bucket_name:
                bucket_id = b["bucketId"]
                break
    if not bucket_id:
        raise B2Error(f"B2 bucket not found: {bucket_name}")

    return {
        "api_url": api_url,
        "download_url": storage["downloadUrl"],
        "bucket_name": bucket_name,
        "bucket_id": bucket_id,
        "token": token,
    }


def list_folders(auth: dict) -> list[int]:
    """Return every numeric folder under ``recit/``, sorted ascending."""
    import requests

    names: set[int] = set()
    start = None
    while True:
        body = {
            "bucketId": auth["bucket_id"],
            "prefix": PREFIX,
            "maxFileCount": 10000,
        }
        if start:
            body["startFileName"] = start
        r = requests.post(
            f"{auth['api_url']}/b2api/v4/b2_list_file_names",
            headers={"Authorization": auth["token"]},
            json=body,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        for f in data.get("files", []):
            name = f.get("fileName", "")
            if not name.startswith(PREFIX):
                continue
            rest = name[len(PREFIX):]
            head = rest.split("/", 1)[0]
            if head.isdigit():
                names.add(int(head))
        start = data.get("nextFileName")
        if not start:
            break
    return sorted(names)


def _download(auth: dict, key: str, dest: Path) -> bool:
    import requests

    url = (
        f"{auth['download_url']}/file/{auth['bucket_name']}/"
        f"{urllib.parse.quote(key, safe='/')}"
    )
    try:
        r = requests.get(url, headers={"Authorization": auth["token"]}, timeout=300)
        r.raise_for_status()
        if not r.content:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def download_folder(auth: dict, number: int, dest: Path) -> tuple[Path, Path]:
    """Download ``short.json`` and ``audio.mp3`` for one folder."""
    short = dest / "short.json"
    audio = dest / "audio.mp3"
    ok_short = _download(auth, f"recit/{number}/short.json", short)
    ok_audio = _download(auth, f"recit/{number}/audio.mp3", audio)
    if not ok_short:
        raise B2Error(f"recit/{number}/short.json is missing or unreadable")
    if not ok_audio:
        raise B2Error(f"recit/{number}/audio.mp3 is missing or unreadable")
    return short, audio
