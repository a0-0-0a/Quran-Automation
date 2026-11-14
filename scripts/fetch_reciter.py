"""Fetch one ayah range for a reciter into a local cache.

Primary   : Backblaze B2  reciters/<reciter>/<SSS><AAA>.opus (then .mp3)
Fallback  : everyayah.com/data/<everyayah_folder>/<SSS><AAA>.mp3

B2 credentials come from the repo secrets (same names as the old repo, with the
AWS-style names from the user's .env accepted as a fallback).  Nothing is printed.
"""
import base64
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

EVERYAYAH = "https://everyayah.com/data/{folder}/{name}"


class ReciterUnavailable(RuntimeError):
    pass


class RangeUnavailable(RuntimeError):
    pass


def _b2_auth():
    key = os.environ.get("B2_AUTH_ID") or os.environ.get("AWS_ACCESS_KEY_ID", "")
    sec = (os.environ.get("B2_APPLICATION_KEY") or os.environ.get("B2_APP_KEY")
           or os.environ.get("AWS_SECRET_ACCESS_KEY", ""))
    bucket = os.environ.get("B2_BUCKET_NAME") or os.environ.get("BUCKET_NAME", "main-one")
    if not key or not sec:
        return None
    import requests

    basic = base64.b64encode(f"{key}:{sec}".encode()).decode()
    r = requests.get(
        "https://api.backblazeb2.com/b2api/v4/b2_authorize_account",
        headers={"Authorization": f"Basic {basic}"},
        timeout=60,
    )
    r.raise_for_status()
    d = r.json()["apiInfo"]["storageApi"]
    return {
        "download_url": d["downloadUrl"],
        "bucket": bucket,
        "auth_token": _auth_token(r),
    }


def _auth_token(r):
    return r.json()["authorizationToken"]


def _download_b2(auth, key: str, dest: Path) -> bool:
    import requests

    url = f"{auth['download_url']}/file/{auth['bucket']}/{urllib.parse.quote(key, safe='/')}"
    try:
        r = requests.get(url, headers={"Authorization": auth["auth_token"]}, timeout=180)
        r.raise_for_status()
        if not r.content:
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def _download_url(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "quran-shorts/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if not data:
            return False
        dest.write_bytes(data)
        return True
    except Exception:
        return False


def fetch(rec: dict, rng: dict, cache_dir: Path) -> Path:
    out = cache_dir / rec["b2"]
    out.mkdir(parents=True, exist_ok=True)
    auth = _b2_auth()
    got = 0
    failed = []
    for ayah in range(int(rng["from"]), int(rng["to"]) + 1):
        name = f"{int(rng['surah']):03d}{ayah:03d}"
        dest = None
        if auth:
            for ext in ("opus", "mp3"):
                d = out / f"{name}.{ext}"
                if _download_b2(auth, f"reciters/{rec['b2']}/{name}.{ext}", d):
                    dest = d
                    break
        if dest is None:
            for ext in ("mp3", "opus"):
                d = out / f"{name}.{ext}"
                if _download_url(EVERYAYAH.format(folder=rec["everyayah"], name=f"{name}.{ext}"), d):
                    dest = d
                    break
        if dest is None:
            failed.append(name)
            continue
        got += 1
    if got == 0:
        raise ReciterUnavailable(f"no audio for {rec['b2']}")
    if failed:
        raise RangeUnavailable(f"missing {len(failed)} ayah(s): {failed[:5]}")
    return out