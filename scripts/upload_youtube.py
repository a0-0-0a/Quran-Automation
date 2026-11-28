#!/usr/bin/env python3
"""Upload one rendered short to YouTube (same method/secrets as the old repo).

Env: YT_CLIENT_ID, YT_CLIENT_SECRET, YT_REFRESH_TOKEN
     YT_PRIVACY (default public), YT_CATEGORY_ID (default 22)
"""
import argparse
import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def upload(video: Path, title: str, description: str) -> str:
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
    )
    creds.refresh(Request())
    yt = build("youtube", "v3", credentials=creds)
    media = MediaFileUpload(str(video), mimetype="video/mp4", resumable=True, chunksize=10 * 1024 * 1024)
    req = yt.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
                "categoryId": os.environ.get("YT_CATEGORY_ID", "22"),
            },
            "status": {
                "privacyStatus": os.environ.get("YT_PRIVACY", "public"),
                "selfDeclaredMadeForKids": False,
            },
        },
        media_body=media,
    )
    response = None
    while response is None:
        status, response = req.next_chunk()
        if status:
            print(f"upload {int(status.progress() * 100)}%", end="\r", flush=True)
    print()
    return response["id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--description", required=True)
    a = ap.parse_args()
    vid = upload(Path(a.video), a.title, a.description)
    print(f"https://www.youtube.com/watch?v={vid}")


if __name__ == "__main__":
    main()