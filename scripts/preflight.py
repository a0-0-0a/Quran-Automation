#!/usr/bin/env python3
"""
preflight.py — Pre-flight checks before video generation starts.

Verifies:
  1. Backblaze B2 connectivity and bucket access
  2. YouTube OAuth token can be refreshed (non-interactive)

Exit code 0 = all good.  Exit code 1 = something failed → workflow stops.
"""
import base64
import os
import sys

import requests


def check_b2(auth_id: str, app_key: str, bucket_name: str) -> bool:
    print("─── Checking Backblaze B2 ───")
    try:
        basic = base64.b64encode(
            f"{auth_id}:{app_key}".encode("utf-8")
        ).decode("ascii")

        resp = requests.get(
            "https://api.backblazeb2.com/b2api/v4/b2_authorize_account",
            headers={"Authorization": f"Basic {basic}"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        storage = data.get("apiInfo", {}).get("storageApi", {})
        allowed = storage.get("allowed", {})

        for b in allowed.get("buckets", []):
            if b.get("name") == bucket_name:
                print(f"✅ B2 OK  — bucket '{bucket_name}' is accessible")
                return True

        # If buckets list is empty, the key might be "all buckets" scope
        # In that case, check by listing with b2_list_buckets
        api_url   = storage.get("apiUrl", "")
        auth_token = data.get("authorizationToken", "")
        if api_url and auth_token:
            r2 = requests.post(
                f"{api_url}/b2api/v4/b2_list_buckets",
                headers={"Authorization": auth_token},
                json={"accountId": data.get("accountId", "")},
                timeout=30,
            )
            r2.raise_for_status()
            for b in r2.json().get("buckets", []):
                if b.get("bucketName") == bucket_name:
                    print(f"✅ B2 OK  — bucket '{bucket_name}' is accessible")
                    return True

        print(f"❌ B2 FAIL — bucket '{bucket_name}' not found / no permission")
        return False

    except Exception as exc:
        print(f"❌ B2 FAIL — {exc}")
        return False


def check_youtube(client_id: str, client_secret: str, refresh_token: str) -> bool:
    print("─── Checking YouTube OAuth ───")
    try:
        resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id":     client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type":    "refresh_token",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if "access_token" in data:
            scope = data.get("scope", "")
            print(f"✅ YouTube OK — token refreshed successfully")
            if "youtube.upload" not in scope and "youtube" not in scope:
                print(
                    "⚠️  Warning: 'youtube' not found in token scope.\n"
                    "   Make sure the OAuth app has YouTube Data API v3 enabled\n"
                    "   and the refresh token was granted the youtube.upload scope."
                )
            return True

        print(f"❌ YouTube FAIL — {data}")
        return False

    except Exception as exc:
        print(f"❌ YouTube FAIL — {exc}")
        return False


def main() -> None:
    b2_auth_id     = os.environ.get("B2_AUTH_ID", "")
    b2_app_key     = os.environ.get("B2_APPLICATION_KEY", "")
    b2_bucket      = os.environ.get("B2_BUCKET_NAME", "main-one")
    yt_client_id   = os.environ.get("YT_CLIENT_ID", "")
    yt_client_sec  = os.environ.get("YT_CLIENT_SECRET", "")
    yt_refresh_tok = os.environ.get("YT_REFRESH_TOKEN", "")

    missing = []
    for name, val in [
        ("B2_AUTH_ID",         b2_auth_id),
        ("B2_APPLICATION_KEY", b2_app_key),
        ("YT_CLIENT_ID",       yt_client_id),
        ("YT_CLIENT_SECRET",   yt_client_sec),
        ("YT_REFRESH_TOKEN",   yt_refresh_tok),
    ]:
        if not val:
            missing.append(name)

    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    b2_ok = check_b2(b2_auth_id, b2_app_key, b2_bucket)
    yt_ok = check_youtube(yt_client_id, yt_client_sec, yt_refresh_tok)

    if b2_ok and yt_ok:
        print("\n✅ All pre-flight checks passed — starting pipeline")
    else:
        print("\n❌ Pre-flight checks FAILED — aborting")
        sys.exit(1)


if __name__ == "__main__":
    main()
