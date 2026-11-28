#!/usr/bin/env python3
"""Download one reviewed `recit/<n>` folder from Backblaze B2.

Used by the manual render workflow and handy for local inspection.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import b2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", type=int, required=True, help="recit/<n> folder number")
    ap.add_argument("--out", type=Path, required=True, help="destination directory")
    args = ap.parse_args()

    auth = b2.authorize()
    if args.out.exists():
        shutil.rmtree(args.out)
    b2.download_folder(auth, args.folder, args.out)
    print(f"downloaded recit/{args.folder} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
