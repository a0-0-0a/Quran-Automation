#!/usr/bin/env python3
"""Generate src/data.rs from the source spreadsheets/CSVs in ../../Assets.

The safe ranges are the most safety-critical data in the project, so they are
baked into the binary (hard-coded) rather than read at runtime.  Re-run this
script only when the source sheet changes:

    python3 tools/gen_data.py --assets ../../Assets

Sources
-------
  ranges/quran_safe_ranges.xlsx     -> SAFE_RANGES  (surah, from_ayah, to_ayah)
  reciters and surah names./surahs.csv         -> SURAHS
  reciters and surah names./reciter_names.csv  -> RECITERS
"""
import argparse
import csv
import re
import zipfile
from pathlib import Path

import openpyxl


def load_ranges(assets: Path):
    wb = openpyxl.load_workbook(assets / "ranges" / "quran_safe_ranges.xlsx", data_only=True)
    ws = wb.worksheets[0]
    out = []
    for row in list(ws.iter_rows(values_only=True))[1:]:
        if not row or not row[4]:
            continue
        m = re.match(r"^(\d+):(\d+)(?:-(\d+))?$", str(row[4]).strip())
        if not m:
            raise SystemExit(f"bad range cell: {row!r}")
        s, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3) or m.group(2))
        out.append((s, a, b))
    out.sort()
    if len(out) != len(set(out)):
        raise SystemExit("duplicate ranges in sheet")
    return out


def load_surahs(assets: Path):
    p = assets / "reciters and surah names." / "surahs.csv"
    out = []
    with p.open(encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if row and row[0].strip().isdigit():
                out.append((int(row[0]), row[1].strip()))
    out.sort()
    return out


def load_reciters(assets: Path):
    p = assets / "reciters and surah names." / "reciter_names.csv"
    out = []
    with p.open(encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip() or row[0].strip() == "English Name":
                continue
            out.append((row[0].strip(), row[1].strip() if len(row) > 1 else ""))
    return out


def rust_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets/data", type=Path)
    ap.add_argument("--out", default="src/data.rs", type=Path)
    a = ap.parse_args()

    ranges = load_ranges(a.assets)
    surahs = load_surahs(a.assets)
    reciters = load_reciters(a.assets)

    lines = []
    lines.append("//! GENERATED FILE - do not edit by hand.")
    lines.append("//! Produced by tools/gen_data.py from ../../Assets.")
    lines.append("//!")
    lines.append("//! The safe ranges are the list of ayah spans the reciters are allowed")
    lines.append("//! to read as one continuous piece.  A render may only use a span that")
    lines.append("//! appears here verbatim: starting or stopping anywhere else can cut an")
    lines.append("//! ayah in half and destroy its meaning.")
    lines.append("")
    lines.append("/// (surah, first_ayah, last_ayah) - a continuous, meaning-safe span.")
    lines.append("pub static SAFE_RANGES: &[(u16, u16, u16)] = &[")
    for s, x, y in ranges:
        lines.append(f"    ({s}, {x}, {y}),")
    lines.append("];")
    lines.append("")
    lines.append("/// (surah_number, arabic_name).")
    lines.append("pub static SURAHS: &[(u16, &str)] = &[")
    for n, name in surahs:
        lines.append(f"    ({n}, {rust_str(name)}),")
    lines.append("];")
    lines.append("")
    lines.append("/// (english_name, arabic_name).")
    lines.append("pub static RECITERS: &[(&str, &str)] = &[")
    for en, ar in reciters:
        lines.append(f"    ({rust_str(en)}, {rust_str(ar)}),")
    lines.append("];")
    lines.append("")
    lines.append("/// True when `(surah, from, to)` is one of the meaning-safe spans.")
    lines.append("pub fn is_safe_range(surah: u16, from: u16, to: u16) -> bool {")
    lines.append("    SAFE_RANGES")
    lines.append("        .binary_search(&(surah, from, to))")
    lines.append("        .is_ok()")
    lines.append("}")
    lines.append("")
    lines.append("/// Arabic name of a surah (empty when out of range).")
    lines.append("pub fn surah_name(surah: u16) -> &'static str {")
    lines.append("    SURAHS")
    lines.append("        .iter()")
    lines.append("        .find(|(n, _)| *n == surah)")
    lines.append("        .map(|(_, name)| *name)")
    lines.append("        .unwrap_or(\"\")")
    lines.append("}")
    lines.append("")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {a.out}: {len(ranges)} ranges, {len(surahs)} surahs, {len(reciters)} reciters")


if __name__ == "__main__":
    main()