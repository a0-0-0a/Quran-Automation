"""Pick a random Arabic title / description for one short."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "title": ROOT / "titling" / "title-short.json",
    "desc": ROOT / "titling" / "discription.json",
}


def _key(count: int) -> str:
    if count <= 1:
        return "single_ayah"
    if count == 2:
        return "two_ayahs"
    return "multiple_ayahs"


def pick(kind: str, reciter_ar: str, surah_ar: str, ayah_range: str, count: int) -> str:
    data = json.loads(FILES[kind].read_text(encoding="utf-8"))
    templates = data[_key(count)]
    return random.choice(templates).format(
        reciter=reciter_ar, surah=surah_ar, ayah=ayah_range
    )