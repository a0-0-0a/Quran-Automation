"""Progress state: the last completed short, stored in the repo."""
import json
from pathlib import Path

STATE = Path(__file__).resolve().parent.parent / "state" / "progress.json"


def load() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"reciter_index": 0, "range_index": 0, "last_posted": None}


def save(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def advance(state: dict, n_ranges: int, n_reciters: int, posted: str | None = None) -> dict:
    """Move one step forward (one range; wraps to the next reciter)."""
    state["range_index"] = int(state.get("range_index", 0)) + 1
    if state["range_index"] >= n_ranges:
        state["range_index"] = 0
        state["reciter_index"] = (int(state.get("reciter_index", 0)) + 1) % n_reciters
    if posted is not None:
        state["last_posted"] = posted
    return state


def advance_reciter(state: dict, n_ranges: int, n_reciters: int) -> dict:
    """Skip to the next reciter (used when a reciter is unavailable)."""
    state["range_index"] = 0
    state["reciter_index"] = (int(state.get("reciter_index", 0)) + 1) % n_reciters
    return state