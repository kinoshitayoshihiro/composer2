from __future__ import annotations

from typing import Dict, Any

QUALITY2ID = {"maj": 0, "min": 1, "dom7": 2, "dim": 3, "aug": 4, "sus4": 5}
FUNCTION2ID = {"T": 0, "S": 1, "D": 2}

NOTE2SEMITONE = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}

def parse_symbol(symbol: str) -> Dict[str, Any]:
    s = symbol.strip()
    if not s:
        return {"root": 0, "quality_id": QUALITY2ID["maj"], "degree": 1, "tensions": 0}
    if len(s) >= 2 and s[:2] in NOTE2SEMITONE:
        root = NOTE2SEMITONE[s[:2]]
        rest = s[2:]
    else:
        root = NOTE2SEMITONE.get(s[0].upper(), 0)
        rest = s[1:]
    qid = QUALITY2ID["maj"]
    if "m" in rest and "maj" not in rest:
        qid = QUALITY2ID["min"]
    if "7" in rest and "maj" not in rest and "m7b5" not in rest:
        qid = QUALITY2ID["dom7"]
    if "sus4" in rest:
        qid = QUALITY2ID["sus4"]
    degree = 1
    tensions = 0
    return {"root": int(root), "quality_id": int(qid), "degree": int(degree), "tensions": int(tensions)}

def harmonic_function(degree: int, quality_id: int) -> int:
    if degree in (1, 3, 6):
        return FUNCTION2ID["T"]
    if degree in (2, 4):
        return FUNCTION2ID["S"]
    if degree in (5, 7):
        return FUNCTION2ID["D"]
    return FUNCTION2ID["T"]

def encode_event(ev: Dict[str, Any]) -> Dict[str, int]:
    p = parse_symbol(ev.get("symbol", ""))
    func = harmonic_function(p["degree"], p["quality_id"])  # 0/1/2
    return {
        "harm_root": p["root"],
        "harm_quality": p["quality_id"],
        "harm_func": func,
        "harm_degree": p["degree"],
        "harm_tensions": p["tensions"],
    }

