from __future__ import annotations
import sys
from pathlib import Path
import yaml
from generator.drum_generator import GM_DRUM_MAP, MISSING_DRUM_MAP_FALLBACK

def collect_instruments(obj):
    """Recursively collect instrument labels from 'pattern' or 'fill_patterns' lists."""
    labels = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in {"pattern", "fill_patterns"} and isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "instrument" in item:
                        labels.add(str(item["instrument"]))
            else:
                labels.update(collect_instruments(v))
    elif isinstance(obj, list):
        for x in obj:
            labels.update(collect_instruments(x))
    return labels

def main() -> int:
    path = Path("data/rhythm_library.yml")
    docs = list(yaml.safe_load_all(path.read_text()))
    labels = set()
    for doc in docs:
        if doc is not None:
            labels.update(collect_instruments(doc))

    rows = []
    warn = False
    for label in sorted(labels):
        alias = MISSING_DRUM_MAP_FALLBACK.get(label)
        inst = alias or label
        pitch = GM_DRUM_MAP.get(inst)
        if pitch is None:
            warn = True
        rows.append((label, alias, inst, pitch))

    print("| Label | Alias/Fallback | GM name | MIDI# |")
    print("|------|---------------|--------|------|")
    for label, alias, inst, pitch in rows:
        mark = " ⚠️" if pitch is None else ""
        alias_disp = alias if alias is not None else ""
        pitch_disp = "" if pitch is None else str(pitch)
        print(f"| {label} | {alias_disp} | {inst} | {pitch_disp}{mark} |")

    return 1 if warn else 0

if __name__ == "__main__":
    sys.exit(main())
