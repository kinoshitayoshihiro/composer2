from __future__ import annotations
import sys
from pathlib import Path
import yaml
import logging

from generator.drum_generator import (
    GM_DRUM_MAP,
    DRUM_ALIAS,
    MISSING_DRUM_MAP_FALLBACK,
)

logging.basicConfig(level=logging.WARNING)

def collect_instruments(obj):
    """Recursively collect all values of keys named ``instrument``."""

    labels = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "instrument":
                labels.add(str(value))
            else:
                labels.update(collect_instruments(value))
    elif isinstance(obj, list):
        for item in obj:
            labels.update(collect_instruments(item))
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
        alias = DRUM_ALIAS.get(label)
        if alias is None:
            alias = MISSING_DRUM_MAP_FALLBACK.get(label)
        gm_name = alias or label
        pitch = GM_DRUM_MAP.get(gm_name)
        if pitch is None:
            logging.warning("Unmapped drum label: %s", label)
            rows.append((label, alias, "–", "–"))
            warn = True
            continue
        rows.append((label, alias, gm_name, str(pitch)))

    print("| Label | Alias/Fallback | GM name | MIDI# |")
    print("|------|---------------|--------|------|")
    for label, alias, gm_name, pitch in rows:
        alias_disp = alias or ""
        print(f"| {label} | {alias_disp} | {gm_name} | {pitch} |")

    return 1 if warn else 0

if __name__ == "__main__":
    sys.exit(main())
