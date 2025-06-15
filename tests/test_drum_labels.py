from pathlib import Path

from utilities.rhythm_library_loader import load_rhythm_library
from utilities.drum_map_registry import DRUM_MAP


def test_all_rhythm_library_labels_mapped():
    lib = load_rhythm_library(Path('data/rhythm_library.yml'))
    labels = set()
    for pat in lib.drum_patterns.values():
        if pat.pattern:
            for ev in pat.pattern:
                if ev.instrument:
                    labels.add(ev.instrument)
        for fill in pat.fill_ins.values():
            for ev in fill:
                if ev.instrument:
                    labels.add(ev.instrument)
    unknown = labels - DRUM_MAP.keys()
    assert not unknown, f"Unknown drum labels: {sorted(unknown)}"
