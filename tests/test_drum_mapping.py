import yaml
from pathlib import Path
from generator.drum_generator import DRUM_ALIAS
from utilities.drum_map_registry import (
    GM_DRUM_MAP,
    MISSING_DRUM_MAP_FALLBACK,
)


def test_all_drum_keys_mapped():
    library = {}
    path = Path('data/drum_patterns.yml')
    with path.open('r', encoding='utf-8') as fh:
        for doc in yaml.safe_load_all(fh):
            if not isinstance(doc, dict):
                continue
            if 'drum_patterns' in doc and isinstance(doc['drum_patterns'], dict):
                library.update(doc['drum_patterns'])
            else:
                library.update(doc)

    from utilities.fill_dsl import parse_fill_dsl

    inst_names = set()
    for pat in library.values():
        pattern_data = pat.get('pattern', [])
        if isinstance(pattern_data, str):
            try:
                events = parse_fill_dsl(pattern_data)
            except Exception:
                continue
        else:
            events = pattern_data
        for ev in events:
            name = ev.get('instrument')
            if name:
                inst_names.add(name.lower())

    for name in inst_names:
        mapped = MISSING_DRUM_MAP_FALLBACK.get(name, name)
        mapped = DRUM_ALIAS.get(mapped, mapped)
        assert mapped in GM_DRUM_MAP, f"{name} -> {mapped} not in GM_DRUM_MAP"
