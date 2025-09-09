import yaml

try:
    from some_module import get_drum_map  # placeholder if library exists
except Exception:  # fallback to internal map
    from generator.drum_generator import GM_DRUM_MAP as _MAP

    def get_drum_map(name: str | None = None):
        return _MAP

ALIASES = {
    "hh": "closed_hi_hat",
    "ghost": "snare",
    "shaker_soft": "shaker",
    "ride_cymbal_swell": "ride",
    "chimes": "triangle",
}


def _gather_instruments(obj):
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key == "instrument":
                yield val
            else:
                yield from _gather_instruments(val)
    elif isinstance(obj, list):
        for item in obj:
            yield from _gather_instruments(item)


def test_drum_map_integrity():
    instruments = set()
    with open("data/drum_patterns.yml", "r", encoding="utf-8") as fh:
        for doc in yaml.safe_load_all(fh):
            instruments.update(_gather_instruments(doc))

    drum_map = get_drum_map("ujam_legend")
    missing = []
    for inst in instruments:
        key = inst if inst in drum_map else ALIASES.get(inst, inst)
        if key not in drum_map:
            missing.append(inst)
    assert not missing, f"Missing drum map entries: {missing}"
