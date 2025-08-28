from pathlib import Path

from tests import _stubs  # ensure pretty_midi stub is registered
from tools.ujam_bridge.ujam_map import pattern_to_keyswitches


def _parse_simple(text: str):
    data = {}
    current = None
    for raw in text.splitlines():
        line = raw.split('#')[0].strip()
        if not line:
            continue
        if line.endswith(':'):
            current = line[:-1]
            data[current] = {}
            continue
        key, val = [x.strip() for x in line.split(':', 1)]
        key = key.strip('"')
        if current:
            if val.startswith('['):
                val = [v.strip().strip('"') for v in val.strip('[]').split(',') if v.strip()]
            else:
                val = int(val) if val.isdigit() else val.strip('"')
            data[current][key] = val
        else:
            data[key] = int(val) if val.isdigit() else val.strip('"')
    return data


def test_pattern_to_keyswitches() -> None:
    config = _parse_simple(Path("tools/ujam_bridge/configs/vg_iron2.yaml").read_text())
    library = _parse_simple(Path("tools/ujam_bridge/patterns/strum_library.yaml").read_text())["patterns"]
    res = pattern_to_keyswitches("D U", library, config["keyswitch"])
    assert res == [36, 37]
