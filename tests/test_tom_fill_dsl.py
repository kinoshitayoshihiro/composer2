import json
from pathlib import Path

import pytest
from music21 import stream, meter

from utilities.fill_dsl import parse_fill_dsl, FillDSLParseError
from generator.drum_generator import DrumGenerator, GM_DRUM_MAP


def _cfg(tmp_path: Path) -> dict:
    heatmap = [{"grid_index": i, "count": 0} for i in range(16)]
    hp = tmp_path / "heatmap.json"
    with hp.open("w") as f:
        json.dump(heatmap, f)
    return {
        "vocal_midi_path_for_drums": "",
        "heatmap_json_path_for_drums": str(hp),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
        "rng_seed": 0,
    }


def test_parse_fill_dsl_basic():
    ev = parse_fill_dsl("(T1 T2 T3 S)x1")
    assert [e["instrument"] for e in ev] == ["tom1", "tom2", "tom3", "snare"]
    assert [e["offset"] for e in ev] == pytest.approx([0.0, 0.25, 0.5, 0.75])


def test_parse_fill_dsl_velocity_and_rest():
    ev = parse_fill_dsl(">1.1 T3 T2 T1 . K")
    assert ev[0]["velocity_factor"] == pytest.approx(1.1)
    assert ev[-1]["instrument"] == "kick"
    assert ev[-1]["offset"] == pytest.approx(1.0)


def test_apply_pattern_from_dsl(tmp_path: Path):
    gen = DrumGenerator(main_cfg=_cfg(tmp_path), part_name="drums", part_parameters={})
    part = stream.Part(id="drums")
    events = parse_fill_dsl("T1 T2")
    gen._apply_pattern(
        part,
        events,
        0.0,
        1.0,
        90,
        "eighth",
        0.5,
        meter.TimeSignature("4/4"),
        {},
    )
    midi = [n.pitch.midi for n in part.flatten().notes]
    assert midi == [GM_DRUM_MAP["tom1"][1], GM_DRUM_MAP["tom2"][1]]


def test_no_duplicate_keys_in_patterns():
    import yaml

    class UniqueLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate key: {key}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )

    with open("data/drum_patterns.yml", "r", encoding="utf-8") as fh:
        yaml.load(fh, Loader=UniqueLoader)


def test_group_repeat_optional():
    ev = parse_fill_dsl("(T1 T2)")
    assert [e["instrument"] for e in ev] == ["tom1", "tom2"]

