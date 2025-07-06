import json
import pretty_midi
from pathlib import Path
from generator.drum_generator import DrumGenerator, GM_DRUM_MAP
from tests.helpers.events import make_event

class SimpleDrum(DrumGenerator):
    def _resolve_style_key(self, mi, ov, section_data=None):
        return "main"

def _cfg(tmp_path: Path, midi_path: str = "", extra_global=None):
    heatmap = [{"grid_index": i, "count": 0} for i in range(16)]
    hp = tmp_path / "heatmap.json"
    with hp.open("w") as f:
        json.dump(heatmap, f)
    cfg = {
        "vocal_midi_path_for_drums": midi_path,
        "heatmap_json_path_for_drums": str(hp),
        "paths": {"rhythm_library_path": "data/rhythm_library.yml"},
    }
    if extra_global:
        cfg["global_settings"] = extra_global
    return cfg

pattern_flam = {
    "main": {
        "pattern": [
            make_event(instrument="snare", offset=0.5, type="flam"),
            make_event(instrument="snare", offset=1.5, type="flam"),
        ],
        "length_beats": 2.0,
    }
}

def test_flam_grace_count(tmp_path: Path):
    cfg = _cfg(tmp_path)
    gen = SimpleDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_flam)
    section = {"absolute_offset": 0.0, "q_length": 2.0, "part_params": {}}
    part = gen.compose(section_data=section)
    notes = list(part.flatten().notes)
    assert len(notes) == 4

pattern_beat = {
    "main": {
        "pattern": [
            make_event(instrument="kick", offset=1.0),
        ],
        "length_beats": 2.0,
    }
}

push_pull = {"push_pull_curve": {"anger": [-40, -30, 0, 0], "sad": [20, 30, 40, 30]}}

def test_push_pull_sign(tmp_path: Path):
    cfg = _cfg(tmp_path, extra_global=push_pull)
    gen = SimpleDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_beat)
    sec_a = {
        "absolute_offset": 0.0,
        "q_length": 2.0,
        "expression_details": {"emotion_bucket": "anger"},
        "part_params": {},
    }
    part_a = gen.compose(section_data=sec_a)
    off_a = list(part_a.flatten().notes)[0].offset
    assert off_a < 1.0

    sec_s = sec_a.copy()
    sec_s["expression_details"] = {"emotion_bucket": "sad"}
    part_s = gen.compose(section_data=sec_s)
    off_s = list(part_s.flatten().notes)[0].offset
    assert off_s > 1.0

pattern_hat = {
    "main": {
        "pattern": [make_event(instrument="chh", offset=i) for i in range(4)],
        "length_beats": 4.0,
    }
}

def _write_vocal(tmp_path: Path) -> str:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0, end=1.99))
    pm.instruments.append(inst)
    path = tmp_path / "voc.mid"
    pm.write(str(path))
    return str(path)

def test_hh_open_replacement(tmp_path: Path):
    midi_path = _write_vocal(tmp_path)
    cfg = _cfg(tmp_path, midi_path=midi_path)
    gen = SimpleDrum(main_cfg=cfg, part_name="drums", part_parameters=pattern_hat)
    section = {"absolute_offset": 0.0, "q_length": 8.0, "length_in_measures": 2, "part_params": {}}
    part = gen.compose(section_data=section)
    notes = list(part.flatten().notes)
    ohh = GM_DRUM_MAP["ohh"][1]
    target = next(n for n in notes if abs(n.offset - 4.0) < 0.01)
    assert target.pitch.midi == ohh
    assert sum(1 for n in notes if n.pitch.midi == ohh) == 1
