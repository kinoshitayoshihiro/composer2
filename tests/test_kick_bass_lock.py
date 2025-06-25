import json

from music21 import instrument

from generator.bass_generator import BassGenerator
from generator.drum_generator import DrumGenerator
from utilities.tempo_utils import TempoMap


def cfg(tmp, extra=None):
    hp = tmp/'heat.json'
    json.dump([{"grid_index":i,"count":0} for i in range(16)], hp.open('w'))
    c = {
        "heatmap_json_path_for_drums": str(hp),
        "paths": {"rhythm_library_path":"data/rhythm_library.yml"},
        "global_settings": {"kick_lock": {"enabled": True, "vel_boost": 10,
                                         "ghost_before_ms": [40,40],
                                         "window_ms": 30,
                                         "ghost_vel_ratio": 0.4,
                                         "random_seed": 0}}}
    if extra:
        c.update(extra)
    return c


def test_lock_and_lift(tmp_path):
    shared = {}
    drum = DrumGenerator(main_cfg=cfg(tmp_path), part_name="drums", part_parameters={})
    cfg_dict = cfg(tmp_path)
    bass = BassGenerator(
        main_cfg=cfg_dict,
        part_name="bass",
        part_parameters={},
        default_instrument=instrument.AcousticBass(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        global_settings=cfg_dict.get("global_settings"),
    )
    section_params = {
        "q_length": 4,
        "chord_symbol_for_voicing": "C",
        "part_params": {"bass": {"rhythm_key": "root_quarters", "velocity": 60}},
    }
    drum.render_kick_track(4.0)
    shared["kick_offsets_sec"] = drum.get_kick_offsets_sec()
    bass_part = bass.compose(section_data=section_params, shared_tracks=shared)
    kicks = shared["kick_offsets_sec"]
    locked = [
        n
        for n in bass_part.notes
        if any(abs((n.offset * 60 / 120) - k) < 0.03 for k in kicks)
    ]
    assert all(n.volume.velocity >= bass.base_velocity + 10 for n in locked)
    ghosts = [n for n in bass_part.notes if n.volume.velocity <= bass.base_velocity * 0.5]
    assert ghosts


def test_lock_and_lift_tempo_map(tmp_path):
    shared = {}
    tempo = TempoMap([
        {"beat": 0, "bpm": 120},
        {"beat": 2, "bpm": 80},
        {"beat": 3, "bpm": 150},
        {"beat": 4, "bpm": 60},
    ])
    drum = DrumGenerator(
        main_cfg=cfg(tmp_path),
        part_name="drums",
        part_parameters={},
        tempo_map=tempo,
    )
    cfg_dict = cfg(tmp_path)
    bass = BassGenerator(
        main_cfg=cfg_dict,
        part_name="bass",
        part_parameters={},
        default_instrument=instrument.AcousticBass(),
        global_tempo=None,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        global_settings=cfg_dict.get("global_settings"),
    )
    bass.tempo_map = tempo
    section_params = {
        "q_length": 4,
        "chord_symbol_for_voicing": "C",
        "part_params": {"bass": {"rhythm_key": "root_quarters", "velocity": 60}},
    }
    drum.render_kick_track(4.0)
    shared["kick_offsets_sec"] = drum.get_kick_offsets_sec()
    bass_part = bass.compose(section_data=section_params, shared_tracks=shared)
    offsets = [n.offset for n in bass_part.notes]
    assert offsets == sorted(offsets)
    ghosts = [n for n in bass_part.notes if n.volume.velocity <= bass.base_velocity * 0.5]
    assert ghosts
