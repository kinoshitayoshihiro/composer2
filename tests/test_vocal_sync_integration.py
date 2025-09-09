import json
import random
from pathlib import Path

import pretty_midi
from music21 import instrument, stream

from generator.bass_generator import BassGenerator
from generator.piano_generator import PianoGenerator
from loaders.section_loader import load_sections
from utilities import vocal_sync
from utilities.arrangement_builder import score_to_pretty_midi
from utilities.midi_export import export_song


def _make_vocal_midi(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    for i in range(4):
        start = i * 1.0
        inst.notes.append(
            pretty_midi.Note(velocity=90, pitch=60, start=start, end=start + 0.25)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def _bass_gen(
    section: dict, tempo: float, *, vocal_metrics=None
) -> pretty_midi.PrettyMIDI:
    gen = BassGenerator(
        global_settings={"tempo_bpm": tempo},
        default_instrument=instrument.ElectricBass(),
        global_tempo=tempo,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        part_name="bass",
        part_parameters={},
    )
    part = gen.compose(
        section_data={"chord_symbol_for_voicing": "C", "q_length": 4},
        vocal_metrics=vocal_metrics,
    )
    score = stream.Score([part])
    return score_to_pretty_midi(score)


def _piano_gen(
    section: dict, tempo: float, *, vocal_metrics=None
) -> pretty_midi.PrettyMIDI:
    random.seed(0)
    gen = PianoGenerator(
        global_settings={"tempo_bpm": tempo},
        default_instrument=instrument.Piano(),
        global_tempo=tempo,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
        part_name="piano",
        part_parameters={},
    )
    parts = gen.compose(
        section_data={"chord_symbol_for_voicing": "C", "q_length": 4},
        vocal_metrics=vocal_metrics,
    )
    if isinstance(parts, dict):
        score = stream.Score(list(parts.values()))
    else:
        score = stream.Score([parts])
    return score_to_pretty_midi(score)


def test_vocal_sync_integration(tmp_path: Path) -> None:
    midi_path = tmp_path / "v.mid"
    _make_vocal_midi(midi_path)
    peaks_path = tmp_path / "c.json"
    peaks_path.write_text(json.dumps({"peaks": [0.5]}))
    sec_yaml = tmp_path / "sections.yaml"
    sec_yaml.write_text(
        f"- vocal_midi_path: {midi_path.name}\n  consonant_json: {peaks_path.name}\n"
    )

    sections = load_sections(sec_yaml)
    out = tmp_path / "song.mid"
    pm = export_song(
        4,
        generators={"bass": _bass_gen, "piano": _piano_gen},
        fixed_tempo=120,
        sections=sections,
        out_path=out,
    )

    pitches = [n.pitch for inst in pm.instruments for n in inst.notes]
    assert any(p < 60 for p in pitches)  # approach note below root
