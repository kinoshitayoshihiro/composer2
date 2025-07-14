from pathlib import Path

import pretty_midi
from music21 import instrument, stream

from generator.bass_generator import BassGenerator
from generator.piano_generator import PianoGenerator
from loaders.section_loader import load_sections
from utilities.midi_export import export_song


def _make_vocal_midi(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0, end=0.25))
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=1, end=1.25))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_vocal_metrics_plumbing(tmp_path: Path) -> None:
    midi_path = tmp_path / "v.mid"
    _make_vocal_midi(midi_path)
    sec_yaml = tmp_path / "sections.yaml"
    sec_yaml.write_text(f"- vocal_midi_path: {midi_path.name}\n")
    sections = load_sections(sec_yaml)

    received = {}

    def bass_gen(section, tempo, *, vocal_metrics=None):
        received["bass"] = vocal_metrics
        gen = BassGenerator(
            global_settings={"tempo_bpm": tempo},
            default_instrument=instrument.ElectricBass(),
            global_tempo=tempo,
            global_time_signature="4/4",
            global_key_signature_tonic="C",
            global_key_signature_mode="major",
            part_name="bass",
        )
        part = gen.compose(section_data=section, vocal_metrics=vocal_metrics)
        score = stream.Score([part])
        return pretty_midi.PrettyMIDI(initial_tempo=tempo)

    def piano_gen(section, tempo, *, vocal_metrics=None):
        received["piano"] = vocal_metrics
        gen = PianoGenerator(
            global_settings={"tempo_bpm": tempo},
            default_instrument=instrument.Piano(),
            global_tempo=tempo,
            global_time_signature="4/4",
            global_key_signature_tonic="C",
            global_key_signature_mode="major",
            part_name="piano",
        )
        parts = gen.compose(section_data=section, vocal_metrics=vocal_metrics)
        return pretty_midi.PrettyMIDI(initial_tempo=tempo)

    export_song(
        2,
        generators={"bass": bass_gen, "piano": piano_gen},
        fixed_tempo=120,
        sections=sections,
        out_path=tmp_path / "out.mid",
    )

    assert received["bass"]["rests"]
    assert received["piano"]["rests"]
