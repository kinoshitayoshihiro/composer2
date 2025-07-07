from pathlib import Path

import pretty_midi

from scripts.evaluate_piano_model import evaluate_piece


def _midi(path: Path, pitches: list[int]) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    for i, p in enumerate(pitches):
        start = i * 0.5
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=p, start=start, end=start + 0.25))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_keys_exist(tmp_path: Path) -> None:
    ref = tmp_path / "ref.mid"
    gen = tmp_path / "gen.mid"
    _midi(ref, [60, 62, 64, 65])
    _midi(gen, [60, 62, 63, 65])
    metrics = evaluate_piece(ref, gen)
    for key in ("pitch_precision", "pitch_recall", "groove_similarity", "velocity_kl"):
        assert key in metrics
