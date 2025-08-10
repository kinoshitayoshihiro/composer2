import csv
from pathlib import Path

import pytest
import importlib.util
import sys

pretty_midi = pytest.importorskip("pretty_midi")

SPEC = importlib.util.spec_from_file_location(
    "rich_note_csv", Path(__file__).resolve().parents[1] / "utilities" / "rich_note_csv.py"
)
rich_note_csv = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["rich_note_csv"] = rich_note_csv
SPEC.loader.exec_module(rich_note_csv)
build_note_csv = rich_note_csv.build_note_csv


def make_midi(
    path: Path,
    *,
    time_signature=(4, 4),
    notes=None,
    cc=False,
    bend=False,
    cc11=False,
):
    pm = pretty_midi.PrettyMIDI()
    num, den = time_signature
    pm.time_signature_changes.append(pretty_midi.TimeSignature(num, den, 0))
    inst = pretty_midi.Instrument(0)
    for start, end, pitch in notes or []:
        inst.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end))
    if cc:
        inst.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=0))
    if cc11:
        inst.control_changes.append(pretty_midi.ControlChange(number=11, value=100, time=0))
    if bend:
        inst.pitch_bends.append(pretty_midi.PitchBend(pitch=8191, time=0))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_bar_position_4_4(tmp_path):
    midi = tmp_path / "four.mid"
    make_midi(midi, time_signature=(4, 4), notes=[(0, 0.5, 60), (2, 2.5, 62)])
    out = tmp_path / "out.csv"
    build_note_csv(tmp_path, out)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["bar"] == "0" and rows[0]["position"] == "0"
    assert rows[1]["bar"] == "1" and rows[1]["position"] == "0"


def test_bar_position_3_4(tmp_path):
    midi = tmp_path / "three.mid"
    make_midi(midi, time_signature=(3, 4), notes=[(0, 0.5, 60), (1.5, 2.0, 62)])
    out = tmp_path / "three.csv"
    build_note_csv(tmp_path, out)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["bar"] == "0" and rows[0]["position"] == "0"
    assert rows[1]["bar"] == "1" and rows[1]["position"] == "0"


def test_cc_bend_flags(tmp_path):
    midi = tmp_path / "ctrl.mid"
    make_midi(midi, notes=[(0, 1, 60)], cc=True, bend=True)
    out1 = tmp_path / "nocc.csv"
    build_note_csv(tmp_path, out1, include_cc=False, include_bend=False)
    with out1.open() as f:
        header = next(csv.reader(f))
    assert "CC64" not in header and "bend" not in header

    out2 = tmp_path / "with.csv"
    build_note_csv(tmp_path, out2)
    with out2.open() as f:
        header2 = next(csv.reader(f))
    assert "CC64" in header2 and "bend" in header2


