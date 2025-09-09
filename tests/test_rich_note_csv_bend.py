import csv
import importlib.util
import sys
from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")

SPEC = importlib.util.spec_from_file_location(
    "rich_note_csv",
    Path(__file__).resolve().parents[1] / "utilities" / "rich_note_csv.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["rich_note_csv"] = module
SPEC.loader.exec_module(module)
build_note_csv = module.build_note_csv


def make_rpn_midi(path: Path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=1.0))
    inst.control_changes.extend(
        [
            pretty_midi.ControlChange(number=101, value=0, time=0.0),
            pretty_midi.ControlChange(number=100, value=0, time=0.0),
            pretty_midi.ControlChange(number=6, value=12, time=0.0),
        ]
    )
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=4096, time=0.5))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_rpn_range(tmp_path: Path):
    midi = tmp_path / "rpn.mid"
    make_rpn_midi(midi)
    out = tmp_path / "out.csv"
    build_note_csv(tmp_path, out)
    with out.open() as f:
        row = next(csv.DictReader(f))
    assert row["bend_range"] == "12"
    assert abs(float(row["bend_max_semi"]) - 6.0) < 1e-3


def make_no_bend_midi(path: Path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_no_pitch_bend_blank(tmp_path: Path):
    midi = tmp_path / "plain.mid"
    make_no_bend_midi(midi)
    out = tmp_path / "plain.csv"
    build_note_csv(tmp_path, out)
    with out.open() as f:
        row = next(csv.DictReader(f))
    assert row["bend"] == ""
    assert row["bend_max_semi"] == ""
    assert row["bend_rms_semi"] == ""
    assert row["vib_rate_hz"] == ""


def make_unsorted_bend_midi(path: Path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=1.0))
    # Add pitch bends out of order
    inst.pitch_bends.extend(
        [
            pretty_midi.PitchBend(pitch=0, time=0.75),
            pretty_midi.PitchBend(pitch=4096, time=0.25),
            pretty_midi.PitchBend(pitch=0, time=0.5),
        ]
    )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_unsorted_bend_events(tmp_path: Path):
    midi = tmp_path / "unsorted.mid"
    make_unsorted_bend_midi(midi)
    out = tmp_path / "unsorted.csv"
    build_note_csv(tmp_path, out)
    with out.open() as f:
        row = next(csv.DictReader(f))
    assert abs(float(row["bend_max_semi"]) - 1.0) < 1e-3


def make_nrpn_then_rpn_midi(path: Path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=1.0))
    inst.control_changes.extend(
        [
            pretty_midi.ControlChange(number=99, value=1, time=0.0),  # NRPN MSB
            pretty_midi.ControlChange(number=98, value=2, time=0.0),  # NRPN LSB
            pretty_midi.ControlChange(
                number=6, value=24, time=0.0
            ),  # Should be ignored
            pretty_midi.ControlChange(number=101, value=0, time=0.1),
            pretty_midi.ControlChange(number=100, value=0, time=0.1),
            pretty_midi.ControlChange(number=6, value=4, time=0.1),
        ]
    )
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=4096, time=0.5))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_nrpn_does_not_set_range(tmp_path: Path):
    midi = tmp_path / "nrpn.mid"
    make_nrpn_then_rpn_midi(midi)
    out = tmp_path / "nrpn.csv"
    build_note_csv(tmp_path, out)
    with out.open() as f:
        row = next(csv.DictReader(f))
    assert row["bend_range"] == "4"
