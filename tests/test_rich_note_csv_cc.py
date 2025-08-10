import csv
import importlib.util
import sys
from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")

SPEC = importlib.util.spec_from_file_location(
    "rich_note_csv", Path(__file__).resolve().parents[1] / "utilities" / "rich_note_csv.py"
)
rich_note_csv = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["rich_note_csv"] = rich_note_csv
SPEC.loader.exec_module(rich_note_csv)
build_note_csv = rich_note_csv.build_note_csv


def make_midi(path: Path, *, cc=False, cc11=False):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0, end=1))
    if cc:
        inst.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=0))
    if cc11:
        inst.control_changes.append(pretty_midi.ControlChange(number=11, value=100, time=0))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_cc11_columns(tmp_path):
    midi = tmp_path / "cc.mid"
    make_midi(midi, cc=True, cc11=True)
    out = tmp_path / "out.csv"
    build_note_csv(tmp_path, out, include_cc=True, include_bend=False, include_cc11=True)
    with out.open() as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    assert header is not None
    assert "cc11_at_onset" in header and "cc11_mean" in header
    assert rows[0]["cc11_at_onset"] == "100"
    assert rows[0]["CC64"] == "127"


def test_no_cc_columns(tmp_path):
    midi = tmp_path / "noc.mid"
    make_midi(midi, cc=True, cc11=True)
    out = tmp_path / "noc.csv"
    build_note_csv(tmp_path, out, include_cc=False, include_bend=False, include_cc11=False)
    with out.open() as f:
        header = next(csv.reader(f))
    assert "CC64" not in header and "cc11_at_onset" not in header


def test_include_cc11_without_events(tmp_path):
    midi = tmp_path / "empty.mid"
    make_midi(midi, cc=False, cc11=False)
    out = tmp_path / "empty.csv"
    build_note_csv(tmp_path, out, include_cc=False, include_bend=False, include_cc11=True)
    with out.open() as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    assert header is not None
    assert "cc11_at_onset" in header and "cc11_mean" in header
    assert rows and "cc11_at_onset" in rows[0]
