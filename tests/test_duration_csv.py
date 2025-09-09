import csv
import logging
from pathlib import Path
import importlib.util

import pretty_midi

spec = importlib.util.spec_from_file_location(
    "duration_csv", Path(__file__).resolve().parents[1] / "utilities" / "duration_csv.py"
)
duration_csv = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(duration_csv)
build_duration_csv = duration_csv.build_duration_csv


def make_midi(path: Path, n_notes: int) -> None:
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for i in range(n_notes):
        inst.notes.append(
            pretty_midi.Note(pitch=60, start=i * 0.5, end=(i + 1) * 0.5, velocity=100)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_instrument_filter_reads_only_matching_files(tmp_path: Path) -> None:
    src = tmp_path / "midi"
    out = tmp_path / "out.csv"
    src.mkdir()
    make_midi(src / "Lead_Guitar.mid", 1)
    make_midi(src / "Piano.mid", 2)
    build_duration_csv(src, out, instrument="gUiTaR")
    with out.open() as fh:
        rows = list(csv.reader(fh))
    # header + one note from Lead_Guitar.mid
    assert len(rows) == 2


def test_instrument_filter_no_matches_warning(tmp_path: Path, caplog) -> None:
    src = tmp_path / "midi"
    out = tmp_path / "out.csv"
    src.mkdir()
    make_midi(src / "Piano.mid", 1)
    with caplog.at_level(logging.WARNING):
        build_duration_csv(src, out, instrument="Flute")
    with out.open() as fh:
        rows = list(csv.reader(fh))
    # Only header should be written
    assert rows == [["duration", "bar", "position", "pitch", "velocity"]]
    assert any("No MIDI files matched" in r.message for r in caplog.records)
