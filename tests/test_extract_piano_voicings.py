from pathlib import Path
import json
import pretty_midi

from scripts.extract_piano_voicings import load_events, chunk_events, write_corpus


def _make_dual_inst(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    lh = pretty_midi.Instrument(program=0, name="LH")
    rh = pretty_midi.Instrument(program=0, name="RH")
    for i in range(4):
        lh.notes.append(pretty_midi.Note(velocity=80, pitch=50, start=i, end=i + 0.5))
        rh.notes.append(pretty_midi.Note(velocity=90, pitch=70, start=i, end=i + 0.5))
    pm.instruments.extend([lh, rh])
    pm.write(str(path))


def _make_single_inst(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=52, start=0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=72, start=1, end=1.5))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_extract_and_chunk(tmp_path: Path) -> None:
    midi1 = tmp_path / "a.mid"
    midi2 = tmp_path / "b.mid"
    _make_dual_inst(midi1)
    _make_single_inst(midi2)

    events1 = load_events(midi1)
    events2 = load_events(midi2)

    assert sum(e["hand"] == "lh" for e in events1) == 4
    assert sum(e["hand"] == "rh" for e in events1) == 4
    assert sum(e["hand"] == "lh" for e in events2) == 1
    assert sum(e["hand"] == "rh" for e in events2) == 1

    chunk = next(chunk_events(events1))
    assert {"bar", "beat", "dur", "note", "velocity", "hand"} <= set(chunk[0].keys())

    out = tmp_path / "out.jsonl"
    write_corpus(tmp_path, out)
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    total = sum(len(item["events"]) for item in lines)
    assert total == 10
