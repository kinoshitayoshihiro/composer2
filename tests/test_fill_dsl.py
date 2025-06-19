from utilities.fill_dsl import parse
from music21 import stream
from generator.drum_generator import FillInserter


def test_parse_ok():
    ev = parse("T1 T2 T3 CRASH", length_beats=2.0)
    assert len(ev) == 4
    assert ev[-1]["instrument"] == "crash"
    assert ev[-1]["offset"] == 1.5


def test_unknown_token():
    import pytest
    with pytest.raises(KeyError):
        parse("T1 XYZ", 1.0)


def test_tom_run_insert(monkeypatch):
    lib = {"run": {"template": "T1 T2 T3 K", "length_beats": 1}}
    fi = FillInserter(lib)
    part = stream.Part(id="drums")
    fi.drum_map = {"tom1": ("", 48), "tom2": ("", 47), "tom3": ("", 45), "kick": ("", 36)}
    fi.insert(part, {"q_length": 4.0, "absolute_offset": 0.0}, "run")
    assert len(list(part.flatten().notes)) == 4
