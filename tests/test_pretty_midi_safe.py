import os
import sys
import tempfile

import pytest

pretty_midi = pytest.importorskip("pretty_midi")

from utilities.pretty_midi_safe import pm_to_mido


def test_pm_to_mido_roundtrip_tmpfile_cleanup(monkeypatch):
    pm = pretty_midi.PrettyMIDI()
    paths: list[str] = []

    orig_ntf = tempfile.NamedTemporaryFile

    def fake_ntf(*args, **kwargs):
        tmp = orig_ntf(*args, **kwargs)
        paths.append(tmp.name)
        return tmp

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", fake_ntf)
    midi = pm_to_mido(pm)
    assert hasattr(midi, "tracks")
    assert paths and not os.path.exists(paths[0])


def test_no_mido_raises_importerror(monkeypatch):
    pm = pretty_midi.PrettyMIDI()
    monkeypatch.setitem(sys.modules, "mido", None)
    with pytest.raises(ImportError):
        pm_to_mido(pm)
