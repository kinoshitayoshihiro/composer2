import pathlib
import os
from unittest import mock

from generator.melody_generator import MelodyGenerator
from music21 import stream, note, instrument


def test_write_creates_section_dir(tmp_path: pathlib.Path) -> None:
    gen = MelodyGenerator(
        part_name="melody",
        default_instrument=instrument.Flute(),
        global_tempo=120,
        global_time_signature="4/4",
        global_key_signature_tonic="C",
        global_key_signature_mode="major",
    )
    part = stream.Part(id="melody")
    part.append(note.Note("C4", quarterLength=1.0))

    project = tmp_path / "project"
    orig_makedirs = os.makedirs
    with mock.patch("os.makedirs") as makedirs:
        makedirs.side_effect = lambda path, exist_ok=True: orig_makedirs(path, exist_ok=exist_ok)
        out = gen.write(part, project, "Verse1")
        assert makedirs.call_args_list[0] == mock.call(project / "Verse1", exist_ok=True)

    assert out.exists()
    assert out.parent.is_dir()
