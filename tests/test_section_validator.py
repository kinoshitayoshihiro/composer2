import pytest
import yaml

pytest.importorskip("pretty_midi")
import pretty_midi

from utilities import section_validator


def _new_pm() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    return pm


def _write_section_midi(path, start_bar, end_bar):
    """Create a simple MIDI file spanning the given bar range.

    Uses 120 BPM and 4/4 time so each bar is 2 seconds long."""

    pm = _new_pm()
    inst = pretty_midi.Instrument(program=0)
    seconds_per_bar = 2.0
    start = (start_bar - 1) * seconds_per_bar
    end = end_bar * seconds_per_bar
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=start, end=end))
    pm.instruments.append(inst)
    pm.write(str(path))


def _write_structure(path, sections):
    with open(path, "w", encoding="utf8") as fh:
        yaml.safe_dump({"sections": sections}, fh)


def test_validate_sections_success(tmp_path):
    sections = []
    labels = ["Verse", "Pre-Chorus", "Chorus", "Bridge"]
    for i, label in enumerate(labels):
        midi_path = tmp_path / f"{label}.mid"
        start = 2 * i + 1
        end = start + 1
        _write_section_midi(midi_path, start, end)
        sections.append(
            {
                "label": label,
                "midi": midi_path.name,
                "start_bar": start,
                "end_bar": end,
            }
        )
    struct_path = tmp_path / "structure.yaml"
    _write_structure(struct_path, sections)
    assert section_validator.validate_sections(struct_path)


def test_validate_sections_bad_label(tmp_path):
    midi_path = tmp_path / "foo.mid"
    _write_section_midi(midi_path, 1, 2)
    struct_path = tmp_path / "structure.yaml"
    _write_structure(
        struct_path,
        [
            {
                "label": "Unknown",
                "midi": midi_path.name,
                "start_bar": 1,
                "end_bar": 2,
            }
        ],
    )
    with pytest.raises(section_validator.SectionValidationError):
        section_validator.validate_sections(struct_path)


def test_section_validator_timesig_change(tmp_path):
    pm = _new_pm()
    pm.time_signature_changes.append(pretty_midi.TimeSignature(3, 4, 4.0))
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=4.0, end=8.5))
    pm.instruments.append(inst)
    midi_path = tmp_path / "chorus.mid"
    pm.write(str(midi_path))
    struct_path = tmp_path / "structure.yaml"
    _write_structure(
        struct_path,
        [
            {
                "label": "Chorus",
                "midi": midi_path.name,
                "start_bar": 3,
                "end_bar": 5,
            }
        ],
    )
    assert section_validator.validate_sections(struct_path)


def test_section_validator_empty_midi(tmp_path):
    pm = _new_pm()
    midi_path = tmp_path / "empty.mid"
    pm.write(str(midi_path))
    struct_path = tmp_path / "structure.yaml"
    _write_structure(
        struct_path,
        [
            {
                "label": "Verse",
                "midi": midi_path.name,
                "start_bar": 1,
                "end_bar": 1,
            }
        ],
    )
    with pytest.raises(
        section_validator.SectionValidationError, match="contains no notes"
    ):
        section_validator.validate_sections(struct_path)


def test_validate_sections_mismatch(tmp_path):
    midi_path = tmp_path / "verse.mid"
    _write_section_midi(midi_path, 1, 2)
    struct_path = tmp_path / "structure.yaml"
    _write_structure(
        struct_path,
        [
            {
                "label": "Verse",
                "midi": midi_path.name,
                "start_bar": 2,  # wrong start
                "end_bar": 3,
            }
        ],
    )
    with pytest.raises(section_validator.SectionValidationError):
        section_validator.validate_sections(struct_path)
