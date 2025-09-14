import pretty_midi
import yaml
from pathlib import Path

from ujam import sparkle_convert as sc


def _dummy_pm(length: float = 2.0):
    class Dummy:
        def __init__(self, length: float) -> None:
            self._length = length
            inst = pretty_midi.Instrument(0)
            inst.notes.append(pretty_midi.Note(velocity=1, pitch=60, start=0.0, end=length))
            inst.is_drum = False
            self.instruments = [inst]
            self.time_signature_changes = []

        def get_beats(self):
            step = 0.5
            n = int(self._length / step) + 1
            return [i * step for i in range(n)]

        def get_downbeats(self):
            return self.get_beats()[::4]

        def get_end_time(self):
            return self._length

        def get_tempo_changes(self):
            return [0.0], [120.0]

        def write(self, path: str) -> None:  # pragma: no cover
            Path(path).write_bytes(b"")

    return Dummy(length)


def test_mapping_template_loads() -> None:
    text = sc.generate_mapping_template(False)
    data = yaml.safe_load(text)
    assert data["swing_unit"] == "1/8"


def test_dry_run_emits_phrase_notes() -> None:
    pm = _dummy_pm()
    chords = [sc.ChordSpan(0.0, 2.0, 0, "maj")]
    mapping = yaml.safe_load(sc.generate_mapping_template(False))
    mapping["cycle_phrase_notes"] = [36]
    out = sc.build_sparkle_midi(pm, chords, mapping, 0.5, "chord", 0.0, 0, "flat", 120.0, 0.0, 0.5)
    phrase_inst = next(i for i in out.instruments if i.name == sc.PHRASE_INST_NAME)
    assert len(phrase_inst.notes) > 0
