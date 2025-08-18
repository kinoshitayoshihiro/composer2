import sys
import types


class ControlChange:
    def __init__(self, number: int, value: int, time: float):
        self.number = int(number)
        self.value = int(value)
        self.time = float(time)


class PitchBend:
    def __init__(self, pitch: int, time: float):
        self.pitch = int(pitch)
        self.time = float(time)


class Note:
    def __init__(self, velocity: int, pitch: int, start: float, end: float):
        self.velocity = int(velocity)
        self.pitch = int(pitch)
        self.start = float(start)
        self.end = float(end)


class Instrument:
    def __init__(self, program: int = 0, name: str | None = None):
        self.program = int(program)
        self.name = name or ""
        self.control_changes: list[ControlChange] = []
        self.pitch_bends: list[PitchBend] = []
        self.notes: list[Note] = []


class PrettyMIDI:
    def __init__(self, path: str | None = None, **kwargs):
        self.instruments: list[Instrument] = []
        self.resolution = 480
        # path/kwargs ignored; stub does not read from disk

    def write(self, path: str) -> None:  # pragma: no cover - stub
        try:
            open(path, "wb").close()
        except Exception:
            pass

    def get_tempo_changes(self):  # pragma: no cover - stub
        return [0.0], [120.0]

    def time_to_tick(self, time: float) -> float:  # pragma: no cover - stub
        return time * self.resolution * 2

    def time_to_beat(self, time: float) -> float:  # pragma: no cover - stub
        return time * 2


pretty_midi = types.SimpleNamespace(
    ControlChange=ControlChange,
    PitchBend=PitchBend,
    Instrument=Instrument,
    PrettyMIDI=PrettyMIDI,
    Note=Note,
)

sys.modules.setdefault("pretty_midi", pretty_midi)
