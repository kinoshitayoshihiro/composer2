from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

try:
    import pretty_midi  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback stub

    class ControlChange:
        def __init__(self, number: int, value: int, time: float):
            self.number = int(number)
            self.value = int(value)
            self.time = float(time)

    class PitchBend:
        def __init__(self, pitch: int, time: float):
            self.pitch = int(pitch)
            self.time = float(time)

    class Instrument:
        def __init__(self, program: int = 0, name: str | None = None):
            self.program = program
            self.name = name or ""
            self.control_changes = []
            self.pitch_bends = []
            self.notes = []

    class PrettyMIDI:
        def __init__(self):
            self.instruments = []

    pretty_midi = sys.modules["pretty_midi"] = type(
        "pretty_midi",
        (),
        {
            "ControlChange": ControlChange,
            "PitchBend": PitchBend,
            "Instrument": Instrument,
            "PrettyMIDI": PrettyMIDI,
        },
    )()

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "utilities.controls_spline", ROOT / "utilities" / "controls_spline.py"
)
module = importlib.util.module_from_spec(spec)
sys.modules.setdefault("utilities", type(sys)("utilities"))
sys.modules["utilities.controls_spline"] = module
assert spec.loader is not None
spec.loader.exec_module(module)
ControlCurve = module.ControlCurve

spec_apply = importlib.util.spec_from_file_location(
    "utilities.apply_controls", ROOT / "utilities" / "apply_controls.py"
)
apply_module = importlib.util.module_from_spec(spec_apply)
apply_module.__package__ = "utilities"
assert spec_apply.loader is not None
spec_apply.loader.exec_module(apply_module)
apply_controls = apply_module.apply_controls


def test_monotone_interpolation_endpoint():
    curve = ControlCurve(target="cc11", knots=[(0.0, 0.0), (1.0, 127.0)])
    t = [0.0, 1.0]
    vals = curve.sample(t)
    assert abs(vals[0] - 0.0) < 1e-6
    assert abs(vals[1] - 127.0) < 1e-6


def test_cc11_value_range():
    curve = ControlCurve(target="cc11", knots=[(0.0, -10.0), (1.0, 200.0)])
    t = [i / 24 for i in range(25)]
    vals = curve.sample(t)
    assert all(v >= 0 for v in vals)
    assert all(v <= 127 for v in vals)


def test_bend_encoding_roundtrip():
    times = [i / 20 for i in range(21)]
    values = [2.0 * math.sin(2 * math.pi * t) for t in times]
    curve = ControlCurve(target="bend", knots=list(zip(times, values)))
    bends = curve.to_pitch_bend(channel=0, range_semitones=2.0)
    peak = max(abs(b.pitch) for b in bends)
    assert 0.9 * 8191 <= peak <= 1.1 * 8191


def test_dedupe():
    times = [0.0, 1.0]
    values = [64.0, 64.0]
    curve = ControlCurve(target="cc11", knots=list(zip(times, values)))
    events = curve.to_midi_cc(channel=0, cc_number=11)
    assert len(events) == 1


def test_from_dense_simplifies():
    times = [i / 99 for i in range(100)]
    values = [t * 127.0 for t in times]
    curve = ControlCurve.from_dense(times, values, tol=0.5, max_knots=256)
    assert len(curve.knots) < 32
    recon = curve.sample(times)
    err = max(abs(a - b) for a, b in zip(recon, values))
    assert err < 0.5


def test_bend_units_and_roundtrip():
    times = [i / 20 for i in range(21)]
    semis = [2.0 * math.sin(2 * math.pi * t) for t in times]
    norms = [1.0 * math.sin(2 * math.pi * t) for t in times]
    curve_semi = ControlCurve(
        target="bend", knots=list(zip(times, semis)), units="semitones"
    )
    curve_norm = ControlCurve(
        target="bend", knots=list(zip(times, norms)), units="normalized"
    )
    bends_semi = curve_semi.to_pitch_bend(channel=0, range_semitones=2.0)
    bends_norm = curve_norm.to_pitch_bend(channel=0, range_semitones=2.0)
    peak_semi = max(abs(b.pitch) for b in bends_semi)
    peak_norm = max(abs(b.pitch) for b in bends_norm)
    assert 0.9 * 8191 <= peak_semi <= 1.1 * 8191
    assert 0.9 * 8191 <= peak_norm <= 1.1 * 8191


def test_domain_beats_with_tempo_map():
    def tempo_map(b: float) -> float:
        return 120.0 if b < 1.0 else 60.0

    curve = ControlCurve(
        target="cc11",
        domain="beats",
        knots=[(0.0, 0.0), (2.0, 127.0)],
        resolution_hz=2.0,
    )
    events = curve.to_midi_cc(channel=0, cc_number=11, tempo_map=tempo_map)
    times = [e.time for e in events]
    values = [e.value for e in events]
    assert abs(times[1] - 0.5) < 1e-6
    assert abs(times[-1] - 1.5) < 1e-6
    assert values == sorted(values)


def test_apply_controls_beats_domain():
    def tempo_map(b: float) -> float:
        return 120.0 if b < 1.0 else 60.0

    pm = pretty_midi.PrettyMIDI()
    curve = ControlCurve(
        target="cc11",
        domain="beats",
        knots=[(0.0, 0.0), (2.0, 127.0)],
        resolution_hz=2.0,
    )
    apply_controls(pm, {"cc11": curve}, {"cc11": 0}, tempo_map=tempo_map)
    inst = [i for i in pm.instruments if i.name == "channel0"][0]
    times = [e.time for e in inst.control_changes]
    values = [e.value for e in inst.control_changes]
    assert abs(times[1] - 0.5) < 1e-6
    assert abs(times[-1] - 1.5) < 1e-6
    assert values == sorted(values)


def test_rpn_emitted_once():
    pm = pretty_midi.PrettyMIDI()
    curve = ControlCurve(target="bend", knots=[(0.0, 0.0), (1.0, 1.0)])
    apply_controls(pm, {"bend": curve}, {"bend": 0})
    apply_controls(pm, {"bend": curve}, {"bend": 0})
    inst = [i for i in pm.instruments if i.name == "channel0"][0]
    nums = inst.control_changes
    assert sum(1 for cc in nums if cc.number == 101 and cc.value == 0) == 1
    assert sum(1 for cc in nums if cc.number == 100 and cc.value == 0) == 1
    assert sum(1 for cc in nums if cc.number == 6) == 1
    assert sum(1 for cc in nums if cc.number == 38) == 1
    first_pb = inst.pitch_bends[0].time
    rpn_time = max(
        cc.time for cc in inst.control_changes if cc.number in {101, 100, 6, 38}
    )
    assert rpn_time <= first_pb


def test_dedupe_epsilon():
    times = [0.0, 1.0, 2.0, 3.0]
    values = [64.0, 64.3, 64.2, 65.0]
    curve = ControlCurve(
        target="cc11", knots=list(zip(times, values)), resolution_hz=1.0
    )
    events = curve.to_midi_cc(channel=0, cc_number=11)
    assert len(events) == 2
    recon_curve = ControlCurve(
        target="cc11", knots=[(e.time, float(e.value)) for e in events]
    )
    recon = recon_curve.sample(times)
    err = max(abs(a - b) for a, b in zip(recon, values))
    assert err <= 0.5


def test_max_events_cap():
    times = [i / 200 for i in range(201)]
    values = [t * 127.0 for t in times]
    curve = ControlCurve(target="cc11", knots=list(zip(times, values)))
    events = curve.to_midi_cc(channel=0, cc_number=11, max_events=10)
    assert len(events) <= 10
    recon_curve = ControlCurve(
        target="cc11", knots=[(e.time, float(e.value)) for e in events]
    )
    recon = recon_curve.sample(times)
    err = max(abs(a - b) for a, b in zip(recon, values))
    assert err < 1.0


def test_instrument_routing():
    pm = pretty_midi.PrettyMIDI()
    cc_curve = ControlCurve(target="cc11", knots=[(0.0, 64.0), (1.0, 64.0)])
    bend_curve = ControlCurve(target="bend", knots=[(0.0, 0.0), (1.0, 1.0)])
    apply_controls(
        pm,
        {"cc11": cc_curve, "bend": bend_curve},
        {"cc11": 0, "bend": 1},
    )
    inst0 = [i for i in pm.instruments if i.name == "channel0"][0]
    inst1 = [i for i in pm.instruments if i.name == "channel1"][0]
    assert inst0.control_changes and not inst0.pitch_bends
    assert inst1.pitch_bends


def test_fractional_bend_range():
    times = [i / 20 for i in range(21)]
    values = [2.5 * math.sin(2 * math.pi * t) for t in times]
    curve = ControlCurve(target="bend", knots=list(zip(times, values)))
    pm = pretty_midi.PrettyMIDI()
    apply_controls(
        pm,
        {"bend": curve},
        {"bend": 0},
        bend_range_semitones=2.5,
    )
    inst = [i for i in pm.instruments if i.name == "channel0"][0]
    peak = max(abs(pb.pitch) for pb in inst.pitch_bends)
    assert 0.9 * 8191 <= peak <= 1.1 * 8191
    msb = [cc.value for cc in inst.control_changes if cc.number == 6][0]
    lsb = [cc.value for cc in inst.control_changes if cc.number == 38][0]
    assert msb == 2
    assert 48 <= lsb <= 52


def test_min_clamp_negative():
    vals = ControlCurve.convert_to_14bit([-10.0], 2.0)
    assert vals[0] == -8192


def test_dedupe_eps():
    times = [0.0, 1.0, 2.0, 3.0]
    values = [64.0, 64.3, 64.2, 64.8]
    default_curve = ControlCurve(
        target="cc11", knots=list(zip(times, values)), resolution_hz=1.0
    )
    default_events = default_curve.to_midi_cc(channel=0, cc_number=11)
    curve = ControlCurve(
        target="cc11", knots=list(zip(times, values)), resolution_hz=1.0, eps_cc=5.0
    )
    events = curve.to_midi_cc(channel=0, cc_number=11)
    assert len(events) < len(default_events)
    recon_curve = ControlCurve(
        target="cc11", knots=[(e.time, float(e.value)) for e in events]
    )
    recon = recon_curve.sample(times)
    err = max(abs(a - b) for a, b in zip(recon, values))
    assert err < 1.0
