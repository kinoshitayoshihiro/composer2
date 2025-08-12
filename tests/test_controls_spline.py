from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

# --- Optional deps ---------------------------------------------------------
# Numpy is required by several tests; skip the whole module if absent.
np = pytest.importorskip("numpy")

# pretty_midi is optional for CI; provide a tiny stub if it's missing so most
# tests can still run. (Tests that genuinely need the real lib will skip.)
try:  # pragma: no cover
    import pretty_midi  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback stub

    class ControlChange:  # minimal API used in tests
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
            self.program = program
            self.name = name or ""
            self.control_changes: list[ControlChange] = []
            self.pitch_bends: list[PitchBend] = []
            self.notes: list[Note] = []

    class PrettyMIDI:
        def __init__(self):
            self.instruments: list[Instrument] = []

    pretty_midi = sys.modules["pretty_midi"] = type(
        "pretty_midi",
        (),
        {
            "ControlChange": ControlChange,
            "PitchBend": PitchBend,
            "Instrument": Instrument,
            "PrettyMIDI": PrettyMIDI,
            "Note": Note,
        },
    )()


# --- Load project modules directly from source ----------------------------
ROOT = Path(__file__).resolve().parent.parent

# utilities.controls_spline -> ControlCurve
spec_cs = importlib.util.spec_from_file_location(
    "utilities.controls_spline", ROOT / "utilities" / "controls_spline.py"
)
module_cs = importlib.util.module_from_spec(spec_cs)
sys.modules.setdefault("utilities", type(sys)("utilities"))
sys.modules["utilities.controls_spline"] = module_cs
assert spec_cs.loader is not None
spec_cs.loader.exec_module(module_cs)
ControlCurve = module_cs.ControlCurve

# utilities.apply_controls -> apply_controls, write_bend_range_rpn
spec_ac = importlib.util.spec_from_file_location(
    "utilities.apply_controls", ROOT / "utilities" / "apply_controls.py"
)
module_ac = importlib.util.module_from_spec(spec_ac)
module_ac.__package__ = "utilities"
assert spec_ac.loader is not None
spec_ac.loader.exec_module(module_ac)
apply_controls = module_ac.apply_controls
if hasattr(module_ac, "write_bend_range_rpn"):
    write_bend_range_rpn = module_ac.write_bend_range_rpn
else:  # pragma: no cover
    def write_bend_range_rpn(*_args, **_kwargs):
        pytest.skip("write_bend_range_rpn not available in utilities.apply_controls")


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _collect_cc(inst: pretty_midi.Instrument, number: int):
    return [c for c in inst.control_changes if c.number == number]


# -------------------------------------------------------------------------
# Tests from the CC/RPN branch (updated API: ControlCurve(times, values, ...))
# -------------------------------------------------------------------------

def test_fit_infer(tmp_path):
    # Import inside the test to avoid skipping the whole file when optional
    # stack (pandas + ml.controls_spline) is unavailable.
    pd = pytest.importorskip("pandas")
    try:
        from ml.controls_spline import fit_controls, infer_controls  # type: ignore
    except Exception:
        pytest.skip("ml.controls_spline not available")

    df = pd.DataFrame({"bend": [0, 1, 2], "cc11": [10, 20, 30]})
    notes = tmp_path / "notes.parquet"
    try:
        df.to_parquet(notes)
    except Exception:
        pytest.skip("parquet engine (pyarrow/fastparquet) not available")

    model = tmp_path / "model.json"
    fit_controls(notes, targets=["bend", "cc11"], out_path=model)
    out = tmp_path / "pred.parquet"
    infer_controls(model, out_path=out)
    assert out.exists()


def test_rpn_lsb_and_null():
    pm = pretty_midi.PrettyMIDI()
    curve = ControlCurve([0.0, 1.0], [0.0, 0.1])
    apply_controls(
        pm,
        {0: {"bend": curve}},
        write_rpn=True,
        bend_range_semitones=2.5,
        lsb_mode="cents",
    )
    # call twice: should not duplicate RPN
    apply_controls(
        pm,
        {0: {"bend": curve}},
        write_rpn=True,
        bend_range_semitones=2.5,
        lsb_mode="cents",
    )
    inst = pm.instruments[0]
    cc_events = inst.control_changes
    pairs = [(c.number, c.value) for c in cc_events]
    assert pairs.count((101, 0)) == 1
    assert pairs.count((100, 0)) == 1
    assert pairs.count((6, 2)) == 1
    assert pairs.count((38, 50)) == 1
    assert pairs.count((101, 127)) == 1
    assert pairs.count((100, 127)) == 1
    rpn_null_time = max(
        c.time for c in cc_events if (c.number, c.value) in {(101, 127), (100, 127)}
    )
    first_bend = min(b.time for b in inst.pitch_bends)
    assert rpn_null_time <= first_bend


def test_rpn_order_coarse_only():
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    pm.instruments.append(inst)
    write_bend_range_rpn(inst, 2.5, coarse_only=True)
    curve = ControlCurve([0.0, 1.0], [0.0, 0.1])
    curve.to_pitch_bend(inst)
    nums = [c.number for c in inst.control_changes]
    assert 38 not in nums  # no LSB when coarse_only
    assert any(c.number == 101 and c.value == 127 for c in inst.control_changes)
    assert any(c.number == 100 and c.value == 127 for c in inst.control_changes)
    rpn_null_time = max(
        c.time
        for c in inst.control_changes
        if c.number in {101, 100} and c.value == 127
    )
    first_bend = min(b.time for b in inst.pitch_bends)
    assert rpn_null_time <= first_bend


def test_rpn_lsb_128th():
    inst = pretty_midi.Instrument(program=0)
    write_bend_range_rpn(inst, 2.5, lsb_mode="128th")
    pairs = [(c.number, c.value) for c in inst.control_changes]
    assert (6, 2) in pairs
    assert (38, 64) in pairs


@pytest.mark.parametrize(
    "bend_range, mode, msb, lsb",
    [
        (2.0, "128th", 2, 0),
        (2.99, "128th", 2, 127),
        (3.0, "128th", 3, 0),
        (2.0, "cents", 2, 0),
        (2.99, "cents", 2, 99),
        (3.0, "cents", 3, 0),
    ],
)
def test_rpn_lsb_rounding(bend_range, mode, msb, lsb):
    inst = pretty_midi.Instrument(program=0)
    write_bend_range_rpn(inst, bend_range, lsb_mode=mode, send_rpn_null=False)
    pairs = {(c.number, c.value) for c in inst.control_changes}
    assert (6, msb) in pairs
    assert (38, lsb) in pairs


def test_rpn_null_idempotent():
    inst = pretty_midi.Instrument(program=0)
    write_bend_range_rpn(inst, 2.5, send_rpn_null=True)
    n = len(inst.control_changes)
    write_bend_range_rpn(inst, 2.5, send_rpn_null=True)
    assert len(inst.control_changes) == n


def test_epsilon_dedupe_cc_and_bend():
    pm = pretty_midi.PrettyMIDI()
    curve_cc = ControlCurve([0, 0.001, 0.002], [64.0, 64.0 + 1e-7, 64.0 - 1e-7])
    apply_controls(pm, {0: {"cc11": curve_cc}})
    inst = pm.instruments[0]
    assert len(_collect_cc(inst, 11)) == 1

    pm2 = pretty_midi.PrettyMIDI()
    curve_b = ControlCurve([0, 0.001, 0.002], [0.0, 0.0 + 1e-7, -1e-7])
    apply_controls(pm2, {0: {"bend": curve_b}})
    inst2 = pm2.instruments[0]
    assert len(inst2.pitch_bends) == 1


def test_beats_domain_piecewise_tempo():
    pm = pretty_midi.PrettyMIDI()
    events = [(0, 120), (1, 60), (2, 60)]
    curve = ControlCurve([0, 1, 2], [0, 64, 127], domain="beats")
    apply_controls(pm, {0: {"cc11": curve}}, tempo_map=events)
    inst = pm.instruments[0]
    times = [c.time for c in _collect_cc(inst, 11)]
    assert times[1] - times[0] == pytest.approx(0.5)
    assert times[-1] - times[1] == pytest.approx(1.0)
    assert times[-1] == pytest.approx(1.5)


def test_beats_domain_constant_bpm():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0, 1, 2], [0, 64, 127], domain="beats")
    curve.to_midi_cc(inst, 11, tempo_map=120.0, sample_rate_hz=2)
    times = [c.time for c in inst.control_changes]
    assert times[0] == pytest.approx(0.0)
    assert times[1] == pytest.approx(0.5)
    assert times[-1] == pytest.approx(1.0)


def test_sparse_tempo_resample_thin_sorted():
    inst = pretty_midi.Instrument(program=0)
    events = [(0, 120), (1.5, 90)]
    curve = ControlCurve([0, 1.5, 3], [0, 64, 127], domain="beats")
    curve.to_midi_cc(
        inst,
        11,
        tempo_map=events,
        sample_rate_hz=10,
        max_events=4,
    )
    times = [c.time for c in inst.control_changes]
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(1.75)
    assert all(t0 <= t1 for t0, t1 in zip(times, times[1:]))


def test_max_events_keeps_endpoints_after_quantization():
    inst = pretty_midi.Instrument(program=0)
    t = np.linspace(0, 1, 50)
    v = np.linspace(0, 127, 50)
    curve = ControlCurve(t, v)
    curve.to_midi_cc(inst, 11, max_events=8)
    times = [c.time for c in inst.control_changes]
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(1.0)
    assert all(t0 <= t1 for t0, t1 in zip(times, times[1:]))
    assert len(times) <= 8


def test_resample_and_thin_preserve_endpoints_cc():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0, 1], [0, 127])
    curve.to_midi_cc(inst, 11, sample_rate_hz=50, max_events=8)
    times = [c.time for c in inst.control_changes]
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(1.0)
    assert all(t0 <= t1 for t0, t1 in zip(times, times[1:]))
    assert len(times) <= 8


def test_resample_and_thin_preserve_endpoints_bend():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0, 1], [0.0, 2.0])
    curve.to_pitch_bend(inst, bend_range_semitones=2.0, sample_rate_hz=50, max_events=8)
    times = [b.time for b in inst.pitch_bends]
    vals = [b.pitch for b in inst.pitch_bends]
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(1.0)
    assert all(t0 <= t1 for t0, t1 in zip(times, times[1:]))
    assert len(times) <= 8
    assert min(vals) >= -8192
    assert max(vals) <= 8191


def test_single_knot_constant_curve():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0.0], [64.0])
    curve.to_midi_cc(inst, 11)
    evs = _collect_cc(inst, 11)
    assert len(evs) == 1
    assert evs[0].value == 64


def test_offset_negative_clamped():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0.0, 1.0], [0.0, 127.0], offset_sec=-0.25)
    curve.to_midi_cc(inst, 11)
    times = [c.time for c in _collect_cc(inst, 11)]
    assert times[0] == pytest.approx(0.0)
    assert all(t >= 0.0 for t in times)


def test_apply_controls_limits():
    pm = pretty_midi.PrettyMIDI()
    t = np.linspace(0, 1, 50)
    v = np.linspace(0, 127, 50)
    bend_v = np.linspace(0.0, 2.0, 50)
    curves = {0: {"cc11": ControlCurve(t, v), "bend": ControlCurve(t, bend_v)}}
    apply_controls(
        pm,
        curves,
        cc_max_events=6,
        bend_max_events=6,
    )
    inst = pm.instruments[0]
    cc_times = [c.time for c in _collect_cc(inst, 11)]
    bend_times = [b.time for b in inst.pitch_bends]
    assert cc_times[0] == pytest.approx(0.0)
    assert cc_times[-1] == pytest.approx(1.0)
    assert bend_times[0] == pytest.approx(0.0)
    assert bend_times[-1] == pytest.approx(1.0)
    assert all(t0 <= t1 for t0, t1 in zip(cc_times, cc_times[1:]))
    assert all(t0 <= t1 for t0, t1 in zip(bend_times, bend_times[1:]))
    assert len(cc_times) <= 6
    assert len(bend_times) <= 6


def test_beats_offset_combo():
    inst = pretty_midi.Instrument(program=0)
    events = [(0, 120), (1, 60), (2, 60)]
    curve = ControlCurve([0, 1, 2], [0, 64, 127], domain="beats", offset_sec=0.5)
    curve.to_midi_cc(inst, 11, tempo_map=events, sample_rate_hz=2)
    times = [c.time for c in inst.control_changes]
    assert times[0] == pytest.approx(0.5)
    assert times[-1] == pytest.approx(2.0)


def test_normalized_units_clip():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0, 1], [-1.5, 2.0])
    curve.to_pitch_bend(inst, units="normalized")
    vals = [b.pitch for b in inst.pitch_bends]
    assert vals[0] == -8192
    assert vals[1] == 8191


def test_lsb_mode_invalid():
    inst = pretty_midi.Instrument(program=0)
    with pytest.raises(ValueError):
        write_bend_range_rpn(inst, 2.0, lsb_mode="bogus")


@pytest.mark.parametrize(
    "events",
    [
        [(1, 120), (0, 120)],  # decreasing beats
        [(0, 120), (1, 0)],  # non-positive bpm
    ],
)
def test_tempo_events_validation(events):
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0, 1], [0, 64], domain="beats")
    with pytest.raises(ValueError):
        curve.to_midi_cc(inst, 11, tempo_map=events)


# -------------------------------------------------------------------------
# Adapted tests from main branch to the new ControlCurve API
# -------------------------------------------------------------------------

def test_monotone_interpolation_endpoint():
    curve = ControlCurve([0.0, 1.0], [0.0, 127.0])
    t = [0.0, 1.0]
    vals = curve.sample(t)
    assert abs(vals[0] - 0.0) < 1e-6
    assert abs(vals[1] - 127.0) < 1e-6


def test_cc11_value_range():
    curve = ControlCurve([0.0, 1.0], [-10.0, 200.0])
    t = [i / 24 for i in range(25)]
    vals = curve.sample(t)
    assert all(v >= 0 for v in vals)
    assert all(v <= 127 for v in vals)


def test_bend_encoding_roundtrip():
    times = [i / 20 for i in range(21)]
    values = [2.0 * math.sin(2 * math.pi * t) for t in times]
    curve = ControlCurve(times, values)  # default units: semitones
    inst = pretty_midi.Instrument(program=0)
    curve.to_pitch_bend(inst, bend_range_semitones=2.0)
    peak = max(abs(b.pitch) for b in inst.pitch_bends)
    assert 0.9 * 8191 <= peak <= 1.1 * 8191


def test_dedupe():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0.0, 1.0], [64.0, 64.0])
    curve.to_midi_cc(inst, 11)
    assert len(_collect_cc(inst, 11)) == 1


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
    inst_a = pretty_midi.Instrument(program=0)
    inst_b = pretty_midi.Instrument(program=0)
    ControlCurve(times, semis, units="semitones").to_pitch_bend(inst_a, bend_range_semitones=2.0)
    ControlCurve(times, norms, units="normalized").to_pitch_bend(inst_b, bend_range_semitones=2.0)
    peak_a = max(abs(b.pitch) for b in inst_a.pitch_bends)
    peak_b = max(abs(b.pitch) for b in inst_b.pitch_bends)
    assert 0.9 * 8191 <= peak_a <= 1.1 * 8191
    assert 0.9 * 8191 <= peak_b <= 1.1 * 8191


def test_apply_controls_beats_domain():
    def tempo_map(b: float) -> float:
        return 120.0 if b < 1.0 else 60.0

    pm = pretty_midi.PrettyMIDI()
    curve = ControlCurve([0.0, 2.0], [0.0, 127.0], domain="beats")
    apply_controls(pm, {0: {"cc11": curve}}, tempo_map=tempo_map)
    inst = pm.instruments[0]
    times = [e.time for e in inst.control_changes]
    values = [e.value for e in inst.control_changes]
    assert abs(times[1] - 0.5) < 1e-6
    assert abs(times[-1] - 1.5) < 1e-6
    assert values == sorted(values)


def test_rpn_emitted_once():
    pm = pretty_midi.PrettyMIDI()
    curve = ControlCurve([0.0, 1.0], [0.0, 1.0])
    apply_controls(pm, {0: {"bend": curve}})
    apply_controls(pm, {0: {"bend": curve}})
    inst = pm.instruments[0]
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
    curve = ControlCurve(times, values, resolution_hz=1.0)
    inst = pretty_midi.Instrument(program=0)
    curve.to_midi_cc(inst, 11)
    events = inst.control_changes
    assert len(events) == 2
    recon_curve = ControlCurve([(e.time) for e in events], [(float(e.value)) for e in events])
    recon = recon_curve.sample(times)
    err = max(abs(a - b) for a, b in zip(recon, values))
    assert err <= 0.5


def test_max_events_cap():
    times = [i / 200 for i in range(201)]
    values = [t * 127.0 for t in times]
    curve = ControlCurve(times, values)
    inst = pretty_midi.Instrument(program=0)
    curve.to_midi_cc(inst, 11, max_events=10)
    events = inst.control_changes
    assert len(events) <= 10
    recon_curve = ControlCurve([e.time for e in events], [float(e.value) for e in events])
    recon = recon_curve.sample(times)
    err = max(abs(a - b) for a, b in zip(recon, values))
    assert err < 1.0


def test_instrument_routing():
    pm = pretty_midi.PrettyMIDI()
    cc_curve = ControlCurve([0.0, 1.0], [64.0, 64.0])
    bend_curve = ControlCurve([0.0, 1.0], [0.0, 1.0])
    apply_controls(pm, {0: {"cc11": cc_curve}, 1: {"bend": bend_curve}})
    assert len(pm.instruments) == 2
    a, b = pm.instruments
    # one inst should have CCs, the other bends
    has_cc_only = bool(_collect_cc(a, 11)) and not a.pitch_bends or bool(_collect_cc(b, 11)) and not b.pitch_bends
    has_bend_only = bool(a.pitch_bends) and not _collect_cc(a, 11) or bool(b.pitch_bends) and not _collect_cc(b, 11)
    assert has_cc_only and has_bend_only


def test_fractional_bend_range():
    times = [i / 20 for i in range(21)]
    values = [2.5 * math.sin(2 * math.pi * t) for t in times]
    curve = ControlCurve(times, values)
    pm = pretty_midi.PrettyMIDI()
    apply_controls(
        pm,
        {0: {"bend": curve}},
        bend_range_semitones=2.5,
    )
    inst = pm.instruments[0]
    peak = max(abs(pb.pitch) for pb in inst.pitch_bends)
    assert 0.9 * 8191 <= peak <= 1.1 * 8191
    msb = [cc.value for cc in inst.control_changes if cc.number == 6][0]
    lsb = [cc.value for cc in inst.control_changes if cc.number == 38][0]
    assert msb == 2
    assert 48 <= lsb <= 52


def test_min_clamp_negative():
    # static method expected on ControlCurve in new API
    vals = ControlCurve.convert_to_14bit([-10.0], 2.0)
    assert vals[0] == -8192
