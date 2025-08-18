from __future__ import annotations

import importlib
import math
import warnings

import pytest

# --- Optional deps ---------------------------------------------------------
try:  # optional numpy
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore
requires_numpy = pytest.mark.skipif(np is None, reason="numpy required")

# pretty_midi is optional for CI; provide a tiny stub if it's missing so most
# tests can still run. (Tests that genuinely need the real lib will skip.)
try:  # pragma: no cover
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover - fallback stub
    from ._stubs import pretty_midi  # type: ignore

apply_controls_mod = importlib.import_module("utilities.apply_controls")
apply_controls = apply_controls_mod.apply_controls
write_bend_range_rpn = apply_controls_mod.write_bend_range_rpn
ControlCurve = importlib.import_module("utilities.controls_spline").ControlCurve
catmull_rom_monotone = importlib.import_module(
    "utilities.controls_spline"
).catmull_rom_monotone

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _collect_cc(inst: pretty_midi.Instrument, number: int):
    return [c for c in inst.control_changes if c.number == number]


def test_resolution_hz_deprecation():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ControlCurve([0.0], [0.0], resolution_hz=10.0)
        ControlCurve([0.0], [0.0], resolution_hz=20.0)
        ControlCurve([0.0], [0.0], sample_rate_hz=30.0, resolution_hz=40.0)
        msgs = [
            str(wr.message) for wr in w if issubclass(wr.category, DeprecationWarning)
        ]
    assert any("use sample_rate_hz" in m for m in msgs)
    assert any("ignored" in m for m in msgs)
    assert len([m for m in msgs if "use sample_rate_hz" in m]) == 1


# -------------------------------------------------------------------------
# Tests from the CC/RPN branch (updated API: ControlCurve(times, values, ...))
# -------------------------------------------------------------------------


def test_fit_infer(tmp_path):
    # Import inside the test to avoid skipping the whole file when optional
    # stack (pandas + ml.controls_spline) is unavailable.
    pd = pytest.importorskip("pandas")
    try:
        from ml.controls_spline import fit_controls  # type: ignore
        from ml.controls_spline import infer_controls
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


def test_rpn_lsb_rounding():
    inst = pretty_midi.Instrument(program=0)
    write_bend_range_rpn(inst, 2.5)
    pairs = {(c.number, c.value) for c in inst.control_changes}
    assert (101, 0) in pairs
    assert (100, 0) in pairs
    assert (6, 2) in pairs
    assert (38, 64) in pairs


def test_rpn_once_before_bend():
    pm = pretty_midi.PrettyMIDI()
    curve = ControlCurve([0.0, 1.0], [0.0, 0.5])
    apply_controls(pm, {0: {"bend": curve}}, write_rpn=True)
    inst = next(i for i in pm.instruments if i.name == "channel0")
    cc_pairs = [(c.number, c.value, c.time) for c in inst.control_changes]
    nums = [(n, v) for n, v, _ in cc_pairs]
    assert nums.count((101, 0)) == 1
    assert nums.count((100, 0)) == 1
    assert nums.count((6, 2)) == 1
    assert nums.count((38, 0)) == 1
    first_bend = min(b.time for b in inst.pitch_bends)
    last_rpn = max(t for n, v, t in cc_pairs if n in {101, 100, 6, 38})
    assert last_rpn <= first_bend


def test_event_ordering_same_time():
    pm = pretty_midi.PrettyMIDI()
    cc_curve = ControlCurve([0.0, 1.0], [0.0, 127.0])
    bend_curve = ControlCurve([0.0, 1.0], [0.0, 1.0])
    apply_controls(pm, {0: {"cc11": cc_curve, "bend": bend_curve}}, write_rpn=True)
    inst = pm.instruments[0]
    rpn_times = [c.time for c in inst.control_changes if c.number in {101, 100, 6, 38}]
    cc_times = [
        c.time for c in inst.control_changes if c.number not in {101, 100, 6, 38}
    ]
    first_bend = min(b.time for b in inst.pitch_bends)
    assert max(rpn_times) <= min(cc_times)
    assert min(cc_times) <= first_bend


def test_dedup_with_eps():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0.0, 0.00005, 1.0], [0.0, 0.0001, 1.0])
    curve.to_midi_cc(inst, 11, dedup_time_epsilon=1e-3, dedup_value_epsilon=1e-3)
    cc = _collect_cc(inst, 11)
    assert len(cc) == 2
    assert cc[0].time == pytest.approx(0.0)
    assert cc[-1].time == pytest.approx(1.0)


def test_ensure_zero_at_edges():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0.0, 0.5, 1.0], [0.0003, 0.1, 0.0003])
    curve.to_pitch_bend(inst)
    vals = [b.pitch for b in inst.pitch_bends]
    assert vals[0] == 0
    assert vals[-1] == 0
    inst2 = pretty_midi.Instrument(program=0)
    curve2 = ControlCurve(
        [0.0, 0.5, 1.0], [0.0003, 0.1, 0.0003], ensure_zero_at_edges=False
    )
    curve2.to_pitch_bend(inst2)
    vals2 = [b.pitch for b in inst2.pitch_bends]
    assert vals2[0] != 0 or vals2[-1] != 0


def test_beats_domain_step_tempo_monotone():
    inst = pretty_midi.Instrument(program=0)
    events = [(0, 120), (1, 60)]
    curve = ControlCurve([0, 1, 2], [0, 64, 127], domain="beats")
    curve.to_midi_cc(inst, 11, tempo_map=events, sample_rate_hz=5)
    times = [c.time for c in inst.control_changes]
    assert all(t0 <= t1 for t0, t1 in zip(times, times[1:]))


def test_value_eps_zero_vs_nonzero():
    curve = ControlCurve([0, 0.001, 0.002], [64.0, 64.6, 64.6])
    inst1 = pretty_midi.Instrument(program=0)
    curve.to_midi_cc(inst1, 11, value_eps=0)
    assert len(_collect_cc(inst1, 11)) == 3
    inst2 = pretty_midi.Instrument(program=0)
    curve.to_midi_cc(inst2, 11, value_eps=0.5)
    assert len(_collect_cc(inst2, 11)) == 2


def test_epsilon_dedupe_cc_and_bend():
    pm = pretty_midi.PrettyMIDI()
    curve_cc = ControlCurve([0, 0.001, 0.002], [64.0, 64.0 + 1e-7, 64.0 - 1e-7])
    apply_controls(pm, {0: {"cc11": curve_cc}})
    inst = pm.instruments[0]
    cc = _collect_cc(inst, 11)
    assert len(cc) == 2
    assert cc[0].time == pytest.approx(0.0)
    assert cc[-1].time == pytest.approx(0.002)

    pm2 = pretty_midi.PrettyMIDI()
    curve_b = ControlCurve([0, 0.001, 0.002], [0.0, 0.0 + 1e-7, -1e-7])
    apply_controls(pm2, {0: {"bend": curve_b}})
    inst2 = pm2.instruments[0]
    assert len(inst2.pitch_bends) == 2
    assert inst2.pitch_bends[0].time == pytest.approx(0.0)
    assert inst2.pitch_bends[-1].time == pytest.approx(0.002)


def test_beats_domain_piecewise_tempo():
    pm = pretty_midi.PrettyMIDI()
    events = [(0, 120), (1, 60), (2, 60)]
    curve = ControlCurve([0, 1, 2], [0, 64, 127], domain="beats")
    apply_controls(
        pm, {0: {"cc11": curve}}, tempo_map=events, sample_rate_hz={"cc11": 0}
    )
    inst = pm.instruments[0]
    times = [c.time for c in _collect_cc(inst, 11)]
    assert times[1] - times[0] == pytest.approx(0.5)
    assert times[-1] - times[1] == pytest.approx(1.0)
    assert times[-1] == pytest.approx(1.5)


def test_beats_domain_extreme_tempo_monotone():
    inst = pretty_midi.Instrument(program=0)
    events = [(0, 30), (1, 240)]
    curve = ControlCurve([0, 1, 2], [0, 64, 127], domain="beats")
    curve.to_midi_cc(inst, 11, tempo_map=events, sample_rate_hz=5)
    times = [c.time for c in inst.control_changes]
    assert all(t0 <= t1 for t0, t1 in zip(times, times[1:]))


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


@requires_numpy
def test_max_events_keeps_endpoints_after_quantization():
    inst = pretty_midi.Instrument(program=0)
    t = np.linspace(0, 1, 500)
    v = np.linspace(0, 127, 500)
    curve = ControlCurve(t, v)
    curve.to_midi_cc(inst, 11, max_events=64)
    events = _collect_cc(inst, 11)
    assert len(events) <= 64
    assert events[0].time == pytest.approx(0.0)
    assert events[-1].time == pytest.approx(1.0)

    def _interp(evts, x):
        for a, b in zip(evts, evts[1:]):
            if a.time <= x <= b.time:
                t0, t1 = a.time, b.time
                v0, v1 = a.value, b.value
                return v0 + (v1 - v0) * (x - t0) / (t1 - t0)
        return evts[-1].value

    mid = _interp(events, 0.5)
    assert abs(mid - 63.5) < 1.0


@requires_numpy
def test_max_events_caps_pitch_bend():
    inst = pretty_midi.Instrument(program=0)
    t = np.linspace(0, 1, 400)
    v = np.linspace(0, 1, 400)
    curve = ControlCurve(t, v)
    curve.to_pitch_bend(inst, max_events=64)
    bends = inst.pitch_bends
    assert len(bends) <= 64

    def _interp(evts, x):
        for a, b in zip(evts, evts[1:]):
            if a.time <= x <= b.time:
                t0, t1 = a.time, b.time
                v0, v1 = a.pitch, b.pitch
                return v0 + (v1 - v0) * (x - t0) / (t1 - t0)
        return evts[-1].pitch

    mid_pitch = _interp(bends, 0.5)
    mid_semi = mid_pitch * 2.0 / 8192.0
    assert abs(mid_semi - 0.5) < 0.05


def test_dedupe_keeps_endpoints():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0.0, 1.0, 2.0], [10.0, 10.0, 10.0])
    curve.to_midi_cc(inst, 11)
    events = _collect_cc(inst, 11)
    assert events[0].time == pytest.approx(0.0)
    assert events[-1].time == pytest.approx(2.0)
    assert len(events) == 2


def test_min_delta_thins_cc():
    curve = ControlCurve([0.0, 0.5, 1.0], [0.0, 64.0, 127.0])
    inst1 = pretty_midi.Instrument(program=0)
    curve.to_midi_cc(inst1, 11)
    inst2 = pretty_midi.Instrument(program=0)
    curve.to_midi_cc(inst2, 11, min_delta=70)
    assert len(_collect_cc(inst2, 11)) < len(_collect_cc(inst1, 11))
    kept = _collect_cc(inst2, 11)
    interp = (kept[-1].value - kept[0].value) * 0.5 / (
        kept[-1].time - kept[0].time
    ) + kept[0].value
    assert abs(interp - 64.0) < 1.0


def test_time_offset_shifts_events():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0.0, 1.0], [0.0, 127.0])
    curve.to_midi_cc(inst, 11, time_offset=1.5)
    times = [c.time for c in _collect_cc(inst, 11)]
    assert times[0] == pytest.approx(1.5)
    assert times[-1] == pytest.approx(2.5)


def test_cc_validation_bounds():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0.0, 1.0], [0.0, 127.0])
    with pytest.raises(ValueError):
        curve.to_midi_cc(inst, 200)


def test_resample_and_thin_preserve_endpoints_cc():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0, 1], [0, 127])
    curve.to_midi_cc(inst, 11, sample_rate_hz=50, max_events=8)
    times = [c.time for c in inst.control_changes]
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(1.0)


def test_convert_to_14bit_endpoints():
    vals = ControlCurve.convert_to_14bit([-2.0, 0.0, 2.0], 2.0, units="semitones")
    assert vals == [-8191, 0, 8191]


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


def test_bend_returns_to_zero():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0, 1], [0.0, 1.0])
    curve.to_pitch_bend(inst, bend_range_semitones=2.0, sample_rate_hz=10)
    assert inst.pitch_bends[-1].pitch == 0


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
    if np is None:
        pytest.skip("numpy required")
    pm = pretty_midi.PrettyMIDI()
    t = np.linspace(0, 1, 50)
    v = np.linspace(0, 127, 50)
    bend_v = np.linspace(0.0, 2.0, 50)
    curves = {0: {"cc11": ControlCurve(t, v), "bend": ControlCurve(t, bend_v)}}
    apply_controls(pm, curves, max_events={"cc": 6, "bend": 6})
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
    curve = ControlCurve([0, 1], [-1.5, 2.0], ensure_zero_at_edges=False)
    curve.to_pitch_bend(inst, units="normalized")
    vals = [b.pitch for b in inst.pitch_bends]
    assert vals[0] == -8192
    assert vals[1] == 8191


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
    t_knots = [0.0, 1.0]
    v_knots = [0.0, 127.0]
    t = [0.0, 1.0]
    vals = catmull_rom_monotone(t_knots, v_knots, t)
    assert abs(vals[0] - 0.0) < 1e-6
    assert abs(vals[1] - 127.0) < 1e-6


def test_cc11_value_range():
    # Verify clamping happens in MIDI domain via to_midi_cc
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0.0, 1.0], [-10.0, 200.0])
    curve.to_midi_cc(inst, 11)
    vals = [c.value for c in _collect_cc(inst, 11)]
    assert min(vals) >= 0
    assert max(vals) <= 127


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


def test_dedupe_tolerance_cc():
    times = [0.0, 1.0]
    values = [64.0, 64.0005]
    curve = ControlCurve(times, values)
    inst = pretty_midi.Instrument(program=0)
    # Use a relatively loose epsilon to dedupe these nearly identical values
    curve.to_midi_cc(inst, 11, value_eps=1e-3)
    events = _collect_cc(inst, 11)
    assert len(events) == 1


def test_catmull_rom_bisect_linear():
    times = [0.0, 1.0, 2.0]
    values = [0.0, 1.0, 2.0]
    query = [0.5, 1.5]
    out = catmull_rom_monotone(times, values, query)
    assert out == pytest.approx([0.5, 1.5])


def test_from_dense_simplifies():
    times = [i / 99 for i in range(100)]
    values = [t * 127.0 for t in times]
    curve = ControlCurve.from_dense(times, values, tol=0.5, max_knots=256)
    # The simplified curve should have far fewer sample points ("knots")
    n_knots = (
        len(curve.times) if hasattr(curve, "times") else len(curve.knots)
    )  # compat
    assert n_knots < 32
    # Reconstruct via the same monotone interpolant used by ControlCurve
    recon = catmull_rom_monotone(
        list(curve.times) if hasattr(curve, "times") else [t for t, _ in curve.knots],
        list(curve.values) if hasattr(curve, "values") else [v for _, v in curve.knots],
        times,
    )
    err = max(abs(a - b) for a, b in zip(recon, values))
    assert err < 0.5


def test_bend_units_and_roundtrip():
    times = [i / 20 for i in range(21)]
    semis = [2.0 * math.sin(2 * math.pi * t) for t in times]
    norms = [1.0 * math.sin(2 * math.pi * t) for t in times]
    inst_a = pretty_midi.Instrument(program=0)
    inst_b = pretty_midi.Instrument(program=0)
    ControlCurve(times, semis).to_pitch_bend(inst_a, bend_range_semitones=2.0)
    ControlCurve(times, norms).to_pitch_bend(
        inst_b, bend_range_semitones=2.0, units="normalized"
    )
    peak_a = max(abs(b.pitch) for b in inst_a.pitch_bends)
    peak_b = max(abs(b.pitch) for b in inst_b.pitch_bends)
    assert 0.9 * 8191 <= peak_a <= 1.1 * 8191
    assert 0.9 * 8191 <= peak_b <= 1.1 * 8191


@pytest.mark.parametrize("bpm", [0.0, -1.0, float("nan")])
def test_beats_domain_invalid_bpm_raises(bpm: float):
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0.0, 1.0], [0.0, 1.0], domain="beats")
    # Use event-list form so validation triggers in tempo_map_from_events
    with pytest.raises(ValueError):
        curve.to_midi_cc(inst, 11, tempo_map=[(0.0, bpm)])


def test_rpn_emitted_once():
    pm = pretty_midi.PrettyMIDI()
    curve = ControlCurve([0.0, 1.0], [0.0, 1.0])
    apply_controls(pm, {0: {"bend": curve}}, write_rpn=True)
    apply_controls(pm, {0: {"bend": curve}}, write_rpn=True)
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
    # Use a larger epsilon to encourage dedupe to 2 events
    curve.to_midi_cc(inst, 11, value_eps=0.5)
    events = inst.control_changes
    assert len(events) == 2
    recon = catmull_rom_monotone(
        [e.time for e in events], [float(e.value) for e in events], times
    )
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
    recon = catmull_rom_monotone(
        [e.time for e in events], [float(e.value) for e in events], times
    )
    err = max(abs(a - b) for a, b in zip(recon, values))
    assert err < 1.0


@pytest.mark.parametrize("mode", ["rdp", "uniform"])
def test_simplify_modes(mode: str):
    times = [i / 200 for i in range(201)]
    values = [t * 127.0 for t in times]
    curve = ControlCurve(times, values)
    inst = pretty_midi.Instrument(program=0)
    curve.to_midi_cc(inst, 11, max_events=10, simplify_mode=mode)
    events = _collect_cc(inst, 11)
    assert len(events) <= 10
    assert events[0].time == pytest.approx(0.0)
    assert events[-1].time == pytest.approx(1.0)


def test_convert_to_14bit_units():
    vals_norm = ControlCurve.convert_to_14bit([-1.0, 1.0], 2.0, units="normalized")
    if np is not None:
        vals_norm = vals_norm.tolist()
    assert vals_norm[0] == -8191
    assert vals_norm[1] == 8191
    vals_semi = ControlCurve.convert_to_14bit([-2.0, 2.0], 2.0, units="semitones")
    if np is not None:
        vals_semi = vals_semi.tolist()
    assert vals_semi[0] == -8191
    assert vals_semi[1] == 8191


def test_convert_to_14bit_numpy_return():
    if np is None:
        pytest.skip("numpy not installed")
    vals = ControlCurve.convert_to_14bit([-10.0, 10.0], 2.0)
    assert isinstance(vals, np.ndarray)


def test_instrument_routing():
    pm = pretty_midi.PrettyMIDI()
    cc_curve = ControlCurve([0.0, 1.0], [64.0, 64.0])
    bend_curve = ControlCurve([0.0, 1.0], [0.0, 1.0])
    apply_controls(pm, {0: {"cc11": cc_curve}, 1: {"bend": bend_curve}})
    assert len(pm.instruments) == 2
    a, b = pm.instruments
    # one inst should have CCs, the other bends
    has_cc_only = (bool(_collect_cc(a, 11)) and not a.pitch_bends) or (
        bool(_collect_cc(b, 11)) and not b.pitch_bends
    )
    has_bend_only = (bool(a.pitch_bends) and not _collect_cc(a, 11)) or (
        bool(b.pitch_bends) and not _collect_cc(b, 11)
    )
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
        write_rpn=True,
    )
    inst = pm.instruments[0]
    peak = max(abs(pb.pitch) for pb in inst.pitch_bends)
    assert 0.9 * 8191 <= peak <= 1.1 * 8191
    msb = [cc.value for cc in inst.control_changes if cc.number == 6][0]
    lsb = [cc.value for cc in inst.control_changes if cc.number == 38][0]
    assert msb == 2
    assert lsb == 64


def test_pitch_bend_clipping():
    inst = pretty_midi.Instrument(program=0)
    curve = ControlCurve([0.0, 1.0], [-3.0, 3.0])
    curve.to_pitch_bend(inst, bend_range_semitones=2.0, sample_rate_hz=1.0)
    pitches = [b.pitch for b in inst.pitch_bends]
    assert min(pitches) == -8192
    assert max(pitches) == 8191


def test_min_clamp_negative():
    # static method expected on ControlCurve in new API
    vals = ControlCurve.convert_to_14bit([-10.0], 2.0)
    assert vals[0] == -8191


def test_to_midi_cc_without_numpy(monkeypatch):
    if np is not None:
        monkeypatch.setattr(module_cs, "np", None)
        monkeypatch.setattr(module_cs, "as_array", lambda xs: [float(x) for x in xs])
        monkeypatch.setattr(
            module_cs,
            "clip",
            lambda xs, lo, hi: [max(lo, min(hi, float(x))) for x in xs],
        )
        monkeypatch.setattr(
            module_cs, "round_int", lambda xs: [int(round(float(x))) for x in xs]
        )
    curve = ControlCurve([0.0, 1.0], [0.0, 127.0])
    inst = pretty_midi.Instrument(program=0)
    curve.to_midi_cc(inst, 11)
    assert len(_collect_cc(inst, 11)) == 2


def test_per_target_sample_rate():
    pm = pretty_midi.PrettyMIDI()
    curves = {
        0: {
            "bend": ControlCurve([0, 1], [0.0, 1.0]),
            "cc11": ControlCurve([0, 1], [0, 127]),
        }
    }
    apply_controls(
        pm,
        curves,
        sample_rate_hz={"bend": 40, "cc11": 5},
        max_events={"bend": 20, "cc11": 4},
    )
    inst = pm.instruments[0]
    assert inst.pitch_bends[0].pitch == 0
    last_time = max(pb.time for pb in inst.pitch_bends)
    assert any(pb.pitch == 0 for pb in inst.pitch_bends if pb.time == last_time)
    assert len(inst.pitch_bends) <= 20
    assert len(_collect_cc(inst, 11)) <= 4


def test_cc_and_bend_same_instrument(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    curves = {
        0: {
            "cc11": ControlCurve([0, 1], [0, 127]),
            "bend": ControlCurve([0, 1], [0, 1]),
        }
    }
    apply_controls(pm, curves, write_rpn=True)
    out = tmp_path / "out.mid"
    pm.write(str(out))
    assert out.exists()
