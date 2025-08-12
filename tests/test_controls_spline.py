import pytest

np = pytest.importorskip("numpy")
pretty_midi = pytest.importorskip("pretty_midi")
pd = pytest.importorskip("pandas")

from ml.controls_spline import fit_controls, infer_controls  # noqa: E402
from utilities.apply_controls import (  # noqa: E402
    apply_controls,
    write_bend_range_rpn,
)
from utilities.controls_spline import ControlCurve  # noqa: E402


def test_fit_infer(tmp_path):
    df = pd.DataFrame({"bend": [0, 1, 2], "cc11": [10, 20, 30]})
    notes = tmp_path / "notes.parquet"
    df.to_parquet(notes)
    model = tmp_path / "model.json"
    fit_controls(notes, targets=["bend", "cc11"], out_path=model)
    out = tmp_path / "pred.parquet"
    infer_controls(model, out_path=out)
    assert out.exists()


def _collect_cc(inst, number):
    return [c for c in inst.control_changes if c.number == number]


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
    assert 38 not in nums
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
