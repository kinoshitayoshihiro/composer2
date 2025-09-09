import importlib

import pytest

try:  # pragma: no cover
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from ._stubs import pretty_midi  # type: ignore

apply_controls = importlib.import_module("utilities.apply_controls")
write_bend_range_rpn = apply_controls.write_bend_range_rpn
ControlCurve = importlib.import_module("utilities.controls_spline").ControlCurve
from utilities import pb_math


def test_rpn_once_and_lsb_rounding():
    curve = ControlCurve([0, 1], [0.0, 1.0])
    pm = pretty_midi.PrettyMIDI()
    apply_controls.apply_controls(
        pm,
        {0: {"bend": curve}},
        write_rpn=True,
        bend_range_semitones=2.5,
    )
    inst = pm.instruments[0]
    rpn_cc = [(c.number, c.value) for c in inst.control_changes]
    assert rpn_cc.count((101, 0)) == 1
    lsb = next(c.value for c in inst.control_changes if c.number == 38)
    assert lsb == 64
    first_cc_time = min(c.time for c in inst.control_changes)
    assert first_cc_time <= inst.pitch_bends[0].time
    apply_controls.apply_controls(
        pm,
        {0: {"bend": curve}},
        write_rpn=True,
        bend_range_semitones=2.5,
    )
    rpn_cc2 = [(c.number, c.value) for c in inst.control_changes]
    assert rpn_cc2.count((101, 0)) == 1


@pytest.mark.parametrize("rng", [0.5, 1.0, 2.0, 12.0, 24.0])
def test_rpn_range_values(rng):
    inst = pretty_midi.Instrument(program=0)
    write_bend_range_rpn(inst, rng)
    msb = int(rng)
    lsb = int(round((rng - msb) * 128))
    pairs = [(c.number, c.value) for c in inst.control_changes]
    assert (6, msb) in pairs
    assert (38, lsb) in pairs


def test_rpn_existing_idempotent():
    inst = pretty_midi.Instrument(program=0)
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=0, time=1.0))
    write_bend_range_rpn(inst, 2.0, at_time=1.0)
    write_bend_range_rpn(inst, 2.0, at_time=1.0)
    pairs = [(c.number, c.value) for c in inst.control_changes]
    assert pairs.count((101, 0)) == 1
    first_bend = inst.pitch_bends[0].time
    rpn_time = min(c.time for c in inst.control_changes if (c.number, c.value) == (101, 0))
    assert rpn_time <= first_bend


def test_skip_preexisting_rpn():
    inst = pretty_midi.Instrument(program=0)
    t = 0.0
    inst.control_changes.extend(
        [
            pretty_midi.ControlChange(number=101, value=0, time=t),
            pretty_midi.ControlChange(number=100, value=0, time=t),
            pretty_midi.ControlChange(number=6, value=2, time=t),
            pretty_midi.ControlChange(number=38, value=0, time=t),
        ]
    )
    write_bend_range_rpn(inst, 2.0, at_time=t)
    pairs = [(c.number, c.value) for c in inst.control_changes]
    assert pairs.count((101, 0)) == 1


def test_rpn_time_clamped_non_negative():
    inst = pretty_midi.Instrument(program=0)
    write_bend_range_rpn(inst, 2.0, at_time=-1.0)
    t = min(c.time for c in inst.control_changes if c.number == 101)
    assert t == pytest.approx(0.0)


def test_rpn_sort_priority_same_timestamp():
    inst = pretty_midi.Instrument(program=0)
    t = 0.0
    inst.control_changes.append(pretty_midi.ControlChange(number=11, value=64, time=t))
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=0, time=t))
    write_bend_range_rpn(inst, 2.0, at_time=t)
    apply_controls._sort_events(inst)
    nums = [c.number for c in inst.control_changes]
    assert nums[:4] == [101, 100, 6, 38]
    assert nums[4] == 11
    assert inst.pitch_bends[0].time == t


def test_strictly_increasing_times():
    curve = ControlCurve([0, 1, 2], [0, 64, 127], domain="beats")
    pm = pretty_midi.PrettyMIDI()
    apply_controls.apply_controls(pm, {0: {"cc11": curve}}, tempo_map=[(0, 120), (1, 60)])
    inst = pm.instruments[0]
    times = [c.time for c in inst.control_changes]
    assert all(t1 >= t0 + 1e-9 for t0, t1 in zip(times, times[1:]))


def test_endpoint_preserved_with_large_eps():
    pm = pretty_midi.PrettyMIDI()
    cc_curve = ControlCurve([0, 1, 2], [0, 64, 127])
    apply_controls.apply_controls(pm, {0: {"cc11": cc_curve}}, value_eps=200)
    inst = pm.instruments[0]
    events = [c for c in inst.control_changes if c.number == 11]
    assert len(events) == 2
    assert events[0].time == pytest.approx(0.0)
    assert events[-1].time == pytest.approx(2.0)

    pm2 = pretty_midi.PrettyMIDI()
    bend_curve = ControlCurve([0, 1, 2], [0, 1, 0])
    apply_controls.apply_controls(pm2, {0: {"bend": bend_curve}}, value_eps=2)
    inst2 = pm2.instruments[0]
    bends = inst2.pitch_bends
    assert len(bends) == 2
    assert bends[0].pitch == 0
    assert bends[-1].pitch == 0


def test_rpn_write_once_across_calls():
    curve = ControlCurve([0, 1], [0.0, 1.0])
    pm = pretty_midi.PrettyMIDI()
    apply_controls.apply_controls(pm, {0: {"bend": curve}}, write_rpn=True)
    apply_controls.apply_controls(pm, {0: {"bend": curve}}, write_rpn=True)
    inst = pm.instruments[0]
    pairs = [(c.number, c.value) for c in inst.control_changes]
    assert pairs.count((101, 0)) == 1
    first_bend = min(pb.time for pb in inst.pitch_bends)
    last_rpn = max(c.time for c in inst.control_changes if c.number in {101, 100, 6, 38})
    assert last_rpn <= first_bend


def test_write_rpn_when_no_pitch_bends():
    pm = pretty_midi.PrettyMIDI()

    class EmptyCurve(ControlCurve):
        def to_pitch_bend(self, inst, **kwargs):  # type: ignore[override]
            return None

    curve = EmptyCurve([], [])
    apply_controls.apply_controls(pm, {0: {"bend": curve}}, write_rpn=True)
    inst = pm.instruments[0]
    assert len(inst.pitch_bends) == 0
    rpn_time = min(c.time for c in inst.control_changes if c.number == 101)
    assert rpn_time == pytest.approx(0.0)


def test_write_rpn_when_first_bend_at_zero():
    pm = pretty_midi.PrettyMIDI()
    curve = ControlCurve([0, 1], [0.0, 1.0])
    apply_controls.apply_controls(pm, {0: {"bend": curve}}, write_rpn=True)
    inst = pm.instruments[0]
    rpn_time = min(c.time for c in inst.control_changes if c.number == 101)
    assert rpn_time == pytest.approx(0.0)


def test_default_sample_rates():
    pm = pretty_midi.PrettyMIDI()
    curve_cc = ControlCurve([0, 1], [0, 127])
    curve_b = ControlCurve([0, 1], [0, 2])
    apply_controls.apply_controls(pm, {0: {"cc11": curve_cc, "bend": curve_b}})
    inst = pm.instruments[0]
    cc_events = [c for c in inst.control_changes if c.number == 11]
    assert len(cc_events) == 31
    assert len(inst.pitch_bends) == 121


def test_tempo_map_validation_and_conversion():
    curve = ControlCurve([0, 4], [0, 127], domain="beats", sample_rate_hz=2)
    pm = pretty_midi.PrettyMIDI()
    apply_controls.apply_controls(pm, {0: {"cc11": curve}}, tempo_map=[(0, 120), (2, 60)])
    inst = pm.instruments[0]
    times = [c.time for c in inst.control_changes]
    assert times == sorted(times)
    assert pytest.approx(times[0], abs=1e-9) == 0.0
    assert pytest.approx(times[-1], abs=1e-6) == 3.0
    pm2 = pretty_midi.PrettyMIDI()
    with pytest.raises(ValueError):
        apply_controls.apply_controls(pm2, {0: {"cc11": curve}}, tempo_map=[(1, 120), (0, 100)])


def test_caps_and_eps():
    curve = ControlCurve([0, 1], [0, 127], sample_rate_hz=100)
    pm = pretty_midi.PrettyMIDI()
    apply_controls.apply_controls(
        pm,
        {0: {"cc11": curve}},
        max_events={"cc11": 4},
        value_eps=0.0,
        time_eps=0.0,
        simplify_mode="uniform",
    )
    inst = pm.instruments[0]
    assert len(inst.control_changes) == 4
    times = [c.time for c in inst.control_changes]
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(1.0)
    for cc in inst.control_changes:
        expected = int(round(cc.time / inst.control_changes[-1].time * 127))
        assert abs(cc.value - expected) <= 1


def test_cc_generic_cap_applies():
    curve = ControlCurve([0, 1], [0, 127], sample_rate_hz=100)
    pm = pretty_midi.PrettyMIDI()
    apply_controls.apply_controls(
        pm, {0: {"cc11": curve}}, max_events={"cc": 6}, simplify_mode="uniform"
    )
    inst = pm.instruments[0]
    cc = [c for c in inst.control_changes if c.number == 11]
    assert len(cc) <= 6
    assert cc[0].time == pytest.approx(0.0)
    assert cc[-1].time == pytest.approx(1.0)


def test_bend_returns_to_zero():
    pm = pretty_midi.PrettyMIDI()
    curve = ControlCurve([0, 1], [0, 1])
    apply_controls.apply_controls(pm, {0: {"bend": curve}})
    inst = pm.instruments[0]
    assert inst.pitch_bends[-1].pitch == 0


def test_total_max_cap_keeps_endpoints():
    curve = ControlCurve([0, 1], [0, 1], sample_rate_hz=100)
    pm = pretty_midi.PrettyMIDI()
    apply_controls.apply_controls(
        pm,
        {0: {"bend": curve, "cc11": curve}},
        total_max_events=4,
    )
    inst = pm.instruments[0]
    cc = [c for c in inst.control_changes if c.number == 11]
    assert len(cc) + len(inst.pitch_bends) <= 4
    assert cc[0].time == pytest.approx(0.0)
    assert cc[-1].time == pytest.approx(1.0)
    assert inst.pitch_bends[0].time == pytest.approx(0.0)
    assert inst.pitch_bends[-1].time == pytest.approx(1.0)


def test_total_cap_proportional():
    curve = ControlCurve([0, 1], [0, 1], sample_rate_hz=100)
    pm = pretty_midi.PrettyMIDI()
    apply_controls.apply_controls(
        pm,
        {0: {"cc11": curve}, 1: {"bend": curve}},
        total_max_events=6,
    )
    inst0, inst1 = pm.instruments
    cc = [c for c in inst0.control_changes if c.number == 11]
    bends = inst1.pitch_bends
    assert len(cc) + len(bends) <= 6
    assert len(cc) >= 2 and len(bends) >= 2
    assert abs(len(cc) - len(bends)) <= 2


@pytest.mark.parametrize(
    "val,expected",
    [(-1.0, pb_math.PB_MIN), (-0.5, -4096), (0.0, 0), (0.5, 4096), (1.0, pb_math.PB_MAX)],
)
def test_to_pitch_bend_normalized_scaling(val, expected):
    curve = ControlCurve([0.0, 1.0], [0.0, val], units="normalized", ensure_zero_at_edges=False)
    inst = pretty_midi.Instrument(program=0)
    curve.to_pitch_bend(inst, units="normalized")
    assert inst.pitch_bends[-1].pitch == expected


@pytest.mark.parametrize(
    "val,expected",
    [(-1.0, pb_math.PB_MIN), (-0.5, -4096), (0.0, 0), (0.5, 4096), (1.0, pb_math.PB_MAX)],
)
def test_convert_to_14bit_normalized_scaling(val, expected):
    res = ControlCurve.convert_to_14bit([val], 2.0, units="normalized")
    assert res == [expected]
