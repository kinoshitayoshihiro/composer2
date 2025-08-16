import json
from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")

from utilities import apply_controls  # noqa: E402
from utilities.apply_controls import write_bend_range_rpn  # noqa: E402
from utilities.controls_spline import ControlCurve  # noqa: E402


def _write_basic_routing(tmp_path: Path) -> tuple[Path, Path]:
    mid = pretty_midi.PrettyMIDI()
    mid_path = tmp_path / "in.mid"
    mid.write(mid_path)
    curve_path = tmp_path / "cc.json"
    with curve_path.open("w") as fh:
        json.dump({"domain": "time", "knots": [[0, 0], [1, 127]]}, fh)
    routing = {"0": {"cc11": str(curve_path)}}
    routing_path = tmp_path / "routing.json"
    with routing_path.open("w") as fh:
        json.dump(routing, fh)
    return mid_path, routing_path


def test_cli_dry_run(tmp_path, capsys):
    mid_path, routing_path = _write_basic_routing(tmp_path)
    apply_controls.main(
        [
            str(mid_path),
            str(routing_path),
            "--dry-run",
            "--out",
            str(tmp_path / "out.mid"),
        ]
    )
    assert not (tmp_path / "out.mid").exists()
    assert "Applied controls" in capsys.readouterr().out


def test_rpn_once_and_lsb_modes():
    curve = ControlCurve([0, 1], [0.0, 1.0])
    pm = pretty_midi.PrettyMIDI()
    apply_controls.apply_controls(
        pm,
        {0: {"bend": curve}},
        write_rpn=True,
        bend_range_semitones=2.5,
        send_rpn_null=False,
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
        send_rpn_null=False,
    )
    assert rpn_cc.count((101, 0)) == 1

    pm2 = pretty_midi.PrettyMIDI()
    apply_controls.apply_controls(
        pm2,
        {0: {"bend": curve}},
        write_rpn=True,
        bend_range_semitones=2.5,
        lsb_mode="cents",
    )
    inst2 = pm2.instruments[0]
    lsb2 = next(c.value for c in inst2.control_changes if c.number == 38)
    assert lsb2 == 50
    assert any(c.number == 101 and c.value == 127 for c in inst2.control_changes)


@pytest.mark.parametrize("rng", [0.5, 1.0, 2.0, 12.0, 24.0])
def test_rpn_range_values(rng):
    inst = pretty_midi.Instrument(program=0)
    write_bend_range_rpn(inst, rng)
    msb = int(rng)
    lsb = int(round((rng - msb) * 128))
    pairs = [(c.number, c.value) for c in inst.control_changes]
    assert (6, msb) in pairs
    assert (38, lsb) in pairs
    inst2 = pretty_midi.Instrument(program=0)
    write_bend_range_rpn(inst2, rng, lsb_mode="cents")
    lsb_c = int(round((rng - msb) * 100))
    pairs2 = [(c.number, c.value) for c in inst2.control_changes]
    assert (6, msb) in pairs2
    assert (38, max(0, min(127, lsb_c))) in pairs2


def test_rpn_existing_null_idempotent():
    inst = pretty_midi.Instrument(program=0)
    inst.control_changes.append(
        pretty_midi.ControlChange(number=101, value=127, time=0.0)
    )
    inst.control_changes.append(
        pretty_midi.ControlChange(number=100, value=127, time=0.0)
    )
    inst.pitch_bends.append(pretty_midi.PitchBend(pitch=0, time=1.0))
    write_bend_range_rpn(inst, 2.0, t=1.0)
    write_bend_range_rpn(inst, 2.0, t=1.0)
    pairs = [(c.number, c.value) for c in inst.control_changes]
    assert pairs.count((101, 0)) == 1
    assert pairs.count((100, 0)) == 1
    first_bend = inst.pitch_bends[0].time
    rpn_time = min(
        c.time for c in inst.control_changes if (c.number, c.value) == (101, 0)
    )
    assert rpn_time <= first_bend


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
    apply_controls.apply_controls(
        pm, {0: {"cc11": curve}}, tempo_map=[(0, 120), (2, 60)]
    )
    inst = pm.instruments[0]
    times = [c.time for c in inst.control_changes]
    assert times == sorted(times)
    assert pytest.approx(times[0], abs=1e-9) == 0.0
    assert pytest.approx(times[-1], abs=1e-6) == 3.0
    pm2 = pretty_midi.PrettyMIDI()
    with pytest.raises(ValueError):
        apply_controls.apply_controls(
            pm2, {0: {"cc11": curve}}, tempo_map=[(1, 120), (0, 100)]
        )


def test_caps_and_eps():
    curve = ControlCurve([0, 1], [0, 127], sample_rate_hz=100)
    pm = pretty_midi.PrettyMIDI()
    apply_controls.apply_controls(
        pm,
        {0: {"cc11": curve}},
        max_events={"cc11": 4},
        value_eps=0.0,
        time_eps=0.0,
    )
    inst = pm.instruments[0]
    assert len(inst.control_changes) == 4
    times = [c.time for c in inst.control_changes]
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(1.0)
    for cc in inst.control_changes:
        expected = int(round(cc.time / inst.control_changes[-1].time * 127))
        assert abs(cc.value - expected) <= 1
