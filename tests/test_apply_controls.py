from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

pretty_midi = pytest.importorskip("pretty_midi")

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
run_cli = apply_module.main


def test_apply_controls_rpn_and_limits():
    pm = pretty_midi.PrettyMIDI()

    c1 = ControlCurve(target="cc11", knots=[(0.0, 0.0), (1.0, 127.0)])
    c64 = ControlCurve(target="cc64", knots=[(0.0, 0.0), (1.0, 127.0)])
    c2 = ControlCurve(target="bend", knots=[(0.0, 0.0), (1.0, 1.0)])
    c3 = ControlCurve(target="bend", knots=[(0.0, 0.0), (1.0, -1.0)])

    routing = {0: {"cc11": c1, "cc64": c64, "bend": c2}, 2: {"bend": c3}}
    apply_controls(
        pm,
        routing,
        bend_range_semitones=2.5,
        per_target_max_events={"bend": 12},
    )

    inst0 = [i for i in pm.instruments if i.name == "channel0"][0]
    inst2 = [i for i in pm.instruments if i.name == "channel2"][0]

    def check(inst: pretty_midi.Instrument) -> None:
        nums = [(cc.number, cc.value) for cc in inst.control_changes]
        assert nums.count((101, 0)) == 1
        assert nums.count((100, 0)) == 1
        assert nums.count((6, 2)) == 1
        assert nums.count((38, 64)) == 1
        assert (101, 127) not in nums
        assert (100, 127) not in nums
        first_pb = inst.pitch_bends[0].time
        rpn_time = max(
            cc.time for cc in inst.control_changes if cc.number in {101, 100, 6, 38}
        )
        assert rpn_time <= first_pb

    check(inst0)
    check(inst2)

    assert len(inst0.pitch_bends) <= 12
    assert len(inst2.pitch_bends) <= 12

    times_cc0 = [cc.time for cc in inst0.control_changes]
    assert times_cc0 == sorted(times_cc0)
    times_pb0 = [pb.time for pb in inst0.pitch_bends]
    assert times_pb0 == sorted(times_pb0)
    times_pb2 = [pb.time for pb in inst2.pitch_bends]
    assert times_pb2 == sorted(times_pb2)

    assert all(0 <= cc.value <= 127 for cc in inst0.control_changes)


def test_apply_controls_rpn_null():
    pm = pretty_midi.PrettyMIDI()
    c1 = ControlCurve(target="bend", knots=[(0.0, 0.0), (1.0, 1.0)])
    c2 = ControlCurve(target="bend", knots=[(0.0, 0.0), (1.0, -1.0)])
    routing = {0: {"bend": c1}, 1: {"bend": c2}}
    apply_controls(pm, routing, rpn_null=True)
    for ch in (0, 1):
        inst = [i for i in pm.instruments if i.name == f"channel{ch}"][0]
        nums = [(cc.number, cc.value) for cc in inst.control_changes]
        assert nums.count((101, 0)) == 1
        assert nums.count((100, 0)) == 1
        assert nums.count((6, 2)) == 1
        assert nums.count((38, 0)) == 1
        assert nums.count((101, 127)) == 1
        assert nums.count((100, 127)) == 1
        first_pb = inst.pitch_bends[0].time
        rpn_time = max(
            cc.time for cc in inst.control_changes if cc.number in {101, 100, 6, 38}
        )
        assert rpn_time <= first_pb


def test_apply_controls_rpn_null_event_limit():
    pm = pretty_midi.PrettyMIDI()
    curve = ControlCurve(target="bend", knots=[(0.0, 0.0), (1.0, 1.0)])
    routing = {0: {"bend": curve}}
    apply_controls(
        pm,
        routing,
        rpn_null=True,
        per_target_max_events={"bend": 5},
    )
    inst = [i for i in pm.instruments if i.name == "channel0"][0]
    assert len(inst.pitch_bends) <= 5
    nums = [(cc.number, cc.value) for cc in inst.control_changes]
    assert nums.count((101, 127)) == 1
    assert nums.count((100, 127)) == 1


def test_beats_domain_apply_controls_with_tempo_map():
    pm = pretty_midi.PrettyMIDI()

    curve = ControlCurve(
        target="cc11", domain="beats", knots=[(0.0, 0.0), (1.0, 127.0)]
    )
    routing = {0: {"cc11": curve}}

    def tempo_map(_b: float) -> float:
        return 120.0

    apply_controls(pm, routing, tempo_map=tempo_map)
    inst = [i for i in pm.instruments if i.name == "channel0"][0]
    times = [cc.time for cc in inst.control_changes if cc.number == 11]
    assert times == sorted(times)
    assert pytest.approx(times[0], abs=1e-9) == 0.0
    assert pytest.approx(times[-1], abs=1e-6) == 0.5


def test_rpn_placed_before_delayed_bend():
    pm = pretty_midi.PrettyMIDI()
    curve = ControlCurve(target="bend", knots=[(1.0, 0.0), (2.0, 1.0)])
    apply_controls(pm, {0: {"bend": curve}})
    inst = [i for i in pm.instruments if i.name == "channel0"][0]
    first_pb = inst.pitch_bends[0].time
    rpn_time = max(
        cc.time for cc in inst.control_changes if cc.number in {101, 100, 6, 38}
    )
    assert first_pb > 0.0
    assert rpn_time < first_pb
    assert pytest.approx(first_pb - rpn_time, abs=1e-6) == 0.005


def test_cli_roundtrip(tmp_path):
    infile = tmp_path / "in.mid"
    shutil.copy(ROOT / "groove.mid", infile)

    curve_data = {"domain": "time", "knots": [[0.0, 0.0], [1.0, 1.0]]}
    expr = tmp_path / "expr.json"
    pedal = tmp_path / "pedal.json"
    bend0 = tmp_path / "bend0.json"
    bend1 = tmp_path / "bend1.json"
    for path in (expr, pedal, bend0, bend1):
        with path.open("w") as fh:
            json.dump(curve_data, fh)

    routing = {
        "0": {"cc11": str(expr), "cc64": str(pedal), "bend": str(bend0)},
        "1": {"bend": str(bend1)},
    }
    routing_json = tmp_path / "routing.json"
    routing_json.write_text(json.dumps(routing))
    outfile = tmp_path / "out.mid"

    pm_out = run_cli(
        [
            str(infile),
            str(routing_json),
            "--out",
            str(outfile),
            "--max-bend",
            "2",
            "--max-cc11",
            "2",
            "--max-cc64",
            "2",
            "--dry-run",
        ]
    )

    inst0 = [i for i in pm_out.instruments if i.name == "channel0"][0]
    inst1 = [i for i in pm_out.instruments if i.name == "channel1"][0]
    assert len([cc for cc in inst0.control_changes if cc.number == 11]) == 2
    assert len([cc for cc in inst0.control_changes if cc.number == 64]) == 2
    assert len(inst0.pitch_bends) <= 2
    assert len(inst1.pitch_bends) <= 2
