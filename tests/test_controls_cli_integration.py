import importlib
import json
from pathlib import Path

try:
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from ._stubs import pretty_midi  # type: ignore

apply_controls_cli = importlib.import_module("utilities.apply_controls_cli")


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    mid = pretty_midi.PrettyMIDI()
    mid_path = tmp_path / "in.mid"
    mid.write(mid_path)
    curves = {
        "bend": {"domain": "time", "knots": [[0.0, 0.0], [1.0, 0.5]]},
        "cc11": {"domain": "time", "knots": [[0.0, 0.0], [1.0, 127.0]]},
    }
    curves_path = tmp_path / "curves.json"
    with curves_path.open("w") as fh:
        json.dump(curves, fh)
    return mid_path, curves_path


def test_cli_basic(tmp_path: Path):
    mid, curves = _prepare(tmp_path)
    pm = apply_controls_cli.main(
        [
            str(mid),
            "--curves",
            str(curves),
            "--controls",
            "bend:on,cc11:on",
            "--bend-range-semitones",
            "2.0",
            "--write-rpn",
            "--dry-run",
        ]
    )
    assert pm.instruments, "no instruments returned"
    inst = pm.instruments[0]
    assert any(cc.number == 11 for cc in inst.control_changes)
    assert inst.pitch_bends
    # RPN written at most once and not after bends
    nums = [c.number for c in inst.control_changes]
    assert nums.count(101) == 1
    assert inst.control_changes[0].time <= inst.pitch_bends[0].time
