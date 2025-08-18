import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    from ._stubs import pretty_midi  # type: ignore


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="integration test")
def test_apply_controls_cli_roundtrip(tmp_path: Path):
    in_mid = tmp_path / "in.mid"
    pm = pretty_midi.PrettyMIDI()
    pm.write(str(in_mid))
    curve = {"domain": "time", "knots": [[0.0, 0.0], [1.0, 1.0]]}
    curve_path = tmp_path / "cc.json"
    curve_path.write_text(json.dumps(curve))
    routing = {"0": {"cc11": str(curve_path)}}
    routing_path = tmp_path / "routing.json"
    routing_path.write_text(json.dumps(routing))
    out_mid = tmp_path / "out.mid"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "utilities.apply_controls",
            str(in_mid),
            str(routing_path),
            "--out",
            str(out_mid),
        ],
        check=True,
    )
    pm2 = pretty_midi.PrettyMIDI(str(out_mid))
    assert any(c.number == 11 for inst in pm2.instruments for c in inst.control_changes)
